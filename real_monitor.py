"""
实盘仓位监控 — 每日检查止损/止盈/调仓信号

用法: python real_monitor.py          查看当前持仓状态
      python real_monitor.py --update  更新持仓（调仓后使用）
"""
import sys
import io
import json
import os
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

POSITION_FILE = "real_positions.json"
STOP_LOSS = 0.92       # 跌破成本92% → 止损
TAKE_PROFIT = 1.25     # 突破成本125% → 止盈
TRAILING_STOP = 0.12   # 从高点回撤12% → 移动止盈


def get_prices(codes: list[str]) -> dict[str, dict]:
    """获取实时价格"""
    import data_source as ds
    try:
        quotes = ds.get_realtime_quotes(codes)
        if quotes.empty:
            return {}
        result = {}
        for _, r in quotes.iterrows():
            code = str(r["code"])
            result[code] = {
                "name": str(r.get("name", "")),
                "price": float(r.get("price", 0)),
                "pct_chg": float(r.get("pct_chg", 0)) if r.get("pct_chg") else 0,
                "pre_close": float(r.get("pre_close", 0)),
            }
        return result
    except Exception as e:
        print(f"  [错误] 获取行情失败: {e}")
        return {}


def load_positions() -> dict:
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_positions(positions: dict):
    with open(POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def init_positions(entries: list[dict]):
    """初始化/更新持仓
    entries: [{"code": "000789", "shares": 7700, "entry_price": 4.42, "entry_date": "2026-05-29"}, ...]
    """
    pos = {}
    for e in entries:
        code = e["code"]
        pos[code] = {
            "shares": e["shares"],
            "entry_price": e["entry_price"],
            "entry_date": e.get("entry_date", ""),
            "peak_price": e.get("peak_price", e["entry_price"]),
        }
    save_positions(pos)
    print(f"  已更新 {len(pos)} 只持仓")


def check_positions(positions: dict, prices: dict) -> list[str]:
    """检查每只股票的触发性信号，返回警报列表"""
    alerts = []

    for code, pos in positions.items():
        info = prices.get(code)
        if not info or info["price"] <= 0:
            alerts.append(f"[数据缺失] {code} 无实时价格，请手动检查")
            continue

        current = info["price"]
        entry = pos["entry_price"]
        peak = pos.get("peak_price", entry)
        shares = pos["shares"]

        # 更新最高价
        if current > peak:
            pos["peak_price"] = current

        pnl_pct = (current / entry - 1) * 100
        peak_dd = (current / peak - 1) * 100

        # 止损
        if current <= entry * STOP_LOSS:
            loss = int(shares * (current - entry))
            alerts.append(
                f"[止损!!] {code} {info['name']} 现价{current:.2f} 跌破成本{entry:.2f}的" +
                f"{(1-STOP_LOSS)*100:.0f}% → 立即卖出{shares}股，亏损约{loss}元"
            )

        # 止盈
        elif current >= entry * TAKE_PROFIT:
            gain = int(shares * (current - entry))
            alerts.append(
                f"[止盈!!] {code} {info['name']} 现价{current:.2f} 突破{TAKE_PROFIT*100:.0f}%目标" +
                f" → 卖出{shares}股，盈利约{gain}元"
            )

        # 移动止盈（已有浮盈后从高点回撤12%）
        elif peak > entry * 1.05 and peak_dd <= -TRAILING_STOP * 100:
            gain = int(shares * (current - entry))
            alerts.append(
                f"[移动止盈] {code} {info['name']} 从高点{peak:.2f}回撤{abs(peak_dd):.1f}%" +
                f" → 建议卖出{shares}股，盈利约{gain}元"
            )

        # 接近止损预警
        elif current <= entry * 0.95:
            alerts.append(
                f"[预警] {code} {info['name']} 浮亏{pnl_pct:+.1f}%，距止损线{(1-STOP_LOSS)*100:.0f}%仅差{abs(-5-pnl_pct):.1f}%"
            )

    return alerts


def print_status(positions: dict, prices: dict):
    """打印完整持仓状态"""
    if not positions:
        print("\n  暂无持仓数据，请先 python real_monitor.py --update")
        return

    total_cost = 0
    total_market = 0
    alerts = check_positions(positions, prices)

    print(f"\n{'=' * 65}")
    print(f"  实盘监控 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 65}")
    print(f"  {'代码':<8} {'名称':<10} {'股数':>6} {'成本':>8} {'现价':>8} {'市值':>10} {'盈亏':>10}")
    print(f"  {'-' * 62}")

    for code, pos in positions.items():
        info = prices.get(code, {})
        price = info.get("price", 0)
        name = info.get("name", "")
        market_val = pos["shares"] * price if price > 0 else 0
        cost_val = pos["shares"] * pos["entry_price"]
        pnl = (price / pos["entry_price"] - 1) * 100 if pos["entry_price"] > 0 else 0
        total_cost += cost_val
        total_market += market_val
        pnl_str = f"{pnl:+.2f}%" if price > 0 else "N/A"
        print(f"  {code:<8} {name:<10} {pos['shares']:>6} {pos['entry_price']:>8.2f} {price:>8.2f} {market_val:>10,.0f} {pnl_str:>10}")

    total_pnl = total_market - total_cost
    print(f"  {'-' * 62}")
    print(f"  总成本: ¥{total_cost:,.0f}  总市值: ¥{total_market:,.0f}  总盈亏: ¥{total_pnl:+,.0f} ({total_pnl/total_cost*100:+.2f}%)" if total_cost > 0 else "")

    if alerts:
        print(f"\n  === 触发性警报 ===")
        for a in alerts:
            print(f"  {a}")
    else:
        print(f"\n  [OK] 无触发性信号，持仓正常")

    print(f"{'=' * 65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=str, default="", help="更新持仓 (JSON格式文件)")
    parser.add_argument("--init-swap", action="store_true", help="初始化6只主板持仓")
    args = parser.parse_args()

    if args.init_swap:
        # 基于最新信号初始化：6只主板，等权约3.4万
        init_positions([
            {"code": "600138", "shares": 4400, "entry_price": 7.73, "entry_date": "2026-05-29"},
            {"code": "000789", "shares": 7600, "entry_price": 4.48, "entry_date": "2026-05-29"},
            {"code": "605368", "shares": 4900, "entry_price": 6.92, "entry_date": "2026-05-29"},
            {"code": "603348", "shares": 2100, "entry_price": 16.15, "entry_date": "2026-05-29"},
            {"code": "000096", "shares": 4200, "entry_price": 8.15, "entry_date": "2026-05-29"},
            {"code": "001390", "shares": 1900, "entry_price": 17.54, "entry_date": "2026-05-29"},
        ])
    elif args.update:
        with open(args.update, "r", encoding="utf-8") as f:
            init_positions(json.load(f))
    else:
        positions = load_positions()
        if positions:
            codes = list(positions.keys())
            prices = get_prices(codes)
            print_status(positions, prices)
        else:
            print("\n  尚未初始化实盘持仓")
            print("  调仓后用: python real_monitor.py --init-swap")
