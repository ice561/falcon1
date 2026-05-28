"""
纸上交易模拟 — 真实环境延迟、成交、成本模拟

核心功能:
  1. 每日跟踪虚拟持仓
  2. T+1 延迟执行
  3. 模拟成交价 = 次日开盘价 (含滑点)
  4. 涨跌停/停牌无法交易
  5. 每日快照保存

用法:
  python paper_trade.py --init        # 初始化账户
  python paper_trade.py --update      # 更新当日净值
  python paper_trade.py --signal      # 根据最新信号调仓
  python paper_trade.py --report      # 查看当前持仓和绩效
"""
import os
import sys
import io
import json
import argparse
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import data_source as ds

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

STATE_FILE = "./paper_trade/state.pkl"
LOG_FILE = "./paper_trade/daily_log.csv"
TRADE_LOG = "./paper_trade/trade_log.csv"
SIGNAL_DIR = "."
os.makedirs("./paper_trade", exist_ok=True)

INITIAL_CAPITAL = 1_000_000  # 初始资金100万
COMMISSION = 0.0003
SLIPPAGE = 0.0002
STAMP_TAX = 0.001  # 卖出印花税


@dataclass
class PaperAccount:
    """虚拟账户"""
    cash: float = INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)  # {code: {"shares": int, "entry_price": float, "entry_date": str, "peak": float}}
    total_invested: float = 0  # 累计投入
    created_at: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": self.positions,
            "total_invested": self.total_invested,
            "created_at": self.created_at,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperAccount":
        return cls(
            cash=d.get("cash", INITIAL_CAPITAL),
            positions=d.get("positions", {}),
            total_invested=d.get("total_invested", 0),
            created_at=d.get("created_at", ""),
            last_update=d.get("last_update", ""),
        )


# ============================================================
# 状态管理
# ============================================================

def load_account() -> PaperAccount:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            data = pickle.load(f)
        if "cash" in data:
            return PaperAccount.from_dict(data)
    return PaperAccount(created_at=datetime.now().strftime("%Y-%m-%d"))


def save_account(acc: PaperAccount):
    acc.last_update = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(STATE_FILE, "wb") as f:
        pickle.dump(acc.to_dict(), f)


def log_daily_snapshot(acc: PaperAccount, prices: dict[str, float]):
    """记录每日快照到CSV"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    mv = get_total_market_value(acc, prices)
    total_value = acc.cash + mv
    pos_count = len(acc.positions)

    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if not exists:
            f.write("date,cash,market_value,total,positions_count\n")
        f.write(f"{date_str},{acc.cash:.2f},{mv:.2f},{total_value:.2f},{pos_count}\n")

    print(f"  日志已记录: {LOG_FILE}")


# ============================================================
# 估值和交易
# ============================================================

def get_realtime_price(code: str) -> float | None:
    """获取实时价格（盘中）或最新收盘价（盘后）"""
    try:
        df = ds.get_realtime_quotes([code])
        if df.empty:
            return None
        price = float(df.iloc[0]["price"])
        return price if price > 0 else None
    except Exception:
        return None


def get_batch_prices(codes: list[str]) -> dict[str, float]:
    """批量获取价格"""
    if not codes:
        return {}
    try:
        df = ds.get_realtime_quotes(codes)
        if df.empty:
            return {}
        prices = {}
        for _, row in df.iterrows():
            code = str(row["code"])
            p = float(row["price"])
            if p > 0:
                prices[code] = p
        return prices
    except Exception:
        return {}


def get_total_market_value(acc: PaperAccount, prices: dict[str, float]) -> float:
    """计算持仓总市值"""
    mv = 0.0
    for code, pos in acc.positions.items():
        p = prices.get(code)
        if p is None:
            p = pos.get("last_price", pos["entry_price"])
        mv += pos["shares"] * p
    return mv


def get_total_value(acc: PaperAccount, prices: dict[str, float]) -> float:
    """总资产 = 现金 + 持仓市值"""
    return acc.cash + get_total_market_value(acc, prices)


# ============================================================
# 交易模拟（真实约束）
# ============================================================

def is_tradable(code: str) -> tuple[bool, str]:
    """
    检查是否可交易
    返回: (可交易, 原因)
    """
    try:
        df = ds.get_realtime_quotes([code])
        if df.empty:
            return False, "无行情数据"
        row = df.iloc[0]
        name = str(row.get("name", ""))
        price = float(row["price"])
        pre_close = float(row.get("pre_close", 0))
        volume = float(row.get("volume", 0))
        amount = float(row.get("amount", 0))

        if "ST" in name or "*ST" in name:
            return False, "ST股票"

        if price <= 0:
            return False, "停牌/无价格"

        if pre_close > 0:
            pct_chg = (price - pre_close) / pre_close
            if pct_chg >= 0.098:
                return False, f"涨停 ({pct_chg:.1%})"
            if pct_chg <= -0.098:
                return False, f"跌停 ({pct_chg:.1%})"

        if amount < 10_000_000:
            return False, f"成交额过低 ({amount/1e4:.0f}万)"

        return True, "可交易"
    except Exception as e:
        return False, f"检查异常: {e}"


def execute_buy(acc: PaperAccount, code: str, weight: float, prices: dict[str, float]) -> dict:
    """
    执行买入（模拟）

    T+1: 今天下单，明天以开盘价成交
    简化: 使用当前价格模拟（实际应延迟一天）
    """
    result = {"code": code, "action": "BUY", "status": "FAILED"}

    current_price = prices.get(code)
    if current_price is None or current_price <= 0:
        result["reason"] = "无有效价格"
        return result

    # 成交价 = 当前价 × (1 + 滑点)，再加佣金
    fill_price = current_price * (1 + SLIPPAGE)

    # 可买入金额
    total_value = get_total_value(acc, prices)
    allocate = total_value * weight
    cost_per_share = fill_price * (1 + COMMISSION)

    if cost_per_share <= 0:
        result["reason"] = "成本计算异常"
        return result

    shares = int(allocate / cost_per_share / 100) * 100  # 手

    if shares < 100:
        result["reason"] = f"资金不足买入1手 (需{cost_per_share * 100:.0f}, 可用{allocate:.0f})"
        result["shares"] = 0
        return result

    total_cost = shares * cost_per_share
    if total_cost > acc.cash:
        shares = int(acc.cash / cost_per_share / 100) * 100
        total_cost = shares * cost_per_share

    if shares < 100:
        result["reason"] = "现金不足"
        result["shares"] = 0
        return result

    acc.cash -= total_cost
    acc.total_invested += total_cost

    date_str = datetime.now().strftime("%Y-%m-%d")

    if code in acc.positions:
        # 加仓: 更新平均成本
        old = acc.positions[code]
        total_shares = old["shares"] + shares
        avg_price = (old["entry_price"] * old["shares"] + fill_price * shares) / total_shares
        acc.positions[code] = {
            "shares": total_shares,
            "entry_price": avg_price,
            "entry_date": old["entry_date"],
            "peak": max(old["peak"], fill_price),
        }
    else:
        acc.positions[code] = {
            "shares": shares,
            "entry_price": fill_price,
            "entry_date": date_str,
            "peak": fill_price,
        }

    result["status"] = "SUCCESS"
    result["shares"] = shares
    result["fill_price"] = fill_price
    result["cost"] = total_cost
    result["weight"] = f"{total_cost / total_value:.2%}"

    return result


def execute_sell(acc: PaperAccount, code: str, shares: int | None = None,
                 prices: dict[str, float] | None = None) -> dict:
    """
    执行卖出

    shares=None 表示清仓
    """
    result = {"code": code, "action": "SELL", "status": "FAILED"}

    if code not in acc.positions:
        result["reason"] = "未持仓"
        return result

    pos = acc.positions[code]
    if shares is None or shares >= pos["shares"]:
        shares = pos["shares"]

    current_price = None
    if prices:
        current_price = prices.get(code)
    if current_price is None:
        current_price = get_realtime_price(code)
    if current_price is None or current_price <= 0:
        result["reason"] = "无有效价格"
        return result

    # 卖出: 价格 × (1 - 滑点) - 佣金 - 印花税
    fill_price = current_price * (1 - SLIPPAGE)
    proceeds = shares * fill_price * (1 - COMMISSION - STAMP_TAX)

    if shares >= pos["shares"]:
        del acc.positions[code]
    else:
        pos["shares"] -= shares

    acc.cash += proceeds

    # 记录到交易日志
    pnl_pct = (fill_price / pos["entry_price"] - 1) * 100
    log_trade(code, "SELL", shares, pos["entry_price"], fill_price, pnl_pct)

    result["status"] = "SUCCESS"
    result["shares"] = shares
    result["fill_price"] = fill_price
    result["proceeds"] = proceeds
    result["pnl_pct"] = f"{pnl_pct:+.2f}%"

    return result


def log_trade(code: str, action: str, shares: int, entry_price: float,
              exit_price: float, pnl_pct: float):
    """记录交易到CSV"""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", encoding="utf-8") as f:
        if not exists:
            f.write("datetime,code,action,shares,entry_price,exit_price,pnl_pct\n")
        f.write(f"{date_str},{code},{action},{shares},{entry_price:.2f},{exit_price:.2f},{pnl_pct:.2f}\n")


# ============================================================
# 信号调仓
# ============================================================

def rebalance_from_signals(acc: PaperAccount, signal_file: str = None):
    """
    根据信号文件调仓
    信号文件: signals_YYYYMMDD.json
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 找最新信号文件
    if signal_file is None:
        json_files = sorted([f for f in os.listdir(SIGNAL_DIR) if f.startswith("signals_") and f.endswith(".json")])
        if not json_files:
            print("[错误] 未找到信号文件，请先运行 daily_signals.py")
            return
        signal_file = json_files[-1]

    print(f"\n  读取信号: {signal_file}")

    with open(signal_file, "r", encoding="utf-8") as f:
        signal_data = json.load(f)

    signals = signal_data.get("signals", [])
    if not signals:
        print("  信号为空，清仓")
        # 清仓
        prices = get_batch_prices(list(acc.positions.keys()))
        for code in list(acc.positions.keys()):
            execute_sell(acc, code, prices=prices)
        print("  已清仓")
        save_account(acc)
        return

    target_codes = {s["代码"] for s in signals}
    current_codes = set(acc.positions.keys())

    # 收集所有需要的价格
    all_codes = list(target_codes | current_codes)
    prices = get_batch_prices(all_codes)

    print(f"\n{'=' * 55}")
    print(f"  调仓执行 — {date_str}")
    print(f"{'=' * 55}")
    print(f"  当前持仓: {len(current_codes)} 只")
    print(f"  目标持仓: {len(target_codes)} 只")

    # 1. 卖出不在新信号中的
    to_sell = current_codes - target_codes
    for code in to_sell:
        r = execute_sell(acc, code, prices=prices)
        status = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"  卖出 {code}: {status} ({r.get('pnl_pct', 'N/A')})")

    # 2. 检查是否需要卖出（止损/止盈）
    for code in list(acc.positions.keys()):
        if code in target_codes:
            pos = acc.positions[code]
            price = prices.get(code)
            if price:
                # 更新peak
                if price > pos["peak"]:
                    pos["peak"] = price

                # 止损
                if price < pos["entry_price"] * 0.92:
                    r = execute_sell(acc, code, prices=prices)
                    print(f"  止损 {code}: {r.get('pnl_pct', 'N/A')}")

                # 移动止盈（从高点回撤12%）
                elif (pos["peak"] > pos["entry_price"] * 1.05 and
                      price < pos["peak"] * 0.88):
                    r = execute_sell(acc, code, prices=prices)
                    print(f"  移动止盈 {code}: {r.get('pnl_pct', 'N/A')}")

    # 3. 买入新信号
    remaining_codes = [c for c in signals if c["代码"] in target_codes]
    active_in_positions = len(acc.positions)
    max_new = 15 - active_in_positions

    to_buy = [c for c in remaining_codes if c["代码"] not in acc.positions]
    to_buy = to_buy[:max_new]

    if to_buy:
        weight_per_stock = 1.0 / (active_in_positions + len(to_buy))
        for s in to_buy:
            code = s["代码"]
            tradable, reason = is_tradable(code)
            if not tradable:
                print(f"  跳过 {code}: {reason}")
                continue
            r = execute_buy(acc, code, weight_per_stock, prices)
            status = "✓" if r["status"] == "SUCCESS" else "✗"
            print(f"  买入 {code}: {status} ({r.get('shares', 0)}股 @ {r.get('fill_price', 0):.2f})")

    save_account(acc)
    print(f"\n  调仓完成。现金: {acc.cash:,.0f}")


# ============================================================
# 报告
# ============================================================

def print_report(acc: PaperAccount):
    """打印当前持仓报告"""
    prices = get_batch_prices(list(acc.positions.keys())) if acc.positions else {}
    total_value = get_total_value(acc, prices)
    mv = get_total_market_value(acc, prices)
    pnl = total_value - INITIAL_CAPITAL

    print(f"\n{'=' * 60}")
    print(f"  纸上交易 — 账户报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'=' * 60}")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"  现金:     {acc.cash:,.0f}")
    print(f"  持仓市值: {mv:,.0f}")
    print(f"  总资产:   {total_value:,.0f}")
    print(f"  累计盈亏: {pnl:+,.0f} ({pnl / INITIAL_CAPITAL:+.2%})")
    print(f"  持仓数量: {len(acc.positions)}")

    if acc.positions:
        print(f"\n{'=' * 60}")
        print(f"  持仓明细")
        print(f"{'=' * 60}")
        print(f"  {'代码':<8} {'股数':>8} {'成本':>8} {'现价':>8} {'市值':>10} {'盈亏':>10}")
        print(f"  {'-' * 54}")

        for code, pos in sorted(acc.positions.items()):
            p = prices.get(code)
            if p is None:
                p = pos.get("last_price", pos["entry_price"])
            shares = pos["shares"]
            cost = pos["entry_price"]
            market_val = shares * p
            pnl_pct = (p / cost - 1) * 100
            print(f"  {code:<8} {shares:>8} {cost:>8.2f} {p:>8.2f} {market_val:>10,.0f} {pnl_pct:>+9.2f}%")

    # 绩效摘要
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        if len(log_df) > 1:
            print(f"\n{'=' * 60}")
            print(f"  每日净值摘要 (最近5日)")
            print(f"{'=' * 60}")
            recent = log_df.tail(5)
            for _, row in recent.iterrows():
                print(f"  {row['date']}: {row['total']:,.0f}")
            if len(log_df) >= 5:
                first = log_df.iloc[0]["total"]
                last = log_df.iloc[-1]["total"]
                if first > 0:
                    period_ret = (last / first - 1) * 100
                    print(f"  累计收益: {period_ret:+.2f}%")

    print(f"{'=' * 60}")


def print_risk_alerts(acc: PaperAccount):
    """风险预警检查"""
    prices = get_batch_prices(list(acc.positions.keys())) if acc.positions else {}
    alerts = []

    # 1. 个股止损预警
    for code, pos in acc.positions.items():
        p = prices.get(code)
        if p is None:
            continue
        dd = (p - pos["entry_price"]) / pos["entry_price"]
        if dd < -0.07:
            alerts.append(f"[WARN] {code} 浮亏 {dd:.1%}，接近止损线 -8%")

        if pos["entry_price"] > 0:
            peak_dd = (p - pos["peak"]) / pos["peak"]
            if pos["peak"] > pos["entry_price"] * 1.05 and peak_dd < -0.10:
                alerts.append(f"[WARN] {code} 从高点回撤 {peak_dd:.1%}，接近移动止盈 -12%")

    # 2. 组合回撤预警
    total_value = get_total_value(acc, prices)
    total_dd = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    if total_dd < -0.10:
        alerts.append(f"[ALERT] 组合回撤 {total_dd:.1%}，超过 -10% 警戒线")

    # 3. 单行业集中度
    sector_count = {}
    try:
        from a_share_strategy import get_board
    except ImportError:
        get_board = lambda c: "未知"
    for code in acc.positions:
        sector = get_board(code)
        sector_count[sector] = sector_count.get(sector, 0) + 1
    for sec, count in sector_count.items():
        if count >= 3:
            alerts.append(f"[INFO] {sec} 持仓 {count} 只（上限3只）")

    if alerts:
        print(f"\n{'=' * 60}")
        print(f"  风险预警")
        print(f"{'=' * 60}")
        for a in alerts:
            print(f"  {a}")
    else:
        print(f"\n  [OK] 无风险预警")

    print(f"{'=' * 60}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猎鹰一号 — 纸上交易")
    parser.add_argument("--init", action="store_true", help="初始化账户")
    parser.add_argument("--update", action="store_true", help="更新当日净值")
    parser.add_argument("--signal", type=str, default=None, help="根据信号调仓")
    parser.add_argument("--report", action="store_true", help="查看持仓报告")
    parser.add_argument("--alerts", action="store_true", help="风险预警检查")
    args = parser.parse_args()

    if args.init:
        if os.path.exists(STATE_FILE):
            r = input("账户已存在，是否重置？(y/n): ")
            if r.lower() != "y":
                print("已取消")
                sys.exit(0)
        acc = PaperAccount(created_at=datetime.now().strftime("%Y-%m-%d"))
        save_account(acc)
        print(f"账户已初始化: {INITIAL_CAPITAL:,} 元")
        print(f"状态文件: {STATE_FILE}")

    elif args.signal is not None or args.signal == "":
        acc = load_account()
        rebalance_from_signals(acc, args.signal if args.signal else None)

    elif args.update:
        acc = load_account()
        prices = get_batch_prices(list(acc.positions.keys()))
        log_daily_snapshot(acc, prices)
        print_report(acc)

    elif args.alerts:
        acc = load_account()
        print_risk_alerts(acc)

    elif args.report:
        acc = load_account()
        print_report(acc)
        print_risk_alerts(acc)

    else:
        # 默认: 显示报告
        acc = load_account()
        print_report(acc)
        print_risk_alerts(acc)
