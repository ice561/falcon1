"""
监控仪表盘 — 持仓盈亏、信号日历、风险一览

用法:
  python dashboard.py               # 终端仪表盘
  python dashboard.py --html        # 生成HTML报告
  python dashboard.py --serve       # 启动简易HTTP服务(需Python http.server)
"""
import os
import sys
import io
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PAPER_TRADE_DIR = "./paper_trade"
STATE_FILE = os.path.join(PAPER_TRADE_DIR, "state.pkl")
LOG_FILE = os.path.join(PAPER_TRADE_DIR, "daily_log.csv")
TRADE_LOG = os.path.join(PAPER_TRADE_DIR, "trade_log.csv")


# ============================================================
# 数据加载
# ============================================================

def load_account():
    import pickle
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "rb") as f:
        return pickle.load(f)


def load_log() -> pd.DataFrame | None:
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def load_trades() -> pd.DataFrame | None:
    if not os.path.exists(TRADE_LOG):
        return None
    df = pd.read_csv(TRADE_LOG)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ============================================================
# 终端仪表盘
# ============================================================

def terminal_dashboard():
    """终端彩色仪表盘"""
    acc = load_account()
    log = load_log()
    trades = load_trades()

    # ANSI颜色
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    now = datetime.now()

    # ===== 顶部标题栏 =====
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  猎鹰一号 — 监控仪表盘{' ' * 37}{now.strftime('%Y-%m-%d %H:%M')}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    if acc is None:
        print(f"\n  {RED}[!] 无账户数据，请先运行 paper_trade.py --init{RESET}")
        return

    # ===== 账户概览 =====
    total_value = acc.get("cash", 0)
    positions = acc.get("positions", {})
    try:
        import data_source as ds
        codes = list(positions.keys())
        if codes:
            quotes = ds.get_realtime_quotes(codes)
            price_map = dict(zip(quotes["code"], quotes["price"])) if not quotes.empty else {}
        else:
            price_map = {}
    except Exception:
        price_map = {}

    mv = 0
    for code, pos in positions.items():
        p = price_map.get(code, pos.get("entry_price", 0))
        mv += pos["shares"] * p
    total_value = acc["cash"] + mv
    pnl = total_value - 1_000_000

    print(f"\n  {BOLD}账户概览{RESET}")
    print(f"  {'─' * 50}")
    print(f"  初始资金:  ¥{1_000_000:>12,}")
    print(f"  当前现金:  ¥{acc['cash']:>12,.0f}")
    color = GREEN if mv > 0 else RESET
    print(f"  持仓市值:  {color}¥{mv:>12,.0f}{RESET}")
    color = GREEN if total_value > 1_000_000 else RED
    print(f"  总资产:    {color}¥{total_value:>12,.0f}{RESET}")
    color = GREEN if pnl >= 0 else RED
    print(f"  累计盈亏:  {color}¥{pnl:>+12,.0f} ({pnl / 1_000_000:+.2%}){RESET}")

    # ===== 绩效指标 =====
    if log is not None and len(log) > 1:
        print(f"\n  {BOLD}绩效指标{RESET}")
        print(f"  {'─' * 50}")

        total_series = log["total"]
        returns = total_series.pct_change().dropna()
        days = len(returns)
        years = days / 252

        total_ret = total_series.iloc[-1] / total_series.iloc[0] - 1
        if years > 0.01:
            annual_ret = (1 + total_ret) ** (1 / years) - 1
        else:
            annual_ret = 0
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = (annual_ret - 0.02) / vol if vol > 0 else 0
        cummax = total_series.cummax()
        dd = (total_series - cummax) / cummax
        max_dd = dd.min()

        print(f"  跟踪天数:  {days}天")
        color = GREEN if total_ret >= 0 else RED
        print(f"  累计收益:  {color}{total_ret:+.2%}{RESET}")
        print(f"  年化收益:  {annual_ret:+.2%}")
        print(f"  年化波动:  {vol:.2%}")
        color = GREEN if sharpe >= 0 else RED
        print(f"  夏普比率:  {color}{sharpe:+.2f}{RESET}")
        color = GREEN if max_dd > -0.15 else RED
        print(f"  最大回撤:  {color}{max_dd:.2%}{RESET}")

    # ===== 当前持仓 =====
    if positions:
        print(f"\n  {BOLD}当前持仓 ({len(positions)}只){RESET}")
        print(f"  {'─' * 60}")
        print(f"  {'代码':<8} {'股数':>8} {'成本':>8} {'现价':>8} {'市值':>10} {'盈亏':>10} {'占比':>8}")
        print(f"  {'-' * 60}")

        for code, pos in sorted(positions.items()):
            shares = pos["shares"]
            cost = pos["entry_price"]
            p = price_map.get(code, cost)
            market_val = shares * p
            pnl_pct = (p / cost - 1) * 100
            weight = market_val / total_value * 100 if total_value > 0 else 0
            color = GREEN if pnl_pct >= 0 else RED
            print(f"  {code:<8} {shares:>8} {cost:>8.2f} {p:>8.2f} {market_val:>10,.0f} "
                  f"{color}{pnl_pct:>+9.2f}%{RESET} {weight:>7.1f}%")

    # ===== 最近交易 =====
    if trades is not None and not trades.empty:
        print(f"\n  {BOLD}最近交易 (最近5笔){RESET}")
        print(f"  {'─' * 60}")
        recent = trades.tail(5)
        for _, t in recent.iterrows():
            pnl = t["pnl_pct"]
            color = GREEN if pnl > 0 else RED
            dt_str = t["datetime"].strftime("%m-%d %H:%M") if hasattr(t["datetime"], "strftime") else str(t["datetime"])[:16]
            print(f"  {dt_str}  {t['code']:<8} {t['action']:<5} {t['shares']:>6}股  "
                  f"@ {t['exit_price']:.2f}  {color}{pnl:+.2f}%{RESET}")

    # ===== 风险指标 =====
    print(f"\n  {BOLD}风险监控{RESET}")
    print(f"  {'─' * 50}")

    # 仓位集中度
    if positions:
        try:
            from a_share_strategy import get_board
        except ImportError:
            get_board = lambda c: "未知"

        sectors = {}
        for code in positions:
            sec = get_board(code)
            sectors[sec] = sectors.get(sec, 0) + 1
        print(f"  板块分布: {', '.join(f'{k}:{v}' for k, v in sectors.items())}")

        # 单票最大仓位
        max_w = max(
            (pos["shares"] * price_map.get(c, pos["entry_price"])) / total_value
            for c, pos in positions.items()
        ) if total_value > 0 else 0
        color = YELLOW if max_w > 0.15 else GREEN
        print(f"  最大单票:  {color}{max_w:.1%}{RESET}")

        # 现金占比
        cash_ratio = acc["cash"] / total_value if total_value > 0 else 1
        color = YELLOW if cash_ratio < 0.05 else GREEN
        print(f"  现金占比:  {color}{cash_ratio:.1%}{RESET}")

    # ===== 信号日历 =====
    print(f"\n  {BOLD}信号日历 (最近){RESET}")
    print(f"  {'─' * 50}")
    signal_files = sorted(
        [f for f in os.listdir(".") if f.startswith("signals_") and f.endswith(".json")]
    )[-5:]
    if signal_files:
        for sf in signal_files:
            dt = sf.replace("signals_", "").replace(".json", "")
            formatted = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = len(data.get("signals", []))
            print(f"  {formatted}: {count} 只推荐")
    else:
        print(f"  {DIM}暂无信号文件{RESET}")

    # ===== 下一调仓日 =====
    print(f"\n  {BOLD}操作提示{RESET}")
    print(f"  {'─' * 50}")
    today = now.date()
    # 简单计算下一个交易日（不考虑节假日）
    days_since = 0
    # 假设每20个交易日调仓
    last_rebalance = acc.get("last_rebalance", acc.get("created_at", ""))
    print(f"  上次调仓:  {last_rebalance[:10] if last_rebalance else 'N/A'}")
    print(f"  下次调仓:  每20个交易日")
    print(f"  更新净值:  python paper_trade.py --update")
    print(f"  执行调仓:  python paper_trade.py --signal")
    print(f"  查看报告:  python paper_trade.py --report")

    print(f"\n{BOLD}{'=' * 70}{RESET}\n")


# ============================================================
# HTML 报告
# ============================================================

def generate_html_report(output: str = "dashboard.html"):
    """生成HTML仪表盘报告"""
    acc = load_account()
    log = load_log()
    trades = load_trades()

    if acc is None:
        print("[错误] 无账户数据")
        return

    total_value = acc["cash"]
    positions = acc.get("positions", {})
    pnl = total_value - 1_000_000

    # 简易内联CSS的HTML
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>猎鹰一号 — 监控仪表盘</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Microsoft YaHei', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
.header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a5f, #0f172a); border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.card {{ background: #1e293b; border-radius: 10px; padding: 16px; }}
.card h3 {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; }}
.card .value {{ font-size: 28px; font-weight: bold; }}
.positive {{ color: #22c55e; }}
.negative {{ color: #ef4444; }}
.neutral {{ color: #f59e0b; }}
table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }}
th {{ background: #334155; padding: 10px 12px; text-align: left; font-size: 12px; color: #94a3b8; }}
td {{ padding: 10px 12px; border-top: 1px solid #334155; font-size: 14px; }}
.section {{ margin-bottom: 24px; }}
.section h2 {{ font-size: 18px; margin-bottom: 12px; color: #94a3b8; }}
.footer {{ text-align: center; color: #475569; font-size: 12px; margin-top: 30px; }}
.alert {{ background: #7f1d1d; border-left: 3px solid #ef4444; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }}
.warn {{ background: #78350f; border-left: 3px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }}
.ok {{ background: #14532d; border-left: 3px solid #22c55e; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>🦅 猎鹰一号 — 监控仪表盘</h1>
  <p style="color:#94a3b8;margin-top:8px">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<div class="grid">
  <div class="card">
    <h3>总资产</h3>
    <div class="value neutral">¥{total_value:,.0f}</div>
  </div>
  <div class="card">
    <h3>累计盈亏</h3>
    <div class="value {'positive' if pnl >= 0 else 'negative'}">{pnl:+,.0f}</div>
  </div>
  <div class="card">
    <h3>现金</h3>
    <div class="value neutral">¥{acc['cash']:,.0f}</div>
  </div>
  <div class="card">
    <h3>持仓数</h3>
    <div class="value neutral">{len(positions)}</div>
  </div>
</div>
"""

    # 持仓表格
    if positions:
        html += """<div class="section">
<h2>当前持仓</h2>
<table>
<tr><th>代码</th><th>股数</th><th>成本</th><th>现价</th><th>市值</th><th>盈亏</th></tr>"""
        for code, pos in positions.items():
            cost = pos["entry_price"]
            market_val = pos["shares"] * cost  # 简化
            pnl_pct = 0
            css = "positive" if pnl_pct >= 0 else "negative"
            html += f"<tr><td>{code}</td><td>{pos['shares']}</td><td>{cost:.2f}</td><td>{cost:.2f}</td><td>{market_val:,.0f}</td><td class='{css}'>{pnl_pct:+.2f}%</td></tr>"
        html += "</table></div>"

    # 绩效
    if log is not None and len(log) > 1:
        total_series = log["total"]
        returns = total_series.pct_change().dropna()
        total_ret = total_series.iloc[-1] / total_series.iloc[0] - 1
        days = len(returns)
        years = days / 252
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0.01 else 0
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = (annual_ret - 0.02) / vol if vol > 0 else 0
        cummax = total_series.cummax()
        max_dd = (total_series - cummax).min() / cummax.min()

        html += f"""<div class="section"><h2>绩效指标</h2>
<div class="grid">
  <div class="card"><h3>跟踪天数</h3><div class="value neutral">{days}</div></div>
  <div class="card"><h3>累计收益</h3><div class="value {'positive' if total_ret >= 0 else 'negative'}">{total_ret:+.2%}</div></div>
  <div class="card"><h3>年化收益</h3><div class="value neutral">{annual_ret:+.2%}</div></div>
  <div class="card"><h3>夏普比率</h3><div class="value {'positive' if sharpe >= 0 else 'negative'}">{sharpe:+.2f}</div></div>
  <div class="card"><h3>最大回撤</h3><div class="value {'positive' if max_dd > -0.15 else 'negative'}">{max_dd:.2%}</div></div>
</div></div>"""

    html += f"""<div class="footer">
猎鹰一号 · 纸上交易监控 · {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</body></html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML报告已生成: {output}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猎鹰一号 — 监控仪表盘")
    parser.add_argument("--html", action="store_true", help="生成HTML报告")
    parser.add_argument("--output", type=str, default="dashboard.html", help="HTML输出路径")
    args = parser.parse_args()

    if args.html:
        generate_html_report(args.output)
    else:
        terminal_dashboard()
