"""全量回测 — 默认参数 + 行业分散"""
import time
from a_share_strategy import StrategyConfig, MultiFactorStrategy
import backtest as bt
import visualize as viz

print("=" * 55)
print("  猎鹰一号 — 全量回测 (默认参数, 行业分散)")
print("=" * 55)

cfg = StrategyConfig(
    start_date="20210101",
    top_k=15,
    rebalance_freq=20,
    stop_loss=0.08,
    take_profit=0.30,
    trailing_stop=0.15,
    max_per_sector=3,
    use_market_timing=True,
    min_score_threshold=0.3,
)
strategy = MultiFactorStrategy(cfg)
t0 = time.time()
result = strategy.run()
elapsed = time.time() - t0

bt.print_metrics(result.metrics)

print("\n  生成图表...")
try:
    viz.plot_equity_curve(result, title="猎鹰一号 — 全量回测", save="falcon1_full_default.png")
    print("  净值图: falcon1_full_default.png")
except Exception as e:
    print(f"  净值图失败: {e}")
try:
    viz.plot_monthly_heatmap(result.returns, title="猎鹰一号 — 月度收益", save="falcon1_full_heatmap.png")
    print("  热力图: falcon1_full_heatmap.png")
except Exception as e:
    print(f"  热力图失败: {e}")

print(f"\n  总耗时: {elapsed/60:.1f} 分钟")

if result.selections:
    print("\n  === 最近5次调仓 ===")
    for d in sorted(result.selections.keys())[-5:]:
        stocks = result.selections[d]
        print(f"  {d.date()}: {', '.join(stocks[:8])}{'...' if len(stocks) > 8 else ''}")
