"""
策略优化工具 — 因子权重搜索 + 参数网格搜索 + 行业分散
用法: python optimize.py
"""
import itertools
import random
import sys
import time
import numpy as np
import pandas as pd
from a_share_strategy import (
    StrategyConfig, MultiFactorStrategy, StrategyResult, get_board,
)


# ============================================================
# 因子权重随机搜索
# ============================================================

def random_weight_trials(n_trials: int = 50, quick: bool = True, **base_kwargs):
    """
    随机搜索因子权重组合
    返回: [(weights_dict, metrics_dict), ...] 按夏普降序
    """
    factor_names = ["w_momentum", "w_reversal", "w_vol_price",
                    "w_low_vol", "w_small_cap", "w_quality"]
    results = []

    print(f"\n{'=' * 55}")
    print(f"  因子权重随机搜索 ({n_trials} 组)")
    print(f"{'=' * 55}")

    for trial in range(n_trials):
        # 生成随机权重（Dirichlet分布）
        raw = np.random.exponential(1, 6)
        w = raw / raw.sum()
        weights = dict(zip(factor_names, w))

        cfg = StrategyConfig(**{**base_kwargs, **weights})
        if quick:
            cfg.quick_mode = True
            cfg.quick_n = 200

        strategy = MultiFactorStrategy(cfg)
        try:
            result = strategy.run()
        except Exception as e:
            print(f"  Trial {trial + 1}: 失败 ({e})")
            continue

        m = result.metrics
        sharpe = m.get("夏普比率", 0)
        ret = m.get("累计收益", "0%")

        results.append((weights, m, result))
        print(f"  Trial {trial + 1:2d}: 夏普={sharpe:5.2f}  收益={str(ret):>10s}  "
              f"w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f},{w[4]:.2f},{w[5]:.2f})")

    results.sort(key=lambda x: x[1].get("夏普比率", -99), reverse=True)
    return results


# ============================================================
# 参数网格搜索
# ============================================================

def grid_search(quick: bool = True, n_stocks: int = 200):
    """
    网格搜索: top_k × rebalance_freq × stop_loss
    """
    param_grid = {
        "top_k": [5, 10, 15, 20],
        "rebalance_freq": [5, 10, 20, 40],
        "stop_loss": [0.05, 0.08, 0.10, 0.12],
    }

    combos = list(itertools.product(*param_grid.values()))
    print(f"\n{'=' * 55}")
    print(f"  参数网格搜索 ({len(combos)} 组)")
    print(f"{'=' * 55}")

    results = []
    best_sharpe = -99

    for i, (top_k, rebalance_freq, stop_loss) in enumerate(combos):
        cfg = StrategyConfig(
            top_k=top_k,
            rebalance_freq=rebalance_freq,
            stop_loss=stop_loss,
        )
        if quick:
            cfg.quick_mode = True
            cfg.quick_n = n_stocks

        strategy = MultiFactorStrategy(cfg)
        try:
            result = strategy.run()
        except Exception as e:
            print(f"  [{i + 1}/{len(combos)}] K={top_k} F={rebalance_freq} SL={stop_loss:.0%} → 失败: {e}")
            continue

        m = result.metrics
        sharpe = m.get("夏普比率", 0)
        ret = m.get("累计收益", "0%")
        dd = m.get("最大回撤", "0%")
        trades = m.get("交易次数", 0)

        results.append({
            "top_k": top_k, "rebalance_freq": rebalance_freq,
            "stop_loss": stop_loss, "sharpe": sharpe,
            "return": ret, "max_dd": dd, "trades": trades,
        })

        marker = " ★" if sharpe > best_sharpe else ""
        best_sharpe = max(best_sharpe, sharpe)
        print(f"  [{i + 1:2d}/{len(combos)}] K={top_k:2d} F={rebalance_freq:2d} SL={stop_loss:.0%}  "
              f"夏普={sharpe:5.2f} 收益={str(ret):>9s} 回撤={str(dd):>8s}{marker}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


# ============================================================
# 综合优化
# ============================================================

def run_full_optimization():
    """分两阶段: (1)中间参数网格搜索 (2)最优参数上做权重微调"""
    t0 = time.time()

    # ---- 阶段1: 网格搜索 ----
    grid_results = grid_search(quick=True, n_stocks=200)

    if not grid_results:
        print("\n[失败] 网格搜索无有效结果")
        return

    best = grid_results[0]
    print(f"\n{'=' * 55}")
    print(f"  网格搜索最佳参数")
    print(f"{'=' * 55}")
    print(f"  top_k={best['top_k']}, rebalance_freq={best['rebalance_freq']}, "
          f"stop_loss={best['stop_loss']:.0%}")
    print(f"  夏普={best['sharpe']:.2f}  收益={best['return']}  回撤={best['max_dd']}")

    # ---- 阶段2: 在最佳参数上随机搜索权重 ----
    weight_results = random_weight_trials(
        n_trials=30, quick=True,
        top_k=best["top_k"],
        rebalance_freq=best["rebalance_freq"],
        stop_loss=best["stop_loss"],
    )

    if weight_results:
        best_w, best_m, _ = weight_results[0]
        print(f"\n{'=' * 55}")
        print(f"  最优因子权重")
        print(f"{'=' * 55}")
        print(f"  动量={best_w['w_momentum']:.0%}  反转={best_w['w_reversal']:.0%}  "
              f"量价={best_w['w_vol_price']:.0%}  低波={best_w['w_low_vol']:.0%}  "
              f"小市值={best_w['w_small_cap']:.0%}  质量={best_w['w_quality']:.0%}")
        print(f"  夏普={best_m.get('夏普比率', 0):.2f}  收益={best_m.get('累计收益', 'N/A')}")

    # ---- 阶段3: 全量回测 ----
    print(f"\n{'=' * 55}")
    print(f"  全量回测 (最优参数)")
    print(f"{'=' * 55}")

    final_cfg = StrategyConfig(
        top_k=best["top_k"],
        rebalance_freq=best["rebalance_freq"],
        stop_loss=best["stop_loss"],
        quick_mode=False,
    )
    if weight_results:
        for k, v in weight_results[0][0].items():
            setattr(final_cfg, k, v)

    strategy = MultiFactorStrategy(final_cfg)
    result = strategy.run()

    bt = sys.modules.get("backtest")
    if bt:
        bt.print_metrics(result.metrics)

    # 图表
    try:
        import visualize as viz
        viz.plot_equity_curve(result, title="猎鹰一号 — 优化后", save="falcon1_optimized.png")
        viz.plot_monthly_heatmap(result.returns, title="月度收益 — 优化后", save="falcon1_optimized_heatmap.png")
    except Exception as e:
        print(f"  图表失败: {e}")

    elapsed = time.time() - t0
    print(f"\n  优化总耗时: {elapsed / 60:.1f} 分钟")

    return result


# ============================================================
# 单次全量回测（当前参数）
# ============================================================

def run_full_backtest(**overrides):
    """全量回测"""
    cfg = StrategyConfig(**overrides)
    strategy = MultiFactorStrategy(cfg)
    result = strategy.run()
    bt = sys.modules.get("backtest")
    if bt:
        bt.print_metrics(result.metrics)
    try:
        import visualize as viz
        viz.plot_equity_curve(result, title="猎鹰一号 — 全量回测", save="falcon1_full.png")
    except Exception:
        pass
    return result


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "optimize", "grid", "weights"], default="full")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.mode == "optimize":
        run_full_optimization()
    elif args.mode == "grid":
        results = grid_search(quick=not args.quick or True, n_stocks=200)
        if results:
            print("\n  Top 10:")
            for i, r in enumerate(results[:10]):
                print(f"  {i + 1}. K={r['top_k']} F={r['rebalance_freq']} SL={r['stop_loss']:.0%}  "
                      f"Sharpe={r['sharpe']:.2f} Ret={r['return']} DD={r['max_dd']}")
    elif args.mode == "weights":
        results = random_weight_trials(n_trials=30, quick=True, top_k=15, rebalance_freq=20, stop_loss=0.08)
        if results:
            print("\n  Top 10:")
            for i, (w, m, _) in enumerate(results[:10]):
                print(f"  {i + 1}. Sharpe={m.get('夏普比率', 0):.2f}  "
                      f"w=({w['w_momentum']:.2f},{w['w_reversal']:.2f},{w['w_vol_price']:.2f},"
                      f"{w['w_low_vol']:.2f},{w['w_small_cap']:.2f},{w['w_quality']:.2f})")
    else:  # full
        run_full_backtest()
