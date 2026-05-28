"""
样本外验证 — 检验策略在未见数据上的表现
将数据分为训练期和测试期，检测过拟合程度

用法: python oos_validation.py
"""
import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass

from a_share_strategy import (
    StrategyConfig, MultiFactorStrategy, StrategyResult,
)
import backtest as bt


@dataclass
class OOSResult:
    """样本外验证结果"""
    train_metrics: dict
    test_metrics: dict
    train_result: StrategyResult
    test_result: StrategyResult
    degradation: dict  # 样本外 vs 样本内 衰减率


def run_oos_validation(
    train_start: str = "20210101",
    train_end: str = "20231231",
    test_start: str = "20240101",
    test_end: str = "",
    **config_overrides,
) -> OOSResult:
    """
    样本外验证:
      - 训练期: 2021-01 ~ 2023-12 (3年)
      - 测试期: 2024-01 ~ 至今 (从未见过的数据)
    """
    print("=" * 60)
    print("  猎鹰一号 — 样本外验证")
    print("=" * 60)

    # ---- 训练期 ----
    print(f"\n{'=' * 60}")
    print(f"  训练期: {train_start} ~ {train_end}")
    print(f"{'=' * 60}")

    cfg_train = StrategyConfig(
        start_date=train_start,
        end_date=train_end,
        **config_overrides,
    )
    strategy_train = MultiFactorStrategy(cfg_train)
    t0 = time.time()
    train_result = strategy_train.run()
    train_elapsed = time.time() - t0

    train_metrics = train_result.metrics
    print(f"\n  训练期耗时: {train_elapsed / 60:.1f} 分钟")
    bt.print_metrics(train_metrics)

    # ---- 测试期 ----
    print(f"\n{'=' * 60}")
    print(f"  测试期 (样本外): {test_start} ~ {test_end or '至今'}")
    print(f"{'=' * 60}")

    cfg_test = StrategyConfig(
        start_date=test_start,
        end_date=test_end,
        **config_overrides,
    )
    strategy_test = MultiFactorStrategy(cfg_test)
    t0 = time.time()
    test_result = strategy_test.run()
    test_elapsed = time.time() - t0

    test_metrics = test_result.metrics
    print(f"\n  测试期耗时: {test_elapsed / 60:.1f} 分钟")
    bt.print_metrics(test_metrics)

    # ---- 衰减分析 ----
    degradation = _compute_degradation(train_metrics, test_metrics)

    print(f"\n{'=' * 60}")
    print(f"  样本外衰减分析")
    print(f"{'=' * 60}")

    for k, v in degradation.items():
        direction = "↓" if v < 0 else "↑"
        print(f"  {k:12s}: {v:+.1%} {direction}")

    _plot_oos_comparison(train_result, test_result)

    # 综合判断
    print(f"\n{'=' * 60}")
    print(f"  综合判断")
    print(f"{'=' * 60}")

    train_sharpe = train_metrics.get("夏普比率", 0)
    test_sharpe = test_metrics.get("夏普比率", 0)
    test_ret = test_metrics.get("累计收益", "0%")
    test_dd = test_metrics.get("最大回撤", "0%")

    checks = []
    if test_sharpe > 0:
        checks.append(("样本外夏普为正", True))
    else:
        checks.append(("样本外夏普为正", False))

    if test_sharpe >= train_sharpe * 0.5:
        checks.append((f"样本外夏普 >= 样本内50% ({train_sharpe * 0.5:.2f})", True))
    else:
        checks.append((f"样本外夏普 >= 样本内50% ({train_sharpe * 0.5:.2f})", False))

    if test_sharpe >= train_sharpe * 0.3:
        checks.append((f"样本外夏普 >= 样本内30% ({train_sharpe * 0.3:.2f})", True))
    else:
        checks.append((f"样本外夏普 >= 样本内30% ({train_sharpe * 0.3:.2f})", False))

    passed = sum(1 for _, ok in checks if ok)
    for desc, ok in checks:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {desc}")

    if passed == len(checks):
        print(f"\n  >>> 综合结论: 通过样本外验证 ({passed}/{len(checks)})")
    elif passed >= len(checks) - 1:
        print(f"\n  >>> 综合结论: 基本通过 ({passed}/{len(checks)})，需关注衰减")
    else:
        print(f"\n  >>> 综合结论: 未通过 ({passed}/{len(checks)})，存在严重过拟合")

    return OOSResult(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_result=train_result,
        test_result=test_result,
        degradation=degradation,
    )


def _compute_degradation(train: dict, test: dict) -> dict:
    """计算样本外相对样本内的衰减率"""
    deg = {}

    # 年化收益
    try:
        train_ret = float(str(train["年化收益"]).replace("%", "")) / 100
        test_ret = float(str(test["年化收益"]).replace("%", "")) / 100
        if abs(train_ret) > 0.001:
            deg["年化收益衰减"] = (test_ret - train_ret) / abs(train_ret)
        else:
            deg["年化收益衰减"] = 0
    except (ValueError, KeyError):
        pass

    # 夏普
    try:
        train_s = float(train["夏普比率"])
        test_s = float(test["夏普比率"])
        if abs(train_s) > 0.01:
            deg["夏普衰减"] = (test_s - train_s) / abs(train_s)
        else:
            deg["夏普衰减"] = 0
    except (ValueError, KeyError):
        pass

    # 最大回撤（正数=恶化）
    try:
        train_dd = abs(float(str(train["最大回撤"]).replace("%", "")) / 100)
        test_dd = abs(float(str(test["最大回撤"]).replace("%", "")) / 100)
        if train_dd > 0.001:
            deg["回撤恶化"] = (test_dd - train_dd) / train_dd
        else:
            deg["回撤恶化"] = 0
    except (ValueError, KeyError):
        pass

    # 日胜率
    try:
        train_wr = float(str(train["日胜率"]).replace("%", "")) / 100
        test_wr = float(str(test["日胜率"]).replace("%", "")) / 100
        if train_wr > 0.001:
            deg["日胜率衰减"] = (test_wr - train_wr) / train_wr
        else:
            deg["日胜率衰减"] = 0
    except (ValueError, KeyError):
        pass

    return deg


def _plot_oos_comparison(train_result, test_result):
    """绘制样本内 vs 样本外净值对比"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # 训练期净值
        teq = train_result.equity
        ax1.plot(teq.index, teq.values, color="#1f77b4", linewidth=1.2, label="策略")
        if not train_result.benchmark.empty:
            bench = train_result.benchmark
            bench = bench / bench.iloc[0] * teq.iloc[0]
            ax1.plot(bench.index, bench.values, color="#d62728", linewidth=0.8, alpha=0.7, label="基准")
        ax1.set_title("训练期 (2021-2023)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 测试期净值
        teq2 = test_result.equity
        ax2.plot(teq2.index, teq2.values, color="#2ca02c", linewidth=1.2, label="策略")
        if not test_result.benchmark.empty:
            bench2 = test_result.benchmark
            bench2 = bench2 / bench2.iloc[0] * teq2.iloc[0]
            ax2.plot(bench2.index, bench2.values, color="#d62728", linewidth=0.8, alpha=0.7, label="基准")
        ax2.set_title("测试期 (2024-至今, 样本外)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("oos_validation.png", dpi=150, bbox_inches="tight")
        print("\n  图表已保存: oos_validation.png")
        plt.close()
    except Exception as e:
        print(f"  图表失败: {e}")


def run_rolling_oos(train_years: int = 2, step_months: int = 6):
    """
    滚动样本外验证：
    每 step_months 滚动一次窗口，看策略稳定性
    """
    print("=" * 60)
    print(f"  滚动样本外验证 (训练窗={train_years}年, 步长={step_months}月)")
    print("=" * 60)

    windows = [
        ("2021-01", "2022-12", "2023-01", "2023-06"),
        ("2021-07", "2023-06", "2023-07", "2023-12"),
        ("2022-01", "2023-12", "2024-01", "2024-06"),
        ("2022-07", "2024-06", "2024-07", "2024-12"),
        ("2023-01", "2024-12", "2025-01", "2025-06"),
    ]

    all_metrics = []
    for train_start, train_end, test_start, test_end in windows:
        print(f"\n  --- 窗口: 训练 {train_start}~{train_end} → 测试 {test_start}~{test_end} ---")

        try:
            cfg_train = StrategyConfig(start_date=train_start, end_date=train_end)
            s1 = MultiFactorStrategy(cfg_train)
            r1 = s1.run()

            cfg_test = StrategyConfig(start_date=test_start, end_date=test_end)
            s2 = MultiFactorStrategy(cfg_test)
            r2 = s2.run()

            train_sharpe = r1.metrics.get("夏普比率", 0)
            test_sharpe = r2.metrics.get("夏普比率", 0)
            train_ret = r1.metrics.get("年化收益", "0%")
            test_ret = r2.metrics.get("年化收益", "0%")

            all_metrics.append({
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_ret": train_ret,
                "test_ret": test_ret,
            })
            print(f"    样本内: 夏普={train_sharpe:.2f}, 收益={train_ret}")
            print(f"    样本外: 夏普={test_sharpe:.2f}, 收益={test_ret}")
        except Exception as e:
            print(f"    窗口失败: {e}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        print(f"\n  --- 滚动汇总 ---")
        test_sharpes = df["test_sharpe"].tolist()
        pos_windows = sum(1 for s in test_sharpes if s > 0)
        avg_test_sharpe = np.mean(test_sharpes)
        print(f"  样本外夏普>0 窗口: {pos_windows}/{len(test_sharpes)}")
        print(f"  平均样本外夏普: {avg_test_sharpe:.2f}")
        print(f"  稳定性评价: {'稳定' if pos_windows >= len(test_sharpes) * 0.6 else '不稳定'}")

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "rolling"], default="split")
    parser.add_argument("--quick", action="store_true", help="快速模式（少量股票）")
    args = parser.parse_args()

    base_config = {}
    if args.quick:
        base_config = {"quick_mode": True, "quick_n": 200}

    if args.mode == "rolling":
        run_rolling_oos()
    else:
        run_oos_validation(**base_config)
