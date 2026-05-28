"""
可视化模块 — 净值曲线 / 回撤图 / 月度热力图 / 因子分析图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curve(result, title: str = "策略净值曲线", save: str = ""):
    """净值曲线 + 回撤图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # 净值
    ax1.plot(result.equity.index, result.equity.values, color="#1f77b4", linewidth=1.2, label="策略")
    ax1.plot(result.benchmark.index, result.benchmark.values, color="#d62728", linewidth=0.8, alpha=0.7, label="买入持有")
    ax1.set_ylabel("净值")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # 回撤
    cummax = result.equity.cummax()
    dd = (result.equity - cummax) / cummax
    ax2.fill_between(dd.index, 0, dd.values, color="#d62728", alpha=0.4)
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save if save else f"{title}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" 图表已保存: {path}")
    plt.close()


def plot_monthly_heatmap(returns: pd.Series, title: str = "月度收益热力图", save: str = ""):
    """月度收益率热力图"""
    if returns.empty:
        return

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period("M")
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    if table.empty:
        return
    # 转置成 (month x year)
    table = table.T
    table.index = table.index.astype(int)

    fig, ax = plt.subplots(figsize=(14, max(3, len(table) * 0.6)))
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)

    for i in range(len(table)):
        for j in range(len(table.columns)):
            v = table.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.08 else "white")

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=0)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = save if save else f"{title}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" 图表已保存: {path}")
    plt.close()


def plot_factor_distribution(factors: pd.DataFrame, target: str = "momentum_5"):
    """单因子分布 + 分位收益对比"""
    if target not in factors.columns:
        print(f"因子 {target} 不存在")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 分布
    axes[0].hist(factors[target].dropna(), bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{target} 分布")
    axes[0].set_xlabel("因子值")
    axes[0].set_ylabel("频次")

    # QQ-plot 近似
    from scipy import stats
    vals = factors[target].dropna()
    if len(vals) > 10:
        stats.probplot(vals, dist="norm", plot=axes[1])
        axes[1].set_title(f"{target} Q-Q Plot")
    else:
        axes[1].text(0.5, 0.5, "数据不足", ha="center", va="center")

    plt.tight_layout()
    path = f"{target}_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" 图表已保存: {path}")
    plt.close()


def plot_factor_correlation(factors: pd.DataFrame, save: str = ""):
    """因子相关性矩阵"""
    corr = factors.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 标注
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("因子相关性矩阵")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = save if save else "factor_correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" 图表已保存: {path}")
    plt.close()


def plot_returns_distribution(returns: pd.Series, save: str = ""):
    """日收益率分布 + 正态对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ret_clean = returns[(returns != 0) & (~returns.isna())]
    ax.hist(ret_clean, bins=80, density=True, alpha=0.6, color="#1f77b4", edgecolor="white")

    # 正态拟合
    from scipy import stats
    mu, sigma = ret_clean.mean(), ret_clean.std()
    x = np.linspace(ret_clean.min(), ret_clean.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#d62728", linewidth=1.5, label=f"N({mu:.4f}, {sigma:.4f})")

    ax.set_xlabel("日收益率")
    ax.set_ylabel("密度")
    ax.set_title("日收益率分布")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save if save else "returns_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f" 图表已保存: {path}")
    plt.close()


def plot_all(result):
    """一键出图：净值 + 回撤 + 月度热力 + 收益分布"""
    plot_equity_curve(result)
    plot_monthly_heatmap(result.returns)
    plot_returns_distribution(result.returns)
