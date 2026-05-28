"""
完整策略示例 — 因子 → 信号 → 回测 → 可视化
演示 3 个策略 + 1 个组合回测
"""
import pandas as pd
import numpy as np
import sys
import data_source as ds
import factors as ft
import backtest as bt
import visualize as viz

# ==================== 策略定义 ====================


def strategy_ma_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    双均线交叉策略
    金叉买入（快线 > 慢线），死叉卖出
    """
    ma_f = ft.sma(df, fast)
    ma_s = ft.sma(df, slow)
    # 持仓信号：快线 > 慢线
    signal = pd.Series(0, index=df.index)
    signal[ma_f > ma_s] = 1
    # 过滤 NaN
    signal = signal.fillna(0)
    return signal


def strategy_momentum_rsi(df: pd.DataFrame,
                          mom_n: int = 20, rsi_n: int = 14,
                          rsi_lo: float = 30, rsi_hi: float = 70) -> pd.Series:
    """
    动量 + RSI 过滤策略
    动量 > 0（趋势向上）且 RSI 不超买 → 持仓
    """
    mom = ft.momentum(df, mom_n)
    rsi_val = ft.rsi(df, rsi_n)
    signal = ((mom > 0) & (rsi_val < rsi_hi)).astype(int)
    return signal


def strategy_multi_factor(df: pd.DataFrame) -> pd.Series:
    """
    多因子打分策略
    综合：动量 + 均线偏离 + RSI位置 + 量比
    打分 > 0 → 持仓
    """
    # 各因子标准化（Z-Score）
    def zscore(s):
        m = s.rolling(252, min_periods=60).mean()
        std = s.rolling(252, min_periods=60).std()
        return (s - m) / std.replace(0, np.nan)

    scores = pd.DataFrame(index=df.index)
    scores["mom_20"] = zscore(ft.momentum(df, 20))       # 动量
    scores["ma_bias"] = zscore(ft.ma_bias(df, 20))        # 均线偏离
    scores["rsi"] = -(ft.rsi(df, 14) - 50) / 20           # RSI 中性化（RSI 低时更好）
    scores["vol"] = zscore(ft.volume_ratio(df, 5))        # 放量

    # 综合得分 = 动量 + 均线 - RSI偏离 + 放量
    composite = scores.mean(axis=1)
    signal = (composite > 0).astype(int)
    return signal


# ==================== 主流程 ====================

def run_one(symbol: str, name: str, strategy_fn, **kwargs):
    """对单只股票运行一个策略"""
    print(f"\n{'=' * 55}")
    print(f"  {name} — {symbol}")
    print(f"{'=' * 55}")

    # 1. 获取数据
    print("  获取K线数据...")
    df = ds.get_daily_kline(symbol, start_date="20200101", adjust="qfq")
    if df.empty:
        print(f"  [失败] 无法获取 {symbol} 数据")
        return None

    print(f"  数据范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}, {len(df)} 条")

    # 统一索引：以 date 为 index
    df = df.set_index("date")

    # 2. 生成信号
    signal = strategy_fn(df, **kwargs)

    # 3. 回测
    engine = bt.BacktestEngine(
        price=df["close"],
        signal=signal,
    )
    result = engine.run()

    # 4. 输出绩效
    bt.print_metrics(result.metrics)
    return result


def run_portfolio(symbols: list[str], name: str, strategy_fn, **kwargs):
    """多股票组合回测"""
    print(f"\n{'=' * 55}")
    print(f"  {name} — {len(symbols)} 只股票组合")
    print(f"{'=' * 55}")

    price_matrix = pd.DataFrame()
    signal_matrix = pd.DataFrame()

    for sym in symbols:
        df = ds.get_daily_kline(sym, start_date="20200101", adjust="qfq")
        if df.empty:
            continue
        df = df.set_index("date")
        price_matrix[sym] = df["close"]
        signal_matrix[sym] = strategy_fn(df, **kwargs)

    if price_matrix.empty:
        print("  [失败] 无有效数据")
        return None

    result = bt.run_portfolio_backtest(price_matrix, signal_matrix)
    bt.print_metrics(result.metrics)
    return result


def run_demo():
    print("=" * 55)
    print("  策略回测 Demo")
    print("=" * 55)

    # ----- 策略1: 双均线，贵州茅台 -----
    r1 = run_one("600519", "双均线交叉 (5/20)", strategy_ma_cross)

    # ----- 策略2: 动量+RSI，比亚迪 -----
    r2 = run_one("002594", "动量+RSI过滤", strategy_momentum_rsi)

    # ----- 策略3: 多因子打分，中国平安 -----
    r3 = run_one("601318", "多因子综合打分", strategy_multi_factor)

    # ----- 因子分析 -----
    if r1 is not None:
        print("\n" + "=" * 55)
        print("  因子分析 — 贵州茅台")
        print("=" * 55)
        df = ds.get_daily_kline("600519", start_date="20200101", adjust="qfq")
        df = df.set_index("date")
        f = ft.compute_all(df)
        print(f"  共 {f.shape[1]} 个因子，{f.dropna().shape[0]} 条有效记录")
        print(f"  近期因子值:\n{f.tail(5).to_string()}")

    # ----- 可视化 (r1) -----
    if r1 is not None:
        print("\n  正在生成图表...")
        try:
            viz.plot_equity_curve(r1, title="贵州茅台 — 双均线策略")
        except Exception as e:
            print(f"  图表显示失败 (无GUI环境): {e}")
            print("  净值数据已保存，可在 Jupyter 中查看")


if __name__ == "__main__":
    run_demo()
