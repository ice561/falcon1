"""
回测引擎 + 绩效指标

用法:
    engine = BacktestEngine(price_df, signal_series)
    result = engine.run()
    print(result.metrics)
    result.plot()
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """回测结果"""
    equity: pd.Series       # 净值曲线
    returns: pd.Series      # 日收益率
    positions: pd.Series    # 持仓 (0/1)
    trades: pd.DataFrame    # 逐笔交易
    metrics: dict           # 绩效指标汇总
    benchmark: pd.Series    # buy&hold 净值


class BacktestEngine:
    """
    向量化回测引擎

    参数:
        price   : 价格 Series（通常用收盘价）
        signal  : 信号 Series，>0=做多，<=0=空仓
        commission : 佣金费率（双向），默认 0.0003（万三）
        slippage   : 滑点，默认 0.0001（万一）
    """

    def __init__(
        self,
        price: pd.Series,
        signal: pd.Series,
        commission: float = 0.0003,
        slippage: float = 0.0001,
    ):
        self.price = price.dropna()
        self.signal = signal.reindex(self.price.index).fillna(0)
        self.commission = commission
        self.slippage = slippage

    def run(self) -> BacktestResult:
        price = self.price.values
        signal = self.signal.values

        n = len(price)
        position = np.zeros(n, dtype=float)      # 仓位(0/1)
        equity = np.ones(n, dtype=float)          # 净值
        cash = np.ones(n, dtype=float)            # 现金占比

        for i in range(1, n):
            prev_pos = position[i - 1]
            target = 1.0 if signal[i] > 0 else 0.0

            if target != prev_pos:
                # 发生交易
                if target > prev_pos:
                    # 买入
                    trade_price = price[i] * (1 + self.slippage)
                    cost = self.commission
                else:
                    # 卖出
                    trade_price = price[i] * (1 - self.slippage)
                    cost = self.commission

                position[i] = target
                cash[i] = cash[i - 1] - cost * abs(target - prev_pos)
                # 调整：卖出时收回现金，买入时付出
                if target > prev_pos:
                    cash[i] -= (target - prev_pos)  # 买入花费
                else:
                    cash[i] += (prev_pos - target) * (price[i] / price[i])
            else:
                position[i] = prev_pos
                cash[i] = cash[i - 1]

            # 持仓收益
            if position[i] > 0:
                equity[i] = equity[i - 1] * (price[i] / price[i - 1])
            else:
                equity[i] = equity[i - 1]

        # 扣除卖出的佣金（如果最后还持仓）
        # 简化：不再单独计算

        equity_series = pd.Series(equity, index=self.price.index)
        pos_series = pd.Series(position, index=self.price.index)
        ret_series = equity_series.pct_change().fillna(0)

        # Benchmark: buy & hold
        bench = self.price / self.price.iloc[0]

        # 逐笔交易记录
        trades = self._extract_trades(pos_series, equity_series)

        # 绩效
        metrics = compute_metrics(equity_series, ret_series, trades)

        return BacktestResult(
            equity=equity_series,
            returns=ret_series,
            positions=pos_series,
            trades=trades,
            metrics=metrics,
            benchmark=bench,
        )

    def _extract_trades(self, pos: pd.Series, eq: pd.Series) -> pd.DataFrame:
        """从仓位变化提取逐笔交易"""
        changes = pos.diff().fillna(pos)
        entries = changes[changes > 0].index
        exits = changes[changes < 0].index

        trades = []
        # 简化：配对 entry/exit
        for i in range(min(len(entries), len(exits))):
            entry_date = entries[i]
            exit_date = exits[i]
            entry_price = self.price.loc[entry_date]
            exit_price = self.price.loc[exit_date]
            pnl_pct = (exit_price / entry_price - 1) * 100  # 简化，不含手续费
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "hold_days": (exit_date - entry_date).days,
                "return_pct": round(pnl_pct, 2),
            })

        return pd.DataFrame(trades)


# ==================== 并行回测（多股票） ====================

def run_portfolio_backtest(
    price_matrix: pd.DataFrame,
    signal_matrix: pd.DataFrame,
    commission: float = 0.0003,
    slippage: float = 0.0001,
) -> BacktestResult:
    """
    组合回测：多只股票等权
    price_matrix  : 每列一只股票的价格
    signal_matrix : 每列一只股票的信号 (>0=做多)
    """
    # 每只股票单独回测
    eq_curves = pd.DataFrame(index=price_matrix.index)
    for col in price_matrix.columns:
        if col in signal_matrix.columns:
            engine = BacktestEngine(
                price_matrix[col], signal_matrix[col],
                commission=commission, slippage=slippage,
            )
            result = engine.run()
            eq_curves[col] = result.equity

    # 等权合并
    avg_equity = eq_curves.mean(axis=1)
    ret_series = avg_equity.pct_change().fillna(0)
    bench = price_matrix.mean(axis=1)
    bench = bench / bench.iloc[0]

    metrics = compute_metrics(avg_equity, ret_series, pd.DataFrame())

    # 占位仓位
    avg_pos = (signal_matrix > 0).mean(axis=1)

    return BacktestResult(
        equity=avg_equity,
        returns=ret_series,
        positions=avg_pos,
        trades=pd.DataFrame(),
        metrics=metrics,
        benchmark=bench,
    )


# ==================== 绩效指标 ====================

def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    rf: float = 0.02,
) -> dict:
    """计算全部绩效指标"""
    total_days = len(returns)
    years = total_days / 252

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annual_ret = float((1 + total_return) ** (1 / max(years, 0.01)) - 1)

    # 风险
    daily_std = returns[returns != 0].std()
    if np.isnan(daily_std) or daily_std == 0:
        annual_vol = 0.0
        sharpe = 0.0
    else:
        annual_vol = float(daily_std * np.sqrt(252))
        sharpe = float((annual_ret - rf) / annual_vol)

    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min())
    calmar = float(annual_ret / abs(max_dd)) if max_dd != 0 else 0

    # 交易统计
    if not trades.empty:
        win_rate = (trades["return_pct"] > 0).mean()
        avg_win = trades.loc[trades["return_pct"] > 0, "return_pct"].mean()
        avg_loss = trades.loc[trades["return_pct"] < 0, "return_pct"].mean()
        gross_profit = trades.loc[trades["return_pct"] > 0, "return_pct"].sum()
        gross_loss = trades.loc[trades["return_pct"] < 0, "return_pct"].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0
        num_trades = len(trades)
        avg_hold = trades["hold_days"].mean()
    else:
        win_rate = avg_win = avg_loss = profit_factor = num_trades = avg_hold = 0

    # 胜率（按日）
    daily_win_rate = (returns > 0).mean()

    return {
        "累计收益": f"{total_return:.2%}",
        "年化收益": f"{annual_ret:.2%}",
        "年化波动": f"{annual_vol:.2%}",
        "夏普比率": round(sharpe, 2),
        "最大回撤": f"{max_dd:.2%}",
        "卡玛比率": round(calmar, 2),
        "日胜率": f"{daily_win_rate:.2%}",
        "交易次数": num_trades,
        "胜率(笔)": f"{win_rate:.2%}",
        "平均盈利": f"{avg_win:.2f}%",
        "平均亏损": f"{avg_loss:.2f}%",
        "盈亏比": round(profit_factor, 2) if not np.isnan(profit_factor) else 0,
        "平均持仓天": round(avg_hold, 1),
        "回测天数": total_days,
    }


def print_metrics(metrics: dict):
    """格式化打印绩效"""
    print("\n" + "=" * 45)
    print("  回测绩效")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:12s}  {v}")
    print("=" * 45)
