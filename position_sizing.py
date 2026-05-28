"""
动态仓位管理 — 基于波动率的风险预算分配
不再等权配置，而是根据每只股票的风险特征动态调整仓位

方法:
  1. 风险平价 (Risk Parity): 波动率越低的股票仓位越大
  2. 波动率目标 (Vol Targeting): 控制组合目标波动率
  3. 凯利准则 (Kelly): 根据胜率和盈亏比调整

用法:
  from position_sizing import PositionSizer
  sizer = PositionSizer(method="risk_parity")
  weights = sizer.allocate(prices_df, selected_codes, total_capital)
"""
import numpy as np
import pandas as pd


class PositionSizer:
    """动态仓位管理器"""

    def __init__(
        self,
        method: str = "risk_parity",  # risk_parity | equal_risk | kelly | vol_target
        target_vol: float = 0.15,      # 年化目标波动率（vol_target模式）
        max_single_position: float = 0.15,  # 单票最大仓位
        min_single_position: float = 0.02,  # 单票最小仓位
        vol_lookback: int = 60,         # 波动率计算窗口
        kelly_fraction: float = 0.5,    # 凯利分数（保守=0.25, 中性=0.5）
    ):
        self.method = method
        self.target_vol = target_vol
        self.max_single = max_single_position
        self.min_single = min_single_position
        self.vol_lookback = vol_lookback
        self.kelly_fraction = kelly_fraction

    def allocate(
        self,
        price_matrix: pd.DataFrame,
        selected_codes: list[str],
        total_capital: float,
        factor_scores: dict[str, float] | None = None,
        trade_history: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        计算每只股票的仓位权重

        参数:
            price_matrix: 价格矩阵 (Date × Stock)
            selected_codes: 选中的股票代码
            total_capital: 总资金
            factor_scores: {code: composite_score} 可选，用于信心加权
            trade_history: 历史交易 DataFrame，用于凯利计算

        返回:
            {code: weight} 仓位权重（和为1）
        """
        valid = [c for c in selected_codes if c in price_matrix.columns]
        if not valid:
            return {}

        if self.method == "risk_parity":
            weights = self._risk_parity(price_matrix, valid)
        elif self.method == "equal_risk":
            weights = self._equal_risk_contribution(price_matrix, valid)
        elif self.method == "kelly":
            weights = self._kelly_allocation(price_matrix, valid, trade_history, factor_scores)
        elif self.method == "vol_target":
            weights = self._vol_target(price_matrix, valid)
        elif self.method == "score_weighted":
            weights = self._score_weighted(valid, factor_scores)
        else:
            weights = self._equal_weight(valid)

        # 应用约束
        weights = self._apply_constraints(weights)
        return weights

    def _risk_parity(self, price_matrix: pd.DataFrame, codes: list[str]) -> dict[str, float]:
        """风险平价：仓位与波动率成反比"""
        vols = {}
        for code in codes:
            rets = price_matrix[code].pct_change().dropna().tail(self.vol_lookback)
            if len(rets) < 20:
                vols[code] = 0.03  # 默认3%日波动
            else:
                vols[code] = rets.std()

        if not vols:
            return {}

        # 仓位 ∝ 1/vol
        inv_vols = {c: 1.0 / max(v, 0.005) for c, v in vols.items()}
        total = sum(inv_vols.values())
        if total == 0:
            return self._equal_weight(codes)

        return {c: v / total for c, v in inv_vols.items()}

    def _equal_risk_contribution(self, price_matrix: pd.DataFrame, codes: list[str]) -> dict[str, float]:
        """等风险贡献（简化版：用波动率倒数近似）"""
        # 完整ERC需要协方差矩阵求逆，这里用简化版
        return self._risk_parity(price_matrix, codes)

    def _kelly_allocation(
        self, price_matrix: pd.DataFrame, codes: list[str],
        trade_history: pd.DataFrame | None, factor_scores: dict[str, float] | None,
    ) -> dict[str, float]:
        """
        凯利准则分配:
        f* = (p * b - (1 - p)) / b  →  fraction of capital
        其中 p=胜率, b=平均盈亏比

        多资产时按凯利分数分配，剩余现金保留
        """
        if trade_history is not None and not trade_history.empty:
            win_rate = (trade_history["return_pct"] > 0).mean()
            avg_win = trade_history.loc[trade_history["return_pct"] > 0, "return_pct"].mean()
            avg_loss = abs(trade_history.loc[trade_history["return_pct"] < 0, "return_pct"].mean())
            if avg_loss > 0:
                b_ratio = avg_win / avg_loss
            else:
                b_ratio = 1.5
        else:
            # 默认：基于IC实证的估计
            win_rate = 0.52
            b_ratio = 1.3

        # 凯利公式
        if b_ratio > 0:
            kelly_f = (win_rate * b_ratio - (1 - win_rate)) / b_ratio
            kelly_f = max(0, min(kelly_f, 0.5))  # 上限50%
        else:
            kelly_f = 0.1

        kelly_f *= self.kelly_fraction  # 半凯利

        # 分配到各股票
        n = len(codes)
        if n == 0:
            return {}

        # 基础仓位 = 凯利比例 / N
        base_weight = kelly_f / n

        weights = {c: base_weight for c in codes}

        # 如有因子得分，微调：得分高的多配
        if factor_scores:
            scores = np.array([factor_scores.get(c, 0) for c in codes])
            if scores.std() > 0:
                score_adj = 1.0 + np.clip(scores / max(abs(scores.max()), 0.01), -0.5, 0.5) * 0.3
                total = sum(base_weight * sa for sa in score_adj)
                if total > 0:
                    weights = {c: base_weight * score_adj[i] / total * kelly_f
                               for i, c in enumerate(codes)}

        return weights

    def _vol_target(self, price_matrix: pd.DataFrame, codes: list[str]) -> dict[str, float]:
        """
        波动率目标：控制组合整体波动率
        仓位 = target_vol / realized_vol
        """
        vols = {}
        for code in codes:
            rets = price_matrix[code].pct_change().dropna().tail(self.vol_lookback)
            if len(rets) < 20:
                vols[code] = 0.03
            else:
                vols[code] = rets.std()

        # 组合波动率 ≈ 平均个股东波动率 / sqrt(N)
        n = len(codes)
        if n == 0:
            return {}
        avg_vol = np.mean(list(vols.values())) * np.sqrt(252)  # 年化
        avg_vol = max(avg_vol, 0.05)

        # 目标仓位 = target_vol / avg_vol
        total_exposure = min(self.target_vol / avg_vol, 1.0)

        # 按1/vol分配
        inv_vols = {c: 1.0 / max(v, 0.005) for c, v in vols.items()}
        inv_total = sum(inv_vols.values())
        if inv_total == 0:
            return self._equal_weight(codes)

        weights = {c: (v / inv_total) * total_exposure for c, v in inv_vols.items()}
        return weights

    def _score_weighted(self, codes: list[str], factor_scores: dict[str, float] | None) -> dict[str, float]:
        """按综合得分加权（需要分数都为正）"""
        if not factor_scores:
            return self._equal_weight(codes)

        scores = {c: max(factor_scores.get(c, 0), 0.01) for c in codes}
        total = sum(scores.values())
        if total == 0:
            return self._equal_weight(codes)
        return {c: s / total for c, s in scores.items()}

    def _equal_weight(self, codes: list[str]) -> dict[str, float]:
        n = len(codes)
        if n == 0:
            return {}
        return {c: 1.0 / n for c in codes}

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """应用仓位上下限约束"""
        if not weights:
            return weights

        # 裁剪到 [min, max]
        clipped = {}
        for c, w in weights.items():
            clipped[c] = np.clip(w, self.min_single, self.max_single)

        # 重新归一化
        total = sum(clipped.values())
        if total == 0:
            return weights

        return {c: w / total for c, w in clipped.items()}


# ============================================================
# 便捷函数
# ============================================================

def compute_position_sizes(
    price_matrix: pd.DataFrame,
    selected_codes: list[str],
    method: str = "risk_parity",
    total_capital: float = 1_000_000,
    **kwargs,
) -> pd.DataFrame:
    """
    计算推荐仓位，返回DataFrame

    返回: DataFrame with columns: 代码, 权重, 金额, 股数(手)
    """
    sizer = PositionSizer(method=method, **kwargs)
    weights = sizer.allocate(price_matrix, selected_codes, total_capital)

    if not weights:
        return pd.DataFrame()

    rows = []
    for code in sorted(selected_codes, key=lambda c: weights.get(c, 0), reverse=True):
        w = weights.get(code, 0)
        amount = total_capital * w

        # 获取最新价格
        if code in price_matrix.columns:
            latest_price = float(price_matrix[code].dropna().iloc[-1])
        else:
            latest_price = 0

        # A股最小交易单位100股（1手）
        if latest_price > 0:
            shares = int(amount / latest_price / 100) * 100
        else:
            shares = 0

        rows.append({
            "代码": code, "权重": f"{w:.2%}",
            "金额": f"{amount:,.0f}",
            "最新价": f"{latest_price:.2f}",
            "建议股数": f"{shares}股 ({shares // 100}手)",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 55)
    print("  仓位管理测试")
    print("=" * 55)

    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    codes = ["000001", "000002", "600036", "600519", "300750"]
    price_data = {}
    for c in codes:
        vol = np.random.uniform(0.015, 0.035)
        rets = np.random.randn(len(dates)) * vol + 0.0005
        price_data[c] = 100 * np.cumprod(1 + rets)

    price_matrix = pd.DataFrame(price_data, index=dates)

    for method in ["risk_parity", "vol_target", "kelly"]:
        print(f"\n  --- {method} ---")
        df = compute_position_sizes(
            price_matrix, codes, method=method,
            total_capital=1_000_000,
        )
        print(df.to_string(index=False))
