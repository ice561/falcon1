"""
因子计算模块 — 技术因子 + 量价因子 + 形态因子
所有因子返回 Series，索引与输入对齐
"""
import numpy as np
import pandas as pd


# ========== 收益率因子 ==========

def momentum(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """N日动量（收益率）"""
    return df["close"].pct_change(n)


def reversal(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """短期反转（负的短期收益 → 正因子值）"""
    return -df["close"].pct_change(n)


def log_return(df: pd.DataFrame, n: int = 1) -> pd.Series:
    """对数收益率"""
    return np.log(df["close"] / df["close"].shift(n))


# ========== 均线因子 ==========

def sma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """简单移动均线"""
    return df["close"].rolling(n).mean()


def ema(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """指数移动均线"""
    return df["close"].ewm(span=n, adjust=False).mean()


def ma_bias(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """均线偏离度 (close - MA) / MA"""
    ma = df["close"].rolling(n).mean()
    return (df["close"] - ma) / ma


def ma_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """均线金叉死叉信号 (1=金叉, -1=死叉, 0=无)"""
    ma_fast = df["close"].rolling(fast).mean()
    ma_slow = df["close"].rolling(slow).mean()
    cross = pd.Series(0, index=df.index)
    cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    cross[cross_up] = 1
    cross[cross_down] = -1
    return cross


def ma_spread(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """双均线差值比率"""
    ma_f = df["close"].rolling(fast).mean()
    ma_s = df["close"].rolling(slow).mean()
    return (ma_f - ma_s) / ma_s


# ========== 波动率因子 ==========

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """平均真实波幅"""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def hist_volatility(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """N日历史波动率（年化）"""
    ret = np.log(df["close"] / df["close"].shift(1))
    return ret.rolling(n).std() * np.sqrt(252)


def bollinger_position(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    """布林带位置 (0~1，>1 突破上轨，<0 突破下轨)"""
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return (df["close"] - lower) / (upper - lower)


def bollinger_width(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    """布林带宽度（波动率指标）"""
    ma = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    return (2 * k * std) / ma


# ========== RSI / 超买超卖 ==========

def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """相对强弱指标"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_divergence(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """RSI 的 Z-Score 标准化（去趋势）"""
    rsi_val = rsi(df, n)
    return (rsi_val - rsi_val.rolling(252).mean()) / rsi_val.rolling(252).std()


# ========== MACD ==========

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指标，返回 DataFrame: dif, dea, hist"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist}, index=df.index)


def macd_hist_factor(df: pd.DataFrame) -> pd.Series:
    """MACD 柱 — 正值且放大 = 强，负值且缩小 = 弱"""
    m = macd(df)
    return m["hist"]


def macd_cross(df: pd.DataFrame) -> pd.Series:
    """MACD 金叉死叉 (1/-1/0)"""
    m = macd(df)
    cross = pd.Series(0, index=df.index)
    up = (m["dif"] > m["dea"]) & (m["dif"].shift(1) <= m["dea"].shift(1))
    down = (m["dif"] < m["dea"]) & (m["dif"].shift(1) >= m["dea"].shift(1))
    cross[up] = 1
    cross[down] = -1
    return cross


# ========== 成交量因子 ==========

def volume_ratio(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """量比（当日成交 / N日均量）"""
    vol_ma = df["volume"].rolling(n).mean()
    return df["volume"] / vol_ma.replace(0, np.nan)


def volume_breakout(df: pd.DataFrame, n: int = 20, multiple: float = 1.5) -> pd.Series:
    """放量突破 (1/0)：成交量超过 N 日均量 * multiple"""
    vol_ma = df["volume"].rolling(n).mean()
    return ((df["volume"] / vol_ma.replace(0, np.nan)) > multiple).astype(int)


def obv(df: pd.DataFrame) -> pd.Series:
    """能量潮 OBV"""
    direction = np.sign(df["close"].diff()).fillna(0)
    obv_val = (direction * df["volume"]).cumsum()
    obv_val.name = "obv"
    return obv_val


def vwap(df: pd.DataFrame) -> pd.Series:
    """日内均价（日线近似：用 amount/volume）"""
    if "amount" in df.columns:
        return df["amount"] / df["volume"].replace(0, np.nan)
    return (df["high"] + df["low"] + df["close"]) / 3


def turnover_factor(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """换手率变化（如果无换手率字段，用量比替代）"""
    if "turnover" in df.columns:
        return df["turnover"] / df["turnover"].rolling(n).mean()
    return volume_ratio(df, n)


# ========== 价格形态 ==========

def higher_high(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """N日最高价是否创新高 (1/0)"""
    return (df["high"] == df["high"].rolling(n).max()).astype(int)


def lower_low(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """N日最低价是否创新低 (1/0)"""
    return (df["low"] == df["low"].rolling(n).min()).astype(int)


def gap_up(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """向上跳空缺口"""
    return ((df["open"] - df["close"].shift(1)) / df["close"].shift(1) > threshold).astype(int)


def gap_down(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """向下跳空缺口"""
    return ((df["open"] - df["close"].shift(1)) / df["close"].shift(1) < -threshold).astype(int)


def max_drawdown_factor(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """N日内从最高点的回撤幅度（正值=回撤大）"""
    rolling_max = df["close"].rolling(n).max()
    return (rolling_max - df["close"]) / rolling_max


# ========== 衍生因子 ==========

def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    经典 Alpha 因子：反转 + 波动率调节
    -(close - close_5) / (std_20 * close)  → 短期反转 / 波动率标准化
    """
    ret5 = df["close"].pct_change(5)
    std20 = df["close"].pct_change().rolling(20).std()
    return -ret5 / std20.replace(0, np.nan)


def alpha_002(df: pd.DataFrame) -> pd.Series:
    """
    量价背离因子：价格上涨但量萎缩 = 负面信号
    sign(close_1 - close_20) * (vol_5 / vol_20 - 1)
    """
    price_dir = np.sign(df["close"].diff(20))
    vol_5 = df["volume"].rolling(5).mean()
    vol_20 = df["volume"].rolling(20).mean()
    vol_change = vol_5 / vol_20.replace(0, np.nan) - 1
    return -price_dir * vol_change


def alpha_003(df: pd.DataFrame) -> pd.Series:
    """
    趋势强度：20日收益率 / 日波动率之和（类似夏普比率）
    """
    ret = df["close"].pct_change()
    ret20 = df["close"].pct_change(20)
    volatility = ret.rolling(20).std() * np.sqrt(20)
    return ret20 / volatility.replace(0, np.nan)


# ========== 批量计算 ==========

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """对 K 线 DataFrame 批量计算全部因子，返回因子矩阵"""
    factors = pd.DataFrame(index=df.index)
    factors["momentum_5"] = momentum(df, 5)
    factors["momentum_20"] = momentum(df, 20)
    factors["reversal_5"] = reversal(df, 5)
    factors["ma_bias_20"] = ma_bias(df, 20)
    factors["ma_spread_5_20"] = ma_spread(df, 5, 20)
    factors["rsi_14"] = rsi(df, 14)
    m = macd(df)
    factors["macd_hist"] = m["hist"]
    factors["vol_ratio_5"] = volume_ratio(df, 5)
    factors["atr_14"] = atr(df, 14)
    factors["hist_vol_20"] = hist_volatility(df, 20)
    factors["bb_position"] = bollinger_position(df, 20)
    factors["bb_width"] = bollinger_width(df, 20)
    factors["alpha_001"] = alpha_001(df)
    factors["alpha_002"] = alpha_002(df)
    factors["alpha_003"] = alpha_003(df)
    factors["drawdown_20"] = max_drawdown_factor(df, 20)

    # 对齐
    factors = factors.replace([np.inf, -np.inf], np.nan)
    return factors
