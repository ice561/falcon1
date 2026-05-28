"""
猎鹰一号 — A股多因子截面排序策略

核心逻辑:
  每20个交易日对全A股截面做6因子加权打分，选Top-K等权持有，
  配合个股止损(-8%)、组合止损(-15%)、T+1执行、ST/次新股/低流动性排除。

用法:
  python a_share_strategy.py              # 完整回测
  python a_share_strategy.py --quick      # 快速测试（少量股票）
"""
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import time
import sys
import numpy as np
import pandas as pd

import data_source as ds
import factors as ft
import backtest as bt
import visualize as viz


# ============================================================
# Step 1: 配置类 + 结果类
# ============================================================

@dataclass
class StrategyConfig:
    """策略参数"""
    # 回测区间
    start_date: str = "20210101"
    end_date: str = ""

    # 选股
    top_k: int = 15
    rebalance_freq: int = 20         # 调仓间隔（交易日）

    # 因子权重（须和为1）— IC实证 v2 (2026-05 更新)
    w_momentum: float = 0.17         # Alpha001 (风险调整反转, IR=+0.59)
    w_reversal: float = 0.17         # 5日反转 (IR=+0.57)
    w_vol_price: float = 0.17        # -RSI(14) (低RSI=高收益, IR=+0.70)
    w_low_vol: float = 0.17          # 低波动 (IR=+0.48)
    w_small_cap: float = 0.15        # 小市值 (IR=+0.19)
    w_quality: float = 0.17          # 长期反转 (IR=+0.59)

    # 风控
    stop_loss: float = 0.08          # 个股止损 8%
    take_profit: float = 0.25        # 个股止盈 25%（0=不限）
    trailing_stop: float = 0.12      # 移动止盈 12%（从最高点回撤，0=不限）
    portfolio_stop: float = 0.15     # 组合止损 15% → 减半仓
    max_per_sector: int = 0          # 单行业上限（0=不限）

    # 择时
    use_market_timing: bool = False   # 大盘趋势过滤
    market_index: str = "000001"     # 上证指数
    ma_period: int = 60              # 均线周期（指数 < MA → 空仓）

    # 信号质量
    min_score_threshold: float = 0.5  # 最低综合得分（Z-Score，0=均值）

    # 成本
    commission: float = 0.0003       # 万三
    slippage: float = 0.0001         # 万一
    t_plus_1: bool = True            # T+1 执行

    # 数据
    adjust: str = "qfq"
    cache_dir: str = "./cache"
    download_workers: int = 4

    # 快速模式
    quick_mode: bool = False
    quick_n: int = 300               # 快速模式下随机选N只股票

    def __post_init__(self):
        ws = [self.w_momentum, self.w_reversal, self.w_vol_price,
              self.w_low_vol, self.w_small_cap, self.w_quality]
        total = sum(ws)
        if abs(total - 1.0) > 0.001:
            print(f"[警告] 因子权重和为 {total:.2f}，已自动归一化")
            self.w_momentum /= total
            self.w_reversal /= total
            self.w_vol_price /= total
            self.w_low_vol /= total
            self.w_small_cap /= total
            self.w_quality /= total


@dataclass
class StrategyResult:
    """策略回测结果（兼容 visualize.py 的 BacktestResult 接口）"""
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame      # (Date × Stock) 持仓权重
    trades: pd.DataFrame         # 逐笔交易
    metrics: dict                # 绩效指标
    benchmark: pd.Series         # 全市场等权基准
    selections: dict = field(default_factory=dict)  # {调仓日: [股票列表]}


# ============================================================
# Step 2-3: 缓存层
# ============================================================

class KlineCache:
    """K线数据缓存到 ./cache/klines/"""

    def __init__(self, cache_dir: str = "./cache"):
        self.dir = os.path.join(cache_dir, "klines")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, code: str, adjust: str) -> str:
        return os.path.join(self.dir, f"{code}_{adjust}.pkl")

    def get(self, code: str, adjust: str, start_date: str, end_date: str):
        """读取缓存，检查日期覆盖"""
        p = self._path(code, adjust)
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_pickle(p)
            if df.empty or "date" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"])
            last = df["date"].max()
            if not end_date:
                req_end = pd.Timestamp.now()
            else:
                req_end = pd.Timestamp(end_date)
            if last >= req_end - pd.Timedelta(days=1):
                # 缓存够新，过滤日期返回
                if start_date:
                    df = df[df["date"] >= pd.Timestamp(start_date)]
                return df
            return None
        except Exception:
            return None

    def put(self, code: str, adjust: str, df: pd.DataFrame):
        if df.empty:
            return
        df.to_pickle(self._path(code, adjust))

    def download_batch(self, codes: list[str], start_date: str, end_date: str,
                       adjust: str, max_workers: int = 4) -> dict[str, pd.DataFrame]:
        """批量下载K线，已缓存的跳过，并发下载缺失的"""
        cached = {}
        missing = []

        for c in codes:
            df = self.get(c, adjust, start_date, end_date)
            if df is not None and len(df) >= 20:
                cached[c] = df.set_index("date") if "date" in df.columns else df
            else:
                missing.append(c)

        print(f"  K线缓存: {len(cached)} 命中, {len(missing)} 需下载")

        if not missing:
            return cached

        downloaded = {}
        total = len(missing)
        done = 0

        def _download(code):
            try:
                df = ds.get_daily_kline(code, start_date=start_date, end_date=end_date, adjust=adjust)
                if not df.empty:
                    self.put(code, adjust, df)
                return code, df
            except Exception as e:
                return code, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download, c): c for c in missing}
            for f in as_completed(futures):
                code, df = f.result()
                done += 1
                if done % 100 == 0:
                    print(f"    下载进度: {done}/{total}")
                if not df.empty:
                    downloaded[code] = df.set_index("date") if "date" in df.columns else df

            # 等待全部完成
            for f in as_completed(futures):
                pass

        print(f"  K线下载完成: {len(downloaded)} 只")
        return {**cached, **downloaded}


class FactorCache:
    """因子缓存到 ./cache/factors/"""

    def __init__(self, cache_dir: str = "./cache"):
        self.dir = os.path.join(cache_dir, "factors")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, code: str) -> str:
        return os.path.join(self.dir, f"{code}.pkl")

    def get(self, code: str) -> pd.DataFrame | None:
        p = self._path(code)
        if not os.path.exists(p):
            return None
        try:
            return pd.read_pickle(p)
        except Exception:
            return None

    def put(self, code: str, df: pd.DataFrame):
        if not df.empty:
            df.to_pickle(self._path(code))

    def compute_batch(self, kline_dict: dict[str, pd.DataFrame],
                      stock_list: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """批量计算因子，已缓存跳过"""
        cached = {}
        missing_codes = []
        for code in kline_dict:
            f = self.get(code)
            if f is not None and len(f) > 20:
                cached[code] = f
            else:
                missing_codes.append(code)

        print(f"  因子缓存: {len(cached)} 命中, {len(missing_codes)} 需计算")

        if not missing_codes:
            return cached

        # 从 stock_list 获取市值基准
        mv_map = {}
        if "code" in stock_list.columns and "total_mv" in stock_list.columns:
            mv_map = dict(zip(stock_list["code"], stock_list["total_mv"]))

        computed = {}
        total = len(missing_codes)
        for i, code in enumerate(missing_codes):
            df = kline_dict.get(code)
            if df is None or df.empty or "close" not in df.columns:
                continue
            try:
                f = compute_factors_for_stock(df, mv_map.get(code))
                self.put(code, f)
                computed[code] = f
                if (i + 1) % 500 == 0:
                    print(f"    因子计算进度: {i + 1}/{total}")
            except Exception:
                continue

        print(f"  因子计算完成: {len(computed)} 只")
        return {**cached, **computed}


# ============================================================
# 板块分类
# ============================================================

def get_board(code: str) -> str:
    """按代码前缀划分交易板块"""
    if code.startswith("60"):
        return "上海主板"
    elif code.startswith(("00", "001", "002", "003")):
        return "深圳主板"
    elif code.startswith(("300", "301")):
        return "创业板"
    elif code.startswith("688"):
        return "科创板"
    elif code.startswith(("83", "87", "92")):
        return "北交所"
    return "其他"


# ============================================================
# Step 4: 自定义因子
# ============================================================

def vol_price_corr(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    量价配合因子：价格方向与成交量变化的N日滚动相关系数
    正相关 = 价涨量增 = 强势
    """
    price_dir = np.sign(df["close"].diff())
    vol_chg = df["volume"].pct_change()
    corr = price_dir.rolling(n).corr(vol_chg)
    return corr


def inverse_log_mv(df: pd.DataFrame, total_mv_base: float | None) -> pd.Series:
    """对数市值倒数因子：市值越小 → 值越大"""
    if total_mv_base is None or np.isnan(total_mv_base) or total_mv_base <= 0:
        return pd.Series(np.nan, index=df.index)
    last_close = df["close"].iloc[-1]
    if last_close <= 0 or np.isnan(last_close):
        return pd.Series(np.nan, index=df.index)
    scale = df["close"] / last_close
    approx_mv = total_mv_base * scale
    return -np.log(approx_mv.replace(0, np.nan))


def compute_factors_for_stock(df: pd.DataFrame, total_mv_base: float | None) -> pd.DataFrame:
    """对单只股票计算全部因子（现有 + 自定义）"""
    base = ft.compute_all(df)
    base["vol_price_corr_10"] = vol_price_corr(df, 10)
    base["inv_log_mv"] = inverse_log_mv(df, total_mv_base)
    base["neg_dd_20"] = -ft.max_drawdown_factor(df, 20)
    base["inv_vol_20"] = 1.0 / ft.hist_volatility(df, 20).replace(0, np.nan)
    base["neg_momentum_20"] = -base["momentum_20"]
    base["neg_rsi_14"] = -base["rsi_14"]  # 长期反转（IC实证：A股20日动量反向）
    base = base.replace([np.inf, -np.inf], np.nan)
    return base


# ============================================================
# Step 5: 数据准备
# ============================================================

class MultiFactorStrategy:
    """猎鹰一号多因子策略"""

    def __init__(self, config: StrategyConfig = None, **kwargs):
        self.cfg = config or StrategyConfig()
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.kline_cache = KlineCache(self.cfg.cache_dir)
        self.factor_cache = FactorCache(self.cfg.cache_dir)
        self._stock_list = None
        self._index_ma = None  # 大盘均线 Series

    def _load_index_data(self) -> tuple[pd.Series, pd.Series]:
        """加载大盘指数 close + MA均线，返回 (index_close, index_ma)"""
        try:
            df = ds.get_index_kline(self.cfg.market_index, start_date=self.cfg.start_date)
            if df.empty:
                print("  [警告] 无法获取指数数据，跳过择时")
                return None, None
            df = df.set_index("date") if "date" in df.columns else df
            close = df["close"]
            ma = close.rolling(self.cfg.ma_period).mean()
            return close, ma
        except Exception as e:
            print(f"  [警告] 指数数据获取失败: {e}")
            return None, None

    def _is_market_ok(self, date, index_close: pd.Series, index_ma: pd.Series) -> bool:
        """检查当日大盘收盘价是否在MA上方（牛市）"""
        if index_ma is None or index_close is None:
            return True
        c = index_close[index_close.index <= date]
        m = index_ma[index_ma.index <= date]
        if c.empty or m.empty or pd.isna(m.iloc[-1]):
            return True
        return float(c.iloc[-1]) > float(m.iloc[-1])

    def run(self) -> StrategyResult:
        t0 = time.time()

        # 1. 获取全市场（带 fallback）
        print("\n[1/5] 获取全A股列表...")
        try:
            self._stock_list = ds.get_stock_list()
        except Exception:
            self._stock_list = pd.DataFrame()
        if self._stock_list.empty:
            # fallback: 从 K线缓存目录提取股票代码
            self._stock_list = self._codes_from_kline_cache()
        print(f"  全市场: {len(self._stock_list)} 只")

        # 2. 确定股票池
        codes = self._get_universe()

        # 3. 下载K线
        print(f"\n[2/5] 下载K线数据 ({len(codes)} 只)...")
        kline_dict = self.kline_cache.download_batch(
            codes, self.cfg.start_date, self.cfg.end_date,
            self.cfg.adjust, self.cfg.download_workers,
        )

        # 4. 计算因子
        print(f"\n[3/5] 计算因子...")
        factor_dict = self.factor_cache.compute_batch(kline_dict, self._stock_list)

        # 5. 构建矩阵
        print(f"\n[4/5] 构建 price/amount 矩阵...")
        price_matrix, amount_matrix = self._build_matrices(kline_dict)

        # 5b. 加载大盘指数（择时用）
        index_close = None
        index_ma = None
        if self.cfg.use_market_timing:
            index_close, index_ma = self._load_index_data()

        # 6. 信号生成
        print(f"\n[5/5] 生成信号并模拟...")
        signal_matrix, weight_matrix, selections = self._generate_signals(
            price_matrix, factor_dict, amount_matrix, index_close, index_ma
        )

        # 7. 组合模拟
        result = self._simulate(price_matrix, weight_matrix)
        result.selections = selections

        # 8. 基准
        result.benchmark = self._compute_benchmark(price_matrix)

        # 9. 绩效
        result.metrics = bt.compute_metrics(result.equity, result.returns, result.trades)

        elapsed = time.time() - t0
        print(f"\n  总耗时: {elapsed / 60:.1f} 分钟")

        return result

    # ---- 内部方法 ----

    def _codes_from_kline_cache(self) -> pd.DataFrame:
        """从 K线缓存目录反向提取股票代码列表"""
        kline_dir = os.path.join(self.cfg.cache_dir, "klines")
        codes = []
        if os.path.exists(kline_dir):
            for fname in os.listdir(kline_dir):
                if fname.endswith(".pkl"):
                    code = fname.split("_")[0]
                    if code.isdigit() and len(code) == 6:
                        codes.append(code)
        codes = list(set(codes))
        # 构造最小 DataFrame
        df = pd.DataFrame({"code": codes, "name": codes, "total_mv": [np.nan] * len(codes)})
        print(f"  [fallback] 从缓存提取 {len(df)} 个代码")
        return df

    def _get_universe(self) -> list[str]:
        """确定候选股票池"""
        df = self._stock_list

        # 排除ST
        df = df[~df["name"].str.contains(r"\*?ST", na=False)]

        # 快速模式：随机选 N 只
        if self.cfg.quick_mode:
            df = df.sample(min(self.cfg.quick_n, len(df)), random_state=42)
            print(f"  [快速模式] 随机选取 {len(df)} 只")

        codes = df["code"].tolist()
        print(f"  候选池: {len(codes)} 只（排除ST后）")
        return codes

    def _build_matrices(self, kline_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """构建 price_matrix 和 amount_matrix"""
        prices = {}
        amounts = {}
        for code, df in kline_dict.items():
            if "close" not in df.columns:
                continue
            s = df["close"].dropna()
            if len(s) >= 60:
                prices[code] = s
            if "amount" in df.columns:
                a = df["amount"].dropna()
                if len(a) >= 60:
                    amounts[code] = a

        price_matrix = pd.DataFrame(prices).sort_index()
        amount_matrix = pd.DataFrame(amounts).sort_index() if amounts else pd.DataFrame(index=price_matrix.index)
        return price_matrix, amount_matrix

    # ---- Step 6: 信号生成 ----

    def _generate_signals(
        self, price_matrix: pd.DataFrame,
        factor_dict: dict[str, pd.DataFrame],
        amount_matrix: pd.DataFrame,
        index_close: pd.Series = None,
        index_ma: pd.Series = None,
    ):
        dates = price_matrix.index.sort_values()

        # 确定调仓日：从第60个交易日开始，每 rebalance_freq 日
        rebalance_dates = []
        for i, d in enumerate(dates):
            if i >= 60 and (i - 60) % self.cfg.rebalance_freq == 0:
                rebalance_dates.append(d)

        rb_str = f"{rebalance_dates[0].date()} ~ {rebalance_dates[-1].date()}" if rebalance_dates else "N/A"
        print(f"  调仓日: {len(rebalance_dates)} 个 ({rb_str})")

        # 因子名映射（基于IC实证）
        factor_cols = {
            "neg_momentum_20": self.cfg.w_quality,   # 长期反转 (IR=+0.59)
            "inv_vol_20": self.cfg.w_low_vol,         # 低波动 (IR=+0.48)
            "reversal_5": self.cfg.w_reversal,        # 5日反转 (IR=+0.57)
            "alpha_001": self.cfg.w_momentum,         # Alpha001 (IR=+0.59)
            "neg_rsi_14": self.cfg.w_vol_price,       # -RSI (IR=+0.70)
            "inv_log_mv": self.cfg.w_small_cap,       # 小市值 (IR=+0.19)
        }
        weights = list(factor_cols.values())

        n_stocks = len([c for c in price_matrix.columns if c in factor_dict])
        n_dates = len(dates)
        signal_matrix = pd.DataFrame(0, index=dates, columns=price_matrix.columns)
        selections = {}
        skip_count = 0

        for di, rd in enumerate(rebalance_dates):
            # ---- 大盘择时：指数 < MA 则空仓 ----
            if self.cfg.use_market_timing and not self._is_market_ok(rd, index_close, index_ma):
                skip_count += 1
                selections[rd] = []  # 空仓
                continue

            # 有效日 = rd 或 下一交易日（T+1）
            idx = dates.get_loc(rd)
            if self.cfg.t_plus_1 and idx + 1 < n_dates:
                eff_date = dates[idx + 1]
            else:
                eff_date = rd

            # ---- 筛选当日有效股票 ----
            valid_codes = []
            factor_rows = []

            for code in price_matrix.columns:
                if code not in factor_dict:
                    continue
                price = price_matrix.loc[rd, code] if rd in price_matrix.index else np.nan
                if pd.isna(price) or price <= 0:
                    continue

                # 流动性：近20日日均成交额 > 5000万
                if not amount_matrix.empty and code in amount_matrix.columns:
                    amt_hist = amount_matrix[code].loc[:rd].tail(20)
                    if len(amt_hist) < 10 or amt_hist.mean() < 50_000_000:
                        continue

                # 因子值
                fdf = factor_dict[code]
                if rd not in fdf.index:
                    continue
                # 确保 neg_momentum_20 存在（兼容旧缓存）
                if "neg_momentum_20" not in fdf.columns and "momentum_20" in fdf.columns:
                    fdf["neg_momentum_20"] = -fdf["momentum_20"]
                if "neg_rsi_14" not in fdf.columns and "rsi_14" in fdf.columns:
                    fdf["neg_rsi_14"] = -fdf["rsi_14"]
                frow = fdf.loc[rd]
                vals = []
                ok = True
                for fc in factor_cols:
                    if fc in frow.index:
                        v = frow[fc]
                        if pd.isna(v):
                            ok = False
                            break
                        vals.append(v)
                    else:
                        ok = False
                        break
                if not ok:
                    continue

                valid_codes.append(code)
                factor_rows.append(vals)

            if len(valid_codes) < self.cfg.top_k:
                continue

            # ---- Z-Score + Winsorize + 加权 ----
            farr = np.array(factor_rows)  # (N_stocks, 6)
            zarr = np.zeros_like(farr)

            for j in range(farr.shape[1]):
                col = farr[:, j]
                mean = np.nanmean(col)
                std = np.nanstd(col)
                if std == 0 or np.isnan(std):
                    zarr[:, j] = 0
                else:
                    z = (col - mean) / std
                    z = np.clip(z, -3.0, 3.0)  # Winsorize
                    zarr[:, j] = z

            composite = zarr @ np.array(weights)

            # ---- 信号质量阈值 ----
            if self.cfg.min_score_threshold > 0:
                valid_indices = [i for i in range(len(composite))
                                 if composite[i] >= self.cfg.min_score_threshold]
                if len(valid_indices) < max(1, self.cfg.top_k // 3):
                    continue  # 合格股票太少，跳过本期
            else:
                valid_indices = list(range(len(composite)))

            # ---- Top-K 选股（含行业分散） ----
            # 只从达标股票中选
            sorted_indices = [i for i in np.argsort(composite)[::-1]
                             if i in set(valid_indices)]
            if self.cfg.max_per_sector > 0:
                selected = []
                sector_counts = {}
                for idx in sorted_indices:
                    code = valid_codes[idx]
                    sector = get_board(code)
                    if sector_counts.get(sector, 0) < self.cfg.max_per_sector:
                        selected.append(code)
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if len(selected) >= self.cfg.top_k:
                        break
            else:
                take_n = min(self.cfg.top_k, len(sorted_indices))
                selected = [valid_codes[i] for i in sorted_indices[:take_n]]

            selections[eff_date] = selected

            # ---- 填入 signal/weight 矩阵 ----
            next_idx = di + 1
            if next_idx < len(rebalance_dates):
                if self.cfg.t_plus_1:
                    next_eff = dates[dates.get_loc(rebalance_dates[next_idx]) + 1] if dates.get_loc(rebalance_dates[next_idx]) + 1 < n_dates else rebalance_dates[next_idx]
                else:
                    next_eff = rebalance_dates[next_idx]
            else:
                next_eff = dates[-1]

            mask = (dates >= eff_date) & (dates < next_eff)
            for c in selected:
                if c in signal_matrix.columns:
                    signal_matrix.loc[mask, c] = 1

        # weight_matrix：等权
        weight_matrix = signal_matrix.div(signal_matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        if skip_count > 0:
            print(f"  择时空仓: {skip_count}/{len(rebalance_dates)} 个调仓日")

        return signal_matrix, weight_matrix, selections

    # ---- Step 7: 组合模拟 ----

    def _simulate(self, price_matrix: pd.DataFrame,
                  weight_matrix: pd.DataFrame) -> StrategyResult:
        """
        逐日模拟：现金 + 持仓市值模型
        positions: {code: shares}  # shares = 买入时投入的金额 / 买入价
        cash: 当前可用现金
        """
        dates = price_matrix.index.sort_values()
        n = len(dates)
        if n == 0:
            return StrategyResult(pd.Series(), pd.Series(), pd.DataFrame(), pd.DataFrame(), {}, pd.Series())

        # 初始：全部现金
        cash = 1.0
        positions = {}        # {code: shares}
        entry_prices = {}     # {code: entry_price} — 用于止损判断
        peak_prices = {}      # {code: peak_price}  — 用于移动止盈
        portfolio_peak = 1.0
        portfolio_stopped = False
        trades = []
        equity = pd.Series(1.0, index=dates, dtype=float)
        prev_weights = pd.Series(0.0, index=weight_matrix.columns)

        for i in range(n):
            d = dates[i]
            curr_weights = weight_matrix.loc[d] if d in weight_matrix.index else prev_weights

            is_rebalance = not curr_weights.equals(prev_weights)

            if is_rebalance:
                # ---- 卖出所有持仓 ----
                for code in list(positions.keys()):
                    sell_price = self._get_price(price_matrix, code, d)
                    if pd.notna(sell_price) and sell_price > 0:
                        cash += positions[code] * sell_price * (1 - self.cfg.commission - self.cfg.slippage)
                        ret = (sell_price / entry_prices[code] - 1) * 100
                        trades.append({
                            "code": code, "entry_date": entry_prices.get(f"{code}_date", d),
                            "exit_date": d, "entry_price": entry_prices[code],
                            "exit_price": sell_price, "return_pct": round(ret, 2),
                            "hold_days": (d - entry_prices.get(f"{code}_date", d)).days,
                        })
                    positions.pop(code, None)
                    entry_prices.pop(code, None)
                    peak_prices.pop(code, None)

                # ---- 买入新持仓 ----
                active = curr_weights[curr_weights > 0]
                K = len(active)
                if K > 0 and cash > 0:
                    total_equity = cash  # 当前总资产（卖出后全是现金）
                    allocate_per_stock = total_equity / K  # 等权分配
                    for code in active.index:
                        buy_price = self._get_price(price_matrix, code, d)
                        if pd.notna(buy_price) and buy_price > 0:
                            cost = buy_price * (1 + self.cfg.slippage + self.cfg.commission)
                            shares = allocate_per_stock / cost
                            positions[code] = shares
                            entry_prices[code] = buy_price
                            peak_prices[code] = buy_price
                            entry_prices[f"{code}_date"] = d
                            cash -= shares * cost
                    # 因取整/价格差异，现金可能略有剩余，保留在 cash 中

                portfolio_stopped = False

            # ---- 每日计价 ----
            stock_value = 0.0
            for code, shares in list(positions.items()):
                p = self._get_price(price_matrix, code, d, fallback=True, prev_d=dates[i - 1] if i > 0 else None)
                if pd.isna(p) or p <= 0:
                    continue

                # 更新持仓期间最高价
                if code in peak_prices and p > peak_prices[code]:
                    peak_prices[code] = p

                exit_price = None
                exit_reason = ""

                # 1. 个股止损
                stop_price = entry_prices[code] * (1 - self.cfg.stop_loss)
                if p < stop_price:
                    exit_price = stop_price
                    exit_reason = "止损"

                # 2. 个股止盈
                elif self.cfg.take_profit > 0:
                    tp_price = entry_prices[code] * (1 + self.cfg.take_profit)
                    if p >= tp_price:
                        exit_price = p
                        exit_reason = "止盈"

                # 3. 移动止盈（从最高点回撤）
                elif self.cfg.trailing_stop > 0 and code in peak_prices:
                    trail_price = peak_prices[code] * (1 - self.cfg.trailing_stop)
                    if p < trail_price and peak_prices[code] > entry_prices[code] * 1.05:
                        exit_price = p
                        exit_reason = "移动止盈"

                if exit_price is not None:
                    cash += shares * exit_price * (1 - self.cfg.commission - self.cfg.slippage)
                    ret = (exit_price / entry_prices[code] - 1) * 100
                    trades.append({
                        "code": code, "entry_date": entry_prices.get(f"{code}_date", d),
                        "exit_date": d, "entry_price": entry_prices[code],
                        "exit_price": exit_price, "return_pct": round(ret, 2),
                        "hold_days": (d - entry_prices.get(f"{code}_date", d)).days,
                    })
                    positions.pop(code, None)
                    entry_prices.pop(code, None)
                    peak_prices.pop(code, None)
                    continue

                stock_value += shares * p

            total_value = cash + stock_value
            equity.iloc[i] = total_value

            # ---- 组合止损 ----
            portfolio_peak = max(portfolio_peak, total_value)
            dd = (total_value - portfolio_peak) / portfolio_peak
            if dd < -self.cfg.portfolio_stop and not portfolio_stopped and positions:
                # 减半仓：卖出一半持仓
                for code in list(positions.keys()):
                    sell_shares = positions[code] / 2
                    sell_price = self._get_price(price_matrix, code, d)
                    if pd.notna(sell_price) and sell_price > 0:
                        cash += sell_shares * sell_price * (1 - self.cfg.commission - self.cfg.slippage)
                        positions[code] -= sell_shares
                portfolio_stopped = True

            prev_weights = curr_weights

        # 期末清仓
        for code in list(positions.keys()):
            last_p = price_matrix[code].dropna().iloc[-1] if code in price_matrix.columns and not price_matrix[code].dropna().empty else 0
            if last_p > 0:
                cash += positions[code] * last_p * (1 - self.cfg.commission - self.cfg.slippage)
                ret = (last_p / entry_prices[code] - 1) * 100
                trades.append({
                    "code": code, "entry_date": entry_prices.get(f"{code}_date", dates[-1]),
                    "exit_date": dates[-1], "entry_price": entry_prices[code],
                    "exit_price": last_p, "return_pct": round(ret, 2),
                    "hold_days": (dates[-1] - entry_prices.get(f"{code}_date", dates[-1])).days,
                })
            del positions[code]
            del entry_prices[code]

        returns = equity.pct_change().fillna(0)

        return StrategyResult(
            equity=equity,
            returns=returns,
            positions=weight_matrix.copy(),
            trades=pd.DataFrame(trades),
            metrics={},
            benchmark=pd.Series(),
        )

    @staticmethod
    def _get_price(price_matrix, code, date, fallback=False, prev_d=None):
        """获取价格，处理 NaN / 停牌"""
        if code not in price_matrix.columns:
            return np.nan
        p = price_matrix.loc[date, code] if date in price_matrix.index else np.nan
        if (pd.isna(p) or p <= 0) and fallback and prev_d is not None:
            p = price_matrix.loc[prev_d, code] if prev_d in price_matrix.index else np.nan
        return p if (pd.notna(p) and p > 0) else np.nan

    # ---- Step 8: 基准 ----

    def _compute_benchmark(self, price_matrix: pd.DataFrame) -> pd.Series:
        """全市场等权基准"""
        bench = price_matrix.mean(axis=1)
        bench = bench.dropna()
        if bench.empty:
            return pd.Series()
        return bench / bench.iloc[0]


# ============================================================
# Step 10: 便捷入口
# ============================================================

def run_full(start_date: str = "20210101", top_k: int = 15, **kwargs) -> StrategyResult:
    """完整回测"""
    cfg = StrategyConfig(start_date=start_date, top_k=top_k, **kwargs)
    strategy = MultiFactorStrategy(cfg)
    return strategy.run()


def run_quick(n_stocks: int = 300, start_date: str = "20230101",
              top_k: int = 10, **kwargs) -> StrategyResult:
    """快速测试：随机选 N 只股票"""
    cfg = StrategyConfig(
        start_date=start_date, top_k=top_k,
        quick_mode=True, quick_n=n_stocks, **kwargs,
    )
    strategy = MultiFactorStrategy(cfg)
    return strategy.run()


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv

    if quick:
        print("=" * 55)
        print("  猎鹰一号 — 快速测试模式 (300只)")
        print("=" * 55)
        result = run_quick(n_stocks=300, start_date="20230101", top_k=10)
    else:
        print("=" * 55)
        print("  猎鹰一号 — 完整回测")
        print("=" * 55)
        result = run_full(start_date="20210101", top_k=15)

    # 打印绩效
    bt.print_metrics(result.metrics)

    # 图表
    print("\n  生成图表...")
    try:
        viz.plot_equity_curve(result, title="猎鹰一号 — 净值曲线", save="falcon1_equity.png")
    except Exception as e:
        print(f"  净值图失败: {e}")
    try:
        viz.plot_monthly_heatmap(result.returns, title="猎鹰一号 — 月度收益热力图", save="falcon1_heatmap.png")
    except Exception as e:
        print(f"  热力图失败: {e}")
    try:
        viz.plot_returns_distribution(result.returns, save="falcon1_dist.png")
    except Exception as e:
        print(f"  收益分布图失败: {e}")

    # 选股摘要
    if result.selections:
        print("\n  === 最近5次调仓 ===")
        dates_sorted = sorted(result.selections.keys())
        for d in dates_sorted[-5:]:
            stocks = result.selections[d]
            print(f"  {d.date()}: {', '.join(stocks[:8])}{'...' if len(stocks) > 8 else ''}")
