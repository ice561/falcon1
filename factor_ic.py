"""
因子IC分析 — 计算各因子对未来收益的预测能力
IC = Spearman rank correlation(因子值, 未来N日收益)
"""
import os
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FACTOR_DIR = "./cache/factors"
KLINE_DIR = "./cache/klines"
REBALANCE_FREQ = 20

FACTOR_COLS = {
    "momentum_20": "20日动量",
    "reversal_5": "5日反转",
    "vol_price_corr_10": "量价配合",
    "inv_vol_20": "低波动",
    "inv_log_mv": "小市值",
    "neg_dd_20": "质量(低回撤)",
    "rsi_14": "RSI(14)",
    "ma_bias_20": "均线偏离",
    "vol_ratio_5": "量比",
}

def _norm_idx(df):
    """统一索引为 date string YYYY-MM-DD"""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index.strftime("%Y-%m-%d")
    return df

def load_sample_factors(n=500):
    files = [f for f in os.listdir(FACTOR_DIR) if f.endswith(".pkl")]
    if len(files) > n:
        files = random.sample(files, n)
    factor_data = {}
    for f in files:
        code = f.replace(".pkl", "")
        df = pd.read_pickle(os.path.join(FACTOR_DIR, f))
        if not df.empty and len(df) > 60:
            factor_data[code] = _norm_idx(df)
    return factor_data

def load_price_data(codes, adjust="qfq"):
    prices = {}
    for code in codes:
        path = os.path.join(KLINE_DIR, f"{code}_{adjust}.pkl")
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        if df.empty:
            continue
        df = _norm_idx(df)
        if "close" in df.columns:
            prices[code] = df["close"].dropna()
    return prices

def compute_ic(factor_data, price_data, forward_days=20, min_stocks=30):
    # 所有日期按时间排序
    all_dates = sorted(set().union(
        *[set(s.index) for s in price_data.values()],
        *[set(f.index) for f in factor_data.values()]
    ))

    # 每个日期的有效股票数
    date_ok = []
    for d in all_dates:
        n = sum(1 for code in factor_data
                if code in price_data
                and d in price_data[code].index
                and d in factor_data[code].index)
        if n >= min_stocks:
            date_ok.append(d)

    # 每20日一个分析截面
    analysis_dates = date_ok[::REBALANCE_FREQ]
    if len(analysis_dates) < 5:
        print(f"  [错误] 分析截面不足: {len(analysis_dates)}")
        return pd.DataFrame()

    print(f"  分析截面: {len(analysis_dates)} ({analysis_dates[0]} ~ {analysis_dates[-1]})")

    results = {name: [] for name in FACTOR_COLS}
    stock_counts = []

    for ad in analysis_dates:
        try:
            idx = all_dates.index(ad)
        except ValueError:
            continue
        future_idx = min(idx + forward_days, len(all_dates) - 1)
        future_date = all_dates[future_idx]

        factor_vals = {name: [] for name in FACTOR_COLS}
        fwd_returns = []
        n_ok = 0

        for code, fdf in factor_data.items():
            if code not in price_data:
                continue
            price_s = price_data[code]

            if ad not in price_s.index or ad not in fdf.index:
                continue

            p_now = price_s.loc[ad]
            try:
                p_now = float(p_now)
            except (ValueError, TypeError):
                continue
            if p_now <= 0:
                continue

            # 未来20日收益
            fwd_s = price_s[price_s.index >= ad]
            if len(fwd_s) <= forward_days:
                continue
            fwd_p = fwd_s.iloc[forward_days]
            try:
                fwd_p = float(fwd_p)
            except (ValueError, TypeError):
                continue
            if fwd_p <= 0:
                continue
            fwd_ret = (fwd_p / p_now - 1)

            frow = fdf.loc[ad]
            try:
                _ = frow.index
            except Exception:
                continue

            ok = True
            for fc in FACTOR_COLS:
                if fc in frow.index:
                    v = frow[fc]
                    if not pd.isna(v):
                        factor_vals[fc].append(float(v))
                    else:
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok:
                fwd_returns.append(fwd_ret)
                n_ok += 1

        stock_counts.append(n_ok)

        if n_ok < min_stocks:
            for fc in FACTOR_COLS:
                results[fc].append(np.nan)
            continue

        for fc in FACTOR_COLS:
            vals = factor_vals[fc]
            try:
                ic, _ = spearmanr(vals, fwd_returns)
                results[fc].append(ic if not np.isnan(ic) else 0)
            except Exception:
                results[fc].append(np.nan)

    # 汇总
    summary = []
    for fc, name in FACTOR_COLS.items():
        ics = [x for x in results[fc] if not np.isnan(x)]
        if len(ics) < 3:
            continue
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive = sum(1 for x in ics if x > 0) / len(ics)
        summary.append({
            "因子": name, "IC均值": round(ic_mean, 4),
            "IC标准差": round(ic_std, 4), "IC_IR": round(ic_ir, 2),
            "IC>0": f"{ic_positive:.0%}", "截面数": len(ics),
        })

    if not summary:
        print("  [错误] 无有效因子IC")
        return pd.DataFrame()

    print(f"  每截面股票: {min(stock_counts):.0f}~{max(stock_counts):.0f} (avg {np.mean(stock_counts):.0f})")
    return pd.DataFrame(summary).sort_values("IC_IR", ascending=False)


if __name__ == "__main__":
    print("=" * 65)
    print("  因子IC分析 (Spearman IC, Forward 20d Return)")
    print("=" * 65)

    random.seed(42)
    print("\n  加载因子数据 (500只样本)...")
    factor_data = load_sample_factors(500)
    codes = list(factor_data.keys())
    print(f"  有效: {len(codes)} 只")

    print("  加载价格数据...")
    price_data = load_price_data(codes)
    print(f"  有效: {len(price_data)} 只")

    print("  计算IC...")
    df = compute_ic(factor_data, price_data, forward_days=20)

    if df.empty:
        print("\n  无法计算IC。")
        exit(1)

    print(f"\n{'=' * 65}")
    print(f"  IC 汇总 (按 IC_IR 排序)")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))

    current_weights = {
        "20日动量": 0.30, "5日反转": 0.15, "量价配合": 0.20,
        "低波动": 0.15, "小市值": 0.10, "质量(低回撤)": 0.10,
    }
    print(f"\n{'=' * 65}")
    print(f"  当前权重 vs IC实证")
    print(f"{'=' * 65}")

    # 用正IC_IR作为新权重建议
    pos_ir = {}
    for _, row in df.iterrows():
        name = row["因子"]
        ir = row["IC_IR"]
        if ir > 0:
            pos_ir[name] = ir

    total = sum(pos_ir.values()) if pos_ir else 1.0

    for _, row in df.iterrows():
        name = row["因子"]
        ic_mean = row["IC均值"]
        ic_ir = row["IC_IR"]
        cur_w = current_weights.get(name, 0)
        sug_w = pos_ir.get(name, 0) / total if name in pos_ir else 0

        if ic_mean > 0.005:
            direction = f"有效 (建议 {sug_w:.0%})"
        elif ic_mean < -0.005:
            direction = "反向 (建议 0%)"
        else:
            direction = "中性"
        marker = ""
        if cur_w > 0 and ic_mean < -0.005:
            marker = "  ← 应剔除!"
        elif ic_mean > 0.005 and cur_w < sug_w - 0.05:
            marker = "  ← 权重偏低"
        print(f"  {name:12s}  IC={ic_mean:+.4f}  IR={ic_ir:+.2f}  {direction}{marker}")
