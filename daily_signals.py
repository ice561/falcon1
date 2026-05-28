"""
每日信号生产 — 输出当日推荐股票列表
用法: python daily_signals.py           # 输出今日信号
      python daily_signals.py --top 20   # 自定义持仓数
      python daily_signals.py --date 2026-05-25  # 指定历史日期
"""
import os
import sys
import io
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import data_source as ds
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CACHE_DIR = "./cache"
FACTOR_DIR = os.path.join(CACHE_DIR, "factors")
KLINE_DIR = os.path.join(CACHE_DIR, "klines")

# 基于IC实证的因子权重
FACTOR_WEIGHTS = {
    "neg_momentum_20": 0.17,
    "inv_vol_20": 0.17,
    "reversal_5": 0.17,
    "alpha_001": 0.17,
    "neg_rsi_14": 0.17,
    "inv_log_mv": 0.15,
}
MIN_DAILY_AMOUNT = 50_000_000  # 5000万
WARMUP_DAYS = 60
MIN_SCORE_THRESHOLD = 0.3
MAX_PER_SECTOR = 3


def get_board(code: str) -> str:
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


def load_factor(code: str) -> pd.DataFrame | None:
    p = os.path.join(FACTOR_DIR, f"{code}.pkl")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p)
        if "neg_momentum_20" not in df.columns and "momentum_20" in df.columns:
            df["neg_momentum_20"] = -df["momentum_20"]
        if "neg_rsi_14" not in df.columns and "rsi_14" in df.columns:
            df["neg_rsi_14"] = -df["rsi_14"]
        return df
    except Exception:
        return None


def load_kline(code: str, adjust: str = "qfq") -> pd.DataFrame | None:
    p = os.path.join(KLINE_DIR, f"{code}_{adjust}.pkl")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_pickle(p)
    except Exception:
        return None


def update_data_for_date(target_date: str, top_n: int = 300):
    """更新到目标日期的数据（只更新需要刷新的股票）"""
    print(f"\n{'=' * 55}")
    print(f"  猎鹰一号 — 每日信号 ({target_date})")
    print(f"{'=' * 55}")

    # 1. 获取股票列表
    print("\n[1/4] 获取全A股列表...")
    try:
        stock_list = ds.get_stock_list()
    except Exception:
        stock_list = pd.DataFrame()
    if stock_list.empty:
        codes = []
        for f in os.listdir(KLINE_DIR):
            if f.endswith(".pkl"):
                code = f.split("_")[0]
                if code.isdigit() and len(code) == 6:
                    codes.append(code)
        codes = list(set(codes))
    else:
        stock_list = stock_list[~stock_list["name"].str.contains(r"\*?ST", na=False)]
        codes = stock_list["code"].tolist()

    print(f"  候选池: {len(codes)} 只（排除ST后）")

    # 2. 检查K线覆盖到目标日期
    print("\n[2/4] 检查K线数据...")
    target_dt = pd.Timestamp(target_date)
    need_update = []
    ready = 0

    for code in codes:
        df = load_kline(code)
        if df is not None and not df.empty:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                last_date = df["date"].max()
            elif isinstance(df.index, pd.DatetimeIndex):
                last_date = df.index.max()
            else:
                need_update.append(code)
                continue
            if last_date >= target_dt - pd.Timedelta(days=1):
                ready += 1
                continue
        need_update.append(code)

    print(f"  数据就绪: {ready} 只, 需更新: {len(need_update)} 只")

    if need_update:
        print(f"  正在更新 {min(len(need_update), 500)} 只...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        update_codes = need_update[:500]
        done = 0
        errors = 0

        def _dl(code):
            try:
                df = ds.get_daily_kline(code, start_date="20240101", end_date=target_date, adjust="qfq")
                return code, df
            except Exception:
                return code, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_dl, c): c for c in update_codes}
            for f in as_completed(futures):
                code, df = f.result()
                done += 1
                if not df.empty:
                    df.to_pickle(os.path.join(KLINE_DIR, f"{code}_qfq.pkl"))
                else:
                    errors += 1
                if done % 100 == 0:
                    print(f"    进度: {done}/{len(update_codes)}")

        print(f"  更新完成: {done - errors} 成功, {errors} 失败")

    # 3. 检查因子覆盖
    print("\n[3/4] 检查因子数据...")
    need_factor = []
    factor_ready = 0
    for code in codes:
        fdf = load_factor(code)
        if fdf is not None and len(fdf) > WARMUP_DAYS:
            idx = fdf.index if isinstance(fdf.index, pd.DatetimeIndex) else pd.to_datetime(fdf.index)
            if idx.max() >= target_dt - pd.Timedelta(days=5):
                factor_ready += 1
                continue
        need_factor.append(code)

    print(f"  因子就绪: {factor_ready} 只, 需计算: {len(need_factor)} 只")

    if need_factor:
        print(f"  正在计算因子 (前{min(len(need_factor), 300)}只)...")
        import factors as ft

        def compute_factors_for_stock(df, total_mv_base=None):
            base = ft.compute_all(df)
            try:
                n = 10
                price_dir = np.sign(df["close"].diff())
                vol_chg = df["volume"].pct_change()
                base["vol_price_corr_10"] = price_dir.rolling(n).corr(vol_chg)
            except Exception:
                base["vol_price_corr_10"] = np.nan
            if total_mv_base and not np.isnan(total_mv_base) and total_mv_base > 0:
                last_close = df["close"].iloc[-1]
                if last_close > 0:
                    scale = df["close"] / last_close
                    approx_mv = total_mv_base * scale
                    base["inv_log_mv"] = -np.log(approx_mv.replace(0, np.nan))
                else:
                    base["inv_log_mv"] = np.nan
            else:
                base["inv_log_mv"] = np.nan
            base["neg_dd_20"] = -ft.max_drawdown_factor(df, 20)
            base["inv_vol_20"] = 1.0 / ft.hist_volatility(df, 20).replace(0, np.nan)
            base["neg_momentum_20"] = -base["momentum_20"]
            base["neg_rsi_14"] = -base["rsi_14"]
            base = base.replace([np.inf, -np.inf], np.nan)
            return base

        mv_map = {}
        if not stock_list.empty and "total_mv" in stock_list.columns:
            mv_map = dict(zip(stock_list["code"], stock_list["total_mv"]))

        computed = 0
        for i, code in enumerate(need_factor[:300]):
            kdf = load_kline(code)
            if kdf is None or kdf.empty:
                continue
            try:
                f = compute_factors_for_stock(kdf, mv_map.get(code))
                f.to_pickle(os.path.join(FACTOR_DIR, f"{code}.pkl"))
                computed += 1
            except Exception:
                continue
        print(f"  因子计算完成: {computed} 只")

    # 4. 生成信号
    print(f"\n[4/4] 生成信号...")
    return generate_signals(codes, target_date, top_n)


def generate_signals(codes: list[str], target_date: str, top_n: int = 15):
    """在 target_date 生成选股信号"""
    target_dt = pd.Timestamp(target_date)

    factor_rows = []
    valid_codes = []
    amount_ok = []

    # 加载股票名称用于过滤退市股
    try:
        stock_info = ds.get_stock_list()
        name_map = dict(zip(stock_info["code"], stock_info["name"])) if not stock_info.empty else {}
    except Exception:
        name_map = {}

    for code in codes:
        # 过滤退市/ST
        name = name_map.get(code, "")
        if any(kw in str(name) for kw in ("退市", "ST", "*ST", "退")):
            continue

        fdf = load_factor(code)
        if fdf is None or fdf.empty:
            continue

        # 统一索引格式
        if "date" in fdf.columns:
            fdf = fdf.copy()
            fdf["date"] = pd.to_datetime(fdf["date"])
            fdf = fdf.set_index("date")
        elif isinstance(fdf.index, pd.DatetimeIndex):
            pass
        else:
            continue

        if target_dt not in fdf.index:
            continue

        kdf = load_kline(code)
        if kdf is None or kdf.empty:
            continue
        if "date" in kdf.columns:
            kdf = kdf.copy()
            kdf["date"] = pd.to_datetime(kdf["date"])
            kdf = kdf.set_index("date")

        if target_dt not in kdf.index:
            continue
        price = kdf.loc[target_dt, "close"]
        if pd.isna(price) or price <= 0:
            continue

        # 流动性检查
        if "amount" in kdf.columns:
            amt_hist = kdf["amount"].loc[:target_dt].tail(20)
            if len(amt_hist) < 10 or amt_hist.mean() < MIN_DAILY_AMOUNT:
                continue

        # 因子值
        frow = fdf.loc[target_dt]
        vals = []
        ok = True
        for fc in FACTOR_WEIGHTS:
            if fc in frow.index:
                v = frow[fc]
                if pd.isna(v):
                    ok = False
                    break
                vals.append(float(v))
            else:
                ok = False
                break
        if not ok:
            continue

        valid_codes.append(code)
        factor_rows.append(vals)

    if len(valid_codes) < top_n:
        print(f"  [警告] 有效股票 {len(valid_codes)} < top_n {top_n}")
        return None

    # Z-Score + Winsorize + 加权
    farr = np.array(factor_rows)
    weights = list(FACTOR_WEIGHTS.values())
    zarr = np.zeros_like(farr)

    for j in range(farr.shape[1]):
        col = farr[:, j]
        mean = np.nanmean(col)
        std = np.nanstd(col)
        if std == 0 or np.isnan(std):
            zarr[:, j] = 0
        else:
            z = (col - mean) / std
            z = np.clip(z, -3.0, 3.0)
            zarr[:, j] = z

    composite = zarr @ np.array(weights)

    # 信号质量阈值
    valid_indices = [i for i in range(len(composite))
                     if composite[i] >= MIN_SCORE_THRESHOLD]
    if len(valid_indices) < max(1, top_n // 3):
        print(f"  [警告] 达标股票 {len(valid_indices)} 太少")

    # Top-K + 行业分散
    sorted_indices = [i for i in np.argsort(composite)[::-1]
                      if i in set(valid_indices)]

    selected = []
    sector_counts = {}
    detail_rows = []

    for idx in sorted_indices:
        code = valid_codes[idx]
        sector = get_board(code)
        if MAX_PER_SECTOR > 0 and sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        selected.append(code)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # 各因子Z-Score
        z_scores = {fc: round(zarr[idx, j], 2) for j, fc in enumerate(FACTOR_WEIGHTS)}
        detail_rows.append({
            "排名": len(selected), "代码": code, "板块": sector,
            "综合得分": round(composite[idx], 2),
            "低波动Z": z_scores["inv_vol_20"],
            "反转Z": z_scores["neg_momentum_20"],
            "5日反转Z": z_scores["reversal_5"],
            "Alpha001": z_scores["alpha_001"],
            "-RSIZ": z_scores["neg_rsi_14"],
            "小市值Z": z_scores["inv_log_mv"],
        })
        if len(selected) >= top_n:
            break

    # 获取股票名称
    try:
        live = ds.get_realtime_quotes(selected)
        name_map = dict(zip(live["code"], live["name"])) if not live.empty else {}
    except Exception:
        name_map = {}

    results = []
    for row in detail_rows:
        code = row["代码"]
        results.append({
            "排名": row["排名"],
            "代码": code,
            "名称": name_map.get(code, ""),
            "板块": row["板块"],
            "综合得分": row["综合得分"],
            "低波动": row.get("低波动Z", 0),
            "反转": row.get("反转Z", 0),
            "5日反转Z": row.get("5日反转Z", 0),
            "Alpha001": row.get("Alpha001", 0),
            "-RSIZ": row.get("-RSIZ", 0),
            "小市值Z": row.get("小市值Z", 0),
        })

    return results


def format_output(results: list[dict], target_date: str):
    """格式化输出信号"""
    if not results:
        print("\n  今日无信号（大盘择时空仓或达标股票不足）")
        return

    print(f"\n{'=' * 70}")
    print(f"  猎鹰一号 — 选股信号 ({target_date})")
    print(f"{'=' * 70}")
    print(f"  {'排名':<4} {'代码':<8} {'名称':<10} {'板块':<10} {'得分':>6} {'低波':>6} {'反转':>6} {'5日反转':>6} {'Alpha1':>6} {'-RSI':>6}")
    print(f"  {'-' * 64}")

    for r in results:
        print(f"  {r['排名']:<4} {r['代码']:<8} {r['名称']:<10} {r['板块']:<10} "
              f"{r['综合得分']:>6.2f} {r['低波动']:>6.2f} {r['反转']:>6.2f} {r['5日反转Z']:>6.2f} {r['Alpha001']:>6.2f} {r['-RSIZ']:>6.2f}")

    print(f"{'=' * 70}")

    # 板块分布
    sectors = {}
    for r in results:
        s = r["板块"]
        sectors[s] = sectors.get(s, 0) + 1
    print(f"  板块分布: {', '.join(f'{k}:{v}只' for k, v in sectors.items())}")

    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = f"signals_{target_date.replace('-', '')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  信号已保存: {csv_path}")

    # 保存JSON（方便程序读取）
    import json
    json_path = f"signals_{target_date.replace('-', '')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"date": target_date, "signals": results}, f, ensure_ascii=False, indent=2)
    print(f"  信号已保存: {json_path}")


def check_market_timing(target_date: str) -> tuple[bool, str]:
    """大盘择时检查"""
    try:
        df = ds.get_index_kline("000001", start_date="20240101", end_date=target_date)
        if df.empty:
            return True, "无法获取指数数据，跳过择时"
        df = df.set_index("date") if "date" in df.columns else df
        close = df["close"]
        ma60 = close.rolling(60).mean()
        target_dt = pd.Timestamp(target_date)
        if target_dt not in close.index or target_dt not in ma60.index:
            return True, f"目标日期 {target_date} 无指数数据"
        current = float(close.loc[target_dt])
        ma = float(ma60.loc[target_dt])
        if pd.isna(ma):
            return True, "MA60 不足，跳过择时"
        if current > ma:
            return True, f"上证 {current:.0f} > MA60 {ma:.0f} → 多头"
        else:
            return False, f"上证 {current:.0f} <= MA60 {ma:.0f} → 空仓"
    except Exception as e:
        return True, f"择时检查异常: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猎鹰一号 — 每日信号")
    parser.add_argument("--date", type=str, default="", help="目标日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--top", type=int, default=15, help="选股数量（默认15）")
    parser.add_argument("--no-timing", action="store_true", help="跳过择时检查")
    args = parser.parse_args()

    if args.date:
        target = args.date
    else:
        target = datetime.now().strftime("%Y-%m-%d")

    # 如果是周末/节假日，往前找最近交易日（简单处理）
    target_dt = pd.Timestamp(target)
    if target_dt.weekday() >= 5:
        print(f"  [注意] {target} 是周末，信号基于最近数据")

    # 择时检查
    if not args.no_timing:
        ok, msg = check_market_timing(target)
        print(f"\n  大盘择时: {msg}")
        if not ok:
            print("\n  [结论] 大盘空头，建议空仓或减仓。")
            print("  如需强制输出信号，使用 --no-timing")
            sys.exit(0)

    # 运行
    results = update_data_for_date(target, top_n=args.top)
    if results:
        format_output(results, target)
    else:
        print(f"\n  [结论] {target} 无有效信号")
