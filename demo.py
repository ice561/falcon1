"""
Demo — A股数据源模块使用示例
数据来源：新浪财经 + 腾讯财经
"""
import data_source as ds

# ---- 1. 全A股列表 ----
print("=" * 60)
print("1. A股全列表（前5只）")
print("=" * 60)
df_all = ds.get_stock_list()
print(df_all[["code", "name", "price", "pct_chg", "pe", "total_mv"]].head())
print(f"\n共 {len(df_all)} 只股票")

# ---- 2. 日K线 ----
print("\n" + "=" * 60)
print("2. 贵州茅台 (600519) 日K线（近10天，前复权）")
print("=" * 60)
df_kline = ds.get_daily_kline("600519", start_date="20250401", adjust="qfq")
print(df_kline.tail(10))
ds.save_to_csv(df_kline, "600519_daily.csv")

# ---- 3. 周K线 ----
print("\n" + "=" * 60)
print("3. 贵州茅台 周K线（近5周）")
print("=" * 60)
df_week = ds.get_weekly_kline("600519", start_date="20250301", adjust="qfq")
print(df_week.tail())

# ---- 4. 分钟K线 ----
print("\n" + "=" * 60)
print("4. 平安银行 (000001) 5分钟K线（最近10根）")
print("=" * 60)
df_min = ds.get_minute_kline("000001", period="5")
print(df_min.tail(10))

# ---- 5. 指数 ----
print("\n" + "=" * 60)
print("5. 上证指数 (000001) 最近收盘价")
print("=" * 60)
df_idx = ds.get_index_kline("000001", start_date="20250501")
print(df_idx[["date", "open", "close", "high", "low", "volume"]].tail(10))

# ---- 6. 涨幅筛选 ----
print("\n" + "=" * 60)
print("6. 涨幅 5%-10% 的股票（前10只）")
print("=" * 60)
up = ds.filter_by_pct_chg(df_all, lo=5, hi=10)
print(up[["code", "name", "price", "pct_chg"]].head(10))

# ---- 7. 换手率筛选 ----
print("\n" + "=" * 60)
print("7. 换手率 > 10% 的股票（前10只）")
print("=" * 60)
hot = ds.filter_by_turnover(df_all, lo=10)
print(hot[["code", "name", "price", "pct_chg", "turnover"]].head(10))

print("\n完成！数据保存在 ./data/ 目录")
