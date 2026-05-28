"""
A股数据源模块 — 新浪 + 腾讯直连 API
无需依赖 AkShare，直接请求 HTTP 接口，绕开东方财富封锁

数据来源：
  新浪财经 — 实时行情、股票列表、分钟K线、指数
  腾讯财经 — 日/周/月K线（支持前复权/后复权）
"""
from datetime import datetime
from typing import Optional
import json
import re
import time
import pandas as pd
import requests

# ==================== 公共 ====================

_SESSION = None
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn",
}


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(_HEADERS)
    return _SESSION


def _get(url, params=None, timeout=15, referer=None):
    """统一 GET 请求，自动重试"""
    s = _get_session()
    headers = {}
    if referer:
        headers["Referer"] = referer
    for attempt in range(3):
        try:
            r = s.get(url, params=params, headers=headers or None, timeout=timeout)
            r.encoding = "gbk" if "sinajs" in url or "qt.gtimg" in url else "utf-8"
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
        return r
    return r


# ==================== 股票列表 ====================

def get_stock_list(force_refresh: bool = False) -> pd.DataFrame:
    """
    获取沪深京全部 A 股实时行情（带本地缓存 fallback）
    数据源：新浪
    """
    import os as _os
    _cache_path = _os.path.join("data", "stock_list_cache.pkl")

    if not force_refresh and _os.path.exists(_cache_path):
        try:
            cached = pd.read_pickle(_cache_path)
            if len(cached) > 1000:
                return cached
        except Exception:
            pass

    all_stocks = []
    page = 1
    page_size = 100  # 新浪API单次上限100

    while True:
        url = (
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "Market_Center.getHQNodeData"
        )
        params = {
            "page": page,
            "num": page_size,
            "sort": "symbol",
            "asc": 1,
            "node": "hs_a",
            "symbol": "",
            "_s_r_a": "auto",
        }
        r = _get(url, params=params, timeout=30, referer="https://finance.sina.com.cn/")
        try:
            data = r.json()
        except Exception:
            time.sleep(1)
            r = _get(url, params=params, timeout=30, referer="https://finance.sina.com.cn/")
            try:
                data = r.json()
            except Exception:
                break
        if not data:
            break
        all_stocks.extend(data)
        page += 1
        time.sleep(0.3)

    if not all_stocks:
        # fallback: 使用缓存
        if _os.path.exists(_cache_path):
            print("  [警告] API 返回空，使用本地缓存")
            return pd.read_pickle(_cache_path)
        return pd.DataFrame()

    df = pd.DataFrame(all_stocks)
    df = df.rename(columns={
        "symbol": "exchange_code", "code": "code", "name": "name",
        "trade": "price", "pricechange": "change", "changepercent": "pct_chg",
        "buy": "bid", "sell": "ask", "settlement": "pre_close",
        "open": "open", "high": "high", "low": "low",
        "volume": "volume", "amount": "amount",
        "per": "pe", "pb": "pb",
        "mktcap": "total_mv", "nmc": "float_mv",
        "turnoverratio": "turnover",
    })
    for col in ["price", "change", "pct_chg", "bid", "ask", "pre_close",
                "open", "high", "low", "volume", "amount",
                "pe", "pb", "total_mv", "float_mv", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 缓存到本地
    try:
        _os.makedirs("data", exist_ok=True)
        df.to_pickle(_cache_path)
    except Exception:
        pass
    return df


def get_stock_info(symbol: str) -> dict:
    """获取单只个股实时行情"""
    df = get_stock_list()
    row = df[df["code"] == symbol]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ==================== 实时行情（轻量，多股） ====================

def get_realtime_quotes(symbols: list[str]) -> pd.DataFrame:
    """
    获取多只股票的实时行情（新浪，速度快）
    返回：code, name, open, pre_close, price, high, low, volume, amount, ...
    """
    # 新浪行情字段顺序
    FIELDS = [
        "name", "open", "pre_close", "price", "high", "low",
        "bid", "ask", "volume", "amount",
        "_b1v", "_b1p", "_b2v", "_b2p", "_b3v", "_b3p", "_b4v", "_b4p", "_b5v", "_b5p",
        "_a1v", "_a1p", "_a2v", "_a2p", "_a3v", "_a3p", "_a4v", "_a4p", "_a5v", "_a5p",
        "date", "time", "status",
    ]

    codes = [f"sh{s}" if s.startswith(("6", "9")) else f"sz{s}" for s in symbols]
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    r = _get(url, timeout=15, referer="https://finance.sina.com.cn/")

    rows = []
    for line in r.text.strip().split("\n"):
        if not line.strip() or "=" not in line:
            continue
        var_part = line.split('"')[1] if '"' in line else ""
        if not var_part:
            continue
        code = line.split("_str_")[1].split("=")[0].replace("sh", "").replace("sz", "") if "_str_" in line else ""
        if not code:
            # 从变量名提取
            code = line.split("=")[0].replace("var hq_str_sh", "").replace("var hq_str_sz", "")
        vals = var_part.split(",")
        row = {"code": code}
        for i, f in enumerate(FIELDS):
            row[f] = vals[i] if i < len(vals) else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    num_cols = ["open", "pre_close", "price", "high", "low", "volume", "amount"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ==================== K线 — 腾讯（支持复权） ====================

def _parse_tencent_kline(raw_data: list, period: str) -> pd.DataFrame:
    """解析腾讯K线数据"""
    if not raw_data:
        return pd.DataFrame()

    key_map = {
        "qfqday": "前复权日线", "hfqday": "后复权日线", "day": "不复权日线",
        "qfqweek": "前复权周线", "hfqweek": "后复权周线", "week": "不复权周线",
        "qfqmonth": "前复权月线", "hfqmonth": "后复权月线", "month": "不复权月线",
    }

    records = []
    for key, entry in raw_data.items():
        if not isinstance(entry, list):
            continue
        adjust_type = key_map.get(key)
        if adjust_type is None:
            continue  # 跳过 qt 等非K线数据
        for row in entry:
            if len(row) < 6:
                continue
            records.append({
                "date": row[0],
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
                "adjust": adjust_type,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_daily_kline(symbol: str, start_date: str = "20200101",
                    end_date: str = "", adjust: str = "qfq") -> pd.DataFrame:
    """
    获取A股日K线（腾讯，支持复权）
    adjust: 'qfq'=前复权, 'hfq'=后复权, ''=不复权
    """
    market = "sh" if symbol.startswith(("6", "9")) else "sz"
    full_code = f"{market}{symbol}"

    # 腾讯接口：参数直接拼 URL，不用 requests params（逗号会被编码成 %2C 导致 param error）
    base_url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    full_url = f"{base_url}?param={full_code},day,,,2000,{adjust}"
    r = _get(full_url, timeout=30, referer="https://finance.qq.com/")
    resp = r.json()
    if resp.get("code") != 0:
        return pd.DataFrame()

    stock_data = resp.get("data", {}).get(full_code, {})
    if not stock_data:
        stock_data = resp.get("data", {}).get(symbol, {})

    df = _parse_tencent_kline(stock_data, "day")
    return _filter_by_date(df, start_date, end_date)


def get_weekly_kline(symbol: str, start_date: str = "20200101",
                     end_date: str = "", adjust: str = "qfq") -> pd.DataFrame:
    """获取周K线"""
    market = "sh" if symbol.startswith(("6", "9")) else "sz"
    full_code = f"{market}{symbol}"
    full_url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},week,,,2000,{adjust}"
    r = _get(full_url, timeout=30, referer="https://finance.qq.com/")
    resp = r.json()
    if resp.get("code") != 0:
        return pd.DataFrame()
    stock_data = resp.get("data", {}).get(full_code, {}) or resp.get("data", {}).get(symbol, {})
    df = _parse_tencent_kline(stock_data, "week")
    return _filter_by_date(df, start_date, end_date)


def get_monthly_kline(symbol: str, start_date: str = "20200101",
                      end_date: str = "", adjust: str = "qfq") -> pd.DataFrame:
    """获取月K线"""
    market = "sh" if symbol.startswith(("6", "9")) else "sz"
    full_code = f"{market}{symbol}"
    full_url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},month,,,2000,{adjust}"
    r = _get(full_url, timeout=30, referer="https://finance.qq.com/")
    resp = r.json()
    if resp.get("code") != 0:
        return pd.DataFrame()
    stock_data = resp.get("data", {}).get(full_code, {}) or resp.get("data", {}).get(symbol, {})
    df = _parse_tencent_kline(stock_data, "month")
    return _filter_by_date(df, start_date, end_date)


def _filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按日期范围过滤"""
    if df.empty:
        return df
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


# ==================== 分钟K线 — 新浪 ====================

def get_minute_kline(symbol: str, period: str = "5") -> pd.DataFrame:
    """
    获取分钟K线（新浪）
    period: '5', '15', '30', '60'
    """
    SCALE_MAP = {"5": 5, "15": 15, "30": 30, "60": 60}
    scale = SCALE_MAP.get(period, 5)
    market = "sh" if symbol.startswith(("6", "9")) else "sz"

    url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/data/CN_MarketDataService.getKLineData"
    params = {
        "symbol": f"{market}{symbol}",
        "scale": scale,
        "ma": "no",
        "datalen": 1500,
    }
    r = _get(url, params=params, timeout=30, referer="https://finance.sina.com.cn/")
    text = r.text

    # 解析 JSONP：data([...]) 或 data(null)
    match = re.search(r"data\((.*?)\);", text, re.DOTALL)
    if not match or match.group(1) == "null":
        return pd.DataFrame()
    arr = json.loads(match.group(1))
    if not arr:
        return pd.DataFrame()

    df = pd.DataFrame(arr)
    df = df.rename(columns={
        "day": "time", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
        "amount": "amount",
    })
    df["time"] = pd.to_datetime(df["time"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


# ==================== 指数数据 — 新浪 ====================

def get_index_kline(index_code: str, start_date: str = "20200101",
                    end_date: str = "") -> pd.DataFrame:
    """
    获取指数日K线（新浪）
    常用指数: 000001=上证, 399001=深证, 399006=创业板, 000688=科创50
    """
    market = "sh" if index_code.startswith(("0", "6")) else "sz"
    url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/data/CN_MarketDataService.getKLineData"
    params = {
        "symbol": f"{market}{index_code}",
        "scale": 240,
        "ma": "no",
        "datalen": 1500,
    }
    r = _get(url, params=params, timeout=30, referer="https://finance.sina.com.cn/")
    text = r.text
    match = re.search(r"data\((.*?)\);", text, re.DOTALL)
    if not match or match.group(1) == "null":
        return pd.DataFrame()
    arr = json.loads(match.group(1))
    if not arr:
        return pd.DataFrame()

    df = pd.DataFrame(arr)
    df = df.rename(columns={
        "day": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _filter_by_date(df.sort_values("date"), start_date, end_date)


# ==================== 行业/概念 — 新浪 ====================

def get_industry_list() -> pd.DataFrame:
    """获取行业板块列表"""
    url = "https://vip.stock.finance.sina.com.cn/q/go.php/vIndustryRank/kind/gszz/p/1/num/2000"
    r = _get(url, timeout=15)
    try:
        data = r.json()
    except json.JSONDecodeError:
        return pd.DataFrame()
    return pd.DataFrame(data)


def get_concept_list() -> pd.DataFrame:
    """获取概念板块列表"""
    url = "https://vip.stock.finance.sina.com.cn/q/go.php/vIndustryRank/kind/gnjj/p/1/num/2000"
    r = _get(url, timeout=15)
    try:
        data = r.json()
    except json.JSONDecodeError:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ==================== 龙虎榜 — 新浪 ====================

def get_lhb_top_list(date: str = "") -> pd.DataFrame:
    """获取龙虎榜（新浪，默认最近交易日）"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    url = f"https://vip.stock.finance.sina.com.cn/q/go.php/vLHBData/kind/topCommCodes/p/1/num/500/date/{date}"
    r = _get(url, timeout=15)
    try:
        data = r.json()
    except json.JSONDecodeError:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ==================== 工具函数 ====================

def save_to_csv(df: pd.DataFrame, filename: str, subdir: str = "data"):
    """DataFrame 存为 CSV，自动创建目录"""
    import os
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已保存: {path}  ({len(df)} 行)")
    return path


def filter_by_pct_chg(df: pd.DataFrame, lo: float = None, hi: float = None) -> pd.DataFrame:
    """按涨跌幅筛选"""
    if "pct_chg" not in df.columns:
        return df
    result = df.copy()
    if lo is not None:
        result = result[result["pct_chg"] >= lo]
    if hi is not None:
        result = result[result["pct_chg"] <= hi]
    return result


def filter_by_mv(df: pd.DataFrame, min_mv: float = 0,
                 max_mv: float = float("inf")) -> pd.DataFrame:
    """按总市值筛选（单位：万元）"""
    if "total_mv" not in df.columns:
        return df
    return df[(df["total_mv"] >= min_mv) & (df["total_mv"] <= max_mv)]


def filter_by_turnover(df: pd.DataFrame, lo: float = 0, hi: float = 100) -> pd.DataFrame:
    """按换手率筛选"""
    if "turnover" not in df.columns:
        return df
    return df[(df["turnover"] >= lo) & (df["turnover"] <= hi)]
