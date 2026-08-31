"""yfinance 일별 OHLCV 수집 + 캐시.

논문 대응:
  TRACE 일중 체결 -> 거래량가중 일별가격 -> 월별 총수익률
  여기서는 yfinance 수정주가(auto_adjust=True, 배당/분할 반영) -> 월별 총수익률
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from . import config as C


def _download_chunk(tks: list[str], start: str, end: str) -> pd.DataFrame:
    last = None
    for attempt in range(C.YF_RETRY):
        try:
            df = yf.download(
                tks, start=start, end=end,
                auto_adjust=True, actions=False,
                group_by="ticker", threads=True,
                progress=False, timeout=60,
            )
            if df is not None and len(df):
                return df
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    if last:
        print(f"  [warn] chunk failed: {last}")
    return pd.DataFrame()


def _tidy(df: pd.DataFrame, tks: list[str]) -> pd.DataFrame:
    """MultiIndex 컬럼 -> long format (date, ticker, open/high/low/close/volume)."""
    if df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):          # 단일 티커
        df = pd.concat({tks[0]: df}, axis=1)
    out = []
    for t in df.columns.get_level_values(0).unique():
        sub = df[t].copy()
        need = {"Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(sub.columns):
            continue
        sub = sub[list(need & set(sub.columns))].dropna(how="all")
        if sub.empty:
            continue
        sub.columns = [c.lower() for c in sub.columns]
        sub["ticker"] = t
        out.append(sub.reset_index().rename(columns={"Date": "date", "index": "date"}))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"]).dt.tz_localize(None)
    return res[["date", "ticker", "open", "high", "low", "close", "volume"]]


def download_daily(tks: list[str], force: bool = False) -> pd.DataFrame:
    """일별 OHLCV long format. data/raw/daily.csv 에 캐시."""
    cache = C.RAW / "daily.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, parse_dates=["date"])
        print(f"[prices] cache hit: {df['ticker'].nunique()} tickers, {len(df):,} rows")
        return df

    frames = []
    for i in range(0, len(tks), C.YF_CHUNK):
        chunk = tks[i:i + C.YF_CHUNK]
        print(f"[prices] {i+1}-{i+len(chunk)} / {len(tks)}")
        frames.append(_tidy(_download_chunk(chunk, C.START, C.END), chunk))
        time.sleep(1.0)

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"[prices] saved {df['ticker'].nunique()} tickers, {len(df):,} rows -> {cache}")
    return df


def add_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["ticker", "date"]).copy()
    d["ret"] = d.groupby("ticker", observed=True)["close"].pct_change()
    # 논문 필터 (v): 가격 하한
    d.loc[d["close"] < C.MIN_PRICE, "ret"] = np.nan
    d["dolvol"] = d["close"] * d["volume"]
    # 극단 오류 제거 (분할 미반영 등)
    d.loc[d["ret"].abs() > 1.0, "ret"] = np.nan
    return d


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """월별 총수익률 (논문 식 21의 주식 대응판; 수정주가라 배당 포함)."""
    d = daily.dropna(subset=["close"]).copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp("M")
    grp = d.groupby(["ticker", "month"], observed=True)
    px = grp["close"].last().rename("close_eom")
    ndays = grp["close"].size().rename("n_days")
    m = pd.concat([px, ndays], axis=1).reset_index()
    m = m.sort_values(["ticker", "month"])
    m["ret_m"] = m.groupby("ticker", observed=True)["close_eom"].pct_change()
    # 관측일이 너무 적은 달은 신뢰할 수 없음 (논문의 유동성 필터 취지)
    m.loc[m["n_days"] < 5, "ret_m"] = np.nan
    return m[["ticker", "month", "close_eom", "n_days", "ret_m"]]


if __name__ == "__main__":
    from .universe import tickers
    dl = download_daily(tickers())
    dl = add_daily_returns(dl)
    print(monthly_returns(dl).tail())
