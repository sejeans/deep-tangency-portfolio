"""거시/시장 시계열 + Fama-French 팩터. 전부 무료·무인증.

[중요] 시장 전체 변수(VIX 수준 등)를 그대로 특성으로 쓰면 안 됩니다.
논문의 전처리는 매월 횡단면 rank 이므로, 모든 자산이 같은 값이면
표준화 후 전부 0이 되어 정보가 사라집니다.
=> 논문이 VIX_BETA / TERM_BETA / DEF_BETA / UNC_BETA 를 쓰는 이유가 이것입니다.
   여기서도 거시 변수는 자산별 베타로 변환해서 특성화합니다 (characteristics.py).

[TERM / DEF 팩터] 논문 3.3절 정의를 ETF 수익률로 그대로 재현합니다.
    TERM = 장기국채 수익률 - 단기국채 수익률   ->  TLT - BIL
    DEF  = 장기회사채 수익률 - 장기국채 수익률  ->  LQD - TLT
  FRED 의 T10Y2Y / BAA10Y 는 '금리 수준'이라 수익률 스프레드인 논문 정의와 다릅니다.
  ETF 방식이 논문에 더 충실하고, 접속도 안정적입니다(FRED는 자주 타임아웃).

데이터 시작 시점 주의: TLT/LQD/IEF/SHY 는 2002년, BIL 은 2007년 상장.
BIL 이 없는 구간은 SHY 로 대체합니다.
"""
from __future__ import annotations

import io
import warnings
import zipfile

import numpy as np
import pandas as pd

from . import config as C
from . import net

FF_ZIP = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_CSV.zip")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# yfinance 프록시
PROXY_TICKERS = ["^VIX", "TLT", "LQD", "IEF", "SHY", "BIL"]
FRED_SERIES = {"VIX": "VIXCLS", "TERM_LVL": "T10Y2Y", "DEF_LVL": "BAA10Y"}
_UA = "Mozilla/5.0 (research script)"


# ----------------------------------------------------------------- yfinance
def market_proxies(force: bool = False) -> pd.DataFrame:
    """일별 시장/채권 프록시. columns: date, VIX, dVIX, TERM, DEF"""
    cache = C.RAW / "macro_proxies.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])

    net.setup()
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(PROXY_TICKERS, start=C.START, end=C.END,
                          auto_adjust=True, progress=False,
                          group_by="ticker", threads=True, timeout=60)

    px = pd.DataFrame({t: raw[t]["Close"] for t in PROXY_TICKERS
                       if t in raw.columns.get_level_values(0)})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    ret = px.pct_change()

    out = pd.DataFrame(index=px.index)
    out["VIX"] = px.get("^VIX")
    out["dVIX"] = out["VIX"].pct_change()

    short_leg = ret["BIL"] if "BIL" in ret else ret.get("SHY")
    if "BIL" in ret and "SHY" in ret:
        short_leg = ret["BIL"].fillna(ret["SHY"])       # BIL 상장 전 구간 보완
    out["TERM"] = ret.get("TLT") - short_leg            # 장기국채 - 단기국채
    out["DEF"] = ret.get("LQD") - ret.get("TLT")        # 장기회사채 - 장기국채

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    out.columns = ["date"] + list(out.columns[1:])
    out.to_csv(cache, index=False)
    print(f"[macro] yfinance 프록시 {len(out):,} rows  "
          f"{out['date'].min():%Y-%m} ~ {out['date'].max():%Y-%m} -> {cache}")
    return out


# ----------------------------------------------------------------- FRED (선택)
def fred_daily(force: bool = False) -> pd.DataFrame:
    """선택 사항. FRED는 접속이 불안정하므로 실패해도 파이프라인은 계속됩니다."""
    cache = C.RAW / "fred_daily.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])

    out = None
    for name, sid in FRED_SERIES.items():
        try:
            r = net.get(FRED.format(sid=sid), user_agent=_UA, retries=2, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"[fred] {name} 실패 (건너뜀): {type(e).__name__}")
            continue
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")
        out = df if out is None else out.merge(df, on="date", how="outer")
        print(f"[fred] {name} ({sid}) {len(df):,} rows")

    if out is None:
        return pd.DataFrame(columns=["date"])
    out = out.sort_values("date").ffill().reset_index(drop=True)
    out.to_csv(cache, index=False)
    return out


# ----------------------------------------------------------------- 팩터
def fama_french_monthly(force: bool = False) -> pd.DataFrame:
    """월별 FF3 + RF. columns: month, MKT_RF, SMB, HML, RF (전부 소수 단위)"""
    cache = C.RAW / "ff_monthly.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["month"])

    r = net.get(FF_ZIP, user_agent=_UA)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")

    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5 or not parts[0].isdigit() or len(parts[0]) != 6:
            if rows:                     # 월별 블록 종료 (연간 블록 시작)
                break
            continue
        rows.append(parts[:5])

    ff = pd.DataFrame(rows, columns=["ym", "MKT_RF", "SMB", "HML", "RF"])
    ff["month"] = (pd.to_datetime(ff["ym"], format="%Y%m")
                     .dt.to_period("M").dt.to_timestamp("M"))
    for c in ["MKT_RF", "SMB", "HML", "RF"]:
        ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0     # 퍼센트 -> 소수
    ff = ff[["month", "MKT_RF", "SMB", "HML", "RF"]].dropna().reset_index(drop=True)
    ff.to_csv(cache, index=False)
    print(f"[ff] {len(ff)} months  {ff['month'].min():%Y-%m} ~ {ff['month'].max():%Y-%m}")
    return ff


# ----------------------------------------------------------------- 베타 입력
def macro_daily_changes(force: bool = False, use_fred: bool = False) -> pd.DataFrame:
    """베타 추정용 일별 팩터 시계열. index=date, columns=[dVIX, dTERM, dDEF]"""
    p = market_proxies(force=force).set_index("date")
    out = pd.DataFrame(index=p.index)
    out["dVIX"] = p["dVIX"]
    out["dTERM"] = p["TERM"]
    out["dDEF"] = p["DEF"]

    if use_fred:
        f = fred_daily(force=force)
        if len(f):
            f = f.set_index("date").reindex(out.index).ffill()
            for c in f.columns:
                out[f"d{c}"] = f[c].diff()

    return out.replace([np.inf, -np.inf], np.nan).astype("float64")


if __name__ == "__main__":
    print(macro_daily_changes().tail())
    print(fama_french_monthly().tail())
