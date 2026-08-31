"""월별 특성(characteristics) 생성.

논문 132개(채권41 + 주식61 + 옵션30)의 무료데이터 대체판.
  - 가격/거래량 파생 특성 : 논문의 채권 기본/수익률분포/유동성 계열 대체
  - 거시변수 베타         : 논문의 TERM_BETA / DEF_BETA / VIX_BETA / UNC_BETA 대응
  - EDGAR 펀더멘털        : 논문의 주식 61개 특성 부분 대체
  - 옵션 30개             : 개별종목 옵션 이력은 무료로 구할 수 없음.
                            VIX 베타로 일부만 근사.
                            (논문의 "옵션이 결정적" 결론은 이 설정에서 검증 불가)

모든 특성은 시점 t 말 기준 정보만 사용합니다. 패널 조립 단계(panel.py)에서
한 달 시차를 줘서 t-1 특성 -> t 수익률 로 정렬합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EOM = "ME"          # pandas 2.2+ 월말 alias


# ----------------------------------------------------------------- 유틸
def _pivot(daily: pd.DataFrame, col: str) -> pd.DataFrame:
    return daily.pivot_table(index="date", columns="ticker", values=col, observed=True)


def _eom(df: pd.DataFrame) -> pd.DataFrame:
    """일별 rolling 결과에서 월말 값만 추출."""
    return df.resample(EOM).last()


def _roll_beta(R: pd.DataFrame, x: pd.Series, win: int, minp: int) -> pd.DataFrame:
    """R의 각 열을 x에 대해 회귀한 rolling 베타."""
    x = x.reindex(R.index)
    cov = R.rolling(win, min_periods=minp).cov(x)
    var = x.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0)


def _long(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.stack(future_stack=True).rename(name).reset_index()
    out.columns = ["month", "ticker", name]
    return out


# ----------------------------------------------------------------- 가격 특성
def price_characteristics(daily: pd.DataFrame, macro_chg: pd.DataFrame) -> pd.DataFrame:
    """일별 데이터 -> 월말 기준 가격/거래량/베타 특성 (long format)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    R = _pivot(d, "ret")                       # 일별 수익률
    CLS = _pivot(d, "close")
    VOLU = _pivot(d, "volume")
    DV = _pivot(d, "dolvol")
    HI, LO = _pivot(d, "high"), _pivot(d, "low")

    w1, w3, w12, w36 = C.WIN_SHORT, C.WIN_3M, C.WIN_12M, C.WIN_36M
    mkt = R.mean(axis=1)                       # 동일가중 시장 (일별)

    feats: dict[str, pd.DataFrame] = {}

    # --- 변동성 / 분포 -------------------------------------------------
    feats["VOL_1M"] = _eom(R.rolling(w1, min_periods=15).std())
    feats["VOL_12M"] = _eom(R.rolling(w12, min_periods=120).std())
    feats["SKEW"] = _eom(R.rolling(w12, min_periods=120).skew())
    feats["KURT"] = _eom(R.rolling(w12, min_periods=120).kurt())
    feats["MAXRET"] = _eom(R.rolling(w1, min_periods=15).max())
    feats["MINRET"] = _eom(R.rolling(w1, min_periods=15).min())
    feats["VAR5"] = _eom(R.rolling(w36, min_periods=250).quantile(0.05))
    feats["VAR10"] = _eom(R.rolling(w36, min_periods=250).quantile(0.10))

    # --- 시장 베타 / 잔차 위험 ------------------------------------------
    beta = _roll_beta(R, mkt, w12, 120)
    feats["MKT_BETA"] = _eom(beta)
    var_i = R.rolling(w12, min_periods=120).var()
    var_m = mkt.rolling(w12, min_periods=120).var()
    resid_var = (var_i - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)
    feats["IVOL"] = _eom(np.sqrt(resid_var))
    feats["MKT_RVAR"] = _eom(resid_var)

    # 체계적 왜도(coskewness): E[e_i * e_m^2] / (sd_i * var_m)
    em = mkt - mkt.rolling(w12, min_periods=120).mean()
    ei = R.sub(R.rolling(w12, min_periods=120).mean())
    cosk_num = ei.mul(em.pow(2), axis=0).rolling(w12, min_periods=120).mean()
    feats["COSKEW"] = _eom(cosk_num.div(np.sqrt(var_i) * var_m, axis=0))

    # --- 거시 베타 (논문 TERM/DEF/VIX 베타 대응) -------------------------
    mc = macro_chg.reindex(R.index).ffill()
    for src, name in [("dVIX", "VIX_BETA"), ("dTERM", "TERM_BETA"), ("dDEF", "DEF_BETA")]:
        if src in mc.columns:
            feats[name] = _eom(_roll_beta(R, mc[src], w12, 120))

    # --- 유동성 --------------------------------------------------------
    amihud = (R.abs() / DV.replace(0, np.nan)) * 1e6
    feats["ILLIQ"] = _eom(amihud.rolling(w12, min_periods=120).mean())
    feats["STD_ILLIQ"] = _eom(amihud.rolling(w12, min_periods=120).std())
    logdv = np.log1p(DV)
    feats["DOLVOL"] = _eom(logdv.rolling(w1, min_periods=15).mean())
    feats["STD_DOLVOL"] = _eom(logdv.rolling(w3, min_periods=40).std())
    zero = ((VOLU.fillna(0) == 0) | (R.fillna(0) == 0)).astype(float)
    feats["ZEROTRADE"] = _eom(zero.rolling(w3, min_periods=40).sum())
    feats["RANGE"] = _eom(((HI - LO) / CLS).rolling(w1, min_periods=15).mean())
    turn = VOLU / VOLU.rolling(w12, min_periods=120).median()
    feats["TURN"] = _eom(turn.rolling(w1, min_periods=15).mean())
    feats["STD_TURN"] = _eom(turn.rolling(w3, min_periods=40).std())

    # --- 낙폭 / 가격수준 ------------------------------------------------
    roll_max = CLS.rolling(w12, min_periods=120).max()
    feats["DRAWDOWN"] = _eom(CLS / roll_max - 1.0)
    feats["LOGPRC"] = _eom(np.log(CLS))

    out = None
    for name, df in feats.items():
        piece = _long(df, name)
        out = piece if out is None else out.merge(piece, on=["month", "ticker"], how="outer")

    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp("M")
    return out


# ----------------------------------------------------------------- 모멘텀
def momentum_characteristics(monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 수익률 -> 모멘텀/반전 계열 (논문 STR, MOM6, MOM12, LTR 대응)."""
    m = monthly.sort_values(["ticker", "month"]).copy()
    m["_lr"] = np.log1p(m["ret_m"].fillna(0.0))

    def cum(lo: int, hi: int) -> pd.Series:
        """t-hi ~ t-lo 구간 누적 로그수익률 (1 <= lo <= hi)."""
        s = m.groupby("ticker", observed=True)["_lr"]
        return s.transform(lambda x: x.shift(lo - 1).rolling(hi - lo + 1, min_periods=1).sum())

    m["STR"] = m["ret_m"]                                # 당월 수익률 (단기반전)
    m["MOM6"] = np.expm1(cum(2, 6))
    m["MOM12"] = np.expm1(cum(2, 12))
    m["MOM36"] = np.expm1(cum(13, 36))
    m["LTR"] = np.expm1(cum(13, 48))
    m["MOM_CHG"] = m["MOM6"] - m["MOM12"]
    m["AGE"] = m.groupby("ticker", observed=True).cumcount() + 1

    cols = ["ticker", "month", "STR", "MOM6", "MOM12", "MOM36", "LTR", "MOM_CHG", "AGE"]
    return m[cols]


# ----------------------------------------------------------------- 펀더멘털
def fundamental_characteristics(fund: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """EDGAR 시점정합 패널 + 시가 -> 밸류/수익성/투자 특성."""
    if fund is None or fund.empty:
        return pd.DataFrame(columns=["ticker", "month"])

    px = monthly[["ticker", "month", "close_eom"]]
    f = fund.merge(px, on=["ticker", "month"], how="left").sort_values(["ticker", "month"])

    def col(name):
        return f[name] if name in f.columns else pd.Series(np.nan, index=f.index)

    shares, equity = col("shares"), col("equity")
    assets, liab = col("assets"), col("liabilities")
    rev, ni = col("revenue"), col("netincome")
    opinc, cogs = col("opincome"), col("cogs")
    cash, cfo = col("cash"), col("cfo")

    f["ME"] = f["close_eom"] * shares
    me = f["ME"].replace(0, np.nan)
    asset_nz = assets.replace(0, np.nan)
    eq_nz = equity.replace(0, np.nan)

    f["BM"] = equity / me
    f["EP"] = ni / me
    f["SP"] = rev / me
    f["CFP"] = cfo / me
    f["LEV"] = liab / asset_nz
    f["ROE"] = ni / eq_nz
    f["ROA"] = ni / asset_nz
    f["GMA"] = (rev - cogs) / asset_nz
    f["OP"] = opinc / eq_nz
    f["CASH"] = cash / asset_nz
    f["ACC"] = (ni - cfo) / asset_nz
    f["LOGME"] = np.log(me)

    g = f.groupby("ticker", observed=True)
    for src, dst in [("assets", "AGR"), ("revenue", "SGR"), ("shares", "NI_ISSUE")]:
        f[dst] = g[src].pct_change(12) if src in f.columns else np.nan

    keep = ["ticker", "month", "ME", "LOGME", "BM", "EP", "SP", "CFP", "LEV", "ROE",
            "ROA", "GMA", "OP", "CASH", "ACC", "AGR", "SGR", "NI_ISSUE"]
    keep = [c for c in keep if c in f.columns]
    return f[keep].replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------- 조립
def build_characteristics(daily: pd.DataFrame, monthly: pd.DataFrame,
                          macro_chg: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    print("[chars] price/volume ...")
    p = price_characteristics(daily, macro_chg)
    print("[chars] momentum ...")
    mo = momentum_characteristics(monthly)
    print("[chars] fundamentals ...")
    fu = fundamental_characteristics(fund, monthly)

    out = monthly[["ticker", "month", "ret_m", "close_eom"]].merge(
        mo, on=["ticker", "month"], how="left")
    out = out.merge(p, on=["ticker", "month"], how="left")
    if len(fu):
        out = out.merge(fu, on=["ticker", "month"], how="left")

    out = out.sort_values(["month", "ticker"]).reset_index(drop=True)
    meta = {"ticker", "month", "ret_m", "close_eom", "ME"}
    chars = [c for c in out.columns if c not in meta]
    print(f"[chars] {len(chars)} characteristics, {len(out):,} ticker-months")
    return out
