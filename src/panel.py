"""패널 조립: 특성 -> 학습용 텐서.

논문 3.5절 전처리를 그대로 따릅니다.
  1) 초과수익률 계산 (r = 수익률 - 무위험이자율)
  2) 매월 시가총액 상위 N개 선택 (논문: 상위 3,000 회사채)
  3) 특성을 매월 '횡단면' rank -> [-1, 1] 스케일, 평균 0, 결측은 0
  4) 특성은 t-1, 수익률은 t 로 시차 정렬 (룩어헤드 차단)
  5) 균형 패널 (T, N, K) 텐서로 저장, 빈 슬롯은 mask 로 표시

출력: data/processed/panel.npz
  X      (T, N, K)  float32  t-1 시점 표준화 특성
  y      (T, N)     float32  t 시점 초과수익률
  mask   (T, N)     float32  1 = 실제 자산, 0 = 패딩 (학습 시 반드시 비중 0 처리)
  bench  (T, P)     float32  벤치마크 팩터 초과수익률 (EW/VW 시장 등)
  months (T,)       str      월말 날짜
  chars  (K,)       str      특성 이름
  bench_names (P,)  str
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

META = {"ticker", "month", "ret_m", "close_eom", "ME", "rf", "exret", "mktcap"}


# ----------------------------------------------------------------- 전처리
def cs_rank_standardize(s: pd.Series) -> pd.Series:
    """횡단면 순위 -> [-1, 1], 평균 0, 결측 0. (논문 3.2절)"""
    r = s.rank(pct=True, method="average")
    z = 2.0 * r - 1.0
    z = z - z.mean(skipna=True)          # 횡단면 평균 0
    return z.fillna(0.0)


def add_excess_return(panel: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(ff[["month", "RF"]], on="month", how="left")
    out = out.rename(columns={"RF": "rf"})
    out["rf"] = out["rf"].fillna(0.0)
    out["exret"] = out["ret_m"] - out["rf"]
    return out


def select_universe(panel: pd.DataFrame, n_assets: int) -> pd.DataFrame:
    """매월 시가총액 상위 N개. ME가 없으면 거래대금(DOLVOL)으로 대체."""
    p = panel.copy()
    size = p["ME"] if "ME" in p.columns else pd.Series(np.nan, index=p.index)
    if size.isna().all() and "DOLVOL" in p.columns:
        print("[panel] ME 없음 -> DOLVOL 기준으로 유니버스 선택")
        size = p["DOLVOL"]
    elif "DOLVOL" in p.columns:
        size = size.fillna(p["DOLVOL"].rank(pct=True))
    p["mktcap"] = size

    p = p[p["exret"].notna()]
    p = p[p["AGE"] >= C.MIN_HISTORY_MONTHS] if "AGE" in p.columns else p
    p["_rk"] = p.groupby("month", observed=True)["mktcap"].rank(ascending=False,
                                                                method="first")
    p = p[p["_rk"] <= n_assets].drop(columns="_rk")
    cnt = p.groupby("month", observed=True).size()
    print(f"[panel] 월별 자산 수: min {cnt.min()}, median {int(cnt.median())}, max {cnt.max()}")
    return p.reset_index(drop=True)


def standardize(panel: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    p = panel.copy()
    for c in char_cols:
        p[c] = p.groupby("month", observed=True)[c].transform(cs_rank_standardize)
    return p


def winsorize_inseample(panel: pd.DataFrame, train_end: str, q: float) -> pd.DataFrame:
    """인샘플 수익률만 매월 횡단면 q / 1-q 분위로 절단. OOS는 건드리지 않음."""
    p = panel.copy()
    is_train = p["month"] <= pd.Timestamp(train_end)
    tr = p.loc[is_train]
    lo = tr.groupby("month", observed=True)["exret"].transform(lambda x: x.quantile(q))
    hi = tr.groupby("month", observed=True)["exret"].transform(lambda x: x.quantile(1 - q))
    p.loc[is_train, "exret"] = tr["exret"].clip(lower=lo, upper=hi)
    return p


# ----------------------------------------------------------------- 벤치마크
def benchmark_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """유니버스 자체에서 동일가중/시총가중 시장 팩터를 만든다 (논문 3.3절)."""
    g = panel.groupby("month", observed=True)
    ew = g["exret"].mean().rename("MKT_EW")

    def _vw(df):
        w = df["mktcap"].clip(lower=0).fillna(0.0)
        return np.nan if w.sum() == 0 else float((w * df["exret"]).sum() / w.sum())

    vw = g[["mktcap", "exret"]].apply(_vw).rename("MKT_VW")
    out = pd.concat([ew, vw], axis=1).reset_index()
    out["MKT_VW"] = out["MKT_VW"].fillna(out["MKT_EW"])
    return out


# ----------------------------------------------------------------- 텐서화
def to_tensors(panel: pd.DataFrame, char_cols: list[str], n_assets: int):
    """t-1 특성 / t 수익률 정렬 + 균형 패널 (T, N, K)."""
    p = panel.sort_values(["ticker", "month"]).copy()

    # 특성을 한 달 뒤로 밀어, 각 행이 'month 시점 수익률 + 직전월 특성' 을 갖게 함
    lagged = p.groupby("ticker", observed=True)[char_cols].shift(1)
    p[char_cols] = lagged
    p = p.dropna(subset=char_cols, how="all")
    p[char_cols] = p[char_cols].fillna(0.0)

    months = np.sort(p["month"].unique())
    K = len(char_cols)
    T = len(months)
    X = np.zeros((T, n_assets, K), dtype=np.float32)
    y = np.zeros((T, n_assets), dtype=np.float32)
    mask = np.zeros((T, n_assets), dtype=np.float32)

    for i, m in enumerate(months):
        sub = p[p["month"] == m].sort_values("mktcap", ascending=False).head(n_assets)
        n = len(sub)
        X[i, :n, :] = sub[char_cols].to_numpy(dtype=np.float32)
        y[i, :n] = sub["exret"].to_numpy(dtype=np.float32)
        mask[i, :n] = 1.0

    return X, y, mask, months


def build_panel(chars: pd.DataFrame, ff: pd.DataFrame,
                n_assets: int = None, save: bool = True) -> dict:
    n_assets = n_assets or C.N_ASSETS

    p = add_excess_return(chars, ff)
    p = p[(p["month"] >= pd.Timestamp(C.TRAIN_START)) & (p["month"] <= pd.Timestamp(C.OOS_END))]

    char_cols = [c for c in p.columns if c not in META]
    print(f"[panel] 특성 {len(char_cols)}개: {', '.join(char_cols)}")

    p = select_universe(p, n_assets)
    p = winsorize_inseample(p, C.TRAIN_END, C.WINSOR_Q)
    bench = benchmark_factors(p)
    p = standardize(p, char_cols)

    X, y, mask, months = to_tensors(p, char_cols, n_assets)

    b = bench.set_index("month").reindex(pd.DatetimeIndex(months)).fillna(0.0)
    bench_arr = b[["MKT_EW", "MKT_VW"]].to_numpy(dtype=np.float32)

    out = {
        "X": X, "y": y, "mask": mask,
        "bench": bench_arr,
        "months": np.array([pd.Timestamp(m).strftime("%Y-%m-%d") for m in months]),
        "chars": np.array(char_cols),
        "bench_names": np.array(["MKT_EW", "MKT_VW"]),
    }

    if save:
        path = C.PROC / "panel.npz"
        np.savez_compressed(path, **out)
        p.to_csv(C.INTERIM / "panel_long.csv", index=False)
        bench.to_csv(C.INTERIM / "benchmark_factors.csv", index=False)
        print(f"[panel] X={X.shape}  y={y.shape}  bench={bench_arr.shape} -> {path}")

        n_is = int((pd.DatetimeIndex(months) <= pd.Timestamp(C.TRAIN_END)).sum())
        print(f"[panel] 인샘플 {n_is}개월 / OOS {len(months) - n_is}개월")
        for name, col in zip(["MKT_EW", "MKT_VW"], bench_arr.T):
            sr = col.mean() / col.std(ddof=1) * np.sqrt(12)
            print(f"[panel] {name} 전체기간 연율 샤프지수 = {sr:.2f}")
    return out
