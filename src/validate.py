"""패널 정합성 검증. 논문 복제에서 가장 치명적인 실수 3가지를 잡는다.

  1) 룩어헤드 : t 시점 정보가 t 시점 특성에 섞이면 성과가 가짜로 좋아진다.
  2) 표준화   : 횡단면 rank 가 아니라 전체기간 통합 표준화를 하면 미래 정보가 샌다.
  3) 패딩     : 균형 패널의 빈 슬롯을 mask 로 막지 않으면 존재하지 않는 자산에
                포트폴리오 비중이 배분된다 (softmax 는 모든 슬롯에 비중을 준다).

사용: python -m src.validate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


def check(verbose: bool = True) -> dict:
    long_path = C.INTERIM / "panel_long.csv"
    npz_path = C.PROC / "panel.npz"
    if not long_path.exists() or not npz_path.exists():
        raise FileNotFoundError("먼저 run_pipeline.py 를 실행하세요.")

    p = pd.read_csv(long_path, parse_dates=["month"]).sort_values(["ticker", "month"])
    d = np.load(npz_path, allow_pickle=True)
    X, y, mask = d["X"], d["y"], d["mask"]
    months = pd.DatetimeIndex(d["months"])
    chars = list(d["chars"])

    res, fail = {}, []

    def ok(name, cond, msg):
        res[name] = bool(cond)
        mark = "OK  " if cond else "FAIL"
        if verbose:
            print(f"  [{mark}] {name}: {msg}")
        if not cond:
            fail.append(name)

    if verbose:
        print("\n=== 패널 검증 ===")

    # --- 1. 결측/무한대 ------------------------------------------------
    ok("no_nan", not (np.isnan(X).any() or np.isnan(y).any()),
       f"X NaN {int(np.isnan(X).sum())}, y NaN {int(np.isnan(y).sum())}")
    ok("no_inf", not (np.isinf(X).any() or np.isinf(y).any()),
       f"X Inf {int(np.isinf(X).sum())}")

    # --- 2. 횡단면 rank 표준화 -----------------------------------------
    ok("x_range", X.min() >= -1.0001 and X.max() <= 1.0001,
       f"X 범위 [{X.min():.3f}, {X.max():.3f}] (논문: [-1, 1])")
    # 표준화 시점(시차 적용 전)의 횡단면 평균은 정확히 0 이어야 한다.
    char_cols = [c for c in chars if c in p.columns]
    pre_lag = p.groupby("month")[char_cols].mean().abs().to_numpy().max()
    ok("cs_demeaned", pre_lag < 1e-8,
       f"시차 적용 전 횡단면 평균 최대 절대값 {pre_lag:.2e} (정확히 0이어야 함)")

    # 시차 적용 후에는 0 이 아니어도 정상이다. t월 행은 t-1월 횡단면에서 표준화된
    # 값을 들고 오는데, 유니버스가 매월 바뀌므로 t월 기준 평균은 0에서 벗어난다.
    valid = mask > 0
    post_lag = np.abs([X[t][valid[t]].mean() for t in range(len(X))]).max()
    if verbose:
        print(f"  [info] 시차 적용 후 횡단면 평균 최대 절대값 {post_lag:.4f} "
              f"(유니버스 변동 때문이며 정상)")

    if "STR" in chars:
        wm = p.groupby("month").apply(
            lambda g: _rank_corr(g["STR"], g["exret"]), include_groups=False).dropna()
        ok("rank_exact", wm.min() > 0.999,
           f"월별 corr(STR, 당월수익률) 최소 {wm.min():.4f} (rank 표준화면 1.0)")

    # --- 3. 룩어헤드 ---------------------------------------------------
    if "STR" in chars:
        q = p.copy()
        q["STR_lag"] = q.groupby("ticker")["STR"].shift(1)
        q["exret_prev"] = q.groupby("ticker")["exret"].shift(1)
        s = q.dropna(subset=["STR_lag", "exret_prev", "exret"])
        c_prev = _rank_corr(s["STR_lag"], s["exret_prev"])
        c_now = _rank_corr(s["STR_lag"], s["exret"])
        ok("lag_aligned", c_prev > 0.5,
           f"corr(시차특성, 전월수익률) = {c_prev:.3f} (양수여야 정상)")
        ok("no_lookahead", abs(c_now) < 0.10,
           f"corr(시차특성, 당월수익률) = {c_now:.3f} (0 근처여야 정상)")

    # --- 4. 마스크 -----------------------------------------------------
    n_valid = mask.sum(1)
    ok("mask_nonempty", n_valid.min() > 0,
       f"월별 유효 자산 수 min {int(n_valid.min())} / max {int(n_valid.max())}")
    pad_x = X[mask == 0]
    ok("pad_zero", pad_x.size == 0 or np.abs(pad_x).max() < 1e-6,
       f"패딩 슬롯 비율 {100 * (1 - mask.mean()):.2f}% (특성은 0으로 채워짐)")

    # --- 5. 표본 분할 --------------------------------------------------
    n_is = int((months <= pd.Timestamp(C.TRAIN_END)).sum())
    n_oos = int((months > pd.Timestamp(C.TRAIN_END)).sum())
    ok("split", n_is > 0 and n_oos > 0,
       f"인샘플 {n_is}개월 / OOS {n_oos}개월 (논문: 120 / 78)")

    # --- 6. 벤치마크 ---------------------------------------------------
    if verbose:
        print("\n=== 벤치마크 팩터 (전체기간) ===")
        for name, col in zip(d["bench_names"], d["bench"].T):
            sr = col.mean() / col.std(ddof=1) * np.sqrt(12)
            print(f"  {name}: 월평균 {col.mean() * 100:6.3f}%  "
                  f"월변동성 {col.std(ddof=1) * 100:5.2f}%  연율 샤프지수 {sr:5.2f}")
        print(f"\n특성 {len(chars)}개 / 자산슬롯 {X.shape[1]} / 기간 "
              f"{months[0]:%Y-%m} ~ {months[-1]:%Y-%m}")
        print("결과:", "전부 통과" if not fail else f"{len(fail)}건 실패 -> {fail}")

    res["_failed"] = fail
    return res


if __name__ == "__main__":
    check()
