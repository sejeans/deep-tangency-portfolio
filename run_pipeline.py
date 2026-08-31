"""Deep Tangency Portfolio - 무료 데이터 수집 파이프라인.

사용법:
    python run_pipeline.py                  # 전체 실행 (캐시 사용)
    python run_pipeline.py --force          # 캐시 무시하고 재다운로드
    python run_pipeline.py --skip-edgar     # 펀더멘털 없이 가격 특성만
    python run_pipeline.py --n-assets 200   # 월별 자산 수 변경
    python run_pipeline.py --universe sec   # 생존편향 최소 (SEC 등록기업 전체, 느림)
    python run_pipeline.py --max-tickers 50 # 소규모 시험 실행

단계:
    1. 유니버스   : --universe sp500 | sec | custom  (생존편향 주의, src/universe.py 참조)
    2. 가격       : yfinance 일별 OHLCV -> 일별/월별 수익률
    3. 거시       : yfinance 프록시(VIX/TERM/DEF) + Ken French FF3/RF
    4. 펀더멘털   : SEC EDGAR companyfacts (시점정합)
    5. 특성       : 월별 characteristics 패널
    6. 텐서       : (T, N, K) 학습용 npz
    7. 검증       : 룩어헤드/표준화/마스크 정합성 체크
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src import config as C
from src import characteristics as ch
from src import edgar, macro, panel, prices, universe, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재다운로드")
    ap.add_argument("--skip-edgar", action="store_true", help="EDGAR 펀더멘털 생략")
    ap.add_argument("--n-assets", type=int, default=C.N_ASSETS)
    ap.add_argument("--max-tickers", type=int, default=0, help="테스트용 티커 수 제한")
    ap.add_argument("--no-validate", action="store_true", help="마지막 검증 단계 생략")
    ap.add_argument("--universe", default="sp500", choices=["sp500", "sec", "custom"],
                    help="sp500=빠름(생존편향 있음) / sec=전체(편향 최소, 느림) / custom=직접지정")
    args = ap.parse_args()

    t0 = time.time()

    # 1 ---------------------------------------------------------- 유니버스
    print("\n=== 1. 유니버스 ===")
    tks = universe.tickers(mode=args.universe, force=args.force)
    if args.max_tickers:
        tks = tks[: args.max_tickers]
    print(f"{len(tks)} tickers")

    # 2 ---------------------------------------------------------- 가격
    print("\n=== 2. 가격 (yfinance) ===")
    daily = prices.download_daily(tks, force=args.force)
    daily = prices.add_daily_returns(daily)
    monthly = prices.monthly_returns(daily)
    print(f"월별 관측치 {len(monthly):,} / 종목 {monthly['ticker'].nunique()}")

    # 3 ---------------------------------------------------------- 거시
    print("\n=== 3. 거시 (FRED) + 팩터 (Ken French) ===")
    macro_chg = macro.macro_daily_changes()
    ff = macro.fama_french_monthly(force=args.force)

    # 4 ---------------------------------------------------------- 펀더멘털
    print("\n=== 4. 펀더멘털 (SEC EDGAR) ===")
    fund = pd.DataFrame()
    if args.skip_edgar or not C.SEC_USER_AGENT.strip():
        print("건너뜀 (config.SEC_USER_AGENT 미설정 또는 --skip-edgar). "
              "가격 특성만으로 패널을 만듭니다.")
    else:
        univ = sorted(monthly["ticker"].unique().tolist())
        facts = edgar.download_facts(univ, force=args.force)
        months = pd.DatetimeIndex(sorted(monthly["month"].unique()))
        fund = edgar.as_of_panel(facts, months)
        print(f"펀더멘털 패널: {fund.shape}")

    # 5 ---------------------------------------------------------- 특성
    print("\n=== 5. 특성 생성 ===")
    chars = ch.build_characteristics(daily, monthly, macro_chg, fund)
    chars.to_csv(C.INTERIM / "characteristics.csv", index=False)

    # 6 ---------------------------------------------------------- 텐서
    print("\n=== 6. 패널 텐서 ===")
    panel.build_panel(chars, ff, n_assets=args.n_assets)

    # 7 ---------------------------------------------------------- 검증
    if not args.no_validate:
        validate.check()

    print(f"\n완료. {time.time() - t0:.0f}초")
    print(f"학습용 파일: {C.PROC / 'panel.npz'}")


if __name__ == "__main__":
    main()
