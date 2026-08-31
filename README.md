# Deep Tangency Portfolio (복제 프로젝트)

Feng, Jiang, Li, Song, Wang, *"Deep Tangency Portfolio"* (SSRN 3971274) 를
유료 데이터(TRACE / FISD / OptionMetrics) 없이 무료 데이터로 복제해보는 프로젝트.

- 논문 요약(한글): [deep_tangency_portfolio_정리.md](deep_tangency_portfolio_정리.md)
  — 목적, 데이터, 방법론(수식 포함), 결과, 복제 가이드
- 데이터 수집 파이프라인 사용법: [README_데이터수집.md](README_데이터수집.md)

## 빠른 시작

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python run_pipeline.py --max-tickers 50 --n-assets 30   # 소규모 시험 실행
```

산출물 `data/processed/panel.npz` 는 학습용 `(T, N, K)` 텐서입니다
(`X` 특성, `y` 수익률, `mask` 유효 자산, `bench` 벤치마크 팩터).

## 구성

```
src/
  config.py           경로·기간·하이퍼파라미터 설정
  net.py              HTTP 요청 (SSL 검사 프록시 환경 대응)
  universe.py         종목 유니버스 (S&P500 / SEC 전체 / 직접 지정)
  prices.py           yfinance 일별 가격 -> 월별 수익률
  macro.py            거시 프록시(VIX/TERM/DEF) + Fama-French 팩터
  edgar.py            SEC EDGAR 시점정합 펀더멘털
  characteristics.py  월별 특성(팩터 후보) 생성
  panel.py            횡단면 표준화 + 시차정렬 + 텐서화
  validate.py         룩어헤드/표준화/마스크 정합성 검증
run_pipeline.py       전체 파이프라인 실행 스크립트
```

## 한계 (실제 논문과의 차이)

무료 데이터로 대체하는 과정에서 생기는 한계는
[README_데이터수집.md](README_데이터수집.md#알려진-한계-결과-해석-시-반드시-고려) 에 정리했습니다.
특히 **옵션 이력 데이터는 무료로 구할 수 없어** 논문의 132개 특성 중 옵션 30개는
VIX 베타 1개로만 근사했고, S&P500 현재 구성종목만 쓰는 기본 모드는 생존편향이 있습니다.
