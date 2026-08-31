# 무료 데이터 수집 파이프라인

논문 *Deep Tangency Portfolio* 를 유료 DB(TRACE / FISD / OptionMetrics) 없이 복제하기 위한
데이터 수집 코드입니다. 논문 정리는 [deep_tangency_portfolio_정리.md](deep_tangency_portfolio_정리.md) 참조.

## 빠른 시작

```bash
pip install yfinance pandas numpy requests lxml truststore

python run_pipeline.py --max-tickers 50 --n-assets 30   # 소규모 시험 (약 1분)
python run_pipeline.py                                  # S&P500 503종목
python run_pipeline.py --universe sec --n-assets 1000   # 생존편향 최소 (수 시간)
```

산출물: `data/processed/panel.npz`

| 배열 | 형태 | 내용 |
|---|---|---|
| `X` | (T, N, K) | t−1 시점 표준화 특성 |
| `y` | (T, N) | t 시점 초과수익률 |
| `mask` | (T, N) | 1 = 실제 자산, **0 = 패딩** |
| `bench` | (T, 2) | MKT_EW, MKT_VW 벤치마크 초과수익률 |
| `months`, `chars`, `bench_names` | | 라벨 |

## 파이프라인 단계

```
1. universe.py        유니버스 선정 (sp500 / sec / custom)
2. prices.py          yfinance 일별 OHLCV -> 일별/월별 수익률
3. macro.py           VIX/TERM/DEF 프록시 + Ken French FF3/RF
4. edgar.py           SEC XBRL companyfacts -> 시점정합 펀더멘털
5. characteristics.py 월별 특성 패널 생성
6. panel.py           표준화 + 시차 정렬 + (T,N,K) 텐서
7. validate.py        룩어헤드 / 표준화 / 마스크 정합성 검증
```

## 논문 대비 매핑

| 논문 | 이 코드 | 비고 |
|---|---|---|
| TRACE 회사채 월수익률 | yfinance 주식 월수익률 (수정주가) | 배당 포함 |
| 매월 시총 상위 3,000 회사채 | 매월 시총 상위 N 주식 | `--n-assets` |
| 채권 특성 41개 | 가격/거래량 특성 32개 | 모멘텀, 변동성, 유동성, 베타 등 |
| 주식 특성 61개 | EDGAR 펀더멘털 15개 | 밸류/수익성/투자 |
| 옵션 특성 30개 | **VIX_BETA 1개뿐** | 아래 한계 참조 |
| TERM 팩터 (장기국채−T빌) | `TLT − BIL` ETF 수익률 스프레드 | 논문 정의 그대로 |
| DEF 팩터 (장기회사채−장기국채) | `LQD − TLT` ETF 수익률 스프레드 | 논문 정의 그대로 |
| EW / VW 시장 벤치마크 | 유니버스에서 직접 계산 | `panel.benchmark_factors` |
| 횡단면 rank → [−1,1], 결측 0 | `panel.cs_rank_standardize` | 논문 3.2절 동일 |
| 인샘플 수익률만 5%/95% 윈저라이즈 | `panel.winsorize_inseample` | 논문 부록 II.2 동일 |
| 인샘플 2004.07–2014.06 / OOS 2014.07–2020.12 | `config.py` 동일 | |

특성 개수는 EDGAR 포함 시 **47개** (논문 132개).

## 알려진 한계 (결과 해석 시 반드시 고려)

1. **생존편향** — `sp500` 모드는 *현재* 구성종목만 담습니다. 위키피디아가 편입/편출
   이력 테이블을 제거해서 자동 복원이 안 됩니다. 실행 시 경고가 출력됩니다.
   완화: `--universe sec` (SEC 등록기업 전체, 상장폐지 기업 포함).

2. **옵션 특성 부재** — 개별종목 옵션의 *과거* 데이터는 무료로 구할 수 없습니다
   (yfinance는 현재 시점 체인만 제공). 논문의 핵심 결론 중 하나인
   "옵션 변수를 빼면 샤프지수 2.13 → 1.01" 은 **이 설정에서 검증 불가**입니다.

3. **펀더멘털 시작 시점** — SEC XBRL 의무화가 2009년경이라 EDGAR 데이터는
   대략 2009년부터입니다. 2004–2009 구간의 펀더멘털 특성은 0으로 채워집니다.
   가격 특성은 전 구간 사용 가능합니다.

4. **자산군이 다름** — 논문은 회사채, 여기는 주식. 절대 수치(2.13)는 재현되지 않는 것이
   정상입니다. 복제 목표는 **"2층 + 소프트맥스 랭킹이 최고"라는 정성적 결론**의 재현입니다.

5. **거래비용 미반영** — 논문 부록 IV의 회전율 기반 비용은 학습 코드 쪽에서 적용하세요.

## 학습 코드에서 주의할 점

```python
import numpy as np
d = np.load("data/processed/panel.npz", allow_pickle=True)
X, y, mask, bench = d["X"], d["y"], d["mask"], d["bench"]
```

- **`mask`를 반드시 쓰세요.** 균형 패널을 위해 빈 슬롯을 0으로 채웠는데,
  softmax는 모든 슬롯에 비중을 배분하므로 마스킹하지 않으면 **존재하지 않는 자산에
  포지션이 잡힙니다.** 소프트맥스 이전에 `-inf`(또는 매우 큰 음수)로 막으세요:

  ```python
  z = model(X_batch)                      # (B, N)
  z = z.masked_fill(mask_batch == 0, -1e9)
  W = softmax_rank_weights(z)             # 패딩 슬롯 비중 ~ 0
  W = W * mask_batch                      # 잔여 수치오차 제거
  ```

- **슬롯 순서는 매월 시총 기준으로 재정렬됩니다.** 같은 인덱스가 같은 종목을 뜻하지
  않습니다. 신경망은 자산을 독립적으로 처리하고 softmax는 월 내 횡단면 연산이라
  문제 없지만, 종목 단위 시계열 분석을 하려면 `data/interim/panel_long.csv` 를 쓰세요.

- **접점 비중 θ는 인샘플에서 고정**하고 OOS에서 재추정하지 마세요 (룩어헤드).

## 네트워크 문제 해결

사내망·백신의 TLS 검사 환경에서 `SSLCertVerificationError: self signed certificate`
가 나면 `src/net.py` 가 처리합니다. 우선순위:

1. `pip install truststore` — OS 인증서 저장소 사용 (가장 간단, 권장)
2. `config.CA_BUNDLE` 에 사내 루트 CA pem 경로 지정
3. `config.VERIFY_SSL = False` — 최후 수단

FRED는 접속이 자주 끊겨서 **선택 사항**으로 강등했습니다. 기본 경로는 yfinance ETF
프록시만 씁니다 (오히려 논문의 TERM/DEF 정의에 더 충실).

## SEC EDGAR 사용 시

`config.SEC_USER_AGENT` 에 **본인 이름과 이메일**을 넣어야 합니다. SEC의 요구사항입니다.

```python
SEC_USER_AGENT = "Hong Gildong hong@example.com"
```

비워두면 EDGAR 단계를 건너뛰고 가격 특성(32개)만으로 패널을 만듭니다.
전체 유니버스 수집은 티커당 약 0.12초 (rate limit 준수) 걸립니다.

## 캐시

`data/raw/` 의 파일이 있으면 재다운로드하지 않습니다. 강제 갱신은 `--force`.
용량이 커지면 `data/raw/daily.csv` 가 가장 큽니다 (티커 500개 기준 수백 MB).
