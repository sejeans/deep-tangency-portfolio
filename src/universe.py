"""자산 유니버스 구성.

논문의 유니버스는 지수가 아니라 "매월 시가총액 상위 3,000 회사채" 입니다.
즉 고정 종목 리스트가 아니라 매월 재선정되는 집합입니다. 그래서 지수 구성종목
목록을 쓰면 논문과 구조가 달라지고, 더 큰 문제로 생존편향이 들어옵니다.

세 가지 모드:

  "sp500"  (빠름 / 생존편향 있음)
      위키피디아의 '현재' S&P500 구성종목 503개.
      과거에 편입되어 있었지만 지금은 빠진 종목(상장폐지·피인수)이 제외되므로
      성과가 낙관적으로 편향됩니다. 파이프라인 시험용으로만 쓰세요.
      (위키피디아가 편출 이력 테이블을 제거해서 자동 복원이 불가능합니다.
       테이블이 다시 생기면 아래 파서가 자동으로 잡습니다.)

  "sec"    (느림 / 생존편향 최소)  <- 논문 구조에 가장 가까움
      SEC 등록 상장기업 전체(약 1만개). 상장폐지 기업도 남아 있어 편향이 작습니다.
      여기서 매월 시총/거래대금 상위 N을 뽑으면 논문의 선정 방식과 같아집니다.
      단, yfinance 다운로드가 오래 걸립니다(수 시간). --max-tickers 로 조절하세요.

  "custom" (직접 지정)
      data/raw/universe_custom.csv 에 ticker 컬럼 한 줄씩. 유료 DB나 다른 소스에서
      시점정합 구성종목을 확보했다면 이 모드를 쓰세요.
"""
from __future__ import annotations

import io
import json

import pandas as pd

from . import config as C
from . import net

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_UA = "Mozilla/5.0 (research script)"

SURVIVORSHIP_WARNING = """
  !! 생존편향 경고 !!
  'sp500' 모드는 현재 구성종목만 담고 있어, 과거에 존재했다 사라진 종목이 빠집니다.
  이 상태로 얻은 샤프지수는 실제보다 높게 나옵니다. 결과를 논문과 직접 비교하지 마세요.
  완화하려면: --universe sec  (SEC 등록기업 전체, 상장폐지 포함)
"""


def _clean(t: str) -> str:
    """위키 표기(BRK.B) -> yfinance 표기(BRK-B)."""
    return str(t).strip().upper().replace(".", "-")


# ----------------------------------------------------------------- sp500
def _sp500() -> pd.DataFrame:
    r = net.get(WIKI_SP500, user_agent=_UA)
    tables = pd.read_html(io.StringIO(r.text))

    current = tables[0]
    tick_col = "Symbol" if "Symbol" in current.columns else current.columns[0]
    rows = [pd.DataFrame({"ticker": current[tick_col].map(_clean),
                          "added": pd.NaT, "removed": pd.NaT})]

    # 편입/편출 이력 테이블이 존재하면 과거 구성종목까지 복원한다.
    n_hist = 0
    for tb in tables[1:]:
        cols = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                for c in tb.columns]
        tb = tb.copy()
        tb.columns = cols
        date_col = next((c for c in cols if "Date" in c), None)
        add_col = next((c for c in cols if "Added" in c and "Ticker" in c), None)
        rem_col = next((c for c in cols if "Removed" in c and "Ticker" in c), None)
        if not date_col:
            continue
        for col, field in [(add_col, "added"), (rem_col, "removed")]:
            if not col:
                continue
            h = tb[[date_col, col]].dropna()
            h.columns = ["date", "ticker"]
            h["ticker"] = h["ticker"].map(_clean)
            h[field] = pd.to_datetime(h["date"], errors="coerce")
            other = "removed" if field == "added" else "added"
            rows.append(h[["ticker", field]].assign(**{other: pd.NaT}))
            n_hist += len(h)

    if n_hist == 0:
        print(SURVIVORSHIP_WARNING)
    else:
        print(f"[universe] 편입/편출 이력 {n_hist}건 반영 (생존편향 부분 완화)")

    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------- sec
def _sec() -> pd.DataFrame:
    ua = C.SEC_USER_AGENT.strip() or _UA
    r = net.get(SEC_TICKERS, user_agent=ua)
    data = json.loads(r.text)
    tks = sorted({_clean(v["ticker"]) for v in data.values() if v.get("ticker")})
    print(f"[universe] SEC 등록기업 {len(tks)}개 (상장폐지 포함 -> 생존편향 최소)")
    return pd.DataFrame({"ticker": tks, "added": pd.NaT, "removed": pd.NaT})


# ----------------------------------------------------------------- custom
def _custom() -> pd.DataFrame:
    path = C.RAW / "universe_custom.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. ticker 컬럼을 가진 csv 를 만들어 주세요.")
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    print(f"[universe] custom {len(df)}개 <- {path}")
    return pd.DataFrame({"ticker": df[col].map(_clean), "added": pd.NaT, "removed": pd.NaT})


# ----------------------------------------------------------------- 진입점
def load_universe(mode: str = "sp500", force: bool = False) -> pd.DataFrame:
    """columns: ticker, added(datetime|NaT), removed(datetime|NaT)"""
    cache = C.RAW / f"universe_{mode}.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, parse_dates=["added", "removed"])
        if mode == "sp500" and df["removed"].notna().sum() == 0:
            print(SURVIVORSHIP_WARNING)
        return df

    builder = {"sp500": _sp500, "sec": _sec, "custom": _custom}
    if mode not in builder:
        raise ValueError(f"mode 는 {list(builder)} 중 하나여야 합니다 (받은 값: {mode})")

    df = builder[mode]()
    df = (df.sort_values(["ticker", "added", "removed"])
            .groupby("ticker", as_index=False)
            .agg({"added": "min", "removed": "max"}))
    df = df[df["ticker"].str.fullmatch(r"[A-Z][A-Z\-]{0,9}")].reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"[universe] {len(df)} tickers -> {cache}")
    return df


def tickers(mode: str = "sp500", force: bool = False) -> list[str]:
    return sorted(load_universe(mode, force)["ticker"].unique().tolist())


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    ts = tickers(m)
    print(f"{len(ts)} tickers: {ts[:15]} ...")
