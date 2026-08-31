"""SEC EDGAR XBRL companyfacts -> 시점정합(point-in-time) 펀더멘털.

왜 EDGAR인가:
  yfinance의 재무제표는 최근 4~8분기만 제공 -> 10년 패널을 못 만듭니다.
  EDGAR companyfacts 는 전체 이력 + 각 수치의 '공시일(filed)' 을 함께 줍니다.
  월말 시점에 filed <= 월말 인 수치만 사용하면 look-ahead bias가 원천 차단됩니다.
  (논문이 특성을 t-1 시점 정보로 제한하는 것과 같은 취지)

요구사항: config.SEC_USER_AGENT 에 "이름 이메일" 형식으로 연락처를 넣어야 합니다.
          SEC 요구사항이며, 비워두면 이 단계는 건너뜁니다 (가격 특성만으로도 동작).
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import config as C
from . import net

TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# us-gaap 태그 -> 우리 변수명. 앞쪽 태그를 우선 사용.
TAGS = {
    "assets":      ["Assets"],
    "liabilities": ["Liabilities"],
    "equity":      ["StockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "revenue":     ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet"],
    "netincome":   ["NetIncomeLoss"],
    "opincome":    ["OperatingIncomeLoss"],
    "cogs":        ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    "cash":        ["CashAndCashEquivalentsAtCarryingValue"],
    "cfo":         ["NetCashProvidedByUsedInOperatingActivities"],
    "shares":      ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding",
                    "WeightedAverageNumberOfSharesOutstandingBasic"],
}


def _session():
    if not C.SEC_USER_AGENT.strip():
        raise RuntimeError("config.SEC_USER_AGENT 를 '이름 이메일' 형식으로 채워주세요.")
    return net.session(user_agent=C.SEC_USER_AGENT)


def cik_map(force: bool = False) -> dict[str, int]:
    cache = C.RAW / "cik_map.json"
    if cache.exists() and not force:
        return {k: int(v) for k, v in json.loads(cache.read_text()).items()}
    s = _session()
    r = s.get(TICKER_MAP, timeout=30)
    r.raise_for_status()
    out = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}
    cache.write_text(json.dumps(out))
    print(f"[edgar] cik map: {len(out)} tickers")
    return out


def _tag_rows(node: dict) -> list[tuple]:
    out = []
    for unit_items in node.get("units", {}).values():
        for it in unit_items:
            if it.get("val") is None or not it.get("filed"):
                continue
            out.append((it.get("end"), it["filed"], float(it["val"])))
    return out


def _extract(facts: dict) -> pd.DataFrame:
    """companyfacts JSON -> long df (var, end, filed, val)

    한 변수에 여러 us-gaap 태그 후보가 있다. 기업마다 쓰는 태그가 다르고
    (예: Apple 은 Revenues 대신 RevenueFromContractWithCustomer... 를 쓴다),
    한 기업이 시기에 따라 태그를 바꾸기도 한다. 그래서
      1) 후보 태그를 전부 수집하고
      2) 같은 (end, filed) 가 겹치면 우선순위가 높은 태그를 채택
    한다. 첫 번째로 '존재하는' 태그에서 멈추면 관측치를 대량으로 잃는다.
    """
    us = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    frames = []
    for var, tags in TAGS.items():
        for prio, tag in enumerate(tags):
            node = us.get(tag) or dei.get(tag)
            if not node:
                continue
            rows = _tag_rows(node)
            if rows:
                f = pd.DataFrame(rows, columns=["end", "filed", "val"])
                f["var"] = var
                f["_prio"] = prio
                frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["var", "end", "filed", "val"])

    df = pd.concat(frames, ignore_index=True)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "filed"])
    df = (df.sort_values(["var", "end", "filed", "_prio"])
            .drop_duplicates(["var", "end", "filed"], keep="first"))
    return df[["var", "end", "filed", "val"]]


def download_facts(tks: list[str], force: bool = False) -> pd.DataFrame:
    """columns: ticker, var, end, filed, val"""
    cache = C.RAW / "edgar_facts.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, parse_dates=["end", "filed"])
        print(f"[edgar] cache hit: {df['ticker'].nunique()} tickers, {len(df):,} facts")
        return df

    cm = cik_map()
    s = _session()
    frames, miss = [], 0
    for i, t in enumerate(tks, 1):
        cik = cm.get(t)
        if cik is None:
            miss += 1
            continue
        try:
            r = s.get(FACTS.format(cik=cik), timeout=30)
            if r.status_code != 200:
                miss += 1
                continue
            d = _extract(r.json())
            if len(d):
                d.insert(0, "ticker", t)
                frames.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {t}: {e}")
            miss += 1
        time.sleep(C.SEC_SLEEP)
        if i % 50 == 0:
            print(f"[edgar] {i}/{len(tks)} (missing {miss})")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(cache, index=False)
    print(f"[edgar] saved {len(df):,} facts -> {cache}")
    return df


def as_of_panel(facts: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """각 (ticker, month) 에 대해 filed <= month 인 가장 최신 수치를 붙인다.

    merge_asof 로 시점정합 조인. 반환: ticker, month, <var...>
    """
    if facts.empty:
        return pd.DataFrame(columns=["ticker", "month"])

    f = facts.sort_values("filed")
    # 같은 (ticker,var,filed) 중복은 가장 최근 회계기간(end) 것을 채택
    f = (f.sort_values(["ticker", "var", "filed", "end"])
           .drop_duplicates(["ticker", "var", "filed"], keep="last"))

    grid = pd.MultiIndex.from_product(
        [sorted(f["ticker"].unique()), months], names=["ticker", "month"]
    ).to_frame(index=False).sort_values("month")

    out = grid.copy()
    for var, sub in f.groupby("var", observed=True):
        sub = sub[["ticker", "filed", "val"]].rename(columns={"val": var}).sort_values("filed")
        merged = pd.merge_asof(
            grid.sort_values("month"), sub,
            left_on="month", right_on="filed", by="ticker",
            direction="backward", allow_exact_matches=True,
        )
        out[var] = merged[var].to_numpy()
    return out


if __name__ == "__main__":
    from .universe import tickers
    print(download_facts(tickers()[:5]).head())
