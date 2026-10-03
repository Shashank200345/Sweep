"""Massive client against httpx.MockTransport fixtures -- no key, no network."""
from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pandas as pd
import pytest

from sweep.data.massive import DataCache, MassiveClient, MassiveError

T0 = int(pd.Timestamp("2024-03-01 14:30", tz="UTC").value // 1_000_000)   # ms


def bar(i: int) -> dict:
    return {"t": T0 + i * 60_000, "o": 500 + i, "h": 501 + i, "l": 499 + i, "c": 500.5 + i, "v": 1000}


class Router:
    def __init__(self, fail_first: int = 0, delay: float = 0.0):
        self.fail_first, self.delay = fail_first, delay
        self.calls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(str(req.url))
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail_first > 0:
                self.fail_first -= 1
                return httpx.Response(429, headers={"retry-after": "0"}, json={"status": "ERROR"})
            assert req.headers["Authorization"] == "Bearer test-key"
            p = req.url.path
            if p.startswith("/v2/aggs/"):
                if req.url.params.get("cursor") == "2":
                    return httpx.Response(200, json={"results": [bar(2), bar(3)], "status": "OK"})
                return httpx.Response(200, json={"results": [bar(0), bar(1)], "status": "OK",
                                                 "next_url": "https://api.massive.com/v2/aggs/next?cursor=2"})
            if p == "/v2/aggs/next":
                return httpx.Response(200, json={"results": [bar(2), bar(3)], "status": "OK"})
            if p == "/stocks/v1/dividends":
                return httpx.Response(200, json={"status": "OK", "results": [
                    {"ex_dividend_date": "2024-03-15", "declaration_date": "2024-03-01", "pay_date": "2024-04-30",
                     "cash_amount": 1.6, "ticker": "SPY"}]})
            if p == "/v3/reference/options/contracts":
                return httpx.Response(200, json={"status": "OK", "results": [
                    {"ticker": "O:SPY240419C00500000", "contract_type": "call", "strike_price": 500,
                     "expiration_date": "2024-04-19", "shares_per_contract": 100},
                    {"ticker": "O:SPY240419P00500000", "contract_type": "put", "strike_price": 500,
                     "expiration_date": "2024-04-19", "shares_per_contract": 100},
                    {"ticker": "O:SPY1240419C00500000", "contract_type": "call", "strike_price": 500,
                     "expiration_date": "2024-04-19", "shares_per_contract": 10}]})
            if p.startswith("/v3/quotes/"):
                t = p.rsplit("/", 1)[1]
                if t.endswith("P00500000"):
                    return httpx.Response(200, json={"status": "OK", "results": []})
                assert req.url.params["order"] == "desc" and req.url.params["limit"] == "1"
                ts = int(req.url.params["timestamp.lte"]) - 7_000_000_000
                return httpx.Response(200, json={"status": "OK", "results": [
                    {"bid_price": 4.1, "ask_price": 4.2, "sip_timestamp": ts}]})
            return httpx.Response(404, json={"status": "NOT_FOUND"})
        finally:
            self.inflight -= 1


def client(router: Router, tmp_path=None, **kw) -> MassiveClient:
    return MassiveClient(api_key="test-key", transport=httpx.MockTransport(router),
                         cache=DataCache(tmp_path), **kw)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(MassiveError, match="--fixture"):
        MassiveClient()


def test_minute_bars_follow_pagination(tmp_path):
    r = Router()
    df = client(r, tmp_path).minute_bars(date(2024, 3, 1), date(2024, 3, 1))
    assert list(df["close"]) == [500.5, 501.5, 502.5, 503.5]
    assert str(df.index.tz) == "UTC" and df.index[0] == pd.Timestamp("2024-03-01 14:30", tz="UTC")


def test_historical_bars_are_cached(tmp_path):
    r = Router()
    client(r, tmp_path).minute_bars(date(2024, 3, 1), date(2024, 3, 1))
    n = len(r.calls)
    df = client(r, tmp_path).minute_bars(date(2024, 3, 1), date(2024, 3, 1))
    assert len(r.calls) == n and len(df) == 4


def test_retry_on_429(tmp_path):
    r = Router(fail_first=2)
    df = client(r, tmp_path).dividends()
    assert len(r.calls) == 3 and df["cash_amount"].iloc[0] == 1.6
    assert df["ex_date"].iloc[0] == pd.Timestamp("2024-03-15")


def test_gives_up_after_retries(tmp_path):
    r = Router(fail_first=99)
    with pytest.raises(MassiveError) as e:
        client(r, tmp_path, max_retries=2).dividends()
    assert e.value.status == 429


def test_contracts_skip_non_standard_deliverables(tmp_path):
    df = client(Router(), tmp_path).contracts(date(2024, 3, 1), date(2024, 4, 1), date(2024, 4, 30))
    assert list(df["ticker"]) == ["O:SPY240419C00500000", "O:SPY240419P00500000"]
    assert df["expiry"].iloc[0] == pd.Timestamp("2024-04-19")


def test_quotes_as_of_never_after_and_missing_is_nan(tmp_path):
    at = pd.Timestamp("2024-03-01 15:30", tz="America/New_York")
    q = client(Router(), tmp_path).quotes(["O:SPY240419C00500000", "O:SPY240419P00500000"], at)
    c = q.loc["O:SPY240419C00500000"]
    assert c["bid"] == 4.1 and c["ask"] == 4.2 and c["ts"] <= at and (at - c["ts"]).total_seconds() == 7
    assert pd.isna(q.loc["O:SPY240419P00500000", "bid"])


def test_quotes_are_fetched_concurrently_within_the_limit(tmp_path):
    r = Router(delay=0.05)
    tickers = [f"O:SPY240419C00{500 + i:03d}000" for i in range(30)]
    client(r, tmp_path, concurrency=8).quotes(tickers, pd.Timestamp("2024-03-01 20:30", tz="UTC"))
    assert r.max_inflight == 8
