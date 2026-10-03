"""Massive (formerly Polygon.io) REST client: request/response only, concurrent via asyncio.

No WebSocket / streaming feed (AGENTS.md non-negotiable 3). Where Massive has no
batch endpoint (historical per-contract quotes), requests for every contract are
fired concurrently, bounded by a semaphore, instead of looped sequentially.

Endpoints used (https://massive.com/docs/rest):
  GET /v2/aggs/ticker/{t}/range/1/minute/{from}/{to}   SPY minute bars
  GET /stocks/v1/dividends                            SPY cash dividends
  GET /v3/reference/options/contracts                 contracts listed as of a past date
  GET /v3/quotes/{optionsTicker}                      historical NBBO (Options Advanced plan;
                                                       history from 2022-03-07 per Massive docs)
  GET /v3/snapshot/options/{underlying}               current chain -- we read ONLY last_quote
                                                       bid/ask/time, never Massive's greeks or IV

Every method returns the schemas documented in ``sweep.data.market``.
Historical responses are cached on disk (``DataCache``); nothing current is cached.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
import pandas as pd

from ..config import UNDERLYING
from .market import ET, empty_quotes

BASE_URL = os.environ.get("MASSIVE_ENDPOINT", "https://api.massive.com")


class MassiveError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class DataCache:
    """Parquet files keyed by sha256 of (endpoint, params). Separate from the Jev JSONL cache."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None

    def _path(self, key: dict) -> Path:
        h = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:32]
        return self.root / f"{h}.parquet"

    def get(self, key: dict) -> pd.DataFrame | None:
        if self.root is None:
            return None
        p = self._path(key)
        return pd.read_parquet(p) if p.exists() else None

    def put(self, key: dict, df: pd.DataFrame) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path(key))


def _today_et() -> date:
    return pd.Timestamp.now(tz=ET).date()


@dataclass
class MassiveClient:
    api_key: str | None = None
    base_url: str = BASE_URL
    concurrency: int = 10
    max_retries: int = 5
    timeout: float = 30.0
    cache: DataCache = field(default_factory=lambda: DataCache(".sweep_cache/massive"))
    transport: httpx.AsyncBaseTransport | None = None      # tests inject httpx.MockTransport
    label: str = "Massive"
    is_fixture: bool = False
    requests_made: int = 0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("MASSIVE_API_KEY")
        if not self.api_key:
            raise MassiveError("MASSIVE_API_KEY is not set. Pass --fixture to run on the offline "
                               "fixture market (reports are labelled 'fixture data — not evidence').")

    # ------------------------------------------------------------------ transport
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, transport=self.transport,
                                 headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "sweep/0.1"})

    async def _get(self, http: httpx.AsyncClient, sem: asyncio.Semaphore, url: str,
                   params: dict | None = None) -> dict:
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            async with sem:
                self.requests_made += 1
                try:
                    r = await http.get(url, params=params)
                except httpx.TransportError as e:
                    err: Exception | None = e
                    r = None
                else:
                    err = None
            if r is not None and r.status_code == 200:
                return r.json()
            retryable = r is None or r.status_code == 429 or r.status_code >= 500
            if retryable and attempt < self.max_retries:
                ra = r.headers.get("retry-after") if r is not None else None
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay + random.random() * delay
                await asyncio.sleep(wait)
                delay = min(delay * 2, 16.0)
                continue
            if r is None:
                raise MassiveError(f"network error: {err}")
            raise MassiveError(f"HTTP {r.status_code} on {url}: {r.text[:300]}", r.status_code)
        raise MassiveError("retries exhausted")

    async def _paged(self, http, sem, url: str, params: dict) -> list[dict]:
        out: list[dict] = []
        body = await self._get(http, sem, url, params)
        out.extend(body.get("results") or [])
        while body.get("next_url"):
            body = await self._get(http, sem, body["next_url"])   # cursor carries the params
            out.extend(body.get("results") or [])
        return out

    # ------------------------------------------------------------------ async API
    async def aminute_bars(self, start: date, end: date, ticker: str = UNDERLYING) -> pd.DataFrame:
        months = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
        chunks = [(max(m.date(), start), min((m + pd.offsets.MonthEnd(0)).date(), end)) for m in months]
        sem = asyncio.Semaphore(self.concurrency)
        async with self._client() as http:
            frames = await asyncio.gather(*[self._bars_chunk(http, sem, ticker, a, b) for a, b in chunks])
        frames = [f for f in frames if len(f)]
        if not frames:
            return _empty_bars()
        df = pd.concat(frames)
        return df[~df.index.duplicated()].sort_index()

    async def _bars_chunk(self, http, sem, ticker: str, a: date, b: date) -> pd.DataFrame:
        key = {"ep": "aggs1m", "t": ticker, "a": a, "b": b}
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        rows = await self._paged(http, sem, f"/v2/aggs/ticker/{ticker}/range/1/minute/{a}/{b}",
                                 {"adjusted": "true", "sort": "asc", "limit": 50000})
        df = _bars_frame(rows)
        if b < _today_et():
            self.cache.put(key, df)
        return df

    async def adividends(self, ticker: str = UNDERLYING) -> pd.DataFrame:
        sem = asyncio.Semaphore(self.concurrency)
        async with self._client() as http:
            rows = await self._paged(http, sem, "/stocks/v1/dividends",
                                     {"ticker": ticker, "limit": 5000, "sort": "ex_dividend_date.asc"})
        return _dividends_frame(rows)

    async def acontracts(self, as_of: date, exp_from: date, exp_to: date, ticker: str = UNDERLYING) -> pd.DataFrame:
        key = {"ep": "contracts", "t": ticker, "as_of": as_of, "a": exp_from, "b": exp_to}
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        sem = asyncio.Semaphore(self.concurrency)
        params = {"underlying_ticker": ticker, "as_of": str(as_of), "expiration_date.gte": str(exp_from),
                  "expiration_date.lte": str(exp_to), "limit": 1000, "order": "asc", "sort": "expiration_date"}
        async with self._client() as http:
            rows = await self._paged(http, sem, "/v3/reference/options/contracts", params)
        df = _contracts_frame(rows)
        if as_of < _today_et():
            self.cache.put(key, df)
        return df

    async def aquotes(self, tickers: Sequence[str], at: pd.Timestamp) -> pd.DataFrame:
        """Last NBBO at or before ``at`` for every ticker, fetched concurrently."""
        at = _utc(at)
        if not len(tickers):
            return empty_quotes(tickers)
        sem = asyncio.Semaphore(self.concurrency)
        async with self._client() as http:
            rows = await asyncio.gather(*[self._quote_one(http, sem, t, at) for t in tickers])
        out = empty_quotes(tickers)
        for t, row in zip(tickers, rows):
            if row is not None:
                out.loc[t, ["bid", "ask"]] = row[0], row[1]
                out.loc[t, "ts"] = row[2]
        return out

    async def _quote_one(self, http, sem, ticker: str, at: pd.Timestamp):
        historical = at < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
        key = {"ep": "quote_asof", "t": ticker, "at": at.value}
        if historical:
            hit = self.cache.get(key)
            if hit is not None:
                return None if hit.empty else (float(hit["bid"].iloc[0]), float(hit["ask"].iloc[0]), hit["ts"].iloc[0])
        body = await self._get(http, sem, f"/v3/quotes/{ticker}", {
            "timestamp.lte": str(at.value), "timestamp.gte": str((at - pd.Timedelta(days=4)).value),
            "order": "desc", "sort": "timestamp", "limit": 1})
        res = body.get("results") or []
        row = None
        if res:
            q = res[0]
            row = (float(q.get("bid_price", np.nan)), float(q.get("ask_price", np.nan)),
                   pd.Timestamp(int(q["sip_timestamp"]), tz="UTC"))
        if historical:
            self.cache.put(key, pd.DataFrame([row], columns=["bid", "ask", "ts"]) if row else
                           pd.DataFrame({"bid": [], "ask": [], "ts": []}))
        return row

    async def achain_snapshot(self, exp_from: date, exp_to: date, ticker: str = UNDERLYING) -> pd.DataFrame:
        """Current chain quotes for the paper loop. Only last_quote is read; Massive's greeks/IV are ignored."""
        sem = asyncio.Semaphore(self.concurrency)
        params = {"expiration_date.gte": str(exp_from), "expiration_date.lte": str(exp_to), "limit": 250}
        async with self._client() as http:
            rows = await self._paged(http, sem, f"/v3/snapshot/options/{ticker}", params)
        recs = []
        for r in rows:
            d, q = r.get("details") or {}, r.get("last_quote") or {}
            ts = q.get("last_updated")
            recs.append({"ticker": d.get("ticker"), "type": d.get("contract_type"), "strike": d.get("strike_price"),
                         "expiry": d.get("expiration_date"), "bid": q.get("bid"), "ask": q.get("ask"),
                         "ts": pd.Timestamp(int(ts), tz="UTC") if ts else pd.NaT})
        df = pd.DataFrame(recs, columns=["ticker", "type", "strike", "expiry", "bid", "ask", "ts"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df.astype({"strike": float, "bid": float, "ask": float})

    # ------------------------------------------------------------------ MarketData (sync)
    def minute_bars(self, start: date, end: date) -> pd.DataFrame:
        return asyncio.run(self.aminute_bars(start, end))

    def dividends(self) -> pd.DataFrame:
        return asyncio.run(self.adividends())

    def contracts(self, as_of: date, exp_from: date, exp_to: date) -> pd.DataFrame:
        return asyncio.run(self.acontracts(as_of, exp_from, exp_to))

    def quotes(self, tickers: Sequence[str], at: pd.Timestamp) -> pd.DataFrame:
        return asyncio.run(self.aquotes(tickers, at))


# ---------------------------------------------------------------------- parsing
def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=float) for k in ("open", "high", "low", "close", "volume")},
                        index=pd.DatetimeIndex([], tz="UTC"))


def _bars_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty_bars()
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"].astype(np.int64), unit="ms", utc=True)
    out = pd.DataFrame({"open": df["o"], "high": df["h"], "low": df["l"], "close": df["c"], "volume": df.get("v", 0.0)})
    out.index = pd.DatetimeIndex(idx)
    return out.astype(float)


def _dividends_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = {"ex_dividend_date": "ex_date", "declaration_date": "declaration_date", "pay_date": "pay_date"}
    out = pd.DataFrame({new: pd.to_datetime(df[old], errors="coerce") if old in df else pd.NaT
                        for old, new in cols.items()})
    out["cash_amount"] = df["cash_amount"].astype(float) if "cash_amount" in df else pd.Series(dtype=float)
    return out.dropna(subset=["ex_date"]).sort_values("ex_date").reset_index(drop=True)


def _contracts_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"ticker": pd.Series(dtype=object), "type": pd.Series(dtype=object),
                             "strike": pd.Series(dtype=float), "expiry": pd.Series(dtype="datetime64[ns]")})
    df = pd.DataFrame(rows)
    df = df[df["contract_type"].isin(["call", "put"])]
    if "shares_per_contract" in df:
        df = df[df["shares_per_contract"].fillna(100) == 100]        # skip adjusted / non-standard deliverables
    return pd.DataFrame({"ticker": df["ticker"].values, "type": df["contract_type"].values,
                         "strike": df["strike_price"].astype(float).values,
                         "expiry": pd.to_datetime(df["expiration_date"]).values}).reset_index(drop=True)


def last_trading_days(n: int, before: date | None = None) -> list[date]:
    """The n most recent weekdays strictly before ``before`` (holidays are skipped later by empty data)."""
    d = before or _today_et()
    days = pd.bdate_range(end=pd.Timestamp(d) - timedelta(days=1), periods=n)
    return [x.date() for x in days]
