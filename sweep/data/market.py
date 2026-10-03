"""The one data interface every consumer (backtester, paper loop, probe) uses.

Both implementations (Massive and the offline fixture) return the same
schemas, so no downstream code knows which one it is talking to except via
``is_fixture`` (used only to label reports "fixture data — not evidence").

Schemas
-------
minute_bars(start, end) -> DataFrame
    index: UTC Timestamp of the minute's START (Massive convention), sorted
    columns: open, high, low, close, volume (float)
dividends() -> DataFrame
    columns: ex_date, declaration_date, pay_date (datetime64[ns], dates; NaT allowed
    for declaration/pay), cash_amount (float)
contracts(as_of, exp_from, exp_to) -> DataFrame
    columns: ticker (str, OCC "O:SPY..."), type ("call"/"put"), strike (float),
    expiry (datetime64[ns] date)
quotes(tickers, at) -> DataFrame
    index: ticker; columns: bid, ask (float, NaN if no quote), ts (UTC Timestamp of the
    quote, NaT if none). The quote is the last NBBO at or before ``at`` -- never after.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

ET = "America/New_York"
YEAR_S = 365.0 * 24 * 3600


class MarketData(Protocol):
    label: str
    is_fixture: bool

    def minute_bars(self, start: date, end: date) -> pd.DataFrame: ...

    def dividends(self) -> pd.DataFrame: ...

    def contracts(self, as_of: date, exp_from: date, exp_to: date) -> pd.DataFrame: ...

    def quotes(self, tickers: Sequence[str], at: pd.Timestamp) -> pd.DataFrame: ...


def occ_ticker(underlying: str, expiry: date, kind: str, strike: float) -> str:
    return f"O:{underlying}{expiry:%y%m%d}{'C' if kind == 'call' else 'P'}{int(round(strike * 1000)):08d}"


def parse_occ(tickers: Sequence[str]) -> pd.DataFrame:
    """Vectorized OCC parse: O:SPY250117C00450000 -> type, strike, expiry."""
    s = pd.Series(list(tickers), dtype="object")
    tail = s.str[-15:]
    exp = pd.to_datetime(tail.str[:6], format="%y%m%d")
    kind = np.where(tail.str[6] == "C", "call", "put")
    strike = tail.str[7:].astype(np.int64) / 1000.0
    return pd.DataFrame({"ticker": s.values, "type": kind, "strike": strike.values, "expiry": exp.values})


def empty_quotes(tickers: Sequence[str]) -> pd.DataFrame:
    n = len(tickers)
    return pd.DataFrame({"bid": np.full(n, np.nan), "ask": np.full(n, np.nan),
                         "ts": pd.DatetimeIndex([pd.NaT] * n, tz="UTC")},
                        index=pd.Index(list(tickers), name="ticker"))


def expiry_close_utc(expiry: pd.Series | pd.DatetimeIndex | np.ndarray) -> pd.DatetimeIndex:
    """Options stop trading at 16:00 ET on the expiration date."""
    d = pd.DatetimeIndex(pd.to_datetime(expiry)).normalize()
    return (d + pd.Timedelta(hours=16)).tz_localize(ET).tz_convert("UTC")


def third_friday(year: np.ndarray, month: np.ndarray) -> pd.DatetimeIndex:
    first = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
    offset = (4 - first.dt.weekday) % 7              # days to the first Friday
    return pd.DatetimeIndex(first + pd.to_timedelta(offset + 14, unit="D"))


def monthly_mask(expiries: pd.Series) -> np.ndarray:
    """True for standard monthly expiries: the third Friday, or the Thursday before it
    when the Friday is a market holiday and only the Thursday is listed."""
    e = pd.DatetimeIndex(pd.to_datetime(expiries)).normalize()
    tf = third_friday(e.year.values, e.month.values)
    tf_listed = np.asarray(tf.isin(e.unique()))
    return np.asarray((e == tf) | ((e == tf - pd.Timedelta(days=1)) & ~tf_listed))
