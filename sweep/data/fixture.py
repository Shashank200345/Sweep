"""Offline fixture market: synthetic SPY-like minutes, dividends, contracts and quotes.

FIXTURE DATA -- NOT EVIDENCE. This exists so every command, test and report runs
without a Massive key. It is NOT a validation simulator: there is no agreed
generative model of SMC structure (blueprint 4.5), prices are a regime-switching
random walk, and option quotes are priced from our own BSM with a made-up vol
surface. Nothing measured on it says anything about SPY. Every report produced
from it is labelled "fixture data — not evidence".

Implements the ``MarketData`` protocol with exactly the Massive schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .. import FIXTURE_LABEL
from ..config import UNDERLYING
from ..options import bsm
from .market import ET, YEAR_S, empty_quotes, expiry_close_utc, occ_ticker, parse_occ, third_friday

MIN_PER_DAY = 390


@dataclass
class FixtureMarket:
    start: date = date(2022, 1, 3)
    days: int = 520
    seed: int = 7
    spot0: float = 450.0
    risk_free: float = 0.04
    label: str = FIXTURE_LABEL
    is_fixture: bool = True
    _m: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        days = pd.bdate_range(self.start, periods=self.days)
        n = self.days
        # daily annualized vol: OU on log-vol; daily drift: regime segments (trends and ranges)
        shocks = rng.normal(0, 0.12, n)
        shocks[0] = 0.0
        lv = np.log(0.16) + lfilter([1.0], [1.0, -0.92], shocks)     # AR(1) log-vol, vectorized
        vol = np.clip(np.exp(lv), 0.07, 0.6)
        seg = np.cumsum(rng.integers(5, 30, n))
        seg_id = np.searchsorted(seg, np.arange(n), side="right")
        seg_drift = rng.choice([-0.3, -0.15, 0.0, 0.0, 0.15, 0.3], seg_id.max() + 1)
        drift_day = seg_drift[seg_id] * vol / np.sqrt(252)          # in daily-sigma units
        drift_day -= drift_day.mean()                               # trends and ranges, no net drift
        sig_min = vol / np.sqrt(252 * MIN_PER_DAY)
        noise = rng.standard_t(5, (n, MIN_PER_DAY)) * np.sqrt(3 / 5)
        noise -= noise.mean()                                       # keep a fixed seed from trending by luck
        r = noise * sig_min[:, None] + (drift_day / MIN_PER_DAY)[:, None]
        r[:, 0] += rng.normal(0, 0.3, n) * vol / np.sqrt(252)       # overnight gap
        logp = np.log(self.spot0) + np.cumsum(r.ravel())
        close = np.exp(logp)
        opn = close / np.exp(r.ravel())                              # = previous minute's close
        wick = np.abs(rng.normal(0, 0.35, close.size)) * np.repeat(sig_min, MIN_PER_DAY)
        high = np.maximum(opn, close) * np.exp(wick)
        low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.35, close.size)) * np.repeat(sig_min, MIN_PER_DAY))
        start_et = (pd.DatetimeIndex(days) + pd.Timedelta(hours=9, minutes=30)).tz_localize(ET).tz_convert("UTC")
        t = (start_et.asi8[:, None] + np.arange(MIN_PER_DAY)[None, :] * 60_000_000_000).ravel()
        vol_shares = rng.lognormal(10, 0.5, close.size)
        self._m = {"t": t, "open": opn, "high": high, "low": low, "close": close, "volume": vol_shares,
                   "days": pd.DatetimeIndex(days), "vol": vol}

    # ------------------------------------------------------------------ MarketData
    def minute_bars(self, start: date, end: date) -> pd.DataFrame:
        m = self._m
        idx = pd.DatetimeIndex(m["t"], tz="UTC")
        et_date = idx.tz_convert(ET).normalize().tz_localize(None)
        sel = (et_date >= pd.Timestamp(start)) & (et_date <= pd.Timestamp(end))
        return pd.DataFrame({k: m[k][sel] for k in ("open", "high", "low", "close", "volume")}, index=idx[sel])

    def dividends(self) -> pd.DataFrame:
        if "div" not in self._m:
            self._m["div"] = self._make_dividends()
        return self._m["div"].copy()

    def _make_dividends(self) -> pd.DataFrame:
        d = self._m["days"]
        years = np.arange(d[0].year, d[-1].year + 1)
        ym = np.array([(y, mo) for y in years for mo in (3, 6, 9, 12)])
        ex = third_friday(ym[:, 0], ym[:, 1])
        ex = ex[(ex >= d[0]) & (ex <= d[-1])]
        spot = self._spot_at((ex - pd.Timedelta(days=10)).tz_localize(ET).tz_convert("UTC") + pd.Timedelta(hours=16))
        return pd.DataFrame({"ex_date": ex, "declaration_date": ex - pd.Timedelta(days=10),
                             "pay_date": ex + pd.Timedelta(days=40), "cash_amount": np.round(0.0033 * spot, 4)})

    def contracts(self, as_of: date, exp_from: date, exp_to: date) -> pd.DataFrame:
        ref = self._spot_at(pd.DatetimeIndex([pd.Timestamp(as_of)]).tz_localize(ET).tz_convert("UTC"))[0]
        if not np.isfinite(ref):
            ref = self.spot0
        fridays = pd.date_range(pd.Timestamp(as_of), pd.Timestamp(exp_to), freq="W-FRI")
        fridays = fridays[fridays >= pd.Timestamp(exp_from)]
        strikes = np.arange(np.floor(ref * 0.75), np.ceil(ref * 1.25) + 1, 1.0)
        e, k, c = (a.ravel() for a in np.meshgrid(fridays, strikes, [0, 1], indexing="ij"))
        kind = np.where(c == 0, "call", "put")
        exp = pd.DatetimeIndex(e)
        tick = [occ_ticker(UNDERLYING, x, y, z) for x, y, z in zip(exp.date, kind, k)]  # string formatting only
        return pd.DataFrame({"ticker": tick, "type": kind, "strike": k.astype(float), "expiry": exp.values})

    def quotes(self, tickers: Sequence[str], at: pd.Timestamp) -> pd.DataFrame:
        at = pd.Timestamp(at).tz_convert("UTC") if pd.Timestamp(at).tzinfo else pd.Timestamp(at, tz="UTC")
        if not len(tickers):
            return empty_quotes(tickers)
        info = parse_occ(tickers)
        S = self._spot_at(pd.DatetimeIndex([at]))[0]
        if not np.isfinite(S):
            return empty_quotes(tickers)
        T = (expiry_close_utc(info["expiry"]).asi8 - at.value) / 1e9 / YEAR_S
        day_vol = self._vol_at(at)
        live = T > 0
        Ts = np.where(live, T, 1e-9)
        m = np.log(info["strike"].values / S) / np.sqrt(np.maximum(Ts, 1 / 365))
        iv = np.clip(day_vol + 0.02 - 0.10 * m + 0.15 * m * m, 0.05, 2.0)
        q = self._div_yield(at, S)
        mid = bsm.price(info["type"].values == "call", S, info["strike"].values, Ts, self.risk_free, q, iv)
        hs = np.maximum(0.01, 0.015 * mid)
        bid = np.floor((mid - hs) * 100) / 100
        ask = np.ceil((mid + hs) * 100) / 100
        bid = np.where(bid < 0.01, 0.0, bid)
        # deterministic pseudo-random quote age in [0, 20] s
        h = pd.util.hash_array(np.asarray(tickers, dtype=object)).astype(np.uint64)
        age_s = ((h ^ np.uint64(at.value // 60_000_000_000)) % np.uint64(21)).astype(float)
        ts = pd.DatetimeIndex(at.value - (age_s * 1e9).astype(np.int64), tz="UTC")
        out = pd.DataFrame({"bid": np.where(live, bid, np.nan), "ask": np.where(live, ask, np.nan),
                            "ts": ts}, index=pd.Index(list(tickers), name="ticker"))
        out.loc[~live, "ts"] = pd.NaT
        return out

    # ------------------------------------------------------------------ helpers
    def _spot_at(self, at: pd.DatetimeIndex) -> np.ndarray:
        """Close of the last minute that has fully closed at or before ``at``."""
        m = self._m
        end = m["t"] + 60_000_000_000
        i = np.searchsorted(end, pd.DatetimeIndex(at).asi8, side="right") - 1
        return np.where(i >= 0, m["close"][np.clip(i, 0, None)], np.nan)

    def _vol_at(self, at: pd.Timestamp) -> float:
        d = pd.Timestamp(at).tz_convert(ET).normalize().tz_localize(None)
        i = int(np.clip(np.searchsorted(self._m["days"], d, side="right") - 1, 0, self.days - 1))
        return float(self._m["vol"][i])

    def _div_yield(self, at: pd.Timestamp, S: float) -> float:
        dv = self.dividends()
        d = pd.Timestamp(at).tz_convert(ET).normalize().tz_localize(None)
        trailing = dv[(dv["ex_date"] <= d) & (dv["ex_date"] > d - pd.Timedelta(days=365))]["cash_amount"].sum()
        return float(trailing / S) if trailing > 0 else 0.013
