"""Candidate legs for a decided trade: code picks strikes and expiry, never Jev.

Only called when the decision layer wants to open a position, so option quotes are
requested for a handful of contracts instead of the whole chain every hour.

  expiry   the standard monthly with dte_min..dte_max calendar days left (else none)
  long     the call/put whose |delta| is closest to long_delta (0.40)
  spread   short leg closest to short_delta (0.30); long wing spread_width ($5)
           further OTM. Bullish -> put credit spread; bearish -> call credit spread.

Strikes are pre-selected with a proxy vol (the latest known ATM IV), then quoted, then
re-ranked with each contract's OWN implied vol from its quote. Contracts without a
valid two-sided quote are skipped. All numeric steps are whole-array operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import DEFAULT, Config
from ..data.market import MarketData, monthly_mask
from . import bsm
from .chain import dividend_yield, mid_iv, session_of, years_to_expiry


@dataclass
class Leg:
    ticker: str
    type: str            # call | put
    strike: float
    expiry: pd.Timestamp
    qty: int             # +1 long, -1 short (per spread unit)
    bid: float
    ask: float
    quote_ts: pd.Timestamp
    iv: float
    delta: float
    gamma: float
    theta: float         # per day
    vega: float          # per vol point

    def to_json(self) -> dict:
        d = asdict(self)
        d["expiry"] = str(pd.Timestamp(self.expiry).date())
        d["quote_ts"] = str(self.quote_ts)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Structure:
    kind: str            # long_call | long_put | put_credit_spread | call_credit_spread
    legs: list[Leg]

    @property
    def expiry(self) -> pd.Timestamp:
        return self.legs[0].expiry


def pick_expiry(ch: pd.DataFrame, at: pd.Timestamp, cfg: Config = DEFAULT) -> pd.Timestamp | None:
    if ch.empty:
        return None
    days = (pd.DatetimeIndex(ch["expiry"]) - session_of(at)).days.to_numpy()
    ok = monthly_mask(ch["expiry"]) & (days >= cfg.options.dte_min) & (days <= cfg.options.dte_max)
    return pd.Timestamp(ch.loc[ok, "expiry"].min()) if ok.any() else None


def _nearest_by_delta(sub: pd.DataFrame, at, spot, r, q, vol, target: float, k: int) -> pd.DataFrame:
    T = years_to_expiry(sub["expiry"], at)
    d = bsm.greeks(sub["type"].to_numpy() == "call", spot, sub["strike"].to_numpy(float), T, r, q, vol)["delta"]
    o = np.argsort(np.abs(np.abs(d) - target), kind="stable")[:k]
    return sub.iloc[o]


def _priced(sub: pd.DataFrame, quotes: pd.DataFrame, at, spot, r, q) -> pd.DataFrame:
    qq = quotes.reindex(sub["ticker"])
    iv = mid_iv(qq, sub, at, spot, r, q)
    T = years_to_expiry(sub["expiry"], at)
    g = bsm.greeks(sub["type"].to_numpy() == "call", spot, sub["strike"].to_numpy(float), T, r, q,
                   np.where(np.isfinite(iv), iv, 0.2))
    return sub.assign(bid=qq["bid"].to_numpy(), ask=qq["ask"].to_numpy(), quote_ts=qq["ts"].to_numpy(), iv=iv,
                      delta=g["delta"], gamma=g["gamma"], theta=bsm.per_day(g["theta"]), vega=bsm.per_vol_point(g["vega"]))


def _leg(row, qty: int) -> Leg:
    return Leg(row["ticker"], row["type"], float(row["strike"]), pd.Timestamp(row["expiry"]), qty, float(row["bid"]),
               float(row["ask"]), pd.Timestamp(row["quote_ts"]), float(row["iv"]), float(row["delta"]),
               float(row["gamma"]), float(row["theta"]), float(row["vega"]))


def shortlist(md: MarketData, kind: str, at: pd.Timestamp, spot: float, div: pd.DataFrame, iv_proxy: float,
              cfg: Config = DEFAULT) -> Structure | None:
    """Concrete legs for ``kind`` at decision time ``at``, or None if no valid candidate exists."""
    o = cfg.options
    d = session_of(at).date()
    ch = md.contracts(d, d + timedelta(days=o.dte_min), d + timedelta(days=o.dte_max))
    exp = pick_expiry(ch, at, cfg)
    if exp is None:
        return None
    ch = ch[ch["expiry"] == exp]
    q = dividend_yield(div, at, spot)
    r = o.risk_free
    vol = iv_proxy if np.isfinite(iv_proxy) and iv_proxy > 0 else 0.2
    opt = "call" if kind in ("long_call", "call_credit_spread") else "put"
    sub = ch[ch["type"] == opt]
    if sub.empty:
        return None
    long_only = kind.startswith("long")
    target = o.long_delta if long_only else o.short_delta
    cand = _nearest_by_delta(sub, at, spot, r, q, vol, target, o.candidate_strikes)
    # second pass: if the quoted candidates imply a vol far from the proxy (stale proxy),
    # re-select the candidate strikes with the vol the market is actually quoting
    iv1 = mid_iv(md.quotes(cand["ticker"].tolist(), at).reindex(cand["ticker"]), cand, at, spot, r, q)
    if np.isfinite(iv1).any():
        vol2 = float(np.nanmedian(iv1))
        if abs(vol2 - vol) > 0.1 * vol:
            cand = _nearest_by_delta(sub, at, spot, r, q, vol2, target, o.candidate_strikes)
    if long_only:
        wings = cand.iloc[:0]
    else:
        off = o.spread_width if opt == "call" else -o.spread_width
        wing_strikes = cand["strike"].to_numpy() + off
        wings = sub[np.isin(sub["strike"].to_numpy(), wing_strikes)]
    tickers = pd.unique(np.concatenate([cand["ticker"].to_numpy(), wings["ticker"].to_numpy()])).tolist()
    quotes = md.quotes(tickers, at)
    pc = _priced(cand, quotes, at, spot, r, q)
    good = np.isfinite(pc["iv"].to_numpy())
    if long_only:
        if not good.any():
            return None
        best = pc[good].iloc[np.abs(np.abs(pc.loc[good, "delta"].to_numpy()) - target).argmin()]
        return Structure(kind, [_leg(best, +1)])
    pw = _priced(wings, quotes, at, spot, r, q).set_index("strike")
    pw = pw[np.isfinite(pw["iv"].to_numpy())]
    wing_ok = np.isin(pc["strike"].to_numpy() + off, pw.index.to_numpy())
    ok = good & wing_ok
    if not ok.any():
        return None
    best = pc[ok].iloc[np.abs(np.abs(pc.loc[ok, "delta"].to_numpy()) - target).argmin()]
    wing = pw.loc[best["strike"] + off]
    wing = wing.copy()
    wing["strike"] = best["strike"] + off
    return Structure(kind, [_leg(best, -1), _leg(wing, +1)])
