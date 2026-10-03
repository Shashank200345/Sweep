"""As-of option chain context: dividend yield, next ex-dividend, and 30-day ATM IV.

Everything is computed by us from quotes (never provider IV/Greeks) and uses only
information knowable at the decision time ``at``:
  * the dividend yield uses dividends whose ex-date is on or before ``at``'s session;
  * the next ex-dividend is only known once it has been declared (or, when the
    declaration date is missing, ``dividend_known_lead_days`` before the ex-date);
  * quotes are the last NBBO at or before ``at``.

The daily ATM IV series is gathered session by session (one small batch of quote
requests per session -- that loop is I/O), then solved in ONE vectorized call.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import DEFAULT, Config
from ..data.market import ET, YEAR_S, MarketData, expiry_close_utc
from . import bsm


def session_of(at: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(at).tz_convert(ET).normalize().tz_localize(None)


def dividend_yield(div: pd.DataFrame, at: pd.Timestamp, spot: float) -> float:
    """Trailing-12-month cash dividends already gone ex by ``at``, over spot."""
    d = session_of(at)
    m = (div["ex_date"] <= d) & (div["ex_date"] > d - pd.Timedelta(days=365))
    s = float(div.loc[m, "cash_amount"].sum())
    return s / spot if spot > 0 and s > 0 else 0.0


def next_dividend(div: pd.DataFrame, at: pd.Timestamp, cfg: Config = DEFAULT) -> tuple[pd.Timestamp | None, float]:
    """The next ex-dividend date after ``at``'s session that was already known at ``at``."""
    d = session_of(at)
    known = div["declaration_date"].fillna(div["ex_date"] - pd.Timedelta(days=cfg.options.dividend_known_lead_days))
    m = (div["ex_date"] > d) & (known <= d)
    if not m.any():
        return None, 0.0
    row = div[m].sort_values("ex_date").iloc[0]
    return row["ex_date"], float(row["cash_amount"])


def years_to_expiry(expiry, at: pd.Timestamp) -> np.ndarray:
    return (expiry_close_utc(expiry).asi8 - pd.Timestamp(at).value) / 1e9 / YEAR_S


def mid_iv(q: pd.DataFrame, info: pd.DataFrame, at, spot: float, r: float, qy: float) -> np.ndarray:
    """IV from the quote midpoint; NaN for one-sided, crossed or missing quotes (vectorized)."""
    bid, ask = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
    good = (bid > 0) & (ask > bid)
    mid = np.where(good, (bid + ask) / 2, np.nan)
    T = years_to_expiry(info["expiry"], at)
    return bsm.implied_vol(mid, info["type"].to_numpy() == "call", spot, info["strike"].to_numpy(float), T, r, qy)


def _atm_pick(ch: pd.DataFrame, at: pd.Timestamp, spot: float, target_days: int) -> pd.DataFrame:
    """ATM call+put for the two expiries bracketing ``target_days`` (or the nearest one)."""
    ch = ch[years_to_expiry(ch["expiry"], at) > 2 / 365]
    if ch.empty:
        return ch
    exps = np.sort(ch["expiry"].unique())
    days = (pd.DatetimeIndex(exps) - session_of(at)).days.to_numpy()
    below, above = exps[days <= target_days], exps[days > target_days]
    pick = ([below[-1]] if len(below) else []) + ([above[0]] if len(above) else [])
    rows = []
    for e in pick:                                    # at most two expiries
        sub = ch[ch["expiry"] == e]
        k = sub["strike"].to_numpy()[np.abs(sub["strike"].to_numpy() - spot).argmin()]
        rows.append(sub[sub["strike"] == k])
    return pd.concat(rows) if rows else ch.iloc[:0]


def atm_iv_from(picks: pd.DataFrame, iv: np.ndarray, meta: pd.DataFrame, target_days: int) -> np.ndarray:
    """30-day ATM IV per session by linear interpolation in total variance (vectorized groupby)."""
    df = picks.assign(iv=iv)
    df = df[np.isfinite(df["iv"])]
    if df.empty:
        return np.full(len(meta), np.nan)
    g = df.groupby(["session", "expiry"], as_index=False).agg(iv=("iv", "mean"), T=("T", "first"))
    tgt = target_days / 365.0
    g["w"] = g["iv"] ** 2 * g["T"]
    lo = g[g["T"] <= tgt].sort_values("T").groupby("session").last()
    hi = g[g["T"] > tgt].sort_values("T").groupby("session").first()
    both = lo.join(hi, lsuffix="_lo", rsuffix="_hi", how="outer")
    wlo, whi = both["w_lo"], both["w_hi"]
    tlo, thi = both["T_lo"], both["T_hi"]
    interp = wlo + (whi - wlo) * (tgt - tlo) / (thi - tlo)
    var = np.where(both["w_lo"].notna() & both["w_hi"].notna(), interp / tgt,
                   np.where(both["iv_lo"].notna(), both["iv_lo"] ** 2, both["iv_hi"] ** 2))
    out = pd.Series(np.sqrt(np.maximum(var, 0)), index=both.index)
    return out.reindex(meta["session"]).to_numpy()


def daily_atm_iv(md: MarketData, closes: pd.DatetimeIndex, spots: np.ndarray, div: pd.DataFrame,
                 cfg: Config = DEFAULT) -> pd.DataFrame:
    """30-day ATM IV at each session close. Index: the close time the value is known at."""
    o = cfg.options
    frames = []
    meta = []
    for at, s in zip(closes, spots):                  # one quote batch per session (I/O loop)
        if not np.isfinite(s):
            continue
        d = session_of(at).date()
        ch = md.contracts(d, d + timedelta(days=1), d + timedelta(days=o.atm_iv_target_days + 40))
        picks = _atm_pick(ch, at, s, o.atm_iv_target_days)
        if picks.empty:
            continue
        q = md.quotes(picks["ticker"].tolist(), at)
        qy = dividend_yield(div, at, s)
        frames.append(picks.assign(bid=q["bid"].to_numpy(), ask=q["ask"].to_numpy(), at=at, spot=s, qy=qy,
                                   session=session_of(at), T=years_to_expiry(picks["expiry"], at)))
        meta.append({"close_time": at, "session": session_of(at), "spot": s})
    if not frames:
        return pd.DataFrame({"atm_iv": pd.Series(dtype=float)}, index=pd.DatetimeIndex([], tz="UTC", name="close_time"))
    allq = pd.concat(frames, ignore_index=True)
    bid, ask = allq["bid"].to_numpy(float), allq["ask"].to_numpy(float)
    mid = np.where((bid > 0) & (ask > bid), (bid + ask) / 2, np.nan)
    iv = bsm.implied_vol(mid, allq["type"].to_numpy() == "call", allq["spot"].to_numpy(float),
                         allq["strike"].to_numpy(float), allq["T"].to_numpy(float), o.risk_free, allq["qy"].to_numpy(float))
    m = pd.DataFrame(meta)
    out = pd.DataFrame({"atm_iv": atm_iv_from(allq, iv, m, o.atm_iv_target_days)},
                       index=pd.DatetimeIndex(m["close_time"], name="close_time"))
    return out
