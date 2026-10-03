"""Session-anchored hourly bars and daily bars from minute bars (vectorized groupby).

Hourly bins are anchored to the 09:30 ET open: [09:30,10:30), ..., [14:30,15:30),
[15:30,16:00) -- the last one is a half bar. Each bar is indexed by its CLOSE time
(UTC), which is also the decision tick. Early-close days simply have fewer/shorter
bins. Only regular-trading-hours minutes are used.

``asof`` drops any bar whose close time is after ``asof`` so the paper loop never
sees a forming bar -- the backtester and the paper loop call the same function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import ET

RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60
BAR_COLS = ["open", "high", "low", "close", "volume"]


def _rth_frame(minutes: pd.DataFrame) -> pd.DataFrame:
    et = minutes.index.tz_convert(ET)
    mod = et.hour * 60 + et.minute
    keep = (mod >= RTH_OPEN_MIN) & (mod < RTH_CLOSE_MIN)
    m = minutes.loc[keep, BAR_COLS].copy()
    et = et[keep]
    m["session"] = et.normalize().tz_localize(None)
    m["bin"] = (np.asarray(mod[keep]) - RTH_OPEN_MIN) // 60
    return m


def _agg(g) -> pd.DataFrame:
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum"))


def hourly_bars(minutes: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Columns: open, high, low, close, volume, session (ET date), bin (0..6). Index: close time UTC."""
    if minutes.empty:
        return _empty(["session", "bin"])
    m = _rth_frame(minutes)
    h = _agg(m.groupby(["session", "bin"], sort=True)).reset_index()
    close_min = np.minimum(RTH_OPEN_MIN + (h["bin"].values + 1) * 60, RTH_CLOSE_MIN)
    close_et = (pd.DatetimeIndex(h["session"]) + pd.to_timedelta(close_min, unit="min")).tz_localize(ET)
    h.index = close_et.tz_convert("UTC")
    h.index.name = "close_time"
    h = h[BAR_COLS + ["session", "bin"]]
    if asof is not None:
        h = h[h.index <= _utc(asof)]
    return h


def daily_bars(minutes: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """One bar per RTH session, indexed by the session's actual last-minute close time (UTC)."""
    if minutes.empty:
        return _empty(["session"])
    m = _rth_frame(minutes)
    m["end"] = m.index + pd.Timedelta(minutes=1)
    d = _agg(m.groupby("session", sort=True))
    end = m.groupby("session", sort=True)["end"].max()
    # a session is only a daily bar once the whole session has closed: use the scheduled
    # 16:00 close, or the last minute seen if that is later (never earlier than 16:00 in
    # normal sessions, so a partial day is not treated as complete).
    sched = (pd.DatetimeIndex(d.index) + pd.Timedelta(hours=16)).tz_localize(ET).tz_convert("UTC")
    close = pd.DatetimeIndex(np.maximum(sched.asi8, pd.DatetimeIndex(end.values).asi8), tz="UTC")
    d = d.reset_index()
    d.index = close
    d.index.name = "close_time"
    d = d[BAR_COLS + ["session"]]
    if asof is not None:
        d = d[d.index <= _utc(asof)]
    return d


def completed_daily_count(h1_close: pd.DatetimeIndex, d1_close: pd.DatetimeIndex) -> np.ndarray:
    """For each hourly close, how many daily bars had closed at or before it."""
    return np.searchsorted(d1_close.asi8, h1_close.asi8, side="right")


def spot_at(minutes: pd.DataFrame, at: pd.DatetimeIndex) -> np.ndarray:
    """Close of the last minute that had fully closed at or before each ``at``."""
    end = minutes.index.asi8 + 60_000_000_000
    i = np.searchsorted(end, pd.DatetimeIndex(at).asi8, side="right") - 1
    c = minutes["close"].to_numpy()
    return np.where(i >= 0, c[np.clip(i, 0, None)], np.nan)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _empty(extra: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLS + extra, index=pd.DatetimeIndex([], tz="UTC", name="close_time"))
