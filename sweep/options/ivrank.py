"""IV rank and IV percentile over the trailing window (vectorized, causal).

rank       = (iv - min) / (max - min) over the last ``window`` daily values, incl. today
percentile = share of those values <= today's
Both are NaN until ``min_history`` valid values exist ("insufficient history").
The hourly view only sees daily values whose close time is at or before its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def iv_rank(iv: np.ndarray, window: int, min_history: int) -> tuple[np.ndarray, np.ndarray]:
    iv = np.asarray(iv, float)
    n = len(iv)
    rank = np.full(n, np.nan)
    pct = np.full(n, np.nan)
    if n == 0:
        return rank, pct
    pad = np.concatenate([np.full(window - 1, np.nan), iv])
    w = sliding_window_view(pad, window)                       # w[t] = iv[t-window+1 .. t]
    valid = np.isfinite(w)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(np.where(valid, w, np.inf), axis=1)
        hi = np.nanmax(np.where(valid, w, -np.inf), axis=1)
        le = (np.where(valid, w, np.inf) <= iv[:, None]).sum(axis=1)
        ok = (count >= min_history) & np.isfinite(iv)
        rng = hi - lo
        rank = np.where(ok & (rng > 0), (iv - lo) / np.where(rng > 0, rng, 1), np.where(ok, 0.5, np.nan))
        pct = np.where(ok, le / np.maximum(count, 1), np.nan)
    return rank, pct


def hourly_view(daily: pd.DataFrame, h1_index: pd.DatetimeIndex, window: int, min_history: int) -> pd.DataFrame:
    """Latest known daily ATM IV / rank / percentile at each hourly close."""
    rank, pct = iv_rank(daily["atm_iv"].to_numpy(), window, min_history)
    k = np.searchsorted(daily.index.asi8, h1_index.asi8, side="right") - 1
    g = k >= 0
    kc = np.clip(k, 0, None)
    pick = lambda a: np.where(g, a[kc], np.nan) if len(a) else np.full(len(h1_index), np.nan)
    return pd.DataFrame({"atm_iv": pick(daily["atm_iv"].to_numpy()), "iv_rank": pick(rank), "iv_pct": pick(pct)},
                        index=h1_index)
