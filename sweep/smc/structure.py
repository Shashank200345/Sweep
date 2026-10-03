"""Break of structure / change of character, trend state, premium/discount position.

A break happens at the first bar whose CLOSE is beyond the most recent confirmed
swing (a wick alone is a liquidity sweep, not a break -- see liquidity.py). Each
swing can be broken once. Trend = direction of the latest break. A break in the
direction of the prevailing trend is a BOS; against it, a CHoCH (the first break
ever is labelled BOS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import ffill_index, take

BOS, CHOCH = 1, 2


def _first_per_id(cond: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """cond, but only the first True bar for each distinct id."""
    pos = np.flatnonzero(cond)
    first = ~pd.Series(ids[pos]).duplicated().to_numpy()
    out = np.zeros(len(cond), dtype=bool)
    out[pos[first]] = True
    return out


def breaks(close: np.ndarray, piv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(close)
    with np.errstate(invalid="ignore"):
        up = _first_per_id(close > piv["sh_level"], piv["sh_idx"])
        dn = _first_per_id(close < piv["sl_level"], piv["sl_idx"])
    both = up & dn                                      # ambiguous inverted range: ignore both
    up, dn = up & ~both, dn & ~both
    dirn = up.astype(int) - dn.astype(int)
    ev = dirn != 0
    last = ffill_index(ev)
    trend = take(dirn, last, 0).astype(int)
    prev_trend = np.concatenate([[0], trend[:-1]])
    kind = np.where(ev, np.where((prev_trend == 0) | (prev_trend == dirn), BOS, CHOCH), 0)
    level = np.where(up, piv["sh_level"], np.where(dn, piv["sl_level"], np.nan))
    return {
        "break_dir": dirn, "break_kind": kind, "trend": trend,
        "last_break_dir": trend, "last_break_kind": take(kind, last, 0).astype(int),
        "last_break_age": np.where(last >= 0, np.arange(n) - last, -1),
        "last_break_level": take(level, last),
        "last_break_idx": last,
    }


def premium_discount(close: np.ndarray, sh_level: np.ndarray, sl_level: np.ndarray) -> np.ndarray:
    """Position of close in the current swing range: 0 = swing low, 1 = swing high.
    Below 0 / above 1 means price is outside the range. NaN without a valid range."""
    rng = sh_level - sl_level
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rng > 0, (close - sl_level) / rng, np.nan)
