"""Swing pivots, recorded at their CONFIRMATION bar.

A swing high at bar i needs ``left`` bars before it with strictly lower highs and
``right`` bars after it with highs no higher. It is therefore only knowable at the
close of bar i + right -- that is when it enters the as-of arrays. (Ties: strict on
the left, non-strict on the right, so a flat top registers once, at its first bar.)
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import ffill_index, prev_event_index, take


def _pivot_mask(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """True at pivot bar i (NOT the confirmation bar)."""
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if n < left + right + 1:
        return out
    w = sliding_window_view(x, left + right + 1)          # w[k] = x[k .. k+left+right], centre k+left
    c = w[:, left]
    lw, rw = w[:, :left], w[:, left + 1:]
    if high:
        ok = (c > lw.max(axis=1) if left else True) & (c >= rw.max(axis=1) if right else True)
    else:
        ok = (c < lw.min(axis=1) if left else True) & (c <= rw.min(axis=1) if right else True)
    out[left:n - right] = ok & np.isfinite(c)
    return out


def swing_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> dict[str, np.ndarray]:
    """As-of swing arrays.

    new_sh / new_sl   True at the bar a swing high / low is confirmed
    sh_level, sh_idx  the most recent confirmed swing high and the bar it formed on
    sh_prev_level     the swing high confirmed before that
    (and the same for lows)
    """
    n = len(high)
    out: dict[str, np.ndarray] = {}
    for tag, x, is_high in (("sh", high, True), ("sl", low, False)):
        piv = _pivot_mask(x, left, right, is_high)
        new = np.zeros(n, dtype=bool)
        new[right:] = piv[:n - right] if right else piv
        pivot_bar = np.arange(n) - right                      # at a confirmation bar c, the pivot is c - right
        last = ffill_index(new)
        prev = prev_event_index(new)
        lvl_at_confirm = np.where(new, x[np.clip(pivot_bar, 0, None)], np.nan)
        out[f"new_{tag}"] = new
        out[f"{tag}_level"] = take(lvl_at_confirm, last)
        out[f"{tag}_idx"] = np.where(last >= 0, pivot_bar[np.clip(last, 0, None)], -1)
        out[f"{tag}_prev_level"] = take(lvl_at_confirm, prev)
    return out
