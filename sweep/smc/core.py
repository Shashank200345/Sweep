"""Causal array primitives shared by every detector (whole-array numpy, no per-bar loops)."""
from __future__ import annotations

import numpy as np

NEVER = np.iinfo(np.int64).max // 4      # "event never happened (in the data we have)"


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Simple-average true range over the last n bars; NaN until n bars exist."""
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
    c = np.concatenate([[0.0], np.cumsum(tr)])
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ffill_index(mask: np.ndarray) -> np.ndarray:
    """At each bar, the index of the most recent True at or before it (-1 if none)."""
    return np.maximum.accumulate(np.where(mask, np.arange(len(mask)), -1))


def take(values: np.ndarray, idx: np.ndarray, fill=np.nan) -> np.ndarray:
    """values[idx] where idx >= 0, else ``fill``."""
    v = np.asarray(values)
    if len(v) == 0:
        return np.full(len(idx), fill)
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], fill)


def prev_event_index(mask: np.ndarray) -> np.ndarray:
    """At each bar, the index of the SECOND most recent True at or before it (-1 if none)."""
    ev = np.flatnonzero(mask)
    k = np.cumsum(mask) - 2                 # ordinal of the previous event
    return np.where(k >= 0, ev[np.clip(k, 0, None)] if len(ev) else -1, -1)


def first_true_after(form: np.ndarray, cond: np.ndarray, max_age: int) -> np.ndarray:
    """For each zone formed at bar ``form[z]``, the first bar j in (form, form+max_age]
    with cond[z, j - form - 1] True, else NEVER. ``cond`` is the (zones x max_age)
    window matrix built by ``window``; bars beyond the data are always False, so a
    truncated history can only report "not yet", never a different answer."""
    if cond.size == 0:
        return np.full(len(form), NEVER, dtype=np.int64)
    hit = cond.any(axis=1)
    return np.where(hit, form + 1 + cond.argmax(axis=1), NEVER).astype(np.int64)


def window(form: np.ndarray, n: int, max_age: int) -> tuple[np.ndarray, np.ndarray]:
    """Bar indices form+1..form+max_age per zone (clipped) and a mask of which exist."""
    j = form[:, None] + np.arange(1, max_age + 1)[None, :]
    return np.clip(j, 0, max(n - 1, 0)), j < n
