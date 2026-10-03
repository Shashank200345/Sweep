"""Liquidity pools and sweeps.

Pools: every confirmed swing high (buy-side liquidity rests above it) and swing low
(sell-side rests below). When two consecutive swing highs are within
``equal_tol_atr`` x ATR of each other the pool is "equal highs" at the higher of the
two (likewise "equal lows" at the lower). A pool exists from its confirmation bar.

After the first bar w that trades beyond the pool level:
  * sweep  -- some close within bars w .. w+k-1 is back inside the level; recorded
              at that reclaim bar (a sweep of buy-side liquidity is bearish, -1; of
              sell-side, bullish, +1);
  * taken  -- no close back inside within k bars; recorded at bar w+k-1, in the
              direction of the breakout (a continuation).
Both are recorded at the bar where they become knowable, never at w.
"""
from __future__ import annotations

import numpy as np

from .core import NEVER, ffill_index, first_true_after, take, window

SWEEP, TAKEN = 1, 2


def detect_liquidity(high, low, close, atr, piv: dict, tol_atr: float, reclaim_bars: int,
                     max_age: int) -> dict[str, np.ndarray]:
    n = len(close)
    t = np.arange(n)
    ev_dir = np.zeros(n, dtype=int)
    ev_kind = np.zeros(n, dtype=int)
    ev_equal = np.zeros(n, dtype=bool)
    ev_level = np.full(n, np.nan)
    pools = []
    for tag, side in (("sh", +1), ("sl", -1)):           # side: +1 buy-side above, -1 sell-side below
        c = np.flatnonzero(piv[f"new_{tag}"]).astype(np.int64)
        if c.size == 0:
            continue
        lvl, prev = piv[f"{tag}_level"][c], piv[f"{tag}_prev_level"][c]
        with np.errstate(invalid="ignore"):
            equal = np.abs(lvl - prev) <= tol_atr * atr[c]
        level = np.where(equal, np.maximum(lvl, prev) if side > 0 else np.minimum(lvl, prev), lvl)
        j, exists = window(c, n, max_age)
        beyond = (high[j] > level[:, None]) if side > 0 else (low[j] < level[:, None])
        w = first_true_after(c, beyond & exists, max_age)
        has = w < NEVER
        wc = np.where(has, w, 0)
        k = np.arange(reclaim_bars)[None, :]
        r_idx = wc[:, None] + k
        r_ok = has[:, None] & (r_idx < n)
        rc = np.clip(r_idx, 0, n - 1)
        back = ((close[rc] < level[:, None]) if side > 0 else (close[rc] > level[:, None])) & r_ok
        swept = back.any(axis=1)
        r_bar = wc + back.argmax(axis=1)
        taken_bar = wc + reclaim_bars - 1
        taken = has & ~swept & (taken_bar < n)
        pools.append({"form": c, "level": level, "side": side, "equal": equal, "breach": w,
                      "event_bar": np.where(swept, r_bar, np.where(taken, taken_bar, NEVER)),
                      "event_kind": np.where(swept, SWEEP, np.where(taken, TAKEN, 0))})
        for sel, kind, d in ((swept, SWEEP, -side), (taken, TAKEN, side)):
            bars = np.where(kind == SWEEP, r_bar, taken_bar)[sel]
            # a sweep beats a continuation on the same bar; later pools overwrite earlier ones
            keep = ev_kind[bars] != SWEEP if kind == TAKEN else np.ones(len(bars), dtype=bool)
            ev_dir[bars[keep]] = d
            ev_kind[bars[keep]] = kind
            ev_equal[bars[keep]] = equal[sel][keep]
            ev_level[bars[keep]] = level[sel][keep]
    last = ffill_index(ev_kind > 0)
    return {
        "liq_event_dir": ev_dir, "liq_event_kind": ev_kind,
        "liq_last_dir": take(ev_dir, last, 0).astype(int),
        "liq_last_kind": take(ev_kind, last, 0).astype(int),
        "liq_last_equal": take(ev_equal, last, False).astype(bool),
        "liq_last_level": take(ev_level, last),
        "liq_last_age": np.where(last >= 0, t - last, -1),
        "sweep_dir": np.where(ev_kind == SWEEP, ev_dir, 0),
        "_pools": pools,
    }
