"""Order blocks: the last opposing candle before the impulse that broke structure.

For a bullish break at bar b (close above the last swing high): within the
``lookback`` bars up to b, the impulse starts at the lowest low; the order block is
the last bearish candle (close < open) at or before that low (or the low bar itself
if there is none). Its band is that candle's [low, high]. Bearish is the mirror.

The block only exists once the break has happened, so ``form = b`` -- never the
earlier bar of the candle itself. It ends when a later bar CLOSES through its far
edge; a touch is "tested".
"""
from __future__ import annotations

import numpy as np

from .zones import Zones


def detect_order_blocks(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        break_dir: np.ndarray, break_kind: np.ndarray, sweep_dir: np.ndarray,
                        lookback: int) -> tuple[Zones, Zones]:
    out = []
    for side in (+1, -1):
        b = np.flatnonzero(break_dir == side).astype(np.int64)
        if b.size == 0:
            out.append(Zones.empty(side))
            continue
        j = b[:, None] - np.arange(lookback, -1, -1)[None, :]       # oldest .. b
        ok = j >= 0
        jc = np.clip(j, 0, None)
        if side > 0:
            ext = np.where(ok, low[jc], np.inf)
            m = ext.argmin(axis=1)
            opposing = close[jc] < open_[jc]
        else:
            ext = np.where(ok, high[jc], -np.inf)
            m = ext.argmax(axis=1)
            opposing = close[jc] > open_[jc]
        cols = np.arange(j.shape[1])[None, :]
        cand = opposing & ok & (cols <= m[:, None])
        last = np.where(cand.any(axis=1), j.shape[1] - 1 - np.argmax(cand[:, ::-1], axis=1), m)
        ob_bar = jc[np.arange(len(b)), last]
        # did a liquidity sweep in the same direction happen in the window before the break?
        sw = np.where(ok, sweep_dir[jc], 0) == side
        out.append(Zones(b, high[ob_bar], low[ob_bar], side,
                         attrs={"candle": ob_bar, "kind": break_kind[b], "after_sweep": sw.any(axis=1)}))
    return out[0], out[1]
