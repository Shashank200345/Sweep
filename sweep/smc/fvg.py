"""Fair value gaps: the three-candle imbalance.

Bullish FVG at bar i: low[i] > high[i-2]; the gap is [high[i-2], low[i]].
Bearish FVG at bar i: high[i] < low[i-2]; the gap is [high[i], low[i-2]].
It is knowable at the close of bar i (its third candle), so ``form = i``. Gaps
smaller than ``min_atr`` x ATR(i) are ignored. A gap ends when price trades
through its far edge ("filled"); before that, a touch is "tested".
"""
from __future__ import annotations

import numpy as np

from .zones import Zones


def detect_fvgs(high: np.ndarray, low: np.ndarray, atr: np.ndarray, min_atr: float) -> tuple[Zones, Zones]:
    n = len(high)
    if n < 3:
        return Zones.empty(+1), Zones.empty(-1)
    i = np.arange(2, n)
    h2, l2 = high[:-2], low[:-2]
    hi, li = high[2:], low[2:]
    a = atr[2:]
    with np.errstate(invalid="ignore"):
        bull = (li > h2) & (li - h2 >= min_atr * a)
        bear = (hi < l2) & (l2 - hi >= min_atr * a)
    b = Zones(i[bull].astype(np.int64), li[bull], h2[bull], +1)
    s = Zones(i[bear].astype(np.int64), l2[bear], hi[bear], -1)
    return b, s
