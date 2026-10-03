"""Zone lifecycle (formed / tested / ended / expired) and the per-bar "nearest active zone".

A zone is a price band known from bar ``form`` onward. ``side`` +1 is a bullish zone
(demand / support, expected below price), -1 bearish (supply / resistance, above).

Lifecycle, all found with a (zones x max_age) window matrix -- no per-bar loop:
  tested_at  first later bar that trades into the band
  filled_at  first later bar that trades through the far edge (wick)
  broken_at  first later bar that CLOSES through the far edge
Each is the bar index where it happened, or NEVER. Looking these up with
``<= t`` at bar t only ever uses bars up to t, which keeps selection causal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import NEVER, first_true_after, window

SELECT_WINDOW = 64      # at each bar, only the 64 most recently formed zones per side are considered


@dataclass
class Zones:
    form: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    side: int
    attrs: dict[str, np.ndarray] = field(default_factory=dict)
    tested_at: np.ndarray | None = None
    filled_at: np.ndarray | None = None
    broken_at: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.form)

    @classmethod
    def empty(cls, side: int) -> "Zones":
        z = np.zeros(0, dtype=np.int64)
        return cls(z, np.zeros(0), np.zeros(0), side, tested_at=z, filled_at=z, broken_at=z)

    def sorted(self) -> "Zones":
        o = np.argsort(self.form, kind="stable")
        return Zones(self.form[o], self.top[o], self.bottom[o], self.side, {k: v[o] for k, v in self.attrs.items()})


def lifecycle(z: Zones, high: np.ndarray, low: np.ndarray, close: np.ndarray, max_age: int) -> Zones:
    z = z.sorted()
    n = len(close)
    if len(z) == 0:
        return Zones.empty(z.side)
    j, exists = window(z.form, n, max_age)
    h, lo, c = high[j], low[j], close[j]
    top, bot = z.top[:, None], z.bottom[:, None]
    if z.side > 0:
        tested, filled, broken = lo <= top, lo <= bot, c < bot
    else:
        tested, filled, broken = h >= bot, h >= top, c > top
    z.tested_at = first_true_after(z.form, tested & exists, max_age)
    z.filled_at = first_true_after(z.form, filled & exists, max_age)
    z.broken_at = first_true_after(z.form, broken & exists, max_age)
    return z


def nearest_active(z: Zones, end_at: np.ndarray, close: np.ndarray, atr: np.ndarray,
                   max_age: int) -> dict[str, np.ndarray]:
    """Per bar: the active zone of this side closest to price (ties: most recent).

    Active at t: form <= t < end_at and t - form <= max_age. Distance is measured from
    the close to the near edge in ATRs (0 when price is inside the band).
    """
    n = len(close)
    zid = np.full(n, -1)
    dist = np.full(n, np.nan)
    if len(z) and n:
        t = np.arange(n)
        hi = np.searchsorted(z.form, t, side="right")                # zones formed at or before t
        cand = hi[:, None] - 1 - np.arange(SELECT_WINDOW)[None, :]   # most recent first
        ok = cand >= 0
        cc = np.clip(cand, 0, None)
        active = ok & (t[:, None] < end_at[cc]) & (t[:, None] - z.form[cc] <= max_age)
        if z.side > 0:
            d = np.maximum(close[:, None] - z.top[cc], 0.0)
        else:
            d = np.maximum(z.bottom[cc] - close[:, None], 0.0)
        d = np.where(active, d, np.inf)
        k = d.argmin(axis=1)
        found = np.isfinite(d[t, k])
        zid = np.where(found, cc[t, k], -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = np.where(found, d[t, k] / atr, np.nan)
    g = zid >= 0
    zc = np.clip(zid, 0, None)
    t = np.arange(n)
    tested = np.zeros(n, dtype=bool)
    if len(z):
        tested = g & (z.tested_at[zc] <= t)
    return {
        "id": zid,
        "dist_atr": dist,
        "tested": tested,
        "age": np.where(g, t - z.form[zc] if len(z) else 0, -1),
        "top": np.where(g, z.top[zc] if len(z) else np.nan, np.nan),
        "bottom": np.where(g, z.bottom[zc] if len(z) else np.nan, np.nan),
    }


def never_like(n: int) -> np.ndarray:
    return np.full(n, NEVER, dtype=np.int64)
