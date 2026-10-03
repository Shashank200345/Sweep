"""Generic look-ahead-bias harness.

``assert_causal(fn, bars)`` checks two things for many random cut points t:

1. truncation: fn(bars[:t+1]) equals fn(bars)[:t+1] -- a detector run on only the
   data available at bar t gives exactly the values the full-history run gave for
   bars 0..t (this is also what makes the paper loop match the backtester);
2. perturbation: changing every bar after t leaves fn's rows 0..t unchanged.

Never loosen these checks to let a detector pass. If a detector fails, it uses
information from after the bar it reports on, and the detector must be fixed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ATOL = 1e-9


def perturb_after(bars: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    """Replace every bar after position t with a different, internally consistent path."""
    rng = np.random.default_rng(seed)
    b = bars.copy()
    n = len(b) - t - 1
    if n <= 0:
        return b
    base = float(b["close"].iloc[t])
    steps = rng.normal(0, 0.004, n) * base
    close = base + np.cumsum(steps) + rng.choice([-1, 1]) * 0.01 * base
    open_ = np.concatenate([[base], close[:-1]])
    wick = np.abs(rng.normal(0, 0.002, (2, n))) * base
    b.iloc[t + 1:, b.columns.get_loc("open")] = open_
    b.iloc[t + 1:, b.columns.get_loc("close")] = close
    b.iloc[t + 1:, b.columns.get_loc("high")] = np.maximum(open_, close) + wick[0]
    b.iloc[t + 1:, b.columns.get_loc("low")] = np.minimum(open_, close) - wick[1]
    return b


def _eq(a: pd.DataFrame, b: pd.DataFrame, what: str, t: int) -> None:
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=ATOL, rtol=0, check_dtype=False)
    except AssertionError as e:
        raise AssertionError(f"look-ahead detected ({what}) at cut t={t}: {e}") from None


def assert_causal(fn: Callable[[pd.DataFrame], pd.DataFrame], bars: pd.DataFrame,
                  n_cuts: int = 12, warmup: int = 30, seed: int = 0) -> None:
    full = fn(bars)
    assert len(full) == len(bars), "detector must return one row per input bar"
    rng = np.random.default_rng(seed)
    cuts = np.unique(np.concatenate([rng.integers(warmup, len(bars) - 1, n_cuts), [len(bars) - 2]]))
    for t in cuts:
        t = int(t)
        _eq(fn(bars.iloc[: t + 1]), full.iloc[: t + 1], "truncation", t)
        _eq(fn(perturb_after(bars, t, seed + t)).iloc[: t + 1], full.iloc[: t + 1], "perturbation", t)
