"""One look-ahead test per structure detector (AGENTS.md: required before a detector is done)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sweep.data.bars import hourly_bars
from sweep.data.fixture import FixtureMarket
from sweep.smc.engine import detect

from .harness import assert_causal

pytestmark = pytest.mark.lookahead

GROUPS = {
    "pivots": ["new_sh", "new_sl", "sh_level", "sh_idx", "sh_prev_level", "sl_level", "sl_idx", "sl_prev_level"],
    "bos_choch": ["break_dir", "break_kind", "trend", "last_break_kind", "last_break_age", "last_break_level"],
    "daily_bias": ["daily_bias", "daily_last_kind", "daily_bars"],
    "fvg": [f"fvg_{s}_{k}" for s in ("bull", "bear") for k in ("id", "dist_atr", "tested", "age", "top", "bottom")],
    "order_blocks": [f"ob_{s}_{k}" for s in ("bull", "bear")
                     for k in ("id", "dist_atr", "tested", "age", "top", "bottom", "kind", "after_sweep", "pd_pos", "fvg_overlap")],
    "liquidity": ["liq_event_dir", "liq_event_kind", "liq_last_dir", "liq_last_kind", "liq_last_equal",
                  "liq_last_level", "liq_last_age", "sweep_dir"],
    "premium_discount": ["pd_pos", "atr"],
}


@pytest.fixture(scope="module", params=[3, 11])
def h1(request):
    m = FixtureMarket(days=160, seed=request.param)
    return hourly_bars(m.minute_bars(date(2022, 1, 1), date(2023, 1, 1)))


@pytest.mark.parametrize("group", list(GROUPS))
def test_detector_is_causal(h1, group):
    cols = GROUPS[group]
    assert_causal(lambda b: detect(b).frame[cols], h1, n_cuts=10, seed=len(group))


def test_whole_frame_is_causal(h1):
    assert_causal(lambda b: detect(b).frame, h1, n_cuts=6, seed=99)


def test_hourly_bars_from_partial_minutes_are_causal():
    m = FixtureMarket(days=12, seed=5).minute_bars(date(2022, 1, 1), date(2022, 2, 1))
    full = hourly_bars(m)
    rng = np.random.default_rng(0)
    for cut in rng.integers(100, len(m) - 1, 15):
        t = m.index[cut] + pd.Timedelta(minutes=1)         # "now": minute ``cut`` just closed
        part = hourly_bars(m.iloc[: cut + 1], asof=t)
        pd.testing.assert_frame_equal(part, full[full.index <= t])


def test_harness_catches_a_peeking_detector(h1):
    """Sanity check on the harness itself: a centred rolling max looks ahead and must fail."""
    peek = lambda b: b[["high"]].rolling(5, center=True, min_periods=1).max()
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_causal(peek, h1, n_cuts=3)
