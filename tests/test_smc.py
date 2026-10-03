"""Hand-built sequences for each detector's geometry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sweep.smc.core import NEVER, atr
from sweep.smc.fvg import detect_fvgs
from sweep.smc.liquidity import SWEEP, TAKEN, detect_liquidity
from sweep.smc.order_blocks import detect_order_blocks
from sweep.smc.pivots import swing_pivots
from sweep.smc.structure import BOS, CHOCH, breaks, premium_discount
from sweep.smc.zones import Zones, lifecycle, nearest_active

nan = np.nan


def test_pivots_recorded_at_confirmation_bar():
    high = np.array([10, 11, 12, 11, 10, 11, 13, 12, 11, 10], float)
    low = np.array([9, 10, 11, 10, 9, 10, 12, 11, 10, 9], float)
    p = swing_pivots(high, low, 2, 2)
    assert list(np.flatnonzero(p["new_sh"])) == [4, 8]            # pivots at 2 and 6
    assert np.isnan(p["sh_level"][3]) and p["sh_level"][4] == 12 and p["sh_level"][8] == 13
    assert p["sh_idx"][8] == 6 and p["sh_prev_level"][8] == 12
    assert list(np.flatnonzero(p["new_sl"])) == [6] and p["sl_level"][6] == 9 and p["sl_idx"][6] == 4


def test_pivot_flat_top_registers_once():
    high = np.array([1, 2, 5, 5, 2, 1, 1], float)
    p = swing_pivots(high, high - 1, 2, 2)
    assert list(np.flatnonzero(p["new_sh"])) == [4] and p["sh_idx"][4] == 2


def test_bos_then_choch():
    piv = {"sh_level": np.array([nan, nan, 10, 10, 10, 10, 10, 10]), "sh_idx": np.array([-1, -1, 0, 0, 0, 0, 0, 0]),
           "sl_level": np.array([nan, nan, nan, 8, 8, 8, 8, 8]), "sl_idx": np.array([-1, -1, -1, 1, 1, 1, 1, 1])}
    close = np.array([9, 9, 9.5, 10.5, 11, 7.5, 7, 9])
    b = breaks(close, piv)
    assert list(b["break_dir"]) == [0, 0, 0, 1, 0, -1, 0, 0]       # each swing broken once, on a close
    assert b["break_kind"][3] == BOS and b["break_kind"][5] == CHOCH
    assert list(b["trend"]) == [0, 0, 0, 1, 1, -1, -1, -1]
    assert b["last_break_age"][7] == 2 and b["last_break_level"][7] == 8


def test_wick_through_is_not_a_break():
    piv = {"sh_level": np.full(4, 10.0), "sh_idx": np.zeros(4, int), "sl_level": np.full(4, 8.0), "sl_idx": np.ones(4, int)}
    assert (breaks(np.array([9.0, 9.9, 9.5, 9.0]), piv)["break_dir"] == 0).all()


def test_fvg_three_candle_gap():
    high = np.array([10, 11, 14, 15], float)
    low = np.array([9, 10, 12, 13], float)
    bull, bear = detect_fvgs(high, low, np.ones(4), 0.1)
    assert list(bull.form) == [2, 3] and list(bull.bottom) == [10, 11] and list(bull.top) == [12, 13]
    assert len(bear) == 0
    small, _ = detect_fvgs(high, low, np.full(4, 100.0), 0.1)       # gap < 0.1 ATR is ignored
    assert len(small) == 0


def test_fvg_lifecycle_tested_then_filled():
    high = np.array([10, 11, 14, 15, 14, 13, 12], float)
    low = np.array([9, 10, 12, 13, 11.5, 11, 9.5], float)
    close = np.array([9.5, 10.5, 13.5, 14.5, 12.5, 12, 10], float)
    bull, _ = detect_fvgs(high, low, np.ones(7), 0.1)
    z = lifecycle(bull, high, low, close, 50)
    assert z.form[0] == 2 and z.tested_at[0] == 4 and z.filled_at[0] == 6      # [10,12]: touched at 4, through 10 at 6
    sel = nearest_active(z, z.filled_at, close, np.ones(7), 50)
    # gap 1 [11,13] (formed at 3) is filled at bar 5 (low 11); gap 0 [10,12] is still live at 5
    assert z.filled_at[1] == 5
    assert sel["id"][5] == 0 and sel["tested"][5] and sel["top"][5] == 12
    assert sel["id"][6] == -1                                             # both gaps filled by bar 6
    assert sel["id"][2] == 0 and not sel["tested"][2] and sel["dist_atr"][2] == 1.5   # close 13.5 vs top 12


def test_order_block_is_last_opposing_candle_before_the_low():
    #            0     1     2     3     4     5
    o = np.array([11, 10.5, 10.8, 9.2, 9.0, 10.5])
    c = np.array([10.5, 10.8, 9.0, 9.0, 10.5, 12.5])
    h = np.array([11.1, 10.9, 10.9, 9.4, 10.6, 12.6])
    l = np.array([10.2, 10.3, 8.5, 8.8, 8.9, 10.4])
    brk = np.array([0, 0, 0, 0, 0, 1])
    bull, bear = detect_order_blocks(o, h, l, c, brk, np.array([0, 0, 0, 0, 0, BOS]), np.zeros(6, int), 10)
    assert len(bull) == 1 and len(bear) == 0
    assert bull.form[0] == 5 and bull.attrs["candle"][0] == 2            # bar 3 is bearish but after the low
    assert bull.bottom[0] == 8.5 and bull.top[0] == 10.9 and bull.attrs["kind"][0] == BOS


def test_order_block_invalidated_on_close_not_wick():
    z = Zones(np.array([0]), np.array([10.0]), np.array([9.0]), +1)
    z = lifecycle(z, np.array([12, 11, 10, 10, 9.5]), np.array([11, 9.8, 8.5, 8.8, 8.7]), np.array([11.5, 10.5, 9.2, 9.1, 8.9]), 50)
    assert z.tested_at[0] == 1 and z.filled_at[0] == 2 and z.broken_at[0] == 4


def _liq(low, close, high=None, k=3):
    low, close = np.asarray(low, float), np.asarray(close, float)
    high = close + 0.3 if high is None else np.asarray(high, float)
    piv = swing_pivots(high, low, 1, 1)
    return detect_liquidity(high, low, close, np.ones(len(low)), piv, 0.1, k, 100)


def test_sell_side_sweep_recorded_at_reclaim_bar():
    r = _liq([10, 9, 10, 10.5, 8.8, 10], [10.5, 9.5, 10.4, 11, 9.2, 10.6])
    assert r["liq_event_kind"][4] == SWEEP and r["liq_event_dir"][4] == +1   # wick below 9, close back above
    assert r["liq_last_dir"][5] == 1 and r["liq_last_age"][5] == 1


def test_sell_side_taken_is_continuation_after_k_bars():
    r = _liq([10, 9, 10, 10.5, 8.8, 8.5, 8.4], [10.5, 9.5, 10.4, 11, 8.9, 8.7, 8.6])
    assert r["liq_event_kind"][4] == 0 and r["liq_event_kind"][5] == 0          # not knowable yet
    assert r["liq_event_kind"][6] == TAKEN and r["liq_event_dir"][6] == -1


def test_equal_lows_pool_uses_the_lower_level():
    low = [10, 9, 10, 9.05, 10, 8.98, 10]
    close = [10.5, 9.5, 10.4, 9.5, 10.4, 9.3, 10.4]
    r = _liq(low, close)
    assert r["liq_event_kind"][5] == SWEEP and r["liq_last_equal"][5]
    assert r["liq_last_level"][5] == 9.0


def test_premium_discount_position():
    pd_ = premium_discount(np.array([90, 100, 110, 120]), np.full(4, 110.0), np.full(4, 90.0))
    assert list(pd_) == [0.0, 0.5, 1.0, 1.5]
    assert np.isnan(premium_discount(np.array([1.0]), np.array([nan]), np.array([0.0]))[0])


def test_atr_simple_average():
    h, l, c = np.array([2, 3, 4.0]), np.array([1, 1, 2.0]), np.array([1.5, 2, 3.0])
    a = atr(h, l, c, 2)
    assert np.isnan(a[0]) and a[1] == (1 + 2) / 2 and a[2] == (2 + 2) / 2
