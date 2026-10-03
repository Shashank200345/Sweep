"""Turns the numeric structure frame into the named-bucket state Jev judges.

Jev never sees a raw number or candle (AGENTS.md non-negotiable 5): every quantity
was computed by deterministic code and is written here as a word from a fixed
vocabulary. The bucketing is vectorized over the whole history (``bucket_frame``);
``state_at`` just assembles one row into the nested JSON a single Jev call carries.

Code, not Jev, also picks WHICH order block / gap is relevant: the one on the side
of the hourly trend (or the nearer one when there is no trend).

Only fields referenced by the question set are included (tests enforce this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .smc.liquidity import SWEEP, TAKEN
from .smc.structure import BOS, CHOCH

NA = "n/a"

TREND = {1: "bullish (the last break of structure was upward)",
         -1: "bearish (the last break of structure was downward)",
         0: "no break of structure yet"}
DAILY = {1: "bullish", -1: "bearish", 0: "no clear daily trend"}
WHEN = ["within the last 3 hours", "earlier today", "1 to 3 days ago", "more than 3 days ago"]
ZONE = ["no valid dealing range", "below the current dealing range", "deep discount", "discount", "equilibrium",
        "premium", "deep premium", "above the current dealing range"]
LOCATION = ["price is inside it", "within 1 ATR of price", "1 to 3 ATR away from price", "more than 3 ATR away from price"]
IV = ["insufficient history", "low (bottom 30% of the past year)", "normal", "elevated (top 30% of the past year)"]
OB_SIDE = {1: "bullish order block (demand below price)", -1: "bearish order block (supply above price)",
           0: "no active order block"}
FVG_SIDE = {1: "bullish fair value gap (below price)", -1: "bearish fair value gap (above price)",
            0: "no active fair value gap"}
RANGE_HALF = ["outside the current dealing range", "in the discount half of the dealing range",
              "in the premium half of the dealing range"]
POSITION_KIND = {"long_call": "long calls (a bullish position)", "long_put": "long puts (a bearish position)",
                 "put_credit_spread": "a put credit spread (a bullish position)",
                 "call_credit_spread": "a call credit spread (a bearish position)"}
PNL = ["at a loss", "about flat", "in profit"]
HELD = ["opened today", "opened 1 to 3 days ago", "opened more than 3 days ago"]


def _when(age: np.ndarray, c: Config) -> np.ndarray:
    s = c.state
    return np.select([age <= s.recent_bars, age <= s.day_bars, age <= s.days3_bars], WHEN[:3], WHEN[3])


def _location(dist: np.ndarray, c: Config) -> np.ndarray:
    s = c.state
    return np.select([dist <= 0, dist <= s.near_atr, dist <= s.mid_atr], LOCATION[:3], LOCATION[3])


def _zone(p: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.select([~np.isfinite(p), p < 0, p < 0.25, p < 0.45, p <= 0.55, p <= 0.75, p <= 1.0],
                         ZONE[:7], ZONE[7])


def _pick(side: np.ndarray, bull: np.ndarray, bear: np.ndarray):
    return np.where(side > 0, bull, bear)


def bucket_frame(f: pd.DataFrame, ivv: pd.DataFrame | None = None, c: Config = DEFAULT) -> pd.DataFrame:
    """Every state bucket for every bar, as strings (vectorized over the whole history)."""
    n = len(f)
    trend = f["trend"].to_numpy()
    out = pd.DataFrame(index=f.index)
    # ---- structure
    out["structure.daily_bias"] = pd.Series(f["daily_bias"]).map(DAILY).to_numpy()
    out["structure.hourly_trend"] = pd.Series(trend).map(TREND).to_numpy()
    bd, bk, ba = f["last_break_dir"].to_numpy(), f["last_break_kind"].to_numpy(), f["last_break_age"].to_numpy()
    ev = np.char.add(np.char.add(np.where(bd > 0, "bullish ", "bearish "), np.where(bk == CHOCH, "CHoCH ", "BOS ")),
                     _when(ba, c).astype(str))
    out["structure.last_event"] = np.where(ba >= 0, ev, "none yet")
    # ---- liquidity
    ld, lk, la = f["liq_last_dir"].to_numpy(), f["liq_last_kind"].to_numpy(), f["liq_last_age"].to_numpy()
    eq = f["liq_last_equal"].to_numpy(bool)
    hi_pool = np.where(eq, "equal highs", "a swing high")
    lo_pool = np.where(eq, "equal lows", "a swing low")
    texts = np.select(
        [(lk == SWEEP) & (ld > 0), (lk == SWEEP) & (ld < 0), (lk == TAKEN) & (ld > 0), (lk == TAKEN) & (ld < 0)],
        [np.char.add(np.char.add("sell-side liquidity below ", lo_pool), " was swept and price closed back above it"),
         np.char.add(np.char.add("buy-side liquidity above ", hi_pool), " was swept and price closed back below it"),
         np.char.add(np.char.add("buy-side liquidity above ", hi_pool), " was taken and price is holding above it"),
         np.char.add(np.char.add("sell-side liquidity below ", lo_pool), " was taken and price is holding below it")],
        "")
    out["liquidity.last_event"] = np.where(la >= 0, np.char.add(np.char.add(texts.astype(str), ", "), _when(la, c).astype(str)),
                                           "no liquidity event yet")
    # ---- relevant side for zones: the trend's side, else the nearer active zone
    def side_for(prefix: str) -> np.ndarray:
        db, ds = f[f"{prefix}_bull_dist_atr"].to_numpy(), f[f"{prefix}_bear_dist_atr"].to_numpy()
        hb, hs = np.isfinite(db), np.isfinite(ds)
        near = np.where(hb & (~hs | (db <= np.where(hs, ds, np.inf))), 1, np.where(hs, -1, 0))
        s = np.where(trend > 0, np.where(hb, 1, 0), np.where(trend < 0, np.where(hs, -1, 0), near))
        return s

    for prefix, key, sides in (("ob", "order_block", OB_SIDE), ("fvg", "fair_value_gap", FVG_SIDE)):
        s = side_for(prefix)
        g = s != 0
        col = lambda k: _pick(s, f[f"{prefix}_bull_{k}"].to_numpy(), f[f"{prefix}_bear_{k}"].to_numpy())
        out[f"{key}.side"] = pd.Series(s).map(sides).to_numpy()
        out[f"{key}.location"] = np.where(g, _location(np.nan_to_num(col("dist_atr").astype(float), nan=np.inf), c), NA)
        tested = col("tested").astype(bool)
        if prefix == "ob":
            out[f"{key}.status"] = np.where(g, np.where(tested, "tested and held so far", "untested"), NA)
            out[f"{key}.origin"] = np.where(g, np.where(col("kind") == CHOCH, "formed on a change of character",
                                                        "formed on a break of structure"), NA)
            out[f"{key}.after_sweep"] = np.where(g, np.where(col("after_sweep").astype(bool),
                                                             "a liquidity sweep came just before it", "no sweep before it"), NA)
            out[f"{key}.gap_overlap"] = np.where(g, np.where(col("fvg_overlap").astype(bool),
                                                             "overlaps a fair value gap on the same side", "no gap overlap"), NA)
            p = col("pd_pos").astype(float)
            with np.errstate(invalid="ignore"):
                half = np.select([~np.isfinite(p) | (p < 0) | (p > 1), p < 0.5], [RANGE_HALF[0], RANGE_HALF[1]], RANGE_HALF[2])
            out[f"{key}.range_half"] = np.where(g, half, NA)
        else:
            out[f"{key}.status"] = np.where(g, np.where(tested, "tested but not filled", "untested"), NA)
    # ---- zone and volatility
    out["zone"] = _zone(f["pd_pos"].to_numpy(float))
    ivp = ivv["iv_pct"].reindex(f.index).to_numpy(float) if ivv is not None else np.full(n, np.nan)
    s = c.state
    with np.errstate(invalid="ignore"):
        out["volatility.iv_rank"] = np.select([~np.isfinite(ivp), ivp < s.iv_low_pct, ivp <= s.iv_high_pct],
                                              IV[:3], IV[3])
    return out


def position_text(kind: str | None, held_bars: int = 0, pnl_frac: float = 0.0, c: Config = DEFAULT) -> str:
    """Bucketed description of the open paper position ('flat' when none)."""
    if not kind:
        return "flat"
    s = c.state
    held = HELD[0] if held_bars < s.day_bars else HELD[1] if held_bars <= s.days3_bars else HELD[2]
    pnl = PNL[1] if abs(pnl_frac) < s.pnl_flat_frac else PNL[2] if pnl_frac > 0 else PNL[0]
    return f"{POSITION_KIND[kind]}, {held}, currently {pnl}"


def state_at(buckets: pd.DataFrame | pd.Series, position: str = "flat") -> dict:
    """Nested JSON for one Jev call from one row of ``bucket_frame``."""
    row = buckets if isinstance(buckets, pd.Series) else buckets.iloc[-1]
    st: dict = {}
    for k, v in row.items():
        parts = k.split(".")
        d = st
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = str(v)
    st["position"] = position
    return st


def leaf_paths(state: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in state.items():
        p = f"{prefix}{k}"
        out += leaf_paths(v, p + ".") if isinstance(v, dict) else [p]
    return out
