"""One entry point: hourly bars in, a causal as-of structure frame out.

``detect(h1)`` is the single implementation called by the backtester (over the whole
history) and by the paper loop (over a trailing window, reading the last row). The
daily bias is computed from daily bars aggregated from the same hourly bars, and an
hourly bar only sees daily bars that had CLOSED by its own close time.

The returned frame is numeric; ``sweep.state`` turns it into named buckets. The zone
tables (``Structure.zones``) carry full lifecycle columns for the dashboard only --
those look forward and must never be read by decision code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT, StructureConfig
from ..data.bars import completed_daily_count
from ..data.market import ET
from .core import atr as _atr
from .core import take
from .fvg import detect_fvgs
from .liquidity import detect_liquidity
from .order_blocks import detect_order_blocks
from .pivots import swing_pivots
from .structure import breaks, premium_discount
from .zones import Zones, lifecycle, nearest_active


@dataclass
class Structure:
    frame: pd.DataFrame                              # causal, one row per hourly bar
    zones: dict[str, pd.DataFrame] = field(default_factory=dict)   # dashboard only (look forward)


def daily_from_hourly(h1: pd.DataFrame) -> pd.DataFrame:
    """Daily bars from hourly bars; a session's daily bar closes at 16:00 ET (or later)."""
    g = h1.groupby("session", sort=True)
    d = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    last = g.apply(lambda x: x.index.max(), include_groups=False) if len(h1) else pd.Series(dtype="datetime64[ns, UTC]")
    sched = (pd.DatetimeIndex(d.index) + pd.Timedelta(hours=16)).tz_localize(ET).tz_convert("UTC")
    close = np.maximum(sched.asi8, pd.DatetimeIndex(last.values).asi8) if len(d) else np.zeros(0, dtype=np.int64)
    d.index = pd.DatetimeIndex(close, tz="UTC", name="close_time")
    return d


def _daily_bias(h1: pd.DataFrame, cfg: StructureConfig) -> dict[str, np.ndarray]:
    d1 = daily_from_hourly(h1)
    hi, lo, cl = (d1[c].to_numpy(float) for c in ("high", "low", "close"))
    br = breaks(cl, swing_pivots(hi, lo, cfg.d1_left, cfg.d1_right))
    k = completed_daily_count(h1.index, d1.index) - 1           # last completed daily bar
    return {"daily_bias": take(br["trend"], k, 0).astype(int),
            "daily_last_kind": take(br["last_break_kind"], k, 0).astype(int),
            "daily_bars": np.maximum(k + 1, 0)}


def _zone_frame(z: Zones, end_at: np.ndarray, index: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    def ts(a):
        a = np.asarray(a)
        ok = (a >= 0) & (a < len(index))
        out = pd.Series(pd.NaT, index=range(len(a)), dtype="datetime64[ns, UTC]")
        out[ok] = index[a[ok]]
        return out                                   # keep tz-aware (``.values`` would drop the tz)
    df = pd.DataFrame({"form": ts(z.form), "top": z.top, "bottom": z.bottom, "side": z.side,
                       "tested": ts(z.tested_at), "end": ts(end_at)})
    for k, v in z.attrs.items():
        df[k] = ts(v) if k == "candle" else v
    df["type"] = name
    return df


def detect(h1: pd.DataFrame, cfg: StructureConfig = DEFAULT.structure) -> Structure:
    o, h, l, c = (h1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    a = _atr(h, l, c, cfg.atr_window)
    piv = swing_pivots(h, l, cfg.h1_left, cfg.h1_right)
    br = breaks(c, piv)
    liq = detect_liquidity(h, l, c, a, piv, cfg.equal_tol_atr, cfg.sweep_reclaim_bars, cfg.zone_max_age)
    fb, fs = (lifecycle(z, h, l, c, cfg.zone_max_age) for z in detect_fvgs(h, l, a, cfg.fvg_min_atr))
    ob, os_ = (lifecycle(z, h, l, c, cfg.zone_max_age) for z in detect_order_blocks(
        o, h, l, c, br["break_dir"], br["break_kind"], liq["sweep_dir"], cfg.ob_lookback))
    pd_pos = premium_discount(c, piv["sh_level"], piv["sl_level"])

    cols: dict[str, np.ndarray] = {"close": c, "atr": a}
    cols.update({k: v for k, v in piv.items()})
    cols.update(br)
    cols.update(_daily_bias(h1, cfg))
    cols.update({k: v for k, v in liq.items() if not k.startswith("_")})
    cols["pd_pos"] = pd_pos
    for tag, z in (("fvg_bull", fb), ("fvg_bear", fs)):
        sel = nearest_active(z, z.filled_at, c, a, cfg.zone_max_age)
        cols.update({f"{tag}_{k}": v for k, v in sel.items()})
    for tag, z in (("ob_bull", ob), ("ob_bear", os_)):
        sel = nearest_active(z, z.broken_at, c, a, cfg.zone_max_age)
        cols.update({f"{tag}_{k}": v for k, v in sel.items()})
        zc = np.clip(sel["id"], 0, None)
        g = sel["id"] >= 0
        cols[f"{tag}_kind"] = np.where(g, z.attrs["kind"][zc] if len(z) else 0, 0)
        cols[f"{tag}_after_sweep"] = g & (z.attrs["after_sweep"][zc] if len(z) else False)
        mid = (sel["top"] + sel["bottom"]) / 2
        cols[f"{tag}_pd_pos"] = premium_discount(mid, piv["sh_level"], piv["sl_level"])
        side = "bull" if tag.endswith("bull") else "bear"
        ft, fbm = cols[f"fvg_{side}_top"], cols[f"fvg_{side}_bottom"]
        with np.errstate(invalid="ignore"):
            cols[f"{tag}_fvg_overlap"] = g & (np.minimum(sel["top"], ft) >= np.maximum(sel["bottom"], fbm))
    frame = pd.DataFrame(cols, index=h1.index)

    zones = {"fvg_bull": _zone_frame(fb, fb.filled_at, h1.index, "fvg"),
             "fvg_bear": _zone_frame(fs, fs.filled_at, h1.index, "fvg"),
             "ob_bull": _zone_frame(ob, ob.broken_at, h1.index, "ob"),
             "ob_bear": _zone_frame(os_, os_.broken_at, h1.index, "ob")}
    for p in liq["_pools"]:
        zp = Zones(p["form"], p["level"], p["level"], p["side"], attrs={"equal": p["equal"], "event_kind": p["event_kind"]})
        zp.tested_at = p["breach"]
        zones[f"pool_{'buy' if p['side'] > 0 else 'sell'}"] = _zone_frame(zp, p["event_bar"], h1.index, "pool")
    return Structure(frame, zones)
