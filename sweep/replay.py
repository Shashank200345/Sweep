"""Self-contained HTML replay (shell adapted from jev_bot/replay.py, MIT).

Same mechanism as the reference: one HTML file, no network, the run's data embedded
as JSON in place of ``/*__DATA__*/null``. Sweep's panels replace the order-book ones:
hourly candles with the SMC overlay (confirmed pivots, BOS/CHoCH, order blocks, fair
value gaps, liquidity pools), paper equity, 30-day ATM IV / IV rank, open-position
Greeks, and for every Jev call the typed answers, gates and the exact state sent.

``build_structure`` renders the same chart without decisions -- the Phase-2 visual
check of the detectors before anything is trusted to Jev.

The zone tables used here carry lifecycle columns that look forward; the dashboard
reveals each item only once the replay cursor passes the bar where it became known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import FIXTURE_LABEL
from .config import DEFAULT, MULTIPLIER
from .data.market import ET, YEAR_S, expiry_close_utc
from .options import bsm
from .questions import decision_questions
from .smc.engine import Structure
from .smc.structure import CHOCH

TEMPLATE = Path(__file__).parent / "assets" / "dashboard.html"


def _r(x, d=4):
    return [None if not np.isfinite(v) else round(float(v), d) for v in np.asarray(x, float)]


def _idx(index: pd.DatetimeIndex, ts) -> int | None:
    if ts is None or pd.isna(ts):
        return None
    i = int(index.searchsorted(pd.Timestamp(ts), side="left"))
    return i if i < len(index) else None


def _structure_payload(h1: pd.DataFrame, s: Structure) -> dict:
    f = s.frame
    n = len(h1)
    piv = []
    for tag, hi in (("new_sh", True), ("new_sl", False)):
        ci = np.flatnonzero(f[tag].to_numpy())
        lvl = f["sh_level" if hi else "sl_level"].to_numpy()[ci]
        pi = f["sh_idx" if hi else "sl_idx"].to_numpy()[ci]
        piv += [{"i": int(p), "ci": int(c), "p": round(float(v), 4), "hi": hi} for p, c, v in zip(pi, ci, lvl)]
    bi = np.flatnonzero(f["break_dir"].to_numpy() != 0)
    brk = [{"i": int(i), "dir": int(f["break_dir"].iat[i]), "kind": "CHoCH" if f["break_kind"].iat[i] == CHOCH else "BOS",
            "level": round(float(f["last_break_level"].iat[i]), 4),
            "from": int(f["sh_idx"].iat[i] if f["break_dir"].iat[i] > 0 else f["sl_idx"].iat[i])} for i in bi]
    zones = []
    for key, z in s.zones.items():
        for r in z.itertuples(index=False):
            zones.append({"type": r.type, "side": int(r.side), "f": _idx(h1.index, r.form), "e": _idx(h1.index, r.end),
                          "top": round(float(r.top), 4), "bottom": round(float(r.bottom), 4),
                          "c": _idx(h1.index, getattr(r, "candle", None)) if r.type == "ob" else None,
                          "eq": bool(getattr(r, "equal", False)) if r.type == "pool" else False})
    et = h1.index.tz_convert(ET)
    return {"bars": {"t": list(et.strftime("%Y-%m-%d %H:%M")), "o": _r(h1["open"]), "h": _r(h1["high"]),
                     "l": _r(h1["low"]), "c": _r(h1["close"])},
            "pivots": piv, "breaks": brk, "zones": [z for z in zones if z["f"] is not None], "n": n}


def _write(data: dict, out: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=str))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _position_greeks(res, h1: pd.DataFrame, a: int, b: int) -> list:
    """Per-bar net Greeks of the open position (BSM at each leg's entry IV), vectorized per trade."""
    out: list = [None] * (b - a)
    idx = h1.index[a:b]
    spot = h1["close"].to_numpy(float)[a:b]
    for t in res.trades:                              # a handful of trades; each is one vectorized call
        i0, i1 = _idx(idx, t.entry_t), _idx(idx, t.exit_t)
        i0 = 0 if i0 is None and pd.Timestamp(t.entry_t) < idx[0] else i0
        if i0 is None:
            continue
        i1 = len(idx) if i1 is None else i1
        if i1 <= i0:
            continue
        sl = slice(i0, i1)
        tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in t.legs_in:
            T = (expiry_close_utc(pd.Series([pd.Timestamp(leg["expiry"])] * (i1 - i0))).asi8 - idx[sl].asi8) / 1e9 / YEAR_S
            g = bsm.greeks(leg["type"] == "call", spot[sl], leg["strike"], np.maximum(T, 1e-6), DEFAULT.options.risk_free,
                           0.013, leg["iv"])
            m = leg["qty"] * t.contracts * MULTIPLIER
            tot = {"delta": tot["delta"] + g["delta"] * m, "gamma": tot["gamma"] + g["gamma"] * m,
                   "theta": tot["theta"] + bsm.per_day(g["theta"]) * m, "vega": tot["vega"] + bsm.per_vol_point(g["vega"]) * m}
        for k in range(i0, i1):
            out[k] = {"kind": t.kind, "n": t.contracts, **{g: round(float(v[k - i0]), 3) for g, v in tot.items()}}
    return out


def build(res, out: str | Path) -> Path:
    f = res.features
    h1 = f.h1
    sess = h1["session"]
    sel = np.flatnonzero(((sess >= pd.Timestamp(res.start)) & (sess <= pd.Timestamp(res.end))).to_numpy())
    a = max(0, (sel[0] if len(sel) else 0) - 140)
    b = (sel[-1] + 1) if len(sel) else len(h1)
    sub = h1.iloc[a:b]
    frame = f.structure.frame.iloc[a:b]
    zones = {k: z[(z["form"] >= sub.index[0]) & (z["form"] <= sub.index[-1])] for k, z in f.structure.zones.items()}
    data = _structure_payload(sub, Structure(frame, zones))
    eq = res.equity.reindex(sub.index)
    decs = []
    for d in res.decisions:
        i = _idx(sub.index, d["t"])
        if i is None:
            continue
        rec = {"i": i, "action": d.get("action"), "reason": d.get("reason", ""), "execution": d.get("execution")}
        if d.get("jev_called"):
            rec.update({"answers": d.get("answers"), "state": d.get("state"), "gates": d.get("gates"),
                        "composite": d.get("composite")})
        if d.get("jev_called") or d.get("execution") or d.get("action") != "hold":
            decs.append(rec)
    data.update({
        "label": res.label, "fixture": res.label == FIXTURE_LABEL, "model": res.model,
        "surrogate": res.model.startswith("surrogate"), "start": int(sel[0] - a) if len(sel) else 0,
        "equity": _r(eq.to_numpy(), 2), "iv": _r(f.ivv["atm_iv"].to_numpy()[a:b]), "ivr": _r(f.ivv["iv_rank"].to_numpy()[a:b]),
        "greeks": _position_greeks(res, h1, a, b), "decisions": decs,
        "qorder": list(decision_questions()),
        "trades": [{"i_in": _idx(sub.index, t.entry_t) or 0, "i_out": _idx(sub.index, t.exit_t), "kind": t.kind,
                    "n": t.contracts, "pnl": round(t.pnl, 2), "why": t.reason_out} for t in res.trades],
    })
    return _write(data, out)


def build_structure(h1: pd.DataFrame, s: Structure, out: str | Path, label: str) -> Path:
    data = _structure_payload(h1, s)
    n = len(h1)
    data.update({"label": label, "fixture": label == FIXTURE_LABEL, "model": "none (structure only, no Jev)",
                 "surrogate": False, "start": n - 1, "equity": [None] * n, "iv": [None] * n, "ivr": [None] * n,
                 "greeks": [None] * n, "decisions": [], "qorder": [], "trades": []})
    return _write(data, out)
