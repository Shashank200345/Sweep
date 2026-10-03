"""Parity: the paper loop and the backtester must make identical decisions and fills on identical data.

The backtester computes features once over the whole history; the paper loop rebuilds
them every tick from a trailing window of data available at that tick. Same functions
(`build_features`, `Trader.on_bar`), so the logs must match exactly -- including across
a restart of the paper loop.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from sweep.config import DEFAULT, PaperConfig
from sweep.data.fixture import FixtureMarket
from sweep.engine import run
from sweep.paper import PaperLoop, bar_closes, next_close_after
from sweep.surrogate import SweepSurrogate

CFG = replace(DEFAULT, paper=PaperConfig(history_days=420))   # a genuinely trailing window (shorter than the data)
N_SESSIONS = 12


@pytest.fixture(scope="module")
def setup():
    md = FixtureMarket(days=330, seed=5)
    days = md._m["days"]
    start, end = str(days[-N_SESSIONS].date()), str(days[-1].date())
    bt = run(md, SweepSurrogate(), start, end, CFG)
    return md, start, end, bt


def _norm(rows):
    return [json.loads(json.dumps(r, default=str)) for r in rows]


def _read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_paper_loop_matches_backtester(setup, tmp_path):
    md, start, end, bt = setup
    loop = PaperLoop(md, SweepSurrogate(), tmp_path, CFG, start=pd.Timestamp(start, tz="America/New_York"))
    loop.run(fake_clock=True)
    paper = _read(tmp_path / "decisions.jsonl")
    ref = _norm(bt.decisions)
    assert len(paper) == len(ref) == N_SESSIONS * 7
    for p, r in zip(paper, ref):
        p.pop("position_greeks", None)
        assert p == r, f"paper and backtest differ at {r['t']}"
    closed = [t.to_json() for t in bt.trades if t.reason_out != "end of backtest window"]
    assert _norm([t.to_json() for t in loop.trader.trades]) == _norm(closed)
    assert any(d.get("execution", "").startswith("opened") for d in ref), "window must contain at least one entry"


def test_paper_loop_resumes_after_restart(setup, tmp_path):
    md, start, end, bt = setup
    s = pd.Timestamp(start, tz="America/New_York")
    first = PaperLoop(md, SweepSurrogate(), tmp_path, CFG, start=s)
    first.run(max_ticks=40, fake_clock=True)
    again = PaperLoop(md, SweepSurrogate(), tmp_path, CFG, start=s)              # a fresh process
    assert again.last_tick == first.last_tick and again.trader.bar_count == 40
    assert (again.trader.position is None) == (first.trader.position is None)
    again.run(fake_clock=True)
    paper = _read(tmp_path / "decisions.jsonl")
    for p in paper:
        p.pop("position_greeks", None)
    assert paper == _norm(bt.decisions)


def test_bar_close_schedule():
    closes = bar_closes(pd.Timestamp("2024-07-01"))
    et = [c.tz_convert("America/New_York").strftime("%H:%M") for c in closes]
    assert et == ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    fri = pd.Timestamp("2024-07-05 16:00", tz="America/New_York")
    assert next_close_after(fri).tz_convert("America/New_York") == pd.Timestamp("2024-07-08 10:30", tz="America/New_York")


def test_no_broker_code_anywhere():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "sweep"
    pat = re.compile(r"place_order|submit_order|broker|alpaca|ibkr|interactive ?brokers|tradier", re.I)
    hits = [str(p) for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8")
                                                             .replace("no broker", "").replace("No broker", ""))]
    assert hits == [], f"paper trading only -- broker-like code found in {hits}"
