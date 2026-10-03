"""Engine-level look-ahead test: changing the future (bars AND option quotes) must not
change any decision, fill or trade made before it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sweep.data.fixture import FixtureMarket
from sweep.engine import run
from sweep.surrogate import SweepSurrogate

pytestmark = pytest.mark.lookahead


class FutureChanged:
    """Wraps a market; everything after ``cut`` is different."""

    def __init__(self, md, cut: pd.Timestamp):
        self.md, self.cut = md, cut
        self.label, self.is_fixture = md.label, md.is_fixture

    def minute_bars(self, start, end):
        m = self.md.minute_bars(start, end).copy()
        late = m.index >= self.cut
        k = np.linspace(1.0, 0.9, late.sum())                 # a 10% slide after the cut
        for c in ("open", "high", "low", "close"):
            m.loc[late, c] = m.loc[late, c].to_numpy() * k
        return m

    def quotes(self, tickers, at):
        q = self.md.quotes(tickers, at)
        if pd.Timestamp(at) >= self.cut:
            q = q.assign(bid=q["bid"] * 1.7, ask=q["ask"] * 1.9)
        return q

    def contracts(self, *a):
        return self.md.contracts(*a)

    def dividends(self):
        return self.md.dividends()


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items()}


def test_future_changes_do_not_alter_past_decisions_or_trades():
    md = FixtureMarket(days=300, seed=5)
    days = md._m["days"]
    start, end = str(days[-45].date()), str(days[-1].date())
    base = run(md, SweepSurrogate(), start, end)
    assert len(base.trades) >= 2, "test needs some trades to be meaningful"
    # cut in the middle of the window, 30 min after a bar close (fills are 60 s after a close)
    t_mid = pd.Timestamp(base.decisions[len(base.decisions) // 2]["t"])
    cut = t_mid + pd.Timedelta(minutes=30)
    alt = run(FutureChanged(md, cut), SweepSurrogate(), start, end)
    before = [d for d in base.decisions if pd.Timestamp(d["t"]) <= t_mid]
    alt_before = [d for d in alt.decisions if pd.Timestamp(d["t"]) <= t_mid]
    assert len(before) == len(alt_before) > 0
    for a, b in zip(before, alt_before):
        assert _strip(a) == _strip(b), f"decision at {a['t']} changed when only the future changed"
    closed = [t.to_json() for t in base.trades if pd.Timestamp(t.exit_t) < cut]
    alt_closed = [t.to_json() for t in alt.trades if pd.Timestamp(t.exit_t) < cut]
    assert closed == alt_closed
    assert any(pd.Timestamp(d["t"]) > cut and d != e for d, e in zip(base.decisions, alt.decisions)), \
        "sanity: the perturbation must actually change something after the cut"
