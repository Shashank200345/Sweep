from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sweep.config import DEFAULT, WalkForwardConfig
from sweep.data.fixture import FixtureMarket
from sweep.metrics import bootstrap_mean, forbidden_metric_names, max_drawdown, summarize
from sweep.report import write_backtest
from sweep.engine import run
from sweep.surrogate import SweepSurrogate
from sweep.walkforward import make_folds, walk_forward, write_walkforward


def test_bootstrap_known_answers():
    c = bootstrap_mean(np.full(20, 5.0), 1000, 0.95, 0)
    assert c.mean == c.lo == c.hi == 5.0 and c.n == 20
    x = np.random.default_rng(1).normal(10, 2, 400)
    ev = bootstrap_mean(x, 4000, 0.95, 0)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert ev.lo < ev.mean < ev.hi
    assert ev.hi - ev.lo == pytest.approx(2 * 1.96 * se, rel=0.1)            # matches the normal-theory width
    assert bootstrap_mean(np.array([]), 100, 0.95, 0).n == 0


def test_max_drawdown_known_answer():
    dd, pct = max_drawdown(np.array([100, 120, 90, 130, 104, 140]))
    assert dd == 30 and pct == pytest.approx(0.25)
    assert max_drawdown(np.array([1, 2, 3])) == (0.0, 0.0)


def test_summary_has_ev_ci_and_drawdown_and_no_hit_rate():
    s = summarize(np.array([100, -50, 30]), np.array([500, 500, 500]), [1000, 1100, 1050, 1080])
    assert {"ev_per_trade", "ev_per_dollar_risked", "max_drawdown", "max_drawdown_pct"} <= set(s)
    assert s["ev_per_dollar_risked"].mean == pytest.approx((0.2 - 0.1 + 0.06) / 3)
    assert not any(k in forbidden_metric_names() for k in s)


def test_folds_are_non_overlapping_and_strictly_forward():
    folds = make_folds("2022-03-07", "2025-12-31")
    assert folds[0].test_start == pd.Timestamp("2023-03-07")
    for f in folds:
        assert f.train_start < f.test_start <= f.test_end
        assert f.test_start - f.train_start >= pd.Timedelta(days=360)
    for a, b in zip(folds, folds[1:]):
        assert b.test_start == a.test_end + pd.Timedelta(days=1)            # contiguous, no overlap
    assert folds[-1].test_end == pd.Timestamp("2025-12-31")


@pytest.fixture(scope="module")
def small():
    cfg = replace(DEFAULT, walkforward=WalkForwardConfig(train_months=11, test_months=1, bootstrap=500, n_random=2))
    return FixtureMarket(days=300, seed=5), cfg


def _no_hit_rate(text: str) -> None:
    low = text.lower()
    assert not any(k in low for k in forbidden_metric_names()), "hit/win rate must never be reported"


def test_backtest_report_headlines_ev_ci_and_drawdown(small, tmp_path):
    md, cfg = small
    days = md._m["days"]
    res = run(md, SweepSurrogate(), str(days[-40].date()), str(days[-1].date()), cfg)
    text = write_backtest(res, tmp_path, cfg)
    assert "expected value per trade" in text and "95% CI" in text and "maximum drawdown" in text
    assert "FIXTURE DATA" in text and "not Jev" in text
    _no_hit_rate(text)
    assert (tmp_path / "decisions.jsonl").exists() and (tmp_path / "trades.jsonl").exists()


def test_walkforward_report(small, tmp_path):
    md, cfg = small
    days = md._m["days"]
    folds, overall = walk_forward(md, SweepSurrogate, str(days[0].date()), str(days[-1].date()), cfg)
    assert len(folds) >= 2
    starts = [f.test_start for f, _, _ in folds]
    assert starts == sorted(starts)
    text = write_walkforward(folds, overall, tmp_path, md.label, "surrogate-local")
    assert text.count("expected value per trade") == len(folds) + 1 and "maximum drawdown" in text
    _no_hit_rate(text)
