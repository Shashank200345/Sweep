"""Walk-forward, strictly out-of-sample evaluation.

The history is cut into consecutive, NON-overlapping test folds (``test_months``).
Each fold is preceded by a ``train_months`` window. All Sweep parameters are frozen
(approved config), so nothing is fitted on the train window today -- it only
guarantees warm-up history. Any future fitted component must use ONLY the train
window of each fold; folds never see their own test data before trading it.

Features are computed once over the whole history with the causal detectors (the
look-ahead suite proves a value at bar t never depends on later bars), and each
fold's trader starts flat with fresh equity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import random_traders
from .config import DEFAULT, Config
from .data.market import MarketData
from .engine import Result, load_features, run
from .metrics import summarize
from .report import DISCLAIMER, header, metrics_table, result_metrics
from .typesafe import SystemOne


@dataclass
class Fold:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(start, end, cfg: Config = DEFAULT) -> list[Fold]:
    wf = cfg.walkforward
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    folds = []
    ts = start + pd.DateOffset(months=wf.train_months)
    while ts < end:
        te = min(ts + pd.DateOffset(months=wf.test_months) - pd.Timedelta(days=1), end)
        folds.append(Fold(ts - pd.DateOffset(months=wf.train_months), ts, te))
        ts = te + pd.Timedelta(days=1)
    return folds


def walk_forward(md: MarketData, jev_factory, start, end, cfg: Config = DEFAULT, random_n: int | None = None,
                 progress: bool = False) -> tuple[list[tuple[Fold, Result, np.ndarray]], dict]:
    folds = make_folds(start, end, cfg)
    feats, div = load_features(md, pd.Timestamp(start), pd.Timestamp(end), cfg)
    out = []
    for f in folds:
        if progress:
            print(f"· fold {f.test_start.date()} → {f.test_end.date()}")
        res = run(md, jev_factory(), f.test_start, f.test_end, cfg, features=feats, div=div)
        sess = feats.h1["session"]
        sel = (sess >= f.test_start) & (sess <= f.test_end)
        bench = random_traders(md, res.trades, feats.h1.index[sel.to_numpy()], feats.spots[sel.to_numpy()],
                               feats.ivv["atm_iv"].to_numpy()[sel.to_numpy()], div, cfg, random_n)
        out.append((f, res, bench))
    pnl = np.concatenate([[t.pnl for t in r.trades] for _, r, _ in out]) if out else np.zeros(0)
    risk = np.concatenate([[t.max_loss for t in r.trades] for _, r, _ in out]) if out else np.zeros(0)
    # chain the fold P/L into one out-of-sample equity curve for the overall drawdown
    curve, base = [], cfg.sizing.starting_equity
    for _, r, _ in out:
        if len(r.equity):
            curve.append(r.equity.to_numpy() - r.starting_equity + base)
            base = curve[-1][-1]
    overall = summarize(pnl, risk, np.concatenate(curve) if curve else [base], cfg.walkforward)
    return out, overall


def write_walkforward(folds, overall: dict, out: Path, label: str, model: str) -> str:
    lines = ["# Sweep walk-forward (out-of-sample folds)", ""] + header(label, model)
    lines += ["## all out-of-sample folds combined", ""] + metrics_table(overall) + [""]
    lines += ["## per fold", ""]
    for f, r, bench in folds:
        m = result_metrics(r)
        lines += [f"### test {f.test_start.date()} → {f.test_end.date()} (train window from {f.train_start.date()})", ""]
        lines += metrics_table(m, bench) + [f"| Jev calls / skipped by the materiality filter | {r.jev_calls} / {r.jev_skipped} |", ""]
    lines += ["Parameters are frozen (approved before the first backtest); nothing was tuned on these folds.", "",
              DISCLAIMER]
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "walkforward.md").write_text(text, encoding="utf-8")
    return text
