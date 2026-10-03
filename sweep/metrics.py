"""Headline metrics: expected value (with a bootstrap confidence interval) and maximum drawdown.

Hit rate / win rate is deliberately NOT computed or reported anywhere (AGENTS.md
non-negotiable 7). With tens to low hundreds of trades a year, a bare EV number is
meaningless -- every EV is reported with its CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT, WalkForwardConfig


@dataclass
class EV:
    mean: float
    lo: float
    hi: float
    n: int

    def fmt(self, unit: str = "$", d: int = 2) -> str:
        if self.n == 0:
            return "n/a (no trades)"
        if unit == "$":
            return f"${self.mean:+,.{d}f}  [95% CI ${self.lo:+,.{d}f} to ${self.hi:+,.{d}f}]  (n={self.n})"
        return f"{self.mean:+.{d}f}{unit}  [95% CI {self.lo:+.{d}f} to {self.hi:+.{d}f}]  (n={self.n})"


def bootstrap_mean(x: np.ndarray, n_boot: int, ci: float, seed: int) -> EV:
    x = np.asarray(x, float)
    if len(x) == 0:
        return EV(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    a = (1 - ci) / 2
    return EV(float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), len(x))


def max_drawdown(equity: pd.Series | np.ndarray) -> tuple[float, float]:
    """(max drawdown in $, as a fraction of the running peak)."""
    e = np.asarray(equity, float)
    if len(e) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(e)
    dd = peak - e
    i = int(dd.argmax())
    return float(dd[i]), float(dd[i] / peak[i]) if peak[i] > 0 else 0.0


def summarize(pnl: np.ndarray, risk: np.ndarray, equity, wf: WalkForwardConfig = DEFAULT.walkforward) -> dict:
    pnl, risk = np.asarray(pnl, float), np.asarray(risk, float)
    dd, ddp = max_drawdown(equity)
    per_risk = np.where(risk > 0, pnl / np.where(risk > 0, risk, 1), np.nan)
    return {
        "trades": int(len(pnl)),
        "net_pnl": float(pnl.sum()),
        "ev_per_trade": bootstrap_mean(pnl, wf.bootstrap, wf.ci, wf.seed),
        "ev_per_dollar_risked": bootstrap_mean(per_risk[np.isfinite(per_risk)], wf.bootstrap, wf.ci, wf.seed + 1),
        "max_drawdown": dd,
        "max_drawdown_pct": ddp,
    }


def forbidden_metric_names() -> tuple[str, ...]:
    return ("hit rate", "win rate", "hit_rate", "win_rate", "accuracy")
