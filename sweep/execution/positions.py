"""Paper positions and trades. Per-share prices; x100 multiplier for dollars.

Sign convention: a position's ``entry`` is the net per-share premium PAID (positive
for long options, negative -- a credit -- for credit spreads). Closing receives the
net per-share value ``exit`` (positive for a long option, negative for buying a
spread back). P/L per contract = (exit - entry) x 100, minus commissions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import MULTIPLIER
from ..options.shortlist import Leg


@dataclass
class Position:
    kind: str
    legs: list[Leg]
    contracts: int
    entry_t: pd.Timestamp
    entry_bar: int
    entry: float                 # net per-share paid (credit < 0)
    entry_fills: list[float]
    commission_in: float
    max_loss: float              # $ for the whole position, incl. round-trip commissions
    size: float
    reason_in: str
    direction: int
    mark: float = float("nan")   # latest net per-share value at mid

    @property
    def tickers(self) -> list[str]:
        return [l.ticker for l in self.legs]

    @property
    def qty(self) -> np.ndarray:
        return np.array([l.qty for l in self.legs])

    @property
    def expiry(self) -> pd.Timestamp:
        return pd.Timestamp(self.legs[0].expiry)

    @property
    def is_spread(self) -> bool:
        return len(self.legs) == 2

    @property
    def credit(self) -> float:
        return -self.entry

    def pnl_at(self, value: float, commission_out: float = 0.0) -> float:
        return (value - self.entry) * MULTIPLIER * self.contracts - self.commission_in - commission_out


@dataclass
class Trade:
    kind: str
    direction: int
    contracts: int
    size: float
    tickers: list[str]
    strikes: list[float]
    expiry: str
    entry_t: str
    exit_t: str
    entry: float
    exit: float
    pnl: float
    max_loss: float
    commissions: float
    held_bars: int
    reason_in: str
    reason_out: str
    last_resort: bool = False
    legs_in: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        for k in ("entry", "exit", "pnl", "max_loss", "commissions"):
            d[k] = round(float(d[k]), 4)
        return d


def net_value(qty: np.ndarray, prices: np.ndarray) -> float:
    """Net per-share value of the legs at ``prices`` (long legs +, short legs -)."""
    return float(np.sum(np.asarray(qty) * np.asarray(prices)))
