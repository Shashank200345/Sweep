"""The single fill model, shared by the backtester and the paper loop (APPROVED rules).

Each leg fills halfway between mid and the far side of its quote -- buys at
mid + 0.5 x (ask - mid), sells at mid - 0.5 x (mid - bid) -- so every fill is
worse than mid. $0.65 per contract per leg per side. A multi-leg order fills
all-or-nothing: no fill if ANY leg's quote is missing, one-sided, crossed, older
than 5 minutes at fill time, or wider than 10% of its mid.

``last_resort`` exists only for forced risk exits that found no valid quote all
the way to their deadline: it fills at the FULL far side of the latest two-sided
quote regardless of age/width, and is flagged in the trade log.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT, FillConfig


@dataclass
class Fill:
    ok: bool
    prices: np.ndarray          # per leg, per share
    reason: str
    at: pd.Timestamp
    last_resort: bool = False


def quote_ok(bid: np.ndarray, ask: np.ndarray, ts, at: pd.Timestamp, f: FillConfig = DEFAULT.fills) -> np.ndarray:
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    ts = pd.DatetimeIndex(ts)
    age = (pd.Timestamp(at).value - ts.asi8) / 1e9
    mid = (bid + ask) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid) & ~ts.isna()
                & (age >= 0) & (age <= f.max_quote_age_s) & ((ask - bid) / mid <= f.max_spread_frac))


def leg_prices(bid, ask, buy: np.ndarray, f: FillConfig = DEFAULT.fills, frac: float | None = None) -> np.ndarray:
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    mid = (bid + ask) / 2
    k = f.slippage_frac if frac is None else frac
    return np.where(buy, mid + k * (ask - mid), mid - k * (mid - bid))


def fill(tickers: list[str], buy: np.ndarray, quotes: pd.DataFrame, at: pd.Timestamp,
         f: FillConfig = DEFAULT.fills) -> Fill:
    q = quotes.reindex(tickers)
    ok = quote_ok(q["bid"], q["ask"], q["ts"], at, f)
    if not ok.all():
        bad = ",".join(t for t, o in zip(tickers, ok) if not o)
        return Fill(False, np.full(len(tickers), np.nan), f"no valid quote: {bad}", at)
    return Fill(True, leg_prices(q["bid"], q["ask"], np.asarray(buy), f), "filled", at)


def last_resort(tickers: list[str], buy: np.ndarray, quotes: pd.DataFrame, at: pd.Timestamp) -> Fill:
    q = quotes.reindex(tickers)
    bid, ask = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
    two = np.isfinite(bid) & np.isfinite(ask) & (ask >= bid) & (ask > 0)
    if not two.all():
        return Fill(False, np.full(len(tickers), np.nan), "no two-sided quote at all", at)
    return Fill(True, np.where(np.asarray(buy), ask, np.maximum(bid, 0.0)), "last-resort far-side fill", at, True)


def commission(contracts: int, legs: int, f: FillConfig = DEFAULT.fills) -> float:
    return f.commission * contracts * legs
