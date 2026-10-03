"""Matched random-entry benchmark (adapted from the reference project's random traders).

Each random trader makes the SAME number of trades as the strategy, with the same
structure types, contract counts and holding periods, but at random entry bars and
in a random direction. Legs are chosen by the same shortlist code and filled with
the same conservative fill model. Exits happen after the matched holding period (no
exit rules), so the benchmark answers: "is the strategy's EV better than trading
the same product mix at random times?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, MULTIPLIER, Config
from .data.market import MarketData
from .execution.fills import commission, fill
from .execution.positions import Trade, net_value
from .options.shortlist import shortlist

FLIP = {"long_call": "long_put", "long_put": "long_call",
        "put_credit_spread": "call_credit_spread", "call_credit_spread": "put_credit_spread"}


def random_traders(md: MarketData, trades: list[Trade], bars: pd.DatetimeIndex, spots: np.ndarray,
                   atm_iv: np.ndarray, div: pd.DataFrame, cfg: Config = DEFAULT, n: int | None = None) -> np.ndarray:
    """Mean P/L per trade for each random trader (NaN when one of its trades could not be placed)."""
    n = cfg.walkforward.n_random if n is None else n
    if not trades or len(bars) < 2:
        return np.zeros(0)
    rng = np.random.default_rng(cfg.walkforward.seed)
    lat = pd.Timedelta(seconds=cfg.fills.latency_s)
    out = np.full(n, np.nan)
    for k in range(n):                                        # I/O-bound: quotes per simulated trade
        pnl = []
        for tr in trades:
            hold = max(1, tr.held_bars)
            for _attempt in range(25):                         # redraw if no contract/quote at that bar
                # (a 30-45 DTE monthly exists on only about half of sessions, so redraws are common)
                i = int(rng.integers(0, max(1, len(bars) - hold)))
                kind = tr.kind if rng.random() < 0.5 else FLIP[tr.kind]
                st = shortlist(md, kind, bars[i], spots[i], div, atm_iv[i], cfg)
                if st is None:
                    continue
                tk = [l.ticker for l in st.legs]
                qty = np.array([l.qty for l in st.legs])
                fo = fill(tk, qty > 0, md.quotes(tk, bars[i] + lat), bars[i] + lat, cfg.fills)
                j = min(i + hold, len(bars) - 1)
                fc = fill(tk, qty < 0, md.quotes(tk, bars[j] + lat), bars[j] + lat, cfg.fills)
                if not (fo.ok and fc.ok):
                    continue
                c = tr.contracts
                pnl.append((net_value(qty, fc.prices) - net_value(qty, fo.prices)) * MULTIPLIER * c
                           - 2 * commission(c, len(tk), cfg.fills))
                break
        if len(pnl) == len(trades):
            out[k] = float(np.mean(pnl))
    return out
