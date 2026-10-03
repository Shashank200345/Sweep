"""The fixed exit rules (APPROVED; fixed before the first backtest, never tuned on results).

Long calls/puts: take profit at +100% of premium, stop at -50%, close at 21 DTE.
Credit spreads: take profit when 50% of the credit is captured, stop when the loss
reaches 2 x the credit, close at 21 DTE. exit_now >= 0.65 (Jev) is applied by the
decision layer.

Early assignment (SPY options are American): a call credit spread is force-closed
on the trading day before a KNOWN ex-dividend date when the short call's extrinsic
value is below the dividend -- the classic early-exercise condition. Together with
the 21 DTE close, no position is ever held into expiry; the engine asserts it.

Profit/stop checks use the conservative EXECUTABLE close value (same fill model),
not the mid, so a rule cannot fire on a price that could not have been traded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT, ExitConfig
from ..options.chain import session_of
from .positions import Position


def dte(pos: Position, at: pd.Timestamp) -> int:
    return int((pos.expiry - session_of(at)).days)


def price_rule(pos: Position, close_value: float, e: ExitConfig = DEFAULT.exits) -> str | None:
    """close_value: net per-share value received on closing now (conservative fills)."""
    if not np.isfinite(close_value):
        return None
    if not pos.is_spread:
        if close_value >= pos.entry * (1 + e.long_take_profit):
            return "take profit (+100% of premium)"
        if close_value <= pos.entry * (1 - e.long_stop):
            return "stop (-50% of premium)"
        return None
    credit = pos.credit
    cost = -close_value                               # paid to buy the spread back
    if cost <= credit * (1 - e.spread_take_profit):
        return "take profit (50% of credit captured)"
    if cost - credit >= e.spread_stop_mult * credit:
        return "stop (loss = 2x credit)"
    return None


def dte_rule(pos: Position, at: pd.Timestamp, e: ExitConfig = DEFAULT.exits) -> str | None:
    return f"{e.close_dte} DTE close" if dte(pos, at) <= e.close_dte else None


def ex_dividend_day_before(at: pd.Timestamp, ex_date: pd.Timestamp | None) -> bool:
    """True when ``at`` is in the last trading session before the ex-dividend date."""
    if ex_date is None:
        return False
    prev = pd.Timestamp(ex_date) - pd.offsets.BDay(1)
    return session_of(at) >= prev and session_of(at) < pd.Timestamp(ex_date)


def ex_dividend_rule(pos: Position, at: pd.Timestamp, spot: float, short_mid: float,
                     ex_date: pd.Timestamp | None, amount: float) -> str | None:
    """Force-close a call credit spread before ex-dividend when early exercise is likely."""
    if pos.kind != "call_credit_spread" or ex_date is None or pd.Timestamp(ex_date) > pos.expiry:
        return None
    if not ex_dividend_day_before(at, ex_date):
        return None
    short = next(l for l in pos.legs if l.qty < 0)
    extrinsic = short_mid - max(spot - short.strike, 0.0)
    if not np.isfinite(extrinsic) or extrinsic < amount:
        return f"ex-dividend forced close (short call extrinsic below the {amount:.2f} dividend)"
    return None
