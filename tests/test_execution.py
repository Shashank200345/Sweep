"""Fill model, exit rules and engine behaviour on scripted quote paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sweep.config import DEFAULT
from sweep.engine import Trader
from sweep.execution.fills import fill, leg_prices, quote_ok
from sweep.execution.positions import Position
from sweep.execution.rules import dte_rule, ex_dividend_day_before, ex_dividend_rule, price_rule
from sweep.options.shortlist import Leg

from .test_decide import resp

T0 = pd.Timestamp("2024-03-04 15:30", tz="America/New_York").tz_convert("UTC")
EXP = pd.Timestamp("2024-04-19")
NO_DIV = pd.DataFrame({"ex_date": pd.to_datetime([]), "declaration_date": pd.to_datetime([]),
                       "pay_date": pd.to_datetime([]), "cash_amount": pd.Series(dtype=float)})


def q(rows: dict, at=T0, age_s=5.0) -> pd.DataFrame:
    return pd.DataFrame({"bid": [b for b, _ in rows.values()], "ask": [a for _, a in rows.values()],
                         "ts": pd.DatetimeIndex([at - pd.Timedelta(seconds=age_s)] * len(rows))},
                        index=pd.Index(list(rows), name="ticker"))


# ---------------------------------------------------------------- fills
def test_every_fill_is_worse_than_mid():
    rng = np.random.default_rng(0)
    bid = rng.uniform(0.5, 20, 1000)
    ask = bid * (1 + rng.uniform(0.001, 0.1, 1000))
    mid = (bid + ask) / 2
    assert (leg_prices(bid, ask, np.ones(1000, bool)) > mid).all()
    assert (leg_prices(bid, ask, np.zeros(1000, bool)) < mid).all()
    np.testing.assert_allclose(leg_prices(np.array([1.0]), np.array([1.2]), np.array([True])), [1.15])


@pytest.mark.parametrize("bid,ask,age,ok", [
    (1.00, 1.05, 5, True), (1.00, 1.05, 299, True), (1.00, 1.05, 301, False),   # 5-minute staleness
    (1.00, 1.12, 5, False), (0.0, 0.05, 5, False), (1.1, 1.0, 5, False),        # >10% wide, no bid, crossed
    (np.nan, 1.0, 5, False)])
def test_quote_validity(bid, ask, age, ok):
    ts = pd.DatetimeIndex([T0 - pd.Timedelta(seconds=age)])
    assert bool(quote_ok(np.array([bid]), np.array([ask]), ts, T0)[0]) is ok


def test_multi_leg_fill_is_all_or_nothing():
    quotes = q({"A": (2.0, 2.1), "B": (1.0, 1.3)})                 # B is 26% wide
    f = fill(["A", "B"], np.array([False, True]), quotes, T0)
    assert not f.ok and "B" in f.reason
    assert fill(["A"], np.array([False]), quotes, T0).ok


# ---------------------------------------------------------------- rules
def leg(t, typ, k, qty):
    return Leg(t, typ, k, EXP, qty, 0, 0, T0, 0.2, 0, 0, 0, 0)


def long_call(entry=5.0):
    return Position("long_call", [leg("C", "call", 500, 1)], 2, T0, 0, entry, [entry], 1.3, 2 * (entry * 100 + 1.3), 1, "", 1)


def put_spread(credit=1.0):
    return Position("put_credit_spread", [leg("P1", "put", 490, -1), leg("P2", "put", 485, 1)], 3, T0, 0, -credit,
                    [credit, 0], 3.9, 3 * ((5 - credit) * 100 + 2.6), 1, "", 1)


def call_spread(credit=1.0):
    return Position("call_credit_spread", [leg("C1", "call", 510, -1), leg("C2", "call", 515, 1)], 1, T0, 0, -credit,
                    [credit, 0], 1.3, (5 - credit) * 100 + 2.6, 1, "", -1)


def test_long_option_price_rules():
    p = long_call(5.0)
    assert price_rule(p, 10.0).startswith("take profit") and price_rule(p, 9.9) is None
    assert price_rule(p, 2.5).startswith("stop") and price_rule(p, 2.6) is None


def test_credit_spread_price_rules():
    p = put_spread(1.0)
    assert price_rule(p, -0.5).startswith("take profit") and price_rule(p, -0.51) is None   # 50% captured
    assert price_rule(p, -3.0).startswith("stop") and price_rule(p, -2.99) is None          # loss = 2 x credit


def test_dte_rule():
    p = long_call()
    assert dte_rule(p, pd.Timestamp("2024-03-29 10:30", tz="America/New_York")) == "21 DTE close"   # 21 days left
    assert dte_rule(p, pd.Timestamp("2024-03-28 16:00", tz="America/New_York")) is None             # 22 days left


def test_ex_dividend_forced_close_only_for_short_calls_with_low_extrinsic():
    ex = pd.Timestamp("2024-03-15")
    day_before = pd.Timestamp("2024-03-14 11:30", tz="America/New_York")
    assert ex_dividend_day_before(day_before, ex) and not ex_dividend_day_before(
        pd.Timestamp("2024-03-13 15:30", tz="America/New_York"), ex)
    p = call_spread()
    # spot 520, short 510 call: intrinsic 10; mid 10.5 -> extrinsic 0.5 < 1.6 dividend -> close
    assert ex_dividend_rule(p, day_before, 520.0, 10.5, ex, 1.6).startswith("ex-dividend")
    assert ex_dividend_rule(p, day_before, 520.0, 12.0, ex, 1.6) is None                     # extrinsic 2.0
    assert ex_dividend_rule(put_spread(), day_before, 520.0, 0.1, ex, 1.6) is None           # puts: n/a
    assert ex_dividend_rule(p, pd.Timestamp("2024-03-12 15:30", tz="America/New_York"), 520.0, 10.5, ex, 1.6) is None


# ---------------------------------------------------------------- engine on scripted quotes
class Scripted:
    label, is_fixture = "scripted", True

    def __init__(self, path: dict, div=NO_DIV):
        self.path, self.div = path, div            # path: {bar close -> {ticker: (bid, ask)}}

    def quotes(self, tickers, at):
        at = pd.Timestamp(at)
        known = [t for t in self.path if t <= at]
        rows = self.path[max(known)] if known else {}
        return q({k: rows.get(k, (np.nan, np.nan)) for k in tickers}, at=at)

    def dividends(self):
        return self.div


class Hold:
    model = "hold"

    def __init__(self, exit_now=0.1):
        self.exit_now = exit_now

    def system_one(self, state, qs):
        return resp(action="no_trade", exit_now=self.exit_now)


def run_path(pos, path, spot=500.0, jev=None, div=NO_DIV):
    md = Scripted(path, div)
    tr = Trader(md, jev or Hold(), div, DEFAULT)
    tr.position = pos
    row = pd.Series({"zone": "equilibrium"})
    for t in sorted(path):
        tr.on_bar(t, row, pd.Series({"atm_iv": 0.2}), spot)
        if tr.position is None:
            break
    return tr


def bars(n, start=T0):
    return [start + pd.Timedelta(hours=h) for h in range(n)]


def test_engine_take_profit_on_executable_price():
    b = bars(3)
    tr = run_path(long_call(5.0), {b[0]: {"C": (6.0, 6.2)}, b[1]: {"C": (9.9, 10.3)}, b[2]: {"C": (10.2, 10.6)}})
    (t,) = tr.trades
    # b1: mid 10.1 but the executable sell is 10.0 = +100% exactly -> fires at b1, fills at b1's quote
    assert t.reason_out.startswith("take profit") and t.exit == pytest.approx(10.0) and t.exit < 10.1
    assert pd.Timestamp(t.exit_t) == b[1] + pd.Timedelta(seconds=DEFAULT.fills.latency_s)
    assert t.pnl == pytest.approx((t.exit - 5.0) * 100 * 2 - 1.3 - 1.3)
    tr2 = run_path(long_call(5.0), {b[0]: {"C": (9.7, 10.3)}, b[1]: {"C": (6.0, 6.2)}})   # mid 10 but sell 9.85
    assert tr2.position is not None and not tr2.trades


def test_engine_stop_on_spread():
    b = bars(2)
    tr = run_path(put_spread(1.0), {b[0]: {"P1": (1.5, 1.6), "P2": (0.5, 0.55)}, b[1]: {"P1": (4.4, 4.6), "P2": (1.2, 1.3)}})
    (t,) = tr.trades
    assert t.reason_out.startswith("stop") and t.exit < -3.0


def test_engine_exit_retries_when_quotes_are_bad_then_fills():
    b = bars(3)
    path = {b[0]: {"C": (5.0, 6.5)}, b[1]: {"C": (5.5, 5.6)}, b[2]: {"C": (5.5, 5.6)}}      # b0: 26% wide
    tr = run_path(long_call(5.0), path, jev=Hold(exit_now=0.9))
    assert "retrying" in tr.decisions[0]["execution"]                     # wanted out, no valid quote
    (t,) = tr.trades
    assert pd.Timestamp(t.exit_t) == b[1] + pd.Timedelta(seconds=DEFAULT.fills.latency_s)
    assert t.exit == pytest.approx(5.525)


def test_engine_exit_now_from_jev():
    b = bars(2)
    tr = run_path(long_call(5.0), {b[0]: {"C": (5.0, 5.1)}, b[1]: {"C": (5.0, 5.1)}}, jev=Hold(exit_now=0.9))
    (t,) = tr.trades
    assert t.reason_out.startswith("exit_now") and len(tr.decisions) == 1


def test_engine_ex_dividend_forced_close():
    div = pd.DataFrame({"ex_date": pd.to_datetime(["2024-03-15"]), "declaration_date": pd.to_datetime(["2024-03-01"]),
                        "pay_date": pd.to_datetime(["2024-04-30"]), "cash_amount": [1.6]})
    t = pd.Timestamp("2024-03-14 14:30", tz="America/New_York").tz_convert("UTC")
    tr = run_path(call_spread(1.0), {t: {"C1": (10.4, 10.6), "C2": (6.0, 6.2)}}, spot=520.0, div=div)
    (tr1,) = tr.trades
    assert tr1.reason_out.startswith("ex-dividend")


def test_engine_refuses_to_hold_into_expiry():
    t = pd.Timestamp("2024-04-19 11:30", tz="America/New_York").tz_convert("UTC")
    with pytest.raises(AssertionError, match="expiry"):
        run_path(long_call(), {t: {"C": (1.0, 1.05)}})
