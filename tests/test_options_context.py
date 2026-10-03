from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sweep.config import DEFAULT
from sweep.data.bars import daily_bars, spot_at
from sweep.data.fixture import FixtureMarket
from sweep.options import bsm
from sweep.options.chain import daily_atm_iv, dividend_yield, next_dividend, session_of
from sweep.options.ivrank import hourly_view, iv_rank
from sweep.options.shortlist import pick_expiry, shortlist


@pytest.fixture(scope="module")
def fx():
    return FixtureMarket(days=90, seed=4)


def test_iv_rank_and_percentile_known_answer():
    iv = np.array([0.10, 0.20, 0.30, 0.20, 0.40, 0.10])
    rank, pct = iv_rank(iv, window=4, min_history=3)
    assert np.isnan(rank[:2]).all() and np.isnan(pct[:2]).all()               # warm-up
    assert rank[2] == pytest.approx(1.0) and pct[2] == pytest.approx(1.0)      # 0.30 over [.1,.2,.3]
    assert rank[3] == pytest.approx(0.5) and pct[3] == pytest.approx(0.75)     # 0.2 over [.1,.2,.3,.2]
    assert rank[5] == pytest.approx(0.0) and pct[5] == pytest.approx(0.25)     # 0.1 over [.3,.2,.4,.1]


@pytest.mark.lookahead
def test_iv_rank_is_causal():
    rng = np.random.default_rng(0)
    iv = rng.uniform(0.1, 0.4, 400)
    full = iv_rank(iv, 252, 100)
    for t in (120, 251, 300, 399):
        part = iv_rank(iv[: t + 1], 252, 100)
        np.testing.assert_array_equal(part[0], full[0][: t + 1])
        iv2 = iv.copy()
        iv2[t + 1:] = 5.0
        np.testing.assert_array_equal(iv_rank(iv2, 252, 100)[1][: t + 1], full[1][: t + 1])


@pytest.mark.lookahead
def test_hourly_view_only_sees_closed_sessions():
    daily = pd.DataFrame({"atm_iv": [0.1, 0.2]}, index=pd.DatetimeIndex(
        ["2024-03-01 21:00", "2024-03-04 21:00"], tz="UTC"))
    h = pd.DatetimeIndex(["2024-03-01 20:30", "2024-03-01 21:00", "2024-03-04 20:30", "2024-03-04 21:00"], tz="UTC")
    v = hourly_view(daily, h, 252, 1)
    assert np.isnan(v["atm_iv"].iloc[0]) and list(v["atm_iv"].iloc[1:]) == [0.1, 0.1, 0.2]


def test_dividends_are_causal():
    div = pd.DataFrame({"ex_date": pd.to_datetime(["2024-03-15", "2024-06-21"]),
                        "declaration_date": pd.to_datetime(["2024-03-01", pd.NaT]),
                        "pay_date": pd.to_datetime(["2024-04-30", "2024-07-31"]), "cash_amount": [1.6, 1.7]})
    at = lambda d: pd.Timestamp(f"{d} 15:30", tz="America/New_York")
    assert dividend_yield(div, at("2024-03-14"), 500) == 0.0
    assert dividend_yield(div, at("2024-03-15"), 500) == pytest.approx(1.6 / 500)
    assert next_dividend(div, at("2024-02-28"))[0] is None                   # not declared yet
    assert next_dividend(div, at("2024-03-01")) == (pd.Timestamp("2024-03-15"), 1.6)
    assert next_dividend(div, at("2024-06-01"))[0] is None                   # no declaration date: known 14 d before
    assert next_dividend(div, at("2024-06-07"))[0] == pd.Timestamp("2024-06-21")


def test_daily_atm_iv_recovers_fixture_vol(fx):
    mins = fx.minute_bars(date(2022, 1, 1), date(2022, 6, 1))
    d1 = daily_bars(mins)
    iv = daily_atm_iv(fx, d1.index[:10], spot_at(mins, d1.index[:10]), fx.dividends())
    assert len(iv) == 10 and np.isfinite(iv["atm_iv"]).all()
    true = np.array([fx._vol_at(t) for t in d1.index[:10]]) + 0.02     # fixture smile level at the money
    np.testing.assert_allclose(iv["atm_iv"].to_numpy(), true, atol=0.01)


def test_pick_expiry_is_monthly_in_window(fx):
    at = pd.Timestamp("2022-02-14 15:30", tz="America/New_York")
    ch = fx.contracts(date(2022, 2, 14), date(2022, 3, 1), date(2022, 4, 30))
    e = pick_expiry(ch, at)
    assert e == pd.Timestamp("2022-03-18") and 30 <= (e - session_of(at)).days <= 45
    at2 = pd.Timestamp("2022-02-28 15:30", tz="America/New_York")         # 18 d to Mar, 46 d to Apr monthly
    assert pick_expiry(fx.contracts(date(2022, 2, 28), date(2022, 3, 1), date(2022, 5, 1)), at2) is None


@pytest.mark.parametrize("kind", ["long_call", "long_put", "put_credit_spread", "call_credit_spread"])
def test_shortlist_legs_hit_targets(fx, kind):
    at = pd.Timestamp("2022-02-14 15:30", tz="America/New_York").tz_convert("UTC")
    mins = fx.minute_bars(date(2022, 2, 14), date(2022, 2, 14))
    s = spot_at(mins, pd.DatetimeIndex([at]))[0]
    st = shortlist(fx, kind, at, s, fx.dividends(), 0.18)
    assert st is not None and st.kind == kind
    o = DEFAULT.options
    if kind.startswith("long"):
        (leg,) = st.legs
        assert leg.qty == 1 and abs(abs(leg.delta) - o.long_delta) < 0.03
        assert leg.type == ("call" if kind == "long_call" else "put")
    else:
        short, wing = st.legs
        assert short.qty == -1 and wing.qty == 1 and abs(abs(short.delta) - o.short_delta) < 0.03
        if kind == "put_credit_spread":
            assert short.type == wing.type == "put" and wing.strike == short.strike - 5 and short.strike < s
        else:
            assert short.type == wing.type == "call" and wing.strike == short.strike + 5 and short.strike > s
    for leg in st.legs:
        assert leg.quote_ts <= at and leg.bid > 0 and leg.ask > leg.bid
        T = (pd.Timestamp(leg.expiry).tz_localize("America/New_York") + pd.Timedelta(hours=16) - at).total_seconds() / (365 * 86400)
        assert bsm.implied_vol((leg.bid + leg.ask) / 2, leg.type == "call", s, leg.strike, T, o.risk_free,
                               dividend_yield(fx.dividends(), at, s)) == pytest.approx(leg.iv)
