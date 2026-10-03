from datetime import date

import numpy as np
import pandas as pd
import pytest

from sweep.cli import main
from sweep.data.bars import completed_daily_count, daily_bars, hourly_bars, spot_at
from sweep.data.fixture import FixtureMarket
from sweep.data.market import monthly_mask, parse_occ, occ_ticker
from sweep.data.probe import run_probe


@pytest.fixture(scope="module")
def fx():
    return FixtureMarket(days=60, seed=3)


def test_hourly_bars_are_session_anchored_with_half_bar(fx):
    m = fx.minute_bars(date(2022, 1, 3), date(2022, 1, 4))
    h = hourly_bars(m)
    et = h.index.tz_convert("America/New_York")
    assert list(et[:7].strftime("%H:%M")) == ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    assert len(h) == 14 and list(h["bin"][:7]) == list(range(7))
    first = m.iloc[:60]
    assert h["open"].iloc[0] == first["open"].iloc[0] and h["close"].iloc[0] == first["close"].iloc[-1]
    assert h["high"].iloc[0] == first["high"].max() and h["low"].iloc[0] == first["low"].min()
    assert h["volume"].iloc[6] == m.iloc[360:390]["volume"].sum()        # the half bar


def test_hourly_bars_across_dst(fx):
    m = FixtureMarket(start=date(2022, 3, 10), days=4).minute_bars(date(2022, 3, 10), date(2022, 3, 15))
    et = hourly_bars(m).index.tz_convert("America/New_York")
    assert set(et.strftime("%H:%M")) == {"10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"}


def test_asof_drops_forming_bar_and_partial_day(fx):
    m = fx.minute_bars(date(2022, 1, 3), date(2022, 1, 4))
    t = pd.Timestamp("2022-01-04 11:45", tz="America/New_York")
    partial = m[m.index + pd.Timedelta(minutes=1) <= t]
    h = hourly_bars(partial, asof=t)
    assert h.index[-1].tz_convert("America/New_York").strftime("%H:%M") == "11:30"
    d = daily_bars(partial, asof=t)
    assert len(d) == 1                                                   # 2022-01-04 not complete yet


def test_daily_availability_mapping(fx):
    m = fx.minute_bars(date(2022, 1, 3), date(2022, 1, 5))
    h, d = hourly_bars(m), daily_bars(m)
    k = completed_daily_count(h.index, d.index)
    assert list(k[:7]) == [0] * 6 + [1] and list(k[7:14]) == [1] * 6 + [2]


def test_spot_at_uses_only_closed_minutes(fx):
    m = fx.minute_bars(date(2022, 1, 3), date(2022, 1, 3))
    t = m.index[10] + pd.Timedelta(seconds=59)
    assert spot_at(m, pd.DatetimeIndex([t]))[0] == m["close"].iloc[9]


def test_occ_round_trip_and_monthly():
    t = occ_ticker("SPY", date(2024, 4, 19), "put", 512.5)
    assert t == "O:SPY240419P00512500"
    p = parse_occ([t])
    assert p["type"][0] == "put" and p["strike"][0] == 512.5 and p["expiry"][0] == pd.Timestamp("2024-04-19")
    e = pd.Series(pd.to_datetime(["2024-04-19", "2024-04-12", "2025-04-17", "2024-05-17"]))
    assert list(monthly_mask(e)) == [True, False, True, True]          # 2025-04-18 was Good Friday


def test_probe_runs_on_fixture(fx):
    days = [d.date() for d in fx._m["days"][-8:-5]]
    res = run_probe(fx, days)
    s = res.summary()
    assert s["ticks"] == 21 and s["quotes_requested"] >= 21 * 4
    assert s["two_sided_pct"] > 90 and all(v in ("quotes found", "no quotes", "no underlying bars") for v in res.history.values())


def test_probe_cli_without_key_is_clear(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        main(["probe"])
    assert "MASSIVE_API_KEY is not set" in str(e.value)
