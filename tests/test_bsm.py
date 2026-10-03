"""Our vectorized BSM against py_vollib (test oracle only; never imported by runtime code)."""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pytest

from sweep.options import bsm

warnings.filterwarnings("ignore", category=DeprecationWarning)
from py_vollib.black_scholes_merton import black_scholes_merton as pv_price  # noqa: E402
from py_vollib.black_scholes_merton.greeks import analytical as pv_greeks  # noqa: E402
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as pv_iv  # noqa: E402

S0 = 100.0
MONEY = np.round(np.arange(0.5, 1.5001, 0.1), 2)
TENORS = [1 / 365, 7 / 365, 30 / 365, 0.25, 0.5, 1.0, 2.0]
VOLS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
RATES = [0.0, 0.05]
DIVS = [0.0, 0.02]
FLAGS = ["c", "p"]

TOL_PRICE = 1e-6
TOL_GREEK = 1e-6
TOL_IV = 1e-5


def grid():
    rows = list(itertools.product(FLAGS, MONEY, TENORS, VOLS, RATES, DIVS))
    f, m, T, v, r, q = (np.array(x) for x in zip(*rows))
    return f, S0 * np.ones(len(rows)), S0 * m.astype(float), T.astype(float), r.astype(float), q.astype(float), v.astype(float)


@pytest.fixture(scope="module")
def g():
    f, S, K, T, r, q, v = grid()
    oracle = {k: np.empty(len(f)) for k in ("price", "delta", "gamma", "theta", "vega")}
    for i in range(len(f)):
        a = (f[i], S[i], K[i], T[i], r[i], v[i], q[i])
        oracle["price"][i] = pv_price(*a)
        oracle["delta"][i] = pv_greeks.delta(*a)
        oracle["gamma"][i] = pv_greeks.gamma(*a)
        oracle["theta"][i] = pv_greeks.theta(*a)
        oracle["vega"][i] = pv_greeks.vega(*a)
    return f, S, K, T, r, q, v, oracle


def test_price_matches_py_vollib(g):
    f, S, K, T, r, q, v, o = g
    ours = bsm.price(f == "c", S, K, T, r, q, v)
    np.testing.assert_allclose(ours, o["price"], atol=TOL_PRICE, rtol=0)


def test_greeks_match_py_vollib(g):
    f, S, K, T, r, q, v, o = g
    gk = bsm.greeks(f == "c", S, K, T, r, q, v)
    np.testing.assert_allclose(gk["delta"], o["delta"], atol=TOL_GREEK, rtol=0)
    np.testing.assert_allclose(gk["gamma"], o["gamma"], atol=TOL_GREEK, rtol=0)
    np.testing.assert_allclose(bsm.per_day(gk["theta"]), o["theta"], atol=TOL_GREEK, rtol=0)
    np.testing.assert_allclose(bsm.per_vol_point(gk["vega"]), o["vega"], atol=TOL_GREEK, rtol=0)


def _well_posed(f, S, K, T, r, q, v):
    p = bsm.price(f == "c", S, K, T, r, q, v)
    lower, upper = bsm.no_arbitrage_bounds(f == "c", S, K, T, r, q)
    vega = bsm.greeks(f == "c", S, K, T, r, q, v)["vega"]
    return p, (vega > 1e-4) & (p - lower > 1e-9) & (upper - p > 1e-9)


def test_iv_recovers_sigma_and_matches_py_vollib(g):
    f, S, K, T, r, q, v, o = g
    p, good = _well_posed(f, S, K, T, r, q, v)
    iv = bsm.implied_vol(p, f == "c", S, K, T, r, q)
    assert good.sum() > 1500
    np.testing.assert_allclose(iv[good], v[good], atol=TOL_IV, rtol=0)
    compared = 0
    for i in np.flatnonzero(good):
        try:
            ref = pv_iv(p[i], S[i], K[i], T[i], r[i], q[i], f[i])
        except Exception:        # py_vollib raises on some edge inputs; ours is compared to truth above
            continue
        if np.isfinite(ref) and ref < 4.9:
            assert abs(iv[i] - ref) < TOL_IV, (f[i], K[i], T[i], v[i], iv[i], ref)
            compared += 1
    assert compared > 1000


def test_price_iv_price_round_trip(g):
    f, S, K, T, r, q, v, _ = g
    p, _ = _well_posed(f, S, K, T, r, q, v)
    iv = bsm.implied_vol(p, f == "c", S, K, T, r, q)
    fin = np.isfinite(iv)
    back = bsm.price(f[fin] == "c", S[fin], K[fin], T[fin], r[fin], q[fin], iv[fin])
    np.testing.assert_allclose(back, p[fin], atol=1e-8, rtol=1e-8)


def test_out_of_bounds_prices_are_nan():
    S, K, T, r, q = 100.0, np.array([80.0, 120.0, 100.0]), 0.25, 0.04, 0.01
    is_call = np.array([True, False, True])
    lower, upper = bsm.no_arbitrage_bounds(is_call, S, K, T, r, q)
    assert np.isnan(bsm.implied_vol(lower - 0.01, is_call, S, K, T, r, q)).all()     # below intrinsic
    assert np.isnan(bsm.implied_vol(lower, is_call, S, K, T, r, q)).all()            # at intrinsic
    assert np.isnan(bsm.implied_vol(upper + 0.01, is_call, S, K, T, r, q)).all()     # above upper bound
    bad = bsm.implied_vol(np.array([0.0, -1.0, np.nan]), is_call, S, K, T, r, q)
    assert np.isnan(bad).all()
    assert np.isnan(bsm.implied_vol(5.0, True, 100.0, 100.0, 0.0, r, q))             # expired


def test_deep_otm_converges_via_bisection_fallback():
    # 1-week 150 call on a 100 stock: vega ~ 0 at the Newton start
    S, K, T, r, q, v = 100.0, np.array([150.0, 140.0, 60.0]), 7 / 365, 0.04, 0.01, np.array([0.9, 0.8, 0.9])
    is_call = np.array([True, True, False])
    p = bsm.price(is_call, S, K, T, r, q, v)
    assert (p > 0).all()
    iv = bsm.implied_vol(p, is_call, S, K, T, r, q)
    assert np.isfinite(iv).all()
    np.testing.assert_allclose(bsm.price(is_call, S, K, T, r, q, iv), p, rtol=1e-6)


def test_bisection_only_path_agrees_with_newton(monkeypatch):
    f, S, K, T, r, q, v = grid()
    p, good = _well_posed(f, S, K, T, r, q, v)
    newton = bsm.implied_vol(p[good], f[good] == "c", S[good], K[good], T[good], r[good], q[good])
    monkeypatch.setattr(bsm, "VEGA_FLOOR", np.inf)          # force every element to bisection
    bis = bsm.implied_vol(p[good], f[good] == "c", S[good], K[good], T[good], r[good], q[good])
    np.testing.assert_allclose(bis, newton, atol=TOL_IV)


def test_vectorized_whole_chain_at_once():
    rng = np.random.default_rng(0)
    n = 50_000
    S = 450.0
    K = rng.uniform(300, 600, n).reshape(250, 200)           # 2-D input keeps its shape
    T = rng.uniform(1 / 365, 1.0, K.shape)
    v = rng.uniform(0.08, 0.8, K.shape)
    c = rng.random(K.shape) < 0.5
    t0 = time.perf_counter()
    p = bsm.price(c, S, K, T, 0.04, 0.013, v)
    iv = bsm.implied_vol(p, c, S, K, T, 0.04, 0.013)
    gk = bsm.greeks(c, S, K, T, 0.04, 0.013, iv)
    assert time.perf_counter() - t0 < 10.0
    assert iv.shape == K.shape and gk["delta"].shape == K.shape


def test_runtime_never_imports_py_vollib_or_numba():
    import pathlib
    import re
    root = pathlib.Path(bsm.__file__).resolve().parents[1]          # the sweep package
    pat = re.compile(r"^\s*(from|import)\s+(py_vollib|vollib|numba)\b", re.M)
    offenders = [str(p) for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
