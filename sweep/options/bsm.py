"""Vectorized Black-Scholes-Merton: price, Greeks and implied volatility for a whole chain.

MODEL CAVEAT -- READ THIS: Black-Scholes-Merton prices *European* options. SPY
options are *American*-style, so everything here is an approximation for SPY.
It is acceptable for the use Sweep makes of it -- selecting and marking OTM legs
at 30-45 DTE, where the early-exercise premium is small -- but it does not model
early exercise or assignment. That risk is handled by the fixed exit rules (close
at 21 DTE, forced close before ex-dividend when a short call's extrinsic value is
below the dividend; see ``sweep.execution.rules``), never by this pricer.

All functions take numpy arrays (or scalars, broadcast together) and operate on the
whole chain at once -- there is no per-contract Python loop. The only Python loops
are over solver *iterations*, each of which is a whole-array operation.

Conventions (inputs): S spot, K strike, T years to expiry, r continuously compounded
risk-free rate, q continuous dividend yield (the "Merton" part; SPY pays dividends),
sigma annualized volatility, ``is_call`` boolean array.
Conventions (outputs): theta is per YEAR and vega per 1.00 of volatility; use
``per_day`` / ``per_vol_point`` to convert (py_vollib reports theta per day and vega
per 1 vol point).

This module is the single implementation used by both the backtester and the paper
loop. py_vollib is used only as a test oracle and must never be imported here.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

IV_LO = 1e-4           # IV search bracket (annualized)
IV_HI = 5.0
VEGA_FLOOR = 1e-8      # below this (price units per 1.00 vol) Newton is abandoned for bisection
NEWTON_TOL = 1e-12
NEWTON_MAX_ITER = 50
BISECT_ITER = 100      # bracket width 5.0 / 2**100 -- far below float precision

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _as_arrays(is_call, *xs):
    arrs = np.broadcast_arrays(np.asarray(is_call, dtype=bool), *[np.asarray(x, dtype=float) for x in xs])
    return [np.array(a) for a in arrs]            # writable copies


def _d1d2(S, K, T, r, q, sigma):
    vs = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vs
    return d1, d1 - vs


def price(is_call, S, K, T, r, q, sigma) -> np.ndarray:
    """European BSM price (approximation for American-style SPY; see module docstring)."""
    c, S, K, T, r, q, sigma = _as_arrays(is_call, S, K, T, r, q, sigma)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    return np.where(c, fs * ndtr(d1) - fk * ndtr(d2), fk * ndtr(-d2) - fs * ndtr(-d1))


def _vega(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * _pdf(d1) * np.sqrt(T)


def greeks(is_call, S, K, T, r, q, sigma) -> dict[str, np.ndarray]:
    """delta, gamma, theta (per year), vega (per 1.00 vol) for every contract at once."""
    c, S, K, T, r, q, sigma = _as_arrays(is_call, S, K, T, r, q, sigma)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    sq = np.sqrt(T)
    n1 = _pdf(d1)
    delta = np.where(c, dq * ndtr(d1), -dq * ndtr(-d1))
    gamma = dq * n1 / (S * sigma * sq)
    common = -S * dq * n1 * sigma / (2.0 * sq)
    theta = np.where(c, common - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1),
                     common + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1))
    vega = S * dq * n1 * sq
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def per_day(theta_per_year):
    return np.asarray(theta_per_year) / 365.0


def per_vol_point(vega_per_unit):
    return np.asarray(vega_per_unit) / 100.0


def no_arbitrage_bounds(is_call, S, K, T, r, q) -> tuple[np.ndarray, np.ndarray]:
    """European bounds: call in (max(Se^-qT - Ke^-rT, 0), Se^-qT); put in (max(Ke^-rT - Se^-qT, 0), Ke^-rT)."""
    c, S, K, T, r, q = _as_arrays(is_call, S, K, T, r, q)
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = np.where(c, np.maximum(fs - fk, 0.0), np.maximum(fk - fs, 0.0))
    upper = np.where(c, fs, fk)
    return lower, upper


def implied_vol(option_price, is_call, S, K, T, r, q) -> np.ndarray:
    """Implied volatility for every contract at once; NaN where no honest answer exists.

    NaN (never a fabricated number) when: inputs are non-finite or non-positive, the
    price is at or below the discounted intrinsic value (European lower bound) or at or
    above the upper bound, or the price lies outside [price(IV_LO), price(IV_HI)].

    Solver: vectorized Newton-Raphson started at the inflection point
    sigma* = sqrt(2|ln(F/K)|/T) (Manaster-Koehler), clipped to [0.05, 3]. Any element whose
    vega is below VEGA_FLOOR (deep OTM), whose Newton step would leave (IV_LO, IV_HI), or
    that has not converged after NEWTON_MAX_ITER falls back to vectorized bisection on
    [IV_LO, IV_HI]. BSM price is strictly increasing in sigma, so bisection always converges.

    European model: an approximation for American-style SPY (see module docstring).
    """
    c, p, S, K, T, r, q = _as_arrays(is_call, option_price, S, K, T, r, q)
    out = np.full(p.shape, np.nan)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(p) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & (S > 0) & (K > 0) & (T > 0) & (p > 0)
    if not ok.any():
        return out
    lower, upper = no_arbitrage_bounds(c, S, K, T, r, q)
    ok &= (p > lower) & (p < upper)
    lo_px = price(c, S, K, T, r, q, np.full(p.shape, IV_LO))
    hi_px = price(c, S, K, T, r, q, np.full(p.shape, IV_HI))
    ok &= (p >= lo_px) & (p <= hi_px)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    cc, pp, SS, KK, TT, rr, qq = (a.ravel()[idx] for a in (c, p, S, K, T, r, q))
    F = SS * np.exp((rr - qq) * TT)
    sig = np.clip(np.sqrt(2.0 * np.abs(np.log(F / KK)) / TT), 0.05, 3.0)

    active = np.ones(idx.size, dtype=bool)
    fallback = np.zeros(idx.size, dtype=bool)
    done = np.zeros(idx.size, dtype=bool)
    for _ in range(NEWTON_MAX_ITER):
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        s = sig[a]
        diff = price(cc[a], SS[a], KK[a], TT[a], rr[a], qq[a], s) - pp[a]
        vg = _vega(SS[a], KK[a], TT[a], rr[a], qq[a], s)
        weak = ~(vg >= VEGA_FLOOR)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = diff / vg
        new = s - step
        escaped = weak | ~np.isfinite(new) | (new <= IV_LO) | (new >= IV_HI)
        conv = ~escaped & (np.abs(step) < NEWTON_TOL)
        sig[a] = np.where(escaped, s, new)
        fallback[a[escaped]] = True
        done[a[conv]] = True
        active[a[escaped | conv]] = False
    fallback |= active                                  # did not converge in time

    b = np.flatnonzero(fallback)
    if b.size:
        lo, hi = np.full(b.size, IV_LO), np.full(b.size, IV_HI)
        for _ in range(BISECT_ITER):
            mid = 0.5 * (lo + hi)
            above = price(cc[b], SS[b], KK[b], TT[b], rr[b], qq[b], mid) > pp[b]
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        sig[b] = 0.5 * (lo + hi)
        done[b] = True
    flat = out.ravel()
    flat[idx[done]] = sig[done]
    return flat.reshape(p.shape)
