from __future__ import annotations

import pytest

from sweep.config import DEFAULT
from sweep.decide import composite_score, decide
from sweep.materiality import MaterialityFilter
from sweep.questions import ACTION_KEYS
from sweep.typesafe import Answer, Response


def choice(c: str, options: tuple, conf: float = 0.9) -> Answer:
    k = len(options)
    pmax = (conf * (k - 1) + 1) / k                      # inverse of TypeSafe's confidence shape
    probs = {o: (pmax if o == c else (1 - pmax) / (k - 1)) for o in options}
    return Answer("choice", choice=c, probabilities=probs, confidence=conf)


def score(s: float, levels: int = 5) -> Answer:
    return Answer("score", score=s, probabilities={str(i): 1.0 if i == round(s) else 0.0 for i in range(levels)}, confidence=0.9)


ACTS = ("buy_calls", "buy_puts", "sell_premium", "no_trade")


def resp(action="buy_calls", bias="bullish", iv="normal", conf=0.9, setup=3.0, actions=None, exit_now=0.1,
         liq=0.8, ob=3.0, fvg="respected", zone=None, ctx_conf=0.9) -> Response:
    zone = zone or ("discount" if bias == "bullish" else "premium")
    a = {"structure_bias": choice(bias, ("bullish", "bearish", "ranging"), ctx_conf),
         "liquidity_event": Answer("noul", noul=liq),
         "ob_confluence": score(ob),
         "fvg_fill_expectation": choice(fvg, ("filled", "respected", "no_gap"), ctx_conf),
         "zone_context": choice(zone, ("premium", "discount", "equilibrium"), ctx_conf),
         "iv_regime": choice(iv, ("elevated", "normal", "low", "unknown")),
         "setup_quality": score(setup),
         "exit_now": Answer("noul", noul=exit_now)}
    for k, act in zip(ACTION_KEYS, actions or [action] * 3):
        a[k] = choice(act, ACTS, conf)
    return Response("test", a)


def test_full_size_long_call():
    d = decide("t", resp(), 0)
    assert d.action == "enter" and d.structure == "long_call" and d.size == 1.0 and d.direction == 1
    assert all(d.gates.values())


def test_half_size_on_medium_confidence():
    d = decide("t", resp(conf=0.5), 0)
    assert d.action == "enter" and d.size == 0.5


def test_low_confidence_blocks():
    d = decide("t", resp(conf=0.2), 0)
    assert d.action == "hold" and not d.gates["confident"]


def test_disagreement_is_no_trade():
    d = decide("t", resp(actions=["buy_calls", "buy_calls", "no_trade"]), 0)
    assert d.action == "hold" and not d.gates["consistent"]


def test_ranging_is_no_trade_even_with_agreeing_actions():
    d = decide("t", resp(action="sell_premium", bias="ranging", iv="elevated"), 0)
    assert d.action == "hold" and not d.gates["not_ranging"]


@pytest.mark.parametrize("bias,action,iv,expect", [
    ("bullish", "buy_calls", "normal", "long_call"),
    ("bullish", "buy_calls", "low", "long_call"),
    ("bearish", "buy_puts", "normal", "long_put"),
    ("bullish", "sell_premium", "elevated", "put_credit_spread"),
    ("bearish", "sell_premium", "elevated", "call_credit_spread"),
])
def test_routing_matrix_allowed(bias, action, iv, expect):
    d = decide("t", resp(action=action, bias=bias, iv=iv), 0)
    assert d.action == "enter" and d.structure == expect


@pytest.mark.parametrize("bias,action,iv,gate", [
    ("bullish", "buy_calls", "elevated", "iv_route"),       # elevated IV must be expressed as a credit spread
    ("bullish", "sell_premium", "low", "iv_route"),         # low IV must not sell premium
    ("bullish", "sell_premium", "normal", "iv_route"),
    ("bullish", "buy_puts", "normal", "direction_matches_bias"),
    ("bullish", "buy_calls", "unknown", "iv_known"),
])
def test_routing_conflicts_are_blocked_not_overridden(bias, action, iv, gate):
    d = decide("t", resp(action=action, bias=bias, iv=iv), 0)
    assert d.action == "hold" and not d.gates[gate] and d.structure is None


def test_setup_and_composite_gates():
    assert not decide("t", resp(setup=1.0), 0).gates["setup"]
    weak = resp(liq=0.0, ob=0.0, fvg="filled", zone="premium", conf=0.36)
    d = decide("t", weak, 0)
    assert d.composite < DEFAULT.policy.min_composite and not d.gates["composite"]


def test_exit_when_positioned():
    d = decide("t", resp(exit_now=0.8), 1)
    assert d.action == "exit"
    assert decide("t", resp(exit_now=0.8), 0).action == "enter"          # exit_now ignored when flat


def test_no_duplicate_entry_when_positioned():
    d = decide("t", resp(), 1)
    assert d.action == "hold" and d.reason == "already positioned"


def test_composite_known_answer():
    r = resp(conf=1.0, liq=1.0, ob=4.0, ctx_conf=1.0)
    w = DEFAULT.policy.weights
    sub_w = sum(w[k] for k in ("structure", "liquidity", "order_block", "fvg", "zone"))
    assert composite_score(r, 1, w) == pytest.approx(1.0)       # every sub-signal term +1, normalized to 1.0
    expected_neg = (-w["structure"] + w["liquidity"] + w["order_block"] + w["fvg"] - w["zone"]) / sub_w
    assert composite_score(r, -1, w) == pytest.approx(expected_neg)
    assert composite_score(r, 0, w) == 0.0


def test_materiality_filter_skips_unchanged_states():
    mf = MaterialityFilter()
    asked = []
    ask = lambda st: asked.append(st) or resp()
    s1 = {"zone": "discount", "position": "flat"}
    r1, c1 = mf.judge("SPY", s1, ask)
    r2, c2 = mf.judge("SPY", dict(s1), ask)
    assert c1 and not c2 and r2 is r1 and len(asked) == 1
    _, c3 = mf.judge("SPY", {"zone": "premium", "position": "flat"}, ask)     # any bucket change fires
    _, c4 = mf.judge("SPY", {"zone": "premium", "position": "long calls"}, ask)
    _, c5 = mf.judge("QQQ", {"zone": "premium", "position": "long calls"}, ask)  # per symbol
    assert c3 and c4 and c5 and mf.calls == 4 and mf.skipped == 1


def test_gate_funnel_diagnostics():
    d_pass = decide("t", resp(), 0)
    assert d_pass.first_failed_gate is None
    assert d_pass.survived_gates == 8
    assert d_pass.candidate_setup is True

    d_inconsistent = decide("t", resp(actions=["buy_calls", "buy_calls", "no_trade"]), 0)
    assert d_inconsistent.first_failed_gate == "consistent"
    assert d_inconsistent.survived_gates == 0
    assert d_inconsistent.candidate_setup is True

    d_low_conf = decide("t", resp(conf=0.2), 0)
    assert d_low_conf.first_failed_gate == "confident"
    assert d_low_conf.survived_gates == 1

    d_ranging = decide("t", resp(action="buy_calls", bias="ranging"), 0)
    assert d_ranging.first_failed_gate == "not_ranging"
    assert d_ranging.survived_gates == 2

    d_no_trade = decide("t", resp(action="no_trade"), 0)
    assert d_no_trade.candidate_setup is False
    assert d_no_trade.first_failed_gate == "consistent"
    assert d_no_trade.survived_gates == 0


def test_sub_signal_agreement_diagnostics():
    # Agreeing setup: bullish action + discount zone + respected FVG + high liquidity
    d_agree = decide("t", resp(action="buy_calls", bias="bullish", zone="discount", fvg="respected", liq=0.9), 0)
    assert d_agree.sub_score > 0
    assert d_agree.action_sub_agree is True

    # Disagreeing setup: bullish action but opposing sub-signals (premium zone, filled FVG, zero liquidity, low OB)
    d_disagree = decide("t", resp(action="buy_calls", bias="bullish", zone="premium", fvg="filled", liq=0.0, ob=0.0), 0)
    assert d_disagree.sub_score < 0
    assert d_disagree.action_sub_agree is False

    # No trade: direction is 0
    d_none = decide("t", resp(action="no_trade"), 0)
    assert d_none.action_sub_agree is None


def test_credit_spreads_only():
    import dataclasses
    p_spreads = dataclasses.replace(DEFAULT.policy, credit_spreads_only=True)
    # When bullish and IV is low, credit_spreads_only routes to put_credit_spread instead of long_call
    d = decide("t", resp(action="buy_calls", bias="bullish", iv="low"), 0, p=p_spreads)
    assert d.action == "enter"
    assert d.structure == "put_credit_spread"
    assert d.direction == 1
    assert all(d.gates.values())

    # When bearish and IV is low, credit_spreads_only routes to call_credit_spread instead of long_put
    d_bear = decide("t", resp(action="buy_puts", bias="bearish", iv="low"), 0, p=p_spreads)
    assert d_bear.action == "enter"
    assert d_bear.structure == "call_credit_spread"
    assert d_bear.direction == -1
    assert all(d_bear.gates.values())


def test_require_ob_near():
    import dataclasses
    p_ob = dataclasses.replace(DEFAULT.policy, require_ob_near=True)
    # State with order block far away
    state_far = {"order_block": {"location": "1 to 3 ATR away from price"}}
    d_far = decide("t", resp(), 0, p=p_ob, state=state_far)
    assert d_far.action == "hold"
    assert not d_far.gates.get("ob_near", False)

    # State with order block inside or within 1 ATR
    state_near = {"order_block": {"location": "within 1 ATR of price"}}
    d_near = decide("t", resp(), 0, p=p_ob, state=state_near)
    assert d_near.action == "enter"
    assert d_near.gates.get("ob_near", False)


