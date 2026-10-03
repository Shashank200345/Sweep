"""From Jev's typed answers to a paper order (mechanism adapted from jev_bot/decide.py, MIT).

Three documented TypeSafe patterns do the work, all in code:
  * composite scoring   -- atomic answers combined with weights we own (APPROVED, config);
  * confidence gating   -- full size at high confidence, half size at medium, none at low;
  * self-consistency    -- the action is asked three ways and all three must agree.

``iv_regime`` is not in the composite; it routes: elevated IV -> the direction is
expressed as a credit spread, low/normal -> long calls/puts. A conflict between the
chosen action and the route is BLOCKED with a named gate, never overridden.
``structure_bias`` ranging -> no trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT, PolicyConfig
from .questions import ACTION_KEYS
from .typesafe import Response

DIRECTION = {"buy_calls": 1, "buy_puts": -1}

GATE_ORDER = (
    "consistent",
    "confident",
    "not_ranging",
    "direction_matches_bias",
    "iv_known",
    "iv_route",
    "setup",
    "composite",
)


@dataclass
class Decision:
    t: str                      # bar close time (ISO)
    action: str                 # enter | exit | hold
    structure: str | None       # long_call | long_put | put_credit_spread | call_credit_spread
    direction: int              # +1 bullish, -1 bearish, 0
    size: float                 # 0, 0.5 or 1.0
    composite: float
    gates: dict[str, bool]
    reason: str
    answers: dict = field(default_factory=dict)
    jev_called: bool = True
    first_failed_gate: str | None = None
    survived_gates: int = 0
    candidate_setup: bool = False
    sub_score: float = 0.0
    action_sub_agree: bool | None = None

    def to_json(self) -> dict:
        return {"t": self.t, "action": self.action, "structure": self.structure, "direction": self.direction,
                "size": self.size, "composite": round(self.composite, 4), "gates": self.gates, "reason": self.reason,
                "jev_called": self.jev_called, "answers": self.answers,
                "first_failed_gate": self.first_failed_gate, "survived_gates": self.survived_gates,
                "candidate_setup": self.candidate_setup, "sub_score": round(self.sub_score, 4),
                "action_sub_agree": self.action_sub_agree}


def _signed(ans, pos: str, neg: str, d: int) -> float:
    return (ans.p(pos) - ans.p(neg)) * d


SUB_SIGNAL_KEYS = ("structure", "liquidity", "order_block", "fvg", "zone")


def sub_signal_score(r: Response, d: int, w: dict) -> float:
    """Weighted sum of sub-signal atomic answers (action excluded) signed toward direction ``d``.
    
    Normalized by the sum of sub-signal weights so the score lies strictly in [-1, 1].
    """
    if d == 0:
        return 0.0
    structure = _signed(r["structure_bias"], "bullish", "bearish", d)
    liquidity = 2 * (r["liquidity_event"].noul or 0.0) - 1
    score_n = lambda a, levels: 2 * (a.score or 0.0) / (levels - 1) - 1
    ob = score_n(r["ob_confluence"], 5)
    fvg = r["fvg_fill_expectation"].p("respected") - r["fvg_fill_expectation"].p("filled")
    zone = _signed(r["zone_context"], "discount", "premium", d)
    sub_w = sum(w.get(k, 0.0) for k in SUB_SIGNAL_KEYS)
    raw = (w.get("structure", 0.0) * structure + w.get("liquidity", 0.0) * liquidity
           + w.get("order_block", 0.0) * ob + w.get("fvg", 0.0) * fvg + w.get("zone", 0.0) * zone)
    return raw / sub_w if sub_w > 0 else raw


def composite_score(r: Response, d: int, w: dict) -> float:
    """Composite score is purely the sub-signals (Pending Decisions §2, Path C).
    
    Action is an independent veto via the consistent and confident gates and is excluded
    from the composite score to eliminate self-reinforcement and double-counting.
    """
    return sub_signal_score(r, d, w)


def route(action: str, bias: int, iv: str, credit_spreads_only: bool = False) -> tuple[str | None, int]:
    """(structure, direction) implied by the action, or (None, 0)."""
    if credit_spreads_only:
        if action == "no_trade" or bias == 0:
            return None, 0
        if bias > 0 and action in ("buy_calls", "sell_premium"):
            return "put_credit_spread", 1
        if bias < 0 and action in ("buy_puts", "sell_premium"):
            return "call_credit_spread", -1
        return None, 0

    if action == "sell_premium":
        if bias > 0:
            return "put_credit_spread", 1
        if bias < 0:
            return "call_credit_spread", -1
        return None, 0
    if action in DIRECTION:
        d = DIRECTION[action]
        return ("long_call" if d > 0 else "long_put"), d
    return None, 0


def decide(t: str, r: Response, position_dir: int, p: PolicyConfig = DEFAULT.policy, state: dict | None = None) -> Decision:
    ans = {k: v.to_json() for k, v in r.answers.items()}
    if position_dir:
        ex = r["exit_now"].noul or 0.0
        if ex >= p.exit_noul:
            return Decision(t, "exit", None, position_dir, 0, 0.0, {"exit_now": True}, f"exit_now {ex:.2f}", ans,
                            first_failed_gate=None, survived_gates=0, candidate_setup=False, sub_score=0.0, action_sub_agree=None)
    bias = {"bullish": 1, "bearish": -1}.get(r["structure_bias"].choice, 0)
    iv = r["iv_regime"].choice
    picks = [r[k].choice for k in ACTION_KEYS]
    conf = min(r[k].confidence or 0.0 for k in ACTION_KEYS)
    side = picks[0]
    structure, d = route(side, bias, iv, credit_spreads_only=p.credit_spreads_only)
    comp = composite_score(r, d, p.weights)
    sub = sub_signal_score(r, d, p.weights)
    wants_long = (side in DIRECTION) and not p.credit_spreads_only
    gates = {
        "consistent": len(set(picks)) == 1 and side != "no_trade",
        "confident": conf >= p.min_action_conf,
        "not_ranging": bias != 0,
        "direction_matches_bias": d == bias and d != 0,
        "iv_known": iv != "unknown",
        "iv_route": (iv != "unknown") if p.credit_spreads_only else ((iv == "elevated") == (side == "sell_premium") if side != "no_trade" else False),
        "setup": (r["setup_quality"].score or 0.0) >= p.min_setup,
        "composite": comp >= p.min_composite,
    }
    if wants_long:
        gates["iv_route"] = gates["iv_route"] and iv in ("low", "normal")
    if p.require_ob_near and state:
        ob_loc = state.get("order_block", {}).get("location", "")
        gates["ob_near"] = ob_loc in ("price is inside it", "within 1 ATR of price")

    # Diagnostics (§1 gate funnel & §2 sub-signal agreement)
    first_failed = None
    survived = 0
    gate_order = list(GATE_ORDER) + (["ob_near"] if p.require_ob_near and "ob_near" in gates else [])
    for g in gate_order:
        if gates.get(g, False):
            survived += 1
        else:
            first_failed = g
            break
    candidate_setup = any(pk != "no_trade" for pk in picks)
    action_sub_agree = (sub > 0) if d != 0 else None

    if not all(gates.values()):
        failed = ",".join(k for k, v in gates.items() if not v)
        return Decision(t, "hold", None, 0, 0, comp, gates, f"blocked: {failed}", ans,
                        first_failed_gate=first_failed, survived_gates=survived,
                        candidate_setup=candidate_setup, sub_score=sub, action_sub_agree=action_sub_agree)
    if position_dir:
        return Decision(t, "hold", None, d, 0, comp, gates, "already positioned", ans,
                        first_failed_gate=first_failed, survived_gates=survived,
                        candidate_setup=candidate_setup, sub_score=sub, action_sub_agree=action_sub_agree)
    size = 1.0 if conf >= p.full_size_conf else 0.5
    return Decision(t, "enter", structure, d, size, comp, gates,
                    f"{structure} conf {conf:.2f} composite {comp:+.2f}", ans,
                    first_failed_gate=first_failed, survived_gates=survived,
                    candidate_setup=candidate_setup, sub_score=sub, action_sub_agree=action_sub_agree)
