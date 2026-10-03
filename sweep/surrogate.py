"""Offline stand-in for Jev (pattern adapted from jev_bot/surrogate.py, MIT).

Answers Sweep's own question set with the same typed shapes (Noul / Choice / Score,
probabilities, confidence) by reading the *named-bucket* state back through the
vocabularies in ``sweep.state``. It never sees anything the real model would not.

It exists so everything runs, tests and renders without a TypeSafe key. It is a
hand-written rule model, NOT Jev, and results produced with it say nothing about
Jev. It knows nothing about questions outside ``questions.decision_questions()``.
"""
from __future__ import annotations

import json
import math
from typing import Any

from . import state as S
from .typesafe import Answer, Choice, Noul, Question, Response, Score, confidence_from


def _softmax(logits: dict[str, float], temp: float = 1.0) -> dict[str, float]:
    m = max(logits.values())
    e = {k: math.exp((v - m) / temp) for k, v in logits.items()}
    z = sum(e.values())
    return {k: round(v / z, 4) for k, v in e.items()}


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class SweepSurrogate:
    model = "surrogate-local"

    @staticmethod
    def _read(st: dict) -> dict[str, float]:
        s, lq, ob, fg = st["structure"], st["liquidity"], st["order_block"], st["fair_value_gap"]
        daily = {"bullish": 1.0, "bearish": -1.0}.get(s["daily_bias"], 0.0)
        hourly = 1.0 if s["hourly_trend"].startswith("bullish") else -1.0 if s["hourly_trend"].startswith("bearish") else 0.0
        ev = s["last_event"]
        choch = "CHoCH" in ev
        recent = S.WHEN[0] in ev or S.WHEN[1] in ev
        le = lq["last_event"]
        sweep = " swept " in le
        sweep_dir = (1.0 if le.startswith("sell-side") else -1.0) if sweep else 0.0
        taken_dir = (1.0 if le.startswith("buy-side") else -1.0) if " taken " in le else 0.0
        liq_recent = S.WHEN[0] in le or S.WHEN[1] in le
        ob_side = 1.0 if ob["side"].startswith("bullish") else -1.0 if ob["side"].startswith("bearish") else 0.0
        loc_w = {S.LOCATION[0]: 1.0, S.LOCATION[1]: 0.8, S.LOCATION[2]: 0.4, S.LOCATION[3]: 0.1}.get(ob["location"], 0.0)
        ob_score = 0.0
        if ob_side:
            ob_score = (0.9 * loc_w + 0.8 * (ob["status"] == "untested") + 0.6 * ("sweep came" in ob["after_sweep"])
                        + 0.5 * ("change of character" in ob["origin"]) + 0.5 * ("overlaps" in ob["gap_overlap"])
                        + 0.4 * ((ob_side > 0 and "discount" in ob["range_half"]) or (ob_side < 0 and "premium" in ob["range_half"])))
        fvg_side = 1.0 if fg["side"].startswith("bullish") else -1.0 if fg["side"].startswith("bearish") else 0.0
        zone = st["zone"]
        zpos = {"deep discount": -1.0, "discount": -0.5, "equilibrium": 0.0, "premium": 0.5, "deep premium": 1.0,
                "below the current dealing range": -1.0, "above the current dealing range": 1.0}.get(zone, 0.0)
        ivr = st["volatility"]["iv_rank"]
        iv = {S.IV[1]: -1.0, S.IV[2]: 0.0, S.IV[3]: 1.0}.get(ivr, float("nan"))
        pos = st["position"]
        pos_dir = 0.0 if pos == "flat" else (1.0 if "bullish position" in pos else -1.0)
        return {"daily": daily, "hourly": hourly, "choch": float(choch), "recent": float(recent), "sweep": float(sweep),
                "sweep_dir": sweep_dir, "taken_dir": taken_dir, "liq_recent": float(liq_recent), "ob_side": ob_side,
                "ob_score": ob_score, "ob_loc": loc_w, "fvg_side": fvg_side, "fvg_tested": float("tested" in fg["status"]),
                "fvg_loc": {S.LOCATION[0]: 1.0, S.LOCATION[1]: 0.7, S.LOCATION[2]: 0.3}.get(fg["location"], 0.0),
                "zpos": zpos, "iv": iv, "pos": pos_dir}

    def system_one(self, state: Any, questions: dict[str, Question]) -> Response:
        f = self._read(state)
        bias = f["hourly"] * (1.0 if f["daily"] * f["hourly"] >= 0 else 0.3)
        d = 1.0 if bias > 0 else -1.0 if bias < 0 else 0.0
        support = (0.35 * (f["sweep_dir"] == d and f["liq_recent"]) + 0.3 * (f["ob_side"] == d) * min(f["ob_score"], 3) / 3
                   + 0.2 * (f["fvg_side"] == d) * f["fvg_loc"] + 0.15 * (-f["zpos"] * d > 0)) if d else 0.0
        conv = abs(bias) * (0.35 + support) + 0.15 * f["recent"] * abs(bias)
        iv_known = not math.isnan(f["iv"])
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            if qid == "structure_bias":
                pr = _softmax({"bullish": 3 * bias, "bearish": -3 * bias, "ranging": 1.2 - 2.5 * abs(bias)}, 0.8)
            elif qid == "liquidity_event":
                out[qid] = Answer("noul", noul=round(_sig(4 * (f["sweep"] * (0.5 + 0.5 * f["liq_recent"]) - 0.4)), 4)); continue
            elif qid == "ob_confluence":
                s = min(f["ob_score"], 4.0) if f["ob_side"] else 0.0
                pr = _softmax({str(i): -abs(s - i) * 1.4 for i in range(5)}, 1.0)
            elif qid == "fvg_fill_expectation":
                if not f["fvg_side"]:
                    pr = _softmax({"filled": 0.0, "respected": 0.0, "no_gap": 3.0})
                else:
                    with_trend = f["fvg_side"] * f["hourly"]
                    pr = _softmax({"respected": 1.5 * with_trend - 0.8 * f["fvg_tested"],
                                   "filled": -1.5 * with_trend + 0.8 * f["fvg_tested"], "no_gap": -2.0}, 0.9)
            elif qid == "zone_context":
                pr = _softmax({"premium": 3 * f["zpos"], "discount": -3 * f["zpos"], "equilibrium": 1.0 - 3 * abs(f["zpos"])}, 0.8)
            elif qid == "iv_regime":
                if not iv_known:
                    pr = _softmax({"elevated": 0.0, "normal": 0.0, "low": 0.0, "unknown": 3.0})
                else:
                    pr = _softmax({"elevated": 3 * f["iv"], "low": -3 * f["iv"], "normal": 1.2 - 2 * abs(f["iv"]), "unknown": -3.0}, 0.8)
            elif qid == "setup_quality":
                s = min(4.0, conv * 4.5)
                pr = _softmax({str(i): -abs(s - i) * 1.3 for i in range(5)}, 1.0)
            elif qid.startswith("action"):
                bar = {"action": 0.55, "action_alt": 0.6, "action_contrarian_check": 0.7}.get(qid, 0.6)
                strength = conv - bar
                sell = iv_known and f["iv"] > 0
                logits = {"buy_calls": -2.0, "buy_puts": -2.0, "sell_premium": -2.0, "no_trade": 0.3}
                if d and iv_known:
                    key = "sell_premium" if sell else ("buy_calls" if d > 0 else "buy_puts")
                    logits[key] = 6 * strength + 0.3
                pr = _softmax(logits, {"action": 0.55, "action_alt": 0.5, "action_contrarian_check": 0.7}.get(qid, 0.6))
            elif qid == "exit_now":
                against = -f["pos"] * bias
                out[qid] = Answer("noul", noul=round(_sig(5 * (against - 0.2 * (f["ob_side"] == f["pos"]))) if f["pos"] else 0.5, 4)); continue
            else:
                pr = self._uniform(q)
            out[qid] = self._answer(q, pr)
        tokens = len(json.dumps({"state": state, "questions": {k: q.to_json() for k, q in questions.items()}})) // 4
        return Response(self.model, out, tokens, 8 * len(questions), 0.0)

    @staticmethod
    def _uniform(q: Question) -> dict[str, float]:
        if isinstance(q, Choice):
            keys = list(q.criteria)
        elif isinstance(q, Score):
            keys = [str(i) for i in range(len(q.criteria))]
        else:
            return {"true": 0.5}
        return {k: 1 / len(keys) for k in keys}

    @staticmethod
    def _answer(q: Question, pr: dict[str, float]) -> Answer:
        if isinstance(q, Noul):
            return Answer("noul", noul=pr.get("true", 0.5))
        conf = round(confidence_from(pr), 4)
        if isinstance(q, Choice):
            return Answer("choice", choice=max(pr, key=pr.get), probabilities=pr, confidence=conf)
        legend = {str(i): (c if isinstance(c, str) else str(c)) for i, c in enumerate(q.criteria)}
        score = round(sum(int(k) * p for k, p in pr.items()), 4)
        return Answer("score", score=score, probabilities=pr, confidence=conf, legend=legend)
