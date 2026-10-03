from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pytest

from sweep.data.bars import hourly_bars
from sweep.data.fixture import FixtureMarket
from sweep.questions import ACTION_KEYS, ACTIONS, decision_questions
from sweep.smc.engine import detect
from sweep.state import bucket_frame, leaf_paths, position_text, state_at
from sweep.surrogate import SweepSurrogate
from sweep.typesafe import Choice, JevClient, Noul, Score

from .test_typesafe import FakeAPI


@pytest.fixture(scope="module")
def buckets():
    m = FixtureMarket(days=120, seed=2)
    h1 = hourly_bars(m.minute_bars(date(2022, 1, 1), date(2023, 1, 1)))
    return bucket_frame(detect(h1).frame)


def _refs(q) -> set[str]:
    text = json.dumps(q.instructions) + json.dumps(getattr(q, "criteria", ""))
    return set(re.findall(r"`([a-z_.]+)`", text))


def test_every_question_reference_exists_and_every_field_is_used(buckets):
    st = state_at(buckets, position_text("put_credit_spread", 10, 0.2))
    leaves = set(leaf_paths(st))
    groups = {p.rsplit(".", 1)[0] for p in leaves if "." in p}
    refs = set().union(*(_refs(q) for q in decision_questions().values()))
    assert refs, "questions must point at state fields by name"
    missing = [r for r in refs if r not in leaves and r not in groups]
    assert missing == [], f"questions reference fields the state does not have: {missing}"
    unused = [p for p in leaves if p not in refs and not any(p.startswith(r + ".") for r in refs)]
    assert unused == [], f"state carries fields no question needs (state minimization): {unused}"


def _leaf(st: dict, path: str):
    for k in path.split("."):
        st = st[k]
    return st


def test_state_is_named_buckets_only(buckets):
    for i in range(0, len(buckets), 50):
        st = state_at(buckets.iloc[i])
        for p in leaf_paths(st):
            v = _leaf(st, p)
            assert isinstance(v, str)
            # prices, levels, ratios or counts would show up as decimals or multi-digit numbers
            assert not re.search(r"\d+\.\d+|\b\d{2,}\b(?!%)", v), f"raw number leaked into state: {p} = {v}"


def test_buckets_cover_every_bar_with_known_words(buckets):
    assert buckets.notna().all().all()
    assert set(buckets["zone"]) <= {"no valid dealing range", "below the current dealing range", "deep discount", "discount",
                                    "equilibrium", "premium", "deep premium", "above the current dealing range"}
    assert (buckets["volatility.iv_rank"] == "insufficient history").all()      # no IV series passed


def test_question_set_shape():
    qs = decision_questions()
    assert len(qs) == 11
    assert {k for k, q in qs.items() if isinstance(q, Noul)} == {"liquidity_event", "exit_now"}
    assert {k for k, q in qs.items() if isinstance(q, Score)} == {"ob_confluence", "setup_quality"}
    for k in ACTION_KEYS:
        assert isinstance(qs[k], Choice) and tuple(qs[k].criteria) == ACTIONS
    for q in qs.values():
        q.to_json()                                   # validates option / level counts


def test_surrogate_answers_every_question_with_typed_shapes(buckets):
    sur = SweepSurrogate()
    qs = decision_questions()
    seen_actions = set()
    for i in range(0, len(buckets), 25):
        r = sur.system_one(state_at(buckets.iloc[i]), qs)
        assert set(r.answers) == set(qs)
        for k, q in qs.items():
            a = r[k]
            if isinstance(q, Noul):
                assert a.type == "noul" and 0 <= a.noul <= 1
            elif isinstance(q, Choice):
                assert a.type == "choice" and a.choice in q.criteria and abs(sum(a.probabilities.values()) - 1) < 1e-3
            else:
                assert a.type == "score" and 0 <= a.score <= len(q.criteria) - 1
        seen_actions.add(r["action"].choice)
    assert "no_trade" in seen_actions


def test_jev_request_carries_the_whole_question_set_in_one_call(buckets):
    api = FakeAPI()
    try:
        JevClient(api_key="k", base_url=api.url).system_one(state_at(buckets.iloc[300]), decision_questions())
        assert len(api.bodies) == 1 and set(api.bodies[0]["questions"]) == set(decision_questions())
        sent = json.dumps(api.bodies[0]["state"])
        assert not re.search(r"\d+\.\d+", sent)                     # no prices, levels or raw values
    finally:
        api.close()
