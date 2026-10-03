"""Jev client tests against a fake local API (ported from the reference project's approach)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sweep.typesafe import (Answer, Choice, JevClient, JevError, Noul, ResponseCache, Score,
                            confidence_from)

ANSWERS = {"q_choice": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.6},
           "q_noul": {"type": "noul", "noul": 0.7},
           "q_score": {"type": "score", "score": 1.4, "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}, "confidence": 0.4}}


class FakeAPI:
    """Serves /v1/systemone and /v1/models; can be told to fail N times first."""

    def __init__(self, fail_first: int = 0, fail_code: int = 429, retry_after: str | None = "0"):
        self.fail_first, self.fail_code, self.retry_after = fail_first, fail_code, retry_after
        self.bodies: list[dict] = []
        self.auth: list[str] = []
        api = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, obj: dict, headers: dict | None = None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                api.bodies.append(json.loads(self.rfile.read(n)))
                api.auth.append(self.headers.get("Authorization", ""))
                if api.fail_first > 0:
                    api.fail_first -= 1
                    hdr = {"retry-after": api.retry_after} if api.retry_after is not None else {}
                    return self._send(api.fail_code, {"error": "busy"}, hdr)
                self._send(200, {"model": "jev-1.13.0", "answers": ANSWERS, "usage": {"input_tokens": 1200, "output_tokens": 30}})

            def do_GET(self):
                self._send(200, {"models": [{"name": "jev-1.13.0", "release_date": "2026-01-01"}]})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


QS = {"q_choice": Choice("pick", {"a": "A", "b": "B"}), "q_noul": Noul("yes?", {"true": "y", "false": "n"}),
      "q_score": Score("how much", ["low", "mid", "high"])}


@pytest.fixture
def api():
    a = FakeAPI()
    yield a
    a.close()


def test_request_body_shape_and_answers(api):
    c = JevClient(api_key="k", base_url=api.url)
    r = c.system_one({"x": "y"}, QS)
    body = api.bodies[0]
    assert body["model"] == "jev-latest" and body["state"] == {"x": "y"}
    assert body["questions"]["q_choice"] == {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}
    assert body["questions"]["q_score"]["type"] == "score" and body["questions"]["q_noul"]["type"] == "noul"
    assert api.auth[0] == "Bearer k"
    assert r.model == "jev-1.13.0" and r["q_choice"].choice == "a" and r["q_noul"].noul == 0.7
    assert r["q_score"].score == 1.4 and c.usage.input_tokens == 1200 and c.usage.calls == 1


def test_retries_on_429_and_529_honouring_retry_after():
    for code in (429, 529):
        a = FakeAPI(fail_first=2, fail_code=code, retry_after="0")
        try:
            r = JevClient(api_key="k", base_url=a.url, max_retries=4).system_one({}, QS)
            assert r["q_noul"].noul == 0.7 and len(a.bodies) == 3
        finally:
            a.close()


def test_gives_up_after_max_retries():
    a = FakeAPI(fail_first=10, fail_code=429, retry_after="0")
    try:
        with pytest.raises(JevError) as e:
            JevClient(api_key="k", base_url=a.url, max_retries=2).system_one({}, QS)
        assert e.value.status == 429 and len(a.bodies) == 3
    finally:
        a.close()


def test_4xx_raises_immediately():
    a = FakeAPI(fail_first=1, fail_code=400)
    try:
        with pytest.raises(JevError) as e:
            JevClient(api_key="k", base_url=a.url).system_one({}, QS)
        assert e.value.status == 400 and len(a.bodies) == 1
    finally:
        a.close()


def test_cache_round_trip_and_cache_only(api, tmp_path):
    path = tmp_path / "jev.jsonl"
    c = JevClient(api_key="k", base_url=api.url, cache=ResponseCache(path))
    r1 = c.system_one({"s": 1}, QS)
    r2 = c.system_one({"s": 1}, QS)
    assert len(api.bodies) == 1 and r2.cached and not r1.cached
    replay = JevClient(cache=ResponseCache(path), cache_only=True)          # no key needed
    assert replay.system_one({"s": 1}, QS)["q_choice"].choice == "a"
    with pytest.raises(JevError):
        replay.system_one({"s": 2}, QS)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient()


def test_models(api):
    assert JevClient(api_key="k", base_url=api.url).models()[0]["name"] == "jev-1.13.0"


def test_confidence_from_and_answer_roundtrip():
    assert confidence_from({"a": 0.5, "b": 0.5}) == 0.0
    assert confidence_from({"a": 1.0, "b": 0.0}) == 1.0
    assert abs(confidence_from({"a": 0.6, "b": 0.2, "c": 0.2}) - 0.4) < 1e-12
    a = Answer.from_json(ANSWERS["q_choice"])
    assert Answer.from_json(a.to_json()) == a and a.p("b") == 0.2


def test_question_validation():
    with pytest.raises(ValueError):
        Choice("x", {"only": "one"}).to_json()
    with pytest.raises(ValueError):
        Score("x", ["one"]).to_json()
