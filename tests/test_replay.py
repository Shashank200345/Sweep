from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from sweep.data.bars import hourly_bars
from sweep.data.fixture import FixtureMarket
from sweep.engine import run
from sweep.replay import build, build_structure
from sweep.smc.engine import detect
from sweep.surrogate import SweepSurrogate

STUB = Path(__file__).parent / "dom_stub.js"


@pytest.fixture(scope="module")
def res():
    md = FixtureMarket(days=300, seed=5)
    days = md._m["days"]
    return run(md, SweepSurrogate(), str(days[-40].date()), str(days[-1].date()))


def _payload(html: str) -> dict:
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    assert m, "data not embedded"
    return json.loads(m.group(1))


def _run_in_stub(html: str, tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = html[html.index("<script>") + 8: html.rindex("</script>")]
    p = tmp_path / "replay.js"
    p.write_text(js, encoding="utf-8")
    out = subprocess.run(["node", str(STUB), str(p)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_replay_is_self_contained_and_embeds_every_jev_call(res, tmp_path):
    html = build(res, tmp_path / "replay.html").read_text(encoding="utf-8")
    assert "<script src" not in html and "http://" not in html and "https://" not in html.split("<script>")[1]
    d = _payload(html)
    assert d["fixture"] and d["surrogate"]
    called = [x for x in res.decisions if x.get("jev_called")]
    embedded = [x for x in d["decisions"] if "answers" in x]
    assert len(embedded) == len(called) > 0
    assert all(set(x["answers"]) == set(d["qorder"]) and x["state"] for x in embedded)
    assert len(d["trades"]) == len(res.trades)
    assert any(z["type"] == "ob" for z in d["zones"]) and d["pivots"] and d["breaks"]
    _run_in_stub(html, tmp_path)


def test_structure_chart(tmp_path):
    md = FixtureMarket(days=40, seed=3)
    h1 = hourly_bars(md.minute_bars(date(2022, 1, 1), date(2022, 3, 1)))
    html = build_structure(h1, detect(h1), tmp_path / "structure.html", md.label).read_text(encoding="utf-8")
    d = _payload(html)
    assert d["n"] == len(h1) and d["decisions"] == [] and "FIXTURE" in html
    for p in d["pivots"]:
        assert p["ci"] >= p["i"] + 3                        # shown at confirmation, never earlier
    _run_in_stub(html, tmp_path)
