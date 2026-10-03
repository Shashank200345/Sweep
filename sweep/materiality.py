"""The Jev materiality filter (AGENTS.md non-negotiable 6 -- never remove it).

A symbol is only sent to Jev when its bucketed state has changed since it was last
judged. The fingerprint is the exact named-bucket state (including the position
text), so any bucket moving -- structure, liquidity, a zone, IV regime, or the
position -- triggers a call; an unchanged state reuses the previous judgment.
This keeps call volume proportional to what is changing, not to the tick count or
the number of tracked symbols.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .typesafe import Response


def fingerprint(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class MaterialityFilter:
    last: dict[str, tuple[str, Response]] = field(default_factory=dict)
    calls: int = 0
    skipped: int = 0

    def judge(self, symbol: str, state: dict, ask) -> tuple[Response, bool]:
        """Return (response, called). ``ask(state)`` is only invoked when the state changed."""
        fp = fingerprint(state)
        prev = self.last.get(symbol)
        if prev is not None and prev[0] == fp:
            self.skipped += 1
            return prev[1], False
        r = ask(state)
        self.last[symbol] = (fp, r)
        self.calls += 1
        return r, True
