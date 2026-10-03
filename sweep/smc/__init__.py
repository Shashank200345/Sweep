"""SMC structure detection: deterministic, vectorized, strictly causal.

Every detector returns "as-of" arrays: the value at bar t uses only bars 0..t, and
an event is recorded at the bar where it becomes KNOWABLE (a pivot at its
confirmation bar, a sweep at its reclaim bar), never at the bar it refers to.
The look-ahead suite (tests/lookahead) enforces this; if a detector fails it, the
detector is wrong -- not the test (AGENTS.md non-negotiable 4).

Jev never sees any of this raw; the state builder turns it into named buckets.
"""
