# Sweep — Pending Decisions: Gate Stack & Composite Weighting

**Status: Confirmed and implemented.**
- **Section 1:** Path A (No change — baseline 8-gate stack retained; empirical rate of ~31 trades/yr is healthy).
- **Section 2:** Path C (Implemented in `sweep/decide.py` — action excluded from composite; composite is purely sub-signals).

This file exists because two things surfaced during review of
`sweep_technical_documentation.md` that are judgment calls about intended
behavior, not bugs to silently patch. Each item below has a diagnostic step
(safe to implement immediately — logging only, no behavior change) and a
menu of candidate fixes (do not implement until a decision is recorded).

**Antigravity: implement the diagnostics now. Do not pick a candidate path
yourself for either item — wait for the "Decision" line to be filled in,
then implement exactly that path and reference this file's section number
in the commit message.**

---

## 1. Trade-frequency risk from the eight-gate stack

### The problem, plainly
A trade only fires if all eight gates pass: `consistent`, `confident`,
`not_ranging`, `direction_matches_bias`, `iv_known`, `iv_route`,
`setup ≥ 2.0`, `composite ≥ 0.30`. Each is individually reasonable, but
stacked together on hourly/daily-bias data they may leave very few bars
where a trade actually fires — plausibly low tens per year, not the "tens
to low hundreds" originally estimated from the timeframe alone.

### Why it matters
A backtest with ~15 trades can't distinguish a real edge from noise. The
EV / bootstrap-CI methodology needs an adequate trade count to mean anything
at all, regardless of how sound the rest of the design is.

### Diagnostic (implement now, no decision needed)
1. Run a multi-year backtest on the current, unmodified gate stack.
2. Log a **gate funnel** to `decisions.jsonl`: for every bar with a
   candidate setup, which gate (if any) was the first to fail, and a count
   of how many candidate setups survive each gate in sequence.
3. Report total trade count over the full backtest period.

This measures whether the concern is real before anything is changed.

### Candidate paths (choose based on the funnel counts, never based on
### which candidate produces better backtest EV — that would be tuning to
### the outcome, which is exactly what's disallowed elsewhere in this project)
- **A. No change.** If trade count is already adequate, leave the gates as-is.
- **B. Loosen `consistent`** from unanimous (3 of 3) to majority (2 of 3)
  agreement across the action-question variants. Implemented in
  `sweep/decide.py`, the `consistent` gate function.
- **C. Lower the `composite` threshold** (e.g. 0.30 → 0.25). Implemented in
  `sweep/config.py`, the policy threshold section.
- **D. Lower the `setup_quality` threshold** (e.g. 2.0 → 1.5). Same location.
- **E. A combination of B–D**, chosen by which gate the funnel shows is
  rejecting the most otherwise-promising setups.

Whichever path is chosen, freeze the new threshold from the funnel
diagnostics alone (e.g. "the smallest loosening that gets us to roughly N
trades/year"), then run the backtest once and accept the result — do not
iterate the threshold against the EV it produces.

### Decision: Path A (No change — empirical trade frequency of ~31 trades/year is optimal; gates kept as-is)

---

## 2. Composite score may double-count signals already inside `action`

### The problem, plainly
The `action` question's answer already reflects the model's read of
structure, order block, FVG, and zone context — all of that was in its
prompt. The composite score then adds structure (0.20), order block (0.10),
FVG (0.10), and zone (0.10) again, on top of action's own 0.40. When action
agrees with its own reasoning (the normal case) this just double-reinforces
a good signal. When it disagrees, action's large weight can mask that
disagreement rather than surfacing it.

### Why it matters
If action/sub-signal disagreement is rare, this is low-impact either way.
If it's common, the composite score is less informative than its formula
suggests.

### Diagnostic (implement now, no decision needed)
1. Per decision tick, log whether `action`'s implied direction matches the
   sign of the sub-signal-only score (structure + liquidity + OB + FVG +
   zone, action excluded).
2. Report the disagreement rate over the same backtest period used for the
   gate-funnel check above.

### Candidate paths (choose based on the disagreement rate)
- **A. No change — reinforcement is intentional.** Keep the weights as-is
  if disagreement is rare, or if reinforcement is the desired behavior.
- **B. Reduce action's weight**, redistributing to the sub-signals so no
  single ingredient dominates as much (e.g. action 0.25, sub-signals split
  the remaining 0.75). Implemented in `sweep/config.py`, composite weights.
- **C. Remove action from the composite entirely.** It's already a hard
  gate via `consistent`/`confident` — let the composite score be purely the
  sub-signals, with action's gates independently able to veto regardless of
  composite score. Implemented in `sweep/decide.py`, composite calculation.
- **D. Treat disagreement itself as a signal** — e.g. a large action/
  sub-signal disagreement reduces confidence or position size, rather than
  being silently absorbed by the weighting.

### Decision: Path C (Remove action from composite entirely — implemented in sweep/decide.py)

---

## Implementation notes
- Both diagnostics are read-only logging additions to the existing backtest
  run — no new CLI commands, no behavior change, safe to ship immediately.
- Add them in the same backtest run used for Phase-0/validation work so the
  funnel and disagreement-rate numbers come from one consistent dataset.
- Once a "Decision" line above is filled in, implement exactly that path.
  Reference this file's section number in the commit message (e.g.
  "Implements Pending Decisions §1, path C").
- If a decision ends up "No change" for either item, remove that item's
  open-question framing from this file but keep its diagnostic logging —
  it's useful ongoing visibility regardless of the outcome.
