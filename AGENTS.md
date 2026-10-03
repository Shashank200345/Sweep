# Sweep — Agent Instructions

## What this project is
Sweep detects Smart Money Concept (SMC) structure in price action and computes
options mechanics (Greeks, IV rank) in deterministic code, then asks Jev
(TypeSafe's System One model) a batch of typed questions per decision tick to
judge confluence, context, and direction. It is a **research and paper-trading
project**, built on the architecture of github.com/L1vsun/JEV-Trading-BOT.

Read this file in full before making changes. Several decisions below were
made deliberately, after explicitly weighing and rejecting a faster or
"more sophisticated" alternative — don't re-introduce those alternatives to
be helpful. If a task seems to require it, stop and ask first instead of
proceeding.

## Tech stack
- **Language:** Python only. One implementation, shared by the backtester and
  the live loop — no second implementation in Rust, numba, or anything else.
- **Structure detection:** pandas / numpy, vectorized (whole-array operations,
  never a per-candle or per-contract Python loop).
- **Options pricing / Greeks:** a custom vectorized Black-Scholes-Merton
  implementation (`sweep/options/bsm.py`) — prices, Greeks, and implied vol
  for a whole chain at once, no per-contract Python loop, no JIT. Includes
  the continuous dividend yield term (SPY pays dividends), a Manaster-Koehler
  initial guess, vectorized Newton-Raphson, and a vectorized bisection
  fallback for deep-OTM contracts where vega is too small for Newton to
  converge. **py_vollib is a test-only oracle** (`tests/test_bsm.py`,
  checked across a wide grid of strikes/expiries/vols) — it is never
  imported at runtime. Do not add py_vollib or QuantLib as a runtime
  dependency; extend the test grid instead if you need more confidence in
  a specific region (e.g. very short-dated or very deep OTM).
- **Options chain data:** Massive (formerly Polygon.io), fetched via batched
  multi-symbol requests where supported, and via concurrent `asyncio`
  requests otherwise. Request/response only — no WebSocket streaming feed
  (deliberately deferred, see Non-Negotiables). See "Decisions already made"
  for the data-provider choice and its caveats.
- **Jev access:** the direct TypeSafe client, reused from the reference
  project's `typesafe.py`. Don't rewrite this.
- **Backtesting engine:** a custom event-driven backtester, not a vectorized
  one — options payoffs are path-dependent (assignment, early exercise, theta
  decay along the path), which a vectorized backtest can silently get wrong.
- **Dashboard:** the reference project's self-contained HTML replay
  generator, extended with SMC-structure and options-Greeks panels.
- **Cache:** JSONL run cache, matching the reference project's `--cache`
  convention.
- **CI:** GitHub Actions running pytest, including a dedicated
  look-ahead-bias test suite.

## What's reused vs. what's new
**Reused near-verbatim** from L1vsun/JEV-Trading-BOT: the Jev client
(auth, retries on 429/529, model pinning), the cache/replay system, CLI
scaffolding, the confidence-gated sizing mechanism, and the self-consistency
pattern (asking the action question multiple ways and requiring agreement).

**Built from scratch, no reuse:** SMC structure detection (swing pivots,
BOS/CHoCH, order blocks, fair value gaps, liquidity sweeps, premium/discount
zone), the options quant engine (Greeks, IV rank), the state builder, the
question set, and the validation harness. The reference project's own quant
code (order-flow imbalance, VPIN, Kyle's lambda) is not reused anywhere.

## Decisions already made — do not re-ask, do not reopen
These were settled by the user. Build to them. If one turns out to be
impractical, stop and explain why before changing anything.

### 1. Underlying and data provider (Phase 0)
- **Underlying:** SPY.
- **Provider:** Massive (formerly Polygon.io) for **both** the backtest and
  the paper loop, so both use one data schema. Do not add a second provider
  for historical data without asking.
- **Before committing to a paid plan:** do a small pull (a few days) to
  confirm historical options *quotes* and IV are available at past
  timestamps. Stock-plan pricing does not cover options, so check the
  options plan specifically.
- **IV and Greeks are computed by us from quotes**, using the same function
  in backtest and live. Do not rely on provider-supplied Greeks or IV.
- **SPY options are American-style.** Short legs carry early-assignment
  risk. Either close spreads before expiry and ex-dividend dates, or make
  the fill model account for it explicitly. Never ignore it silently.

### 2. Timeframe
- **1h candles drive structure detection, with a daily bias.** This suits
  30–45 DTE monthlies and keeps Jev call volume low.
- Decision ticks happen **once per hour, on candle close**. A sub-second or
  seconds-level loop is not a goal for the decision layer, so don't build
  toward one.
- Expect only tens to low hundreds of independent setups per year. Backtests
  need multi-year history, and every EV figure must be reported with a
  confidence interval, not as a bare point estimate.

### 3. What "sell premium" means in the paper executor
- **Defined-risk credit spreads only**, oriented by `structure_bias`:
  bullish → put credit spread, bearish → call credit spread.
- **Ranging → no trade** for now. Iron condors and single short options are
  out of scope until the user says otherwise.
- **Fill model must be conservative** (worse than mid) and **identical** in
  backtest and paper. Spread costs are where options backtests quietly fail.
- **Exit rules are fixed before the first backtest run** and are never tuned
  against backtest results.

## Non-negotiables — do not change these to "optimize" or "simplify"
1. **Paper trading only.** Never add live order placement, broker execution,
   or any code path that could touch real money. If a task seems to require
   broker connectivity, stop and ask the user first — do not add it
   speculatively "for later."
2. **One implementation only.** Do not add a second implementation of
   structure-detection or Greeks logic in a compiled language or JIT path
   (Rust, numba, Cython, etc.), even if it would be faster. The backtester and
   the live loop must call the *exact same* Python functions. Two
   implementations of the same logic can silently drift apart and invalidate
   what the backtest was supposed to prove — this was a deliberate trade-off
   of some raw speed for that guarantee, not an oversight.
3. **No streaming data feed.** Data fetching stays request/response (batched
   + concurrent via `asyncio`), not a persistent WebSocket stream. Also a
   deliberate, discussed scope decision, not a gap to fill in.
4. **Never weaken the look-ahead-bias test suite to make a feature pass.**
   SMC patterns (an order block, a fair value gap) are partly defined by what
   price did *afterward*, which makes them easy to detect with hindsight and
   easy to get wrong causally. Detection code must only use information
   available at the moment of the decision. If a new detector fails this
   test, the detector is wrong — not the test.
5. **Jev never detects patterns from raw price data.** Pattern detection
   (pivots, BOS/CHoCH, order blocks, FVGs, sweeps) is deterministic code,
   always. Jev only receives already-detected, named-bucket state and judges
   confluence, context, direction, and confidence over it — never raw candles.
6. **Preserve the Jev materiality filter.** Never remove the "only call Jev
   when a symbol's state has changed meaningfully since it was last judged"
   gate. It protects both API cost and rate-limit exposure as the number of
   tracked underlyings grows — removing it to "simplify the loop" reintroduces
   both problems at once.
7. **The success metric is expected value and maximum drawdown — never hit
   rate.** Do not report, optimize toward, or headline a win-rate percentage
   as evidence that a change is an improvement. A high hit rate with poor EV
   is not progress here.

## Code quality rules
- Vectorize. If you find yourself writing a Python `for` loop over contracts
  or candles for something numeric, it almost certainly should be a
  numpy/pandas array operation instead.
- State sent to Jev includes only the fields the current question set
  actually needs — don't pass a full record "just in case," it measurably
  hurts Jev's accuracy on the fields that matter.
- New Jev questions follow the existing typed pattern (Choice / Score / Noul),
  batched into one call per decision tick. Never ask Jev to generate free
  text — it can't do this reliably and it isn't what it's for.

## Testing requirements
- Every new structure detector (order block, FVG, sweep, etc.) needs a
  corresponding look-ahead-bias test before it's considered done.
- Backtests must be walk-forward and strictly out-of-sample. In-sample
  results alone are never sufficient evidence a change helped.
- Every backtest report must include expected value and maximum drawdown,
  not just cumulative PnL or a win/loss count.

## Ask before doing, don't just do
- Adding any new external data provider or paid API key.
- Changing the underlying, the candle timeframe, or the spread type
  (see "Decisions already made").
- Changing confidence thresholds or composite-scoring weights in the
  decision logic.
- Anything that expands the bot's capability toward live execution, even
  behind a flag.

## Disclaimer
This is a research and education project. Nothing it produces is financial
advice, and no backtested or paper-traded result implies anything about
real or live-market performance.
