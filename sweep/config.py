"""Every fixed parameter in one place.

The values marked APPROVED were signed off by the user before the first
backtest. Changing any of them requires asking the user first (AGENTS.md:
"Ask before doing"). They must never be tuned against backtest results.

Values marked DETECTION are geometric details of the SMC detectors that the
approved plan left to implementation; they are also frozen before the first
backtest and must not be tuned on backtest results either.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNDERLYING = "SPY"          # APPROVED (AGENTS.md "Decisions already made")
MULTIPLIER = 100            # shares per contract


@dataclass(frozen=True)
class StructureConfig:
    h1_left: int = 3                 # APPROVED: 1h pivots 3 left / 3 right
    h1_right: int = 3
    d1_left: int = 2                 # APPROVED: daily pivots 2 / 2
    d1_right: int = 2
    atr_window: int = 14             # DETECTION: causal ATR length
    equal_tol_atr: float = 0.1       # APPROVED: equal highs/lows within 0.1 x ATR
    sweep_reclaim_bars: int = 3      # DETECTION: a sweep must close back inside within k bars
    ob_lookback: int = 10            # DETECTION: bars searched back from a break for the order block
    fvg_min_atr: float = 0.1         # DETECTION: ignore gaps smaller than 0.1 x ATR
    zone_max_age: int = 140          # DETECTION: zones expire after ~20 trading days of 1h bars
    max_active_zones: int = 10       # DETECTION: cap on active zones kept per side for display/state
    recent_bars: int = 7             # DETECTION: "recent" = within one trading day of 1h bars


@dataclass(frozen=True)
class OptionsConfig:
    dte_min: int = 30                # APPROVED: standard monthly, 30-45 DTE at entry
    dte_max: int = 45
    long_delta: float = 0.40         # APPROVED: long calls/puts ~0.40 |delta|
    short_delta: float = 0.30        # APPROVED: credit spread short leg ~0.30 |delta|
    spread_width: float = 5.0        # APPROVED: $5 wide
    risk_free: float = 0.04          # APPROVED: configurable constant
    iv_rank_window: int = 252        # APPROVED plan: IV rank / percentile over 252 trading days
    iv_rank_min_history: int = 252   # warm-up: IV rank is NaN ("insufficient history") before this
    atm_iv_target_days: int = 30     # 30-day ATM IV (interpolated in total variance)
    candidate_strikes: int = 5       # strikes per leg pre-selected by estimated delta before quoting
    dividend_known_lead_days: int = 14  # if declaration_date is missing, treat as known 14 days before ex


@dataclass(frozen=True)
class SizingConfig:
    starting_equity: float = 100_000.0  # APPROVED (configurable)
    full_risk: float = 0.02          # APPROVED: max loss 2% of equity at full size
    half_risk: float = 0.01          # APPROVED: 1% at half size


@dataclass(frozen=True)
class ExitConfig:
    long_take_profit: float = 1.00   # APPROVED: +100% of premium
    long_stop: float = 0.50          # APPROVED: -50% of premium
    close_dte: int = 21              # APPROVED: close at 21 DTE
    exit_noul: float = 0.65          # APPROVED: exit_now >= 0.65
    spread_take_profit: float = 0.50  # APPROVED: 50% of credit captured
    spread_stop_mult: float = 2.0    # APPROVED: stop at loss = 2 x credit


@dataclass(frozen=True)
class FillConfig:
    slippage_frac: float = 0.5       # APPROVED: halfway from mid to the far side
    commission: float = 0.65         # APPROVED: $ per contract per leg per side
    max_quote_age_s: float = 300.0   # APPROVED: no fill if quote older than 5 minutes
    max_spread_frac: float = 0.10    # APPROVED: no fill if leg spread > 10% of mid
    latency_s: float = 60.0          # decision at bar close, fill 60 s later


@dataclass(frozen=True)
class PolicyConfig:
    min_action_conf: float = 0.35    # APPROVED: half size
    full_size_conf: float = 0.65     # APPROVED: full size
    min_setup: float = 2.0           # APPROVED: setup_quality >= 2 (0..4)
    min_composite: float = 0.30      # APPROVED
    exit_noul: float = 0.65          # APPROVED
    credit_spreads_only: bool = False # Defined-risk credit spreads only (AGENTS.md §3)
    require_ob_near: bool = False     # Require price inside or within 1 ATR of order block
    weights: dict = field(default_factory=lambda: {   # APPROVED composite weights
        "action": 0.40, "structure": 0.20, "liquidity": 0.10,
        "order_block": 0.10, "fvg": 0.10, "zone": 0.10})


@dataclass(frozen=True)
class StateConfig:
    # STATE BUCKETS: edges that turn numbers into the words Jev reads. Fixed before
    # the first backtest like everything else; never tuned on backtest results.
    iv_low_pct: float = 0.30         # IV percentile below this -> "low"
    iv_high_pct: float = 0.70        # above this -> "elevated"
    near_atr: float = 1.0            # zone "within 1 ATR"
    mid_atr: float = 3.0             # "1 to 3 ATR away"
    recent_bars: int = 3             # event "within the last 3 hours"
    day_bars: int = 7                # "within the last day"
    days3_bars: int = 21             # "1 to 3 days ago"
    pnl_flat_frac: float = 0.10      # |P/L| < 10% of max loss -> "about flat"


@dataclass(frozen=True)
class PaperConfig:
    bar_close_delay_s: float = 90.0  # wait after each 1h close before fetching
    greeks_recompute_move: float = 0.002  # recompute display Greeks when SPY moved > 0.2%
    history_days: int = 500          # calendar days of history loaded for detectors


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 12           # window available to a (currently frozen) fit step
    test_months: int = 6             # non-overlapping out-of-sample folds
    bootstrap: int = 10_000
    ci: float = 0.95
    n_random: int = 100              # random-entry benchmark traders
    seed: int = 1


@dataclass(frozen=True)
class Config:
    structure: StructureConfig = field(default_factory=StructureConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    fills: FillConfig = field(default_factory=FillConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    state: StateConfig = field(default_factory=StateConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


DEFAULT = Config()
