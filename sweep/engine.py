"""The event-driven engine: one hourly bar at a time.

``Trader.on_bar`` is the ONE per-tick routine. The backtester calls it for every
historical bar; the paper loop calls it once per live bar close. Per bar:

  1. mark the open position from quotes at the bar close; apply the fixed exit rules
     (take profit / stop on the executable close value, 21 DTE, ex-dividend);
  2. build the named-bucket state (incl. the bucketed position) and pass it through
     the materiality filter -- Jev is only called when the state changed;
  3. decide (composite, gates, self-consistency, confidence-gated size);
  4. exits and entries fill ``latency_s`` after the bar close with the shared
     conservative fill model, using quotes at the fill time and never later.

Path dependence (theta along the path, stops, forced pre-ex-dividend closes) is why
this is event-driven and not vectorized. Structure features are precomputed with
the vectorized, causal detectors; nothing here reads a bar after the current one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from . import FIXTURE_LABEL
from .config import DEFAULT, MULTIPLIER, UNDERLYING, Config
from .data.bars import hourly_bars, spot_at
from .data.market import ET, MarketData
from .decide import Decision, decide
from .execution.fills import commission, fill, last_resort, leg_prices, quote_ok
from .execution.positions import Position, Trade, net_value
from .execution.rules import dte, dte_rule, ex_dividend_day_before, ex_dividend_rule, price_rule
from .materiality import MaterialityFilter
from .options.chain import daily_atm_iv, next_dividend, session_of
from .options.ivrank import hourly_view
from .options.shortlist import shortlist
from .questions import decision_questions
from .smc.engine import Structure, detect
from .state import bucket_frame, position_text, state_at
from .typesafe import JevError, SystemOne

WARMUP_DAYS = 400          # calendar days of history before the first decision (IV rank needs 252 sessions)


# ---------------------------------------------------------------------- features
@dataclass
class Features:
    h1: pd.DataFrame
    structure: Structure
    ivv: pd.DataFrame
    buckets: pd.DataFrame
    spots: np.ndarray
    minutes: pd.DataFrame


def build_features(md: MarketData, minutes: pd.DataFrame, div: pd.DataFrame, cfg: Config = DEFAULT,
                   daily_iv: pd.DataFrame | None = None, asof: pd.Timestamp | None = None) -> Features:
    """Everything the decision layer reads, from data up to ``asof`` (the same call in backtest and paper)."""
    h1 = hourly_bars(minutes, asof=asof)
    s = detect(h1, cfg.structure)
    if daily_iv is None:
        closes = _session_closes(h1, asof)
        daily_iv = daily_atm_iv(md, closes, spot_at(minutes, closes), div, cfg)
    ivv = hourly_view(daily_iv, h1.index, cfg.options.iv_rank_window, cfg.options.iv_rank_min_history)
    return Features(h1, s, ivv, bucket_frame(s.frame, ivv, cfg), h1["close"].to_numpy(float), minutes)


def _session_closes(h1: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """The last hourly close of each COMPLETE session (where that session's ATM IV becomes known).

    With ``asof`` (the paper loop), the session in progress only counts once its 16:00 bar
    has closed. (On an early-close day the paper loop therefore learns that day's IV at
    the next morning's first tick, one bar later than the backtest -- a documented gap.)
    """
    last = ~h1["session"].duplicated(keep="last").to_numpy()
    closes = h1.index[last]
    if asof is not None and len(closes):
        et = closes[-1].tz_convert(ET)
        today = session_of(pd.Timestamp(asof))
        if pd.Timestamp(h1["session"].iloc[-1]) == today and et.hour * 60 + et.minute < 16 * 60:
            closes = closes[:-1]
    return closes


# ---------------------------------------------------------------------- trader
@dataclass
class Trader:
    md: MarketData
    jev: SystemOne
    div: pd.DataFrame
    cfg: Config = DEFAULT
    symbol: str = UNDERLYING
    cash: float = field(default=None)
    position: Position | None = None
    trades: list[Trade] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    equity: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    materiality: MaterialityFilter = field(default_factory=MaterialityFilter)
    bar_count: int = 0
    jev_errors: int = 0

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.cfg.sizing.starting_equity
        self.questions = decision_questions()

    # -------------------------------------------------------------- per bar
    def on_bar(self, t: pd.Timestamp, buckets_row: pd.Series, iv_row: pd.Series, spot: float) -> dict:
        cfg = self.cfg
        t = pd.Timestamp(t)
        fill_at = t + pd.Timedelta(seconds=cfg.fills.latency_s)
        log: dict = {"t": str(t), "spot": round(float(spot), 4)}
        exited = False
        pos = self.position
        if pos is not None:
            assert session_of(t) < pos.expiry, f"position {pos.tickers} reached its expiry date -- exit rules failed"
            reason, forced, deadline = self._rule_exit(pos, t, spot, log)
            if reason:
                exited = self._close(fill_at, reason, allow_last_resort=forced and deadline, log=log)
        # ---- state -> materiality -> Jev -> decision
        pos = self.position
        st = state_at(buckets_row, self._position_text(pos))
        try:
            r, called = self.materiality.judge(self.symbol, st, lambda s: self.jev.system_one(s, self.questions))
        except JevError as e:
            self.jev_errors += 1
            log.update({"action": "hold", "reason": f"Jev error: {e}", "state": st})
            self._record(t, log)
            return log
        d = decide(str(t), r, pos.direction if pos else 0, cfg.policy, state=st)
        d.jev_called = called
        log.update(d.to_json())
        log["state"] = st
        if d.action == "exit" and pos is not None:
            self._close(fill_at, d.reason, allow_last_resort=False, log=log)
        elif d.action == "enter" and pos is None and not exited:
            self._open(d, t, fill_at, spot, iv_row, log)
        self._record(t, log)
        return log

    # -------------------------------------------------------------- helpers
    def _position_text(self, pos: Position | None) -> str:
        if pos is None:
            return "flat"
        upnl = pos.pnl_at(pos.mark) if np.isfinite(pos.mark) else 0.0
        return position_text(pos.kind, self.bar_count - pos.entry_bar, upnl / pos.max_loss, self.cfg)

    def _rule_exit(self, pos: Position, t: pd.Timestamp, spot: float, log: dict) -> tuple[str | None, bool, bool]:
        q = self.md.quotes(pos.tickers, t).reindex(pos.tickers)
        bid, ask = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
        two = np.isfinite(bid) & np.isfinite(ask) & (ask >= bid) & (bid >= 0)
        if two.all():
            pos.mark = net_value(pos.qty, (bid + ask) / 2)
        ok = quote_ok(bid, ask, q["ts"], t, self.cfg.fills)
        close_value = net_value(pos.qty, leg_prices(bid, ask, pos.qty < 0, self.cfg.fills)) if ok.all() else float("nan")
        log["position_mark"] = None if not np.isfinite(pos.mark) else round(pos.mark, 4)
        ex_date, amount = next_dividend(self.div, t, self.cfg)
        short_mid = float("nan")
        if pos.kind == "call_credit_spread":
            i = int(np.flatnonzero(pos.qty < 0)[0])
            short_mid = (bid[i] + ask[i]) / 2 if two[i] else float("nan")
        et = t.tz_convert(ET)
        last_bar_of_day = et.hour * 60 + et.minute >= 16 * 60
        exd = ex_dividend_rule(pos, t, spot, short_mid, ex_date, amount)
        if exd:
            return exd, True, last_bar_of_day
        pr = price_rule(pos, close_value, self.cfg.exits)
        if pr:
            return pr, False, False
        dr = dte_rule(pos, t, self.cfg.exits)
        if dr:
            return dr, True, last_bar_of_day and dte(pos, t) <= 1
        return None, False, False

    def _open(self, d: Decision, t: pd.Timestamp, fill_at: pd.Timestamp, spot: float, iv_row: pd.Series, log: dict) -> None:
        cfg = self.cfg
        st = shortlist(self.md, d.structure, t, spot, self.div, float(iv_row.get("atm_iv", np.nan)), cfg)
        if st is None:
            log["execution"] = "no candidate contract (no 30-45 DTE monthly or no valid quotes)"
            return
        tickers = [l.ticker for l in st.legs]
        qty = np.array([l.qty for l in st.legs])
        f = fill(tickers, qty > 0, self.md.quotes(tickers, fill_at), fill_at, cfg.fills)
        if not f.ok:
            log["execution"] = f"not filled: {f.reason}"
            return
        entry = net_value(qty, f.prices)
        comm_rt = 2 * cfg.fills.commission * len(tickers)          # per contract, open + close
        if len(tickers) == 2:
            credit = -entry
            if credit <= 0:
                log["execution"] = "not filled: spread would not collect a credit"
                return
            loss_pc = (cfg.options.spread_width - credit) * MULTIPLIER
        else:
            loss_pc = entry * MULTIPLIER
        budget = self.cash * (cfg.sizing.full_risk if d.size >= 1.0 else cfg.sizing.half_risk)
        n = int(math.floor(budget / (loss_pc + comm_rt)))
        if n < 1:
            log["execution"] = "not filled: max loss of one contract exceeds the risk budget"
            return
        comm_in = commission(n, len(tickers), cfg.fills)
        self.position = Position(st.kind, st.legs, n, fill_at, self.bar_count, entry, list(map(float, f.prices)),
                                 comm_in, n * (loss_pc + comm_rt), d.size, d.reason, d.direction, mark=entry)
        log["execution"] = f"opened {n} x {st.kind} @ {entry:+.4f}/share"
        log["legs"] = [l.to_json() for l in st.legs]

    def _close(self, at: pd.Timestamp, reason: str, allow_last_resort: bool, log: dict) -> bool:
        pos = self.position
        buy = pos.qty < 0                                             # buy back shorts, sell longs
        q = self.md.quotes(pos.tickers, at)
        f = fill(pos.tickers, buy, q, at, self.cfg.fills)
        if not f.ok and allow_last_resort:
            f = last_resort(pos.tickers, buy, q, at)
        if not f.ok:
            log["execution"] = f"exit wanted ({reason}) but {f.reason}; retrying next bar"
            return False
        exit_value = net_value(pos.qty, f.prices)
        comm_out = commission(pos.contracts, len(pos.legs), self.cfg.fills)
        pnl = pos.pnl_at(exit_value, comm_out)
        self.cash += pnl
        self.trades.append(Trade(pos.kind, pos.direction, pos.contracts, pos.size, pos.tickers,
                                 [l.strike for l in pos.legs], str(pos.expiry.date()), str(pos.entry_t), str(at),
                                 pos.entry, exit_value, pnl, pos.max_loss, pos.commission_in + comm_out,
                                 self.bar_count - pos.entry_bar, pos.reason_in, reason, f.last_resort,
                                 [l.to_json() for l in pos.legs]))
        log["execution"] = f"closed ({reason}) @ {exit_value:+.4f}/share, P/L {pnl:+.2f}"
        self.position = None
        return True

    def _record(self, t: pd.Timestamp, log: dict) -> None:
        pos = self.position
        eq = self.cash + (pos.pnl_at(pos.mark) if pos is not None and np.isfinite(pos.mark) else 0.0)
        log["equity"] = round(eq, 2)
        self.equity.append((t, eq))
        self.decisions.append(log)
        self.bar_count += 1


# ---------------------------------------------------------------------- backtest
@dataclass
class Result:
    label: str
    model: str
    start: str
    end: str
    trades: list[Trade]
    decisions: list[dict]
    equity: pd.Series
    features: Features
    usage: dict
    jev_calls: int
    jev_skipped: int
    starting_equity: float

    @property
    def is_fixture(self) -> bool:
        return self.label == FIXTURE_LABEL


def load_features(md: MarketData, start, end, cfg: Config = DEFAULT) -> tuple[Features, pd.DataFrame]:
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    minutes = md.minute_bars(start - timedelta(days=WARMUP_DAYS), end)
    div = md.dividends()
    return build_features(md, minutes, div, cfg), div


def run(md: MarketData, jev: SystemOne, start, end, cfg: Config = DEFAULT, features: Features | None = None,
        div: pd.DataFrame | None = None, progress: bool = False) -> Result:
    if features is None:
        features, div = load_features(md, start, end, cfg)
    div = md.dividends() if div is None else div
    f = features
    sess = f.h1["session"]
    sel = np.flatnonzero((sess >= pd.Timestamp(start)).to_numpy() & (sess <= pd.Timestamp(end)).to_numpy())
    tr = Trader(md, jev, div, cfg)
    for k, i in enumerate(sel):                       # the event loop: one bar at a time
        tr.on_bar(f.h1.index[i], f.buckets.iloc[i], f.ivv.iloc[i], f.spots[i])
        if progress and k % 500 == 0:
            print(f"  bar {k}/{len(sel)} · {len(tr.trades)} trades · Jev calls {tr.materiality.calls}")
    if tr.position is not None and len(sel):          # close anything open at the end of the window
        last = f.h1.index[sel[-1]] + pd.Timedelta(seconds=cfg.fills.latency_s)
        log: dict = {}
        if not tr._close(last, "end of backtest window", allow_last_resort=True, log=log):
            raise RuntimeError(f"could not close the final position: {log}")
    u = getattr(jev, "usage", None)
    usage = {} if u is None else {"calls": u.calls, "input_tokens": u.input_tokens, "cost_usd": round(u.cost_usd, 4),
                                  "p50_latency_ms": float(np.median(u.latency_ms)) if u.latency_ms else None}
    eq = pd.Series([e for _, e in tr.equity], index=pd.DatetimeIndex([t for t, _ in tr.equity]), dtype=float)
    if tr.trades and len(eq):
        eq.iloc[-1] = tr.cash
    return Result(md.label, getattr(jev, "model", "?"), str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()),
                  tr.trades, tr.decisions, eq, f, usage, tr.materiality.calls, tr.materiality.skipped,
                  cfg.sizing.starting_equity)
