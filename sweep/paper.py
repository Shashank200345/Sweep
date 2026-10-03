"""The hourly paper-trading loop. PAPER ONLY: there is no broker code anywhere in Sweep.

Once per 1h bar close (+ ``bar_close_delay_s``, so the bar and the fill-time quotes
exist) it pulls data by request/response (no streaming), rebuilds the features from a
trailing window with the SAME ``build_features`` the backtester uses, and hands the
last bar to the SAME ``Trader.on_bar``. So decisions, fills and exit rules are the
backtester's, by construction; ``tests/test_paper.py`` proves it on identical data.

Recompute only on change:
  * structure is recomputed only when a new hourly bar has closed (one tick per bar);
  * the daily ATM IV series is cached on disk; only newly completed sessions are priced;
  * the Jev materiality filter skips the call when the bucketed state is unchanged;
  * position Greeks (display only) are re-priced only after SPY moves more than
    ``greeks_recompute_move``.

State (cash, open position, closed trades, materiality memory, last tick) is saved
after every tick, so the loop resumes where it stopped after a restart.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT, MULTIPLIER, Config
from .data.bars import hourly_bars, spot_at
from .data.market import ET, YEAR_S, MarketData, expiry_close_utc
from .engine import Trader, _session_closes, build_features
from .execution.positions import Position, Trade
from .options import bsm
from .options.chain import daily_atm_iv
from .options.shortlist import Leg
from .typesafe import Answer, Response, SystemOne

CLOSE_MINUTES = [10 * 60 + 30, 11 * 60 + 30, 12 * 60 + 30, 13 * 60 + 30, 14 * 60 + 30, 15 * 60 + 30, 16 * 60]


# ---------------------------------------------------------------------- (de)serialization
def _leg_from(d: dict) -> Leg:
    return Leg(d["ticker"], d["type"], float(d["strike"]), pd.Timestamp(d["expiry"]), int(d["qty"]), float(d["bid"]),
               float(d["ask"]), pd.Timestamp(d["quote_ts"]), float(d["iv"]), float(d["delta"]), float(d["gamma"]),
               float(d["theta"]), float(d["vega"]))


def position_to_json(p: Position | None) -> dict | None:
    if p is None:
        return None
    d = {k: v for k, v in asdict(p).items() if k != "legs"}
    d["legs"] = [l.to_json() for l in p.legs]
    d["entry_t"] = str(p.entry_t)
    return d


def position_from_json(d: dict | None) -> Position | None:
    if d is None:
        return None
    d = dict(d)
    legs = [_leg_from(x) for x in d.pop("legs")]
    d["entry_t"] = pd.Timestamp(d["entry_t"])
    return Position(legs=legs, **d)


def response_to_json(r: Response) -> dict:
    return {"model": r.model, "answers": {k: a.to_json() for k, a in r.answers.items()}}


def response_from_json(d: dict) -> Response:
    return Response(d["model"], {k: Answer.from_json(v) for k, v in d["answers"].items()})


# ---------------------------------------------------------------------- clock
def bar_closes(day: pd.Timestamp) -> list[pd.Timestamp]:
    """Scheduled 1h bar closes of a weekday session, UTC (holidays simply produce no bar)."""
    d = pd.Timestamp(day).normalize()
    return [(d + pd.Timedelta(minutes=m)).tz_localize(ET).tz_convert("UTC") for m in CLOSE_MINUTES]


def next_close_after(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t).tz_convert("UTC")
    day = t.tz_convert(ET).normalize().tz_localize(None)
    for k in range(10):
        d = day + pd.Timedelta(days=k)
        if d.weekday() >= 5:
            continue
        for c in bar_closes(d):
            if c > t:
                return c
    raise RuntimeError("no bar close found")


def last_close_before(t: pd.Timestamp) -> pd.Timestamp:
    """Scheduled 1h bar close strictly before or at t, UTC."""
    t = pd.Timestamp(t).tz_convert("UTC")
    day = t.tz_convert(ET).normalize().tz_localize(None)
    for k in range(10):
        d = day - pd.Timedelta(days=k)
        if d.weekday() >= 5:
            continue
        for c in reversed(bar_closes(d)):
            if c <= t:
                return c
    raise RuntimeError("no prior bar close found")


# ---------------------------------------------------------------------- loop
class PaperLoop:
    def __init__(self, md: MarketData, jev: SystemOne, state_dir: Path, cfg: Config = DEFAULT, verbose: bool = False,
                 start: pd.Timestamp | None = None):
        self.md, self.jev, self.cfg, self.verbose = md, jev, cfg, verbose
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.iv_path = self.dir / "daily_iv.parquet"
        self.start = start
        self._div_day = None
        self._div = None
        self._greeks_spot = None
        self.trader = Trader(md, jev, self._dividends(pd.Timestamp.now(tz="UTC")), cfg)
        self.last_tick: pd.Timestamp | None = None
        self._restore()

    # -------------------------------------------------------------- persistence
    def _restore(self) -> None:
        if not self.state_path.exists():
            return
        s = json.loads(self.state_path.read_text(encoding="utf-8"))
        tr = self.trader
        tr.cash = s["cash"]
        tr.position = position_from_json(s["position"])
        tr.trades = [Trade(**t) for t in s["trades"]]
        tr.bar_count = s["bar_count"]
        tr.materiality.calls, tr.materiality.skipped = s["jev_calls"], s["jev_skipped"]
        tr.materiality.last = {k: (fp, response_from_json(r)) for k, (fp, r) in s["materiality"].items()}
        self.last_tick = pd.Timestamp(s["last_tick"]) if s["last_tick"] else None

    def _save(self) -> None:
        tr = self.trader
        s = {"cash": tr.cash, "position": position_to_json(tr.position), "trades": [t.to_json() for t in tr.trades],
             "bar_count": tr.bar_count, "jev_calls": tr.materiality.calls, "jev_skipped": tr.materiality.skipped,
             "materiality": {k: [fp, response_to_json(r)] for k, (fp, r) in tr.materiality.last.items()},
             "last_tick": str(self.last_tick) if self.last_tick is not None else None,
             "note": "paper trading only"}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    def _append_logs(self) -> None:
        tr = self.trader
        with (self.dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
            for d in tr.decisions:
                fh.write(json.dumps(d, default=str) + "\n")
        with (self.dir / "equity.jsonl").open("a", encoding="utf-8") as fh:
            for t, e in tr.equity:
                fh.write(json.dumps({"t": str(t), "equity": round(e, 2)}) + "\n")
        tr.decisions.clear()
        tr.equity.clear()

    # -------------------------------------------------------------- data
    def _dividends(self, t: pd.Timestamp) -> pd.DataFrame:
        day = pd.Timestamp(t).tz_convert(ET).date()
        if self._div_day != day:
            self._div, self._div_day = self.md.dividends(), day
        return self._div

    def _daily_iv(self, minutes: pd.DataFrame, h1: pd.DataFrame, div: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
        cached = pd.read_parquet(self.iv_path) if self.iv_path.exists() else None
        closes = _session_closes(h1, t)
        have = cached.index if cached is not None else pd.DatetimeIndex([], tz="UTC")
        missing = closes[~closes.isin(have)]
        if len(missing):
            new = daily_atm_iv(self.md, missing, spot_at(minutes, missing), div, self.cfg)
            cached = new if cached is None else pd.concat([cached, new]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            cached.to_parquet(self.iv_path)
        if cached is None:
            return pd.DataFrame({"atm_iv": pd.Series(dtype=float)}, index=pd.DatetimeIndex([], tz="UTC", name="close_time"))
        return cached[cached.index.isin(closes)]

    # -------------------------------------------------------------- one tick
    def tick(self, t: pd.Timestamp) -> dict | None:
        """Process the bar that closed at ``t``. Returns the decision log, or None if there was no bar."""
        cfg = self.cfg
        t = pd.Timestamp(t).tz_convert("UTC")
        end = t.tz_convert(ET).date()
        minutes = self.md.minute_bars(end - timedelta(days=cfg.paper.history_days), end)
        minutes = minutes[minutes.index + pd.Timedelta(minutes=1) <= t]          # closed minutes only
        h1 = hourly_bars(minutes, asof=t)
        if h1.empty or h1.index[-1] != t:
            self.last_tick = t
            self._save()
            return None
        div = self._dividends(t)
        self.trader.div = div
        feats = build_features(self.md, minutes, div, cfg, daily_iv=self._daily_iv(minutes, h1, div, t), asof=t)
        log = self.trader.on_bar(t, feats.buckets.iloc[-1], feats.ivv.iloc[-1], feats.spots[-1])
        g = self._maybe_greeks(t, feats.spots[-1])
        if g:
            log["position_greeks"] = g
        self.last_tick = t
        self._append_logs()
        self._save()
        try:
            from .replay import _structure_payload
            st_data = _structure_payload(feats.h1, feats.structure)
            (self.dir / "structure.json").write_text(json.dumps(st_data, default=str), encoding="utf-8")
        except Exception:
            pass
        if self.verbose or log.get("execution"):
            et = t.tz_convert(ET).strftime("%Y-%m-%d %H:%M")
            called = "Jev" if log.get("jev_called") else "reused"
            print(f"{et} ET  {log.get('action', '?'):5} [{called}] {log.get('reason', '')}"
                  + (f"  -> {log['execution']}" if log.get("execution") else "") + f"  equity ${log['equity']:,.2f}", flush=True)
        return log

    def _maybe_greeks(self, t: pd.Timestamp, spot: float) -> dict | None:
        pos = self.trader.position
        if pos is None:
            self._greeks_spot = None
            return None
        if self._greeks_spot is not None and abs(spot / self._greeks_spot - 1) <= self.cfg.paper.greeks_recompute_move:
            return None
        self._greeks_spot = spot
        tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in pos.legs:
            T = float((expiry_close_utc(pd.Series([leg.expiry])).asi8[0] - t.value) / 1e9 / YEAR_S)
            gk = bsm.greeks(leg.type == "call", spot, leg.strike, max(T, 1e-6), self.cfg.options.risk_free, 0.013, leg.iv)
            m = leg.qty * pos.contracts * MULTIPLIER
            tot = {"delta": tot["delta"] + float(gk["delta"]) * m, "gamma": tot["gamma"] + float(gk["gamma"]) * m,
                   "theta": tot["theta"] + float(bsm.per_day(gk["theta"])) * m,
                   "vega": tot["vega"] + float(bsm.per_vol_point(gk["vega"])) * m}
        return {k: round(v, 3) for k, v in tot.items()}

    # -------------------------------------------------------------- run
    def run(self, max_ticks: int | None = None, fake_clock: bool = False) -> None:
        print(f"paper trading (paper only) · data {self.md.label} · model {getattr(self.jev, 'model', '?')} · "
              f"state {self.dir} · Ctrl-C to stop", flush=True)
        n = 0
        for t in (self._fake_ticks() if fake_clock else self._real_ticks()):
            self.tick(t)
            n += 1
            if max_ticks is not None and n >= max_ticks:
                break
        tr = self.trader
        print(f"stopped · cash ${tr.cash:,.2f} · {len(tr.trades)} closed trades · "
              f"{'open ' + tr.position.kind if tr.position else 'flat'} · Jev calls {tr.materiality.calls}, "
              f"skipped {tr.materiality.skipped}", flush=True)

    def _fake_ticks(self):
        """Replay the data's own bar closes with no waiting (fixture / recorded data)."""
        days = getattr(self.md, "_m", {}).get("days")
        last_day = days[-1].date() if days is not None else pd.Timestamp.now(tz=ET).date()
        first_day = self.start.date() if self.start is not None else (days[-30].date() if days is not None else last_day)
        mins = self.md.minute_bars(first_day, last_day)
        for t in hourly_bars(mins).index:
            if self.last_tick is not None and t <= self.last_tick:
                continue
            if self.start is not None and t < self.start:
                continue
            yield t

    def _real_ticks(self):
        now = pd.Timestamp.now(tz="UTC")
        if self.last_tick is None:
            try:
                prev = last_close_before(now)
                if now - prev < pd.Timedelta(days=1):
                    print(f"· evaluating latest completed 1h bar ({prev.tz_convert(ET).strftime('%Y-%m-%d %H:%M')} ET)...", flush=True)
                    yield prev
            except Exception:
                pass
        t = self.last_tick if self.last_tick is not None and self.last_tick > now - pd.Timedelta(days=5) else now
        while True:
            t = next_close_after(t)
            due = t + pd.Timedelta(seconds=self.cfg.paper.bar_close_delay_s)
            wait = (due - pd.Timestamp.now(tz="UTC")).total_seconds()
            if wait > 0:
                print(f"· waiting for next 1h bar close at {t.tz_convert(ET).strftime('%Y-%m-%d %H:%M')} ET ({wait / 60:.1f} min remaining)...", flush=True)
                time.sleep(wait)
            yield t
