"""Phase-0 probe: is the data good enough to backtest on, before paying for a plan?

For a few sample trading days, at every hourly close, it picks shortlist-like SPY
contracts (standard monthly expiry at 30-45 DTE; calls and puts at and out of the
money) and asks the data source for the last NBBO at or before that close. It then
reports how usable those quotes are for Sweep's fixed fill rules:

* share with a valid two-sided quote (bid > 0, ask > bid)
* share fresh enough to fill (age <= 5 minutes) and tight enough (spread <= 10% of mid)
* median quote age and median spread as % of mid
* whether quotes exist on older dates (how far back history goes for this key)

Massive's docs say historical option quotes need the Options Advanced plan and go back
to 2022-03-07; this probe is how you check what YOUR key actually returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT, Config
from .bars import hourly_bars, spot_at
from .market import MarketData, monthly_mask

MONEYNESS = {"call": (1.00, 1.02, 1.04), "put": (1.00, 0.98, 0.96)}
HISTORY_CHECKS = (date(2021, 6, 15), date(2022, 3, 15), date(2023, 3, 15), date(2024, 3, 15))


@dataclass
class ProbeResult:
    rows: pd.DataFrame          # one row per (tick, contract)
    history: dict[str, str]     # date -> "quotes found" / "no quotes" / "error: ..."
    label: str

    def summary(self, cfg: Config = DEFAULT) -> dict:
        r = self.rows
        if r.empty:
            return {"quotes_requested": 0}
        f = cfg.fills
        mid = (r["bid"] + r["ask"]) / 2
        two_sided = (r["bid"] > 0) & (r["ask"] > r["bid"])
        spread_pct = (r["ask"] - r["bid"]) / mid
        fresh = r["age_s"] <= f.max_quote_age_s
        tight = spread_pct <= f.max_spread_frac
        return {
            "quotes_requested": int(len(r)),
            "two_sided_pct": float(two_sided.mean() * 100),
            "fillable_pct": float((two_sided & fresh & tight).mean() * 100),
            "median_age_s": float(r.loc[two_sided, "age_s"].median()) if two_sided.any() else float("nan"),
            "median_spread_pct_of_mid": float(spread_pct[two_sided].median() * 100) if two_sided.any() else float("nan"),
            "ticks": int(r["tick"].nunique()),
        }


def run_probe(md: MarketData, days: list[date], cfg: Config = DEFAULT) -> ProbeResult:
    o = cfg.options
    out = []
    for d in days:                                   # a handful of days: one request batch each
        mins = md.minute_bars(d, d)
        if mins.empty:
            continue
        h1 = hourly_bars(mins)
        ticks = h1.index
        spots = spot_at(mins, ticks)
        # quote quality only: take the nearest monthly at >= dte_min even if it is past dte_max
        # (a 30-45 DTE monthly exists on only about half of trading days)
        ch = md.contracts(d, d + timedelta(days=o.dte_min), d + timedelta(days=o.dte_max + 35))
        ch = ch[monthly_mask(ch["expiry"])] if len(ch) else ch
        if ch.empty:
            continue
        expiry = ch["expiry"].min()
        ch = ch[ch["expiry"] == expiry]
        for t, s in zip(ticks, spots):
            picks = []
            for kind, ms in MONEYNESS.items():
                k = ch[ch["type"] == kind]
                if k.empty:
                    continue
                strikes = k["strike"].to_numpy()
                idx = np.abs(strikes[None, :] - s * np.asarray(ms)[:, None]).argmin(axis=1)
                picks += k.iloc[np.unique(idx)]["ticker"].tolist()
            q = md.quotes(picks, t)
            q = q.assign(tick=t, spot=s, age_s=(t - q["ts"]).dt.total_seconds())
            out.append(q.reset_index())
    rows = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ticker", "bid", "ask", "ts", "tick", "age_s"])
    return ProbeResult(rows, history_depth(md, cfg), md.label)


def history_depth(md: MarketData, cfg: Config = DEFAULT) -> dict[str, str]:
    """One quote request at 15:30 ET on each of a few old dates, for a near-the-money monthly."""
    o = cfg.options
    res = {}
    for d in HISTORY_CHECKS:
        try:
            mins = md.minute_bars(d, d)
            if mins.empty:
                res[str(d)] = "no underlying bars"
                continue
            t = pd.Timestamp(f"{d} 15:30", tz="America/New_York").tz_convert("UTC")
            s = spot_at(mins, pd.DatetimeIndex([t]))[0]
            ch = md.contracts(d, d + timedelta(days=o.dte_min), d + timedelta(days=o.dte_max + 35))
            ch = ch[monthly_mask(ch["expiry"])] if len(ch) else ch
            if ch.empty:
                res[str(d)] = "no contracts"
                continue
            ch = ch[ch["expiry"] == ch["expiry"].min()]
            c = ch.iloc[[np.abs(ch["strike"].to_numpy() - s).argmin()]]
            q = md.quotes(c["ticker"].tolist(), t)
            res[str(d)] = "quotes found" if np.isfinite(q["bid"].iloc[0]) else "no quotes"
        except Exception as e:                       # plan limits show up as HTTP errors
            res[str(d)] = f"error: {str(e)[:120]}"
    return res


def report(p: ProbeResult, cfg: Config = DEFAULT) -> str:
    s = p.summary(cfg)
    lines = [f"# Sweep Phase-0 data probe", "", f"source: {p.label}", ""]
    if not s.get("quotes_requested"):
        lines.append("No quotes could be requested (no bars or no monthly contracts on the sample days).")
    else:
        lines += ["| metric | value |", "|---|---|",
                  f"| hourly ticks sampled | {s['ticks']} |",
                  f"| contract quotes requested | {s['quotes_requested']} |",
                  f"| two-sided quote | {s['two_sided_pct']:.1f}% |",
                  f"| fillable under Sweep's rules (fresh <= {cfg.fills.max_quote_age_s:.0f} s, spread <= "
                  f"{cfg.fills.max_spread_frac:.0%} of mid) | {s['fillable_pct']:.1f}% |",
                  f"| median quote age | {s['median_age_s']:.1f} s |",
                  f"| median spread | {s['median_spread_pct_of_mid']:.2f}% of mid |"]
    lines += ["", "## history depth (one near-the-money monthly at 15:30 ET)", "", "| date | result |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in p.history.items()]
    lines += ["", "A backtest needs multi-year history with most ticks fillable. If older dates show "
              "'no quotes' or errors, the plan on this key does not cover them."]
    return "\n".join(lines) + "\n"


def write_report(p: ProbeResult, out_dir: Path, cfg: Config = DEFAULT) -> str:
    text = report(p, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.md").write_text(text, encoding="utf-8")
    p.rows.to_csv(out_dir / "probe_quotes.csv", index=False)
    return text
