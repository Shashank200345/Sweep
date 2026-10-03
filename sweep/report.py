"""Backtest / walk-forward reports. Headline = EV with CI and max drawdown. No hit rate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import FIXTURE_LABEL
from .config import DEFAULT, Config
from .engine import Result
from .metrics import summarize

DISCLAIMER = ("Research and education only; paper trading. Nothing here is financial advice, and no backtested "
              "or paper-traded result implies anything about live performance.")


def result_metrics(res: Result, cfg: Config = DEFAULT) -> dict:
    pnl = np.array([t.pnl for t in res.trades])
    risk = np.array([t.max_loss for t in res.trades])
    return summarize(pnl, risk, res.equity.to_numpy() if len(res.equity) else [res.starting_equity], cfg.walkforward)


def header(res_label: str, model: str) -> list[str]:
    lines = []
    if res_label == FIXTURE_LABEL:
        lines += ["> **FIXTURE DATA — NOT EVIDENCE.** Synthetic prices and quotes; nothing measured here says "
                  "anything about SPY or about the strategy.", ""]
    if model.startswith("surrogate"):
        lines += ["> Decisions by the offline surrogate, **not Jev**.", ""]
    return lines


def metrics_table(m: dict, bench: np.ndarray | None = None) -> list[str]:
    lines = ["| metric | value |", "|---|---|",
             f"| expected value per trade | {m['ev_per_trade'].fmt('$')} |",
             f"| expected value per $1 of max risk | {m['ev_per_dollar_risked'].fmt(' ', 3)} |",
             f"| maximum drawdown | ${m['max_drawdown']:,.2f} ({m['max_drawdown_pct'] * 100:.2f}% of peak equity) |",
             f"| trades | {m['trades']} |",
             f"| net P/L | ${m['net_pnl']:+,.2f} |"]
    if bench is not None and len(bench):
        b = bench[np.isfinite(bench)]
        if len(b):
            ev = m["ev_per_trade"].mean
            lines.append(f"| matched random-entry benchmark: mean EV per trade | ${b.mean():+,.2f} "
                         f"(strategy EV beats {np.mean(b < ev) * 100:.0f}% of {len(b)} random traders) |")
    return lines


def gate_funnel_summary(decisions: list[dict]) -> dict:
    from .decide import GATE_ORDER
    total_ticks = len(decisions)
    candidate_ticks = [d for d in decisions if d.get("candidate_setup")]
    n_cand = len(candidate_ticks)

    survived_counts = {g: 0 for g in GATE_ORDER}
    first_failed_counts = {g: 0 for g in GATE_ORDER}
    all_passed = 0

    for d in candidate_ticks:
        surv = d.get("survived_gates", 0)
        for i, g in enumerate(GATE_ORDER):
            if surv > i:
                survived_counts[g] += 1
        ff = d.get("first_failed_gate")
        if ff in first_failed_counts:
            first_failed_counts[ff] += 1
        elif surv == len(GATE_ORDER):
            all_passed += 1

    return {
        "total_ticks": total_ticks,
        "candidate_setups": n_cand,
        "survived_counts": survived_counts,
        "first_failed_counts": first_failed_counts,
        "all_passed": all_passed,
    }


def sub_signal_disagreement_summary(decisions: list[dict]) -> dict:
    directional = [d for d in decisions if d.get("action_sub_agree") is not None]
    n_dir = len(directional)
    if n_dir == 0:
        return {"total_directional": 0, "agreed": 0, "disagreed": 0, "disagreement_pct": 0.0}
    agreed = sum(1 for d in directional if d.get("action_sub_agree") is True)
    disagreed = sum(1 for d in directional if d.get("action_sub_agree") is False)
    disagreement_pct = (disagreed / n_dir) * 100.0 if n_dir > 0 else 0.0
    return {
        "total_directional": n_dir,
        "agreed": agreed,
        "disagreed": disagreed,
        "disagreement_pct": disagreement_pct,
    }


def write_backtest(res: Result, out: Path, cfg: Config = DEFAULT, bench: np.ndarray | None = None) -> str:
    m = result_metrics(res, cfg)
    lines = ["# Sweep backtest", ""] + header(res.label, res.model)
    lines += [f"window {res.start} → {res.end} · data: {res.label} · model `{res.model}`", ""]
    lines += metrics_table(m, bench)
    lines += [f"| Jev calls / skipped by the materiality filter | {res.jev_calls} / {res.jev_skipped} |"]
    if res.usage:
        u = res.usage
        lines.append(f"| API usage | {u['calls']} calls · {u['input_tokens']:,} input tokens · ${u['cost_usd']:.4f} |")
    last_resort = sum(t.last_resort for t in res.trades)
    if last_resort:
        lines.append(f"| exits that needed a last-resort far-side fill | {last_resort} |")

    funnel = gate_funnel_summary(res.decisions)
    disagree = sub_signal_disagreement_summary(res.decisions)
    from .decide import GATE_ORDER

    lines += ["", "## gate funnel diagnostic (Pending Decisions §1)", "",
              f"Candidate setups evaluated: {funnel['candidate_setups']} (out of {funnel['total_ticks']} total decision ticks)", "",
              "| gate (in sequence) | setups surviving | survival % | first to fail here |",
              "|---|---|---|---|"]
    n_c = max(1, funnel["candidate_setups"])
    for g in GATE_ORDER:
        surv = funnel["survived_counts"][g]
        failed_here = funnel["first_failed_counts"][g]
        pct = (surv / n_c) * 100.0 if funnel["candidate_setups"] > 0 else 0.0
        lines.append(f"| `{g}` | {surv} | {pct:.1f}% | {failed_here} |")
    all_pct = (funnel["all_passed"] / n_c) * 100.0 if funnel["candidate_setups"] > 0 else 0.0
    lines.append(f"| *all 8 gates passed* | {funnel['all_passed']} | {all_pct:.1f}% | — |")

    lines += ["", "## composite sub-signal agreement diagnostic (Pending Decisions §2)", "",
              "| metric | value |", "|---|---|",
              f"| decision ticks with directional action | {disagree['total_directional']} |",
              f"| action agrees with sub-signals | {disagree['agreed']} ({100.0 - disagree['disagreement_pct']:.1f}%) |",
              f"| action disagrees with sub-signals | {disagree['disagreed']} ({disagree['disagreement_pct']:.1f}%) |",
              f"| sub-signal disagreement rate | {disagree['disagreement_pct']:.2f}% |"]

    lines += ["", "In-sample single-window result. Walk-forward, out-of-sample folds (`sweep walkforward`) are the "
              "evidence standard; this alone is not.", "",
              "## trades", "", "| # | structure | contracts | entry | exit | P/L | max loss | held (bars) | exit reason |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(res.trades, 1):
        lines.append(f"| {i} | {t.kind} | {t.contracts} | {t.entry_t[:16]} @ {t.entry:+.2f} | {t.exit_t[:16]} @ {t.exit:+.2f} "
                     f"| {t.pnl:+,.2f} | {t.max_loss:,.2f} | {t.held_bars} | {t.reason_out} |")
    lines += ["", DISCLAIMER]
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for d in res.decisions:
            fh.write(json.dumps(d, default=str) + "\n")
    with (out / "trades.jsonl").open("w", encoding="utf-8") as fh:
        for t in res.trades:
            fh.write(json.dumps(t.to_json(), default=str) + "\n")
    return text
