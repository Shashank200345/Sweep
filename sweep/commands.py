"""Subcommands beyond ``models``. Each is a thin wrapper over library code."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT


def market_data(args):
    """The data source for a command: the offline fixture market, TradingView, Yahoo Finance, or Massive."""
    if getattr(args, "fixture", False):
        from .data.fixture import FixtureMarket
        return FixtureMarket(seed=getattr(args, "seed", 7))
    if getattr(args, "tv", False):
        from .data.tradingview import TradingViewMarket
        return TradingViewMarket()
    if getattr(args, "yahoo", False):
        from .data.yahoo import YahooMarket
        return YahooMarket()
    from .cli import UserFacingError
    from .data.massive import MassiveClient, MassiveError
    try:
        return MassiveClient()
    except MassiveError as e:
        raise UserFacingError(str(e)) from None



def _window(args, md) -> tuple[str, str]:
    """Default window: the fixture's last ~11 months, or 30 days for Yahoo, else required."""
    if md.is_fixture:
        days = md._m["days"]
        return args.start or str(days[-230].date()), args.end or str(days[-1].date())
    if getattr(md, "label", "") == "YahooFinance" and not (args.start and args.end):
        end_d = pd.Timestamp.now(tz="America/New_York").date()
        start_d = end_d - pd.Timedelta(days=30)
        return args.start or str(start_d), args.end or str(end_d)
    from .cli import UserFacingError
    if not (args.start and args.end):
        raise UserFacingError("--start and --end are required with real data (YYYY-MM-DD)")
    return args.start, args.end


def cmd_probe(args) -> None:
    from .data.massive import last_trading_days
    from .data.probe import run_probe, write_report
    md = market_data(args)
    if md.is_fixture:
        days = [d.date() for d in md._m["days"][-args.days - 5:-5]]
    elif getattr(md, "label", "") == "YahooFinance":
        now_d = pd.Timestamp.now(tz="America/New_York").date()
        b_days = pd.bdate_range(end=now_d, periods=args.days + 1)
        days = [d.date() for d in b_days[:-1]]
    else:
        days = last_trading_days(args.days)
    res = run_probe(md, days, DEFAULT)
    print(write_report(res, Path(args.out), DEFAULT))
    print(f"· report {Path(args.out) / 'probe.md'}")



def cmd_backtest(args) -> None:
    import dataclasses
    from .benchmark import random_traders
    from .cli import make_jev
    from .engine import run
    from .replay import build as build_replay
    from .report import write_backtest
    md = market_data(args)
    start, end = _window(args, md)
    jev = make_jev(args)
    print(f"· backtest {start} → {end} · data: {md.label} · model {getattr(jev, 'model', '?')}")
    cfg = DEFAULT
    ex_noul = getattr(args, "exit_noul", None)
    if getattr(args, "spreads", False):
        spread_exit = ex_noul if ex_noul is not None else 0.85
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, credit_spreads_only=True, exit_noul=spread_exit),
            options=dataclasses.replace(cfg.options, short_delta=0.25)
        )
    elif ex_noul is not None:
        cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy, exit_noul=ex_noul))

    if getattr(args, "ob_near", False):
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, require_ob_near=True)
        )
    res = run(md, jev, start, end, cfg, progress=True)
    f = res.features
    sess = f.h1["session"]
    sel = ((sess >= pd.Timestamp(start)) & (sess <= pd.Timestamp(end))).to_numpy()
    bench = random_traders(md, res.trades, f.h1.index[sel], f.spots[sel], f.ivv["atm_iv"].to_numpy()[sel],
                           md.dividends(), cfg, args.random)
    out = Path(args.out)
    print(write_backtest(res, out, cfg, bench))
    html = build_replay(res, out / "replay.html")
    print(f"· report {out / 'report.md'}\n· log    {out / 'decisions.jsonl'}\n· replay {html}")


def cmd_walkforward(args) -> None:
    from .cli import make_jev
    from .walkforward import walk_forward, write_walkforward
    md = market_data(args)
    start, end = (args.start, args.end)
    if md.is_fixture:
        days = md._m["days"]
        start, end = start or str(days[0].date()), end or str(days[-1].date())
    elif not (start and end):
        from .cli import UserFacingError
        raise UserFacingError("--start and --end are required with real data (YYYY-MM-DD)")
    folds, overall = walk_forward(md, lambda: make_jev(args), start, end, DEFAULT, args.random, progress=True)
    model = getattr(make_jev(args), "model", "?")
    print(write_walkforward(folds, overall, Path(args.out), md.label, model))
    print(f"· report {Path(args.out) / 'walkforward.md'}")


def cmd_ask(args) -> None:
    from .cli import make_jev
    from .questions import decision_questions
    jev = make_jev(args)
    if args.state:
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    else:
        from .engine import load_features
        from .state import state_at
        md = market_data(args)
        at = pd.Timestamp(args.at, tz="America/New_York") if args.at else None
        start, end = _window(args, md) if md.is_fixture else (str((at or pd.Timestamp.now()).date()),) * 2
        feats, _ = load_features(md, start, end, DEFAULT)
        i = len(feats.h1) - 1 if at is None else int(feats.h1.index.searchsorted(at.tz_convert("UTC"), side="right") - 1)
        state = state_at(feats.buckets.iloc[i])
        print(f"state at {feats.h1.index[i]} ({md.label}):")
    print(json.dumps(state, indent=2))
    r = jev.system_one(state, decision_questions())
    print(f"\nmodel {r.model} · {r.input_tokens} input tokens · {r.latency_ms:.0f} ms")
    for k, a in r.answers.items():
        if a.type == "noul":
            print(f"  {k:26} noul   {a.noul:.2f}")
        elif a.type == "choice":
            print(f"  {k:26} choice {a.choice:14} conf {a.confidence:.2f}")
        else:
            print(f"  {k:26} score  {a.score:.2f}  conf {a.confidence:.2f}")


def cmd_structure(args) -> None:
    from .data.bars import hourly_bars
    from .replay import build_structure
    from .smc.engine import detect
    md = market_data(args)
    start, end = _window(args, md)
    mins = md.minute_bars(pd.Timestamp(start).date(), pd.Timestamp(end).date())
    h1 = hourly_bars(mins)
    html = build_structure(h1, detect(h1, DEFAULT.structure), Path(args.out) / "structure.html", md.label)
    print(f"· {len(h1)} hourly bars · chart {html}")


def cmd_paper(args) -> None:
    import dataclasses
    from .cli import make_jev
    from .paper import PaperLoop
    md = market_data(args)
    cfg = DEFAULT
    ex_noul = getattr(args, "exit_noul", None)
    if getattr(args, "spreads", False):
        spread_exit = ex_noul if ex_noul is not None else 0.85
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, credit_spreads_only=True, exit_noul=spread_exit),
            options=dataclasses.replace(cfg.options, short_delta=0.25)
        )
    elif ex_noul is not None:
        cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy, exit_noul=ex_noul))

    if getattr(args, "ob_near", False):
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, require_ob_near=True)
        )
    state_dir = Path(getattr(args, "state_dir", None) or getattr(args, "out", "paper_state"))
    loop = PaperLoop(md, make_jev(args), state_dir, cfg, verbose=getattr(args, "verbose", True))
    loop.run(max_ticks=args.ticks, fake_clock=md.is_fixture)


def cmd_replay(args) -> None:
    import webbrowser
    target = Path(args.target) if args.target else Path("runs/diagnostics_750d")
    if target.is_dir():
        for cand in ("replay.html", "structure.html"):
            if (target / cand).is_file():
                target = target / cand
                break
    if not target.is_file():
        from .cli import UserFacingError
        raise UserFacingError(f"no replay or structure HTML found at '{args.target}'")
    print(f"· opening dashboard: {target.resolve()}")
    webbrowser.open(target.resolve().as_uri())


def cmd_live(args) -> None:
    import webbrowser
    url = f"http://127.0.0.1:{args.port}/live.html"
    print(f"· opening live real-time dashboard: {url}")
    webbrowser.open(url)


def _data_flags(p, default_out: str, window: bool = True) -> None:
    p.add_argument("--fixture", action="store_true", help="use the offline fixture market (not evidence)")
    p.add_argument("--yahoo", action="store_true", help="use free Yahoo Finance market data (no API key or tax ID required)")
    p.add_argument("--tv", action="store_true", help="connect directly to TradingView live data feed (completely free, no TradingView premium required)")
    p.add_argument("--spreads", action="store_true", help="trade defined-risk credit spreads only (pure premium selling, short delta ~0.25)")
    p.add_argument("--ob-near", action="store_true", help="require price to be inside or within 1 ATR of order block")
    p.add_argument("--exit-noul", type=float, default=None, help="override exit_noul threshold (default: 0.85 with --spreads, 0.65 otherwise)")
    p.add_argument("--seed", type=int, default=7, help="fixture market seed")
    p.add_argument("--out", default=default_out)
    if window:
        p.add_argument("--start", help="first session (YYYY-MM-DD)")
        p.add_argument("--end", help="last session (YYYY-MM-DD)")



def register(sp) -> None:
    p = sp.add_parser("probe", help="Phase-0 check: are historical option quotes usable? (needs MASSIVE_API_KEY)")
    p.add_argument("--days", type=int, default=3)
    _data_flags(p, "runs/probe", window=False)
    p.set_defaults(fn=cmd_probe)

    b = sp.add_parser("backtest", help="event-driven backtest over one window")
    _data_flags(b, "runs/backtest")
    b.add_argument("--random", type=int, default=None, help="random-entry benchmark traders (default from config)")
    b.set_defaults(fn=cmd_backtest)

    w = sp.add_parser("walkforward", help="walk-forward, out-of-sample folds with EV + CI and max drawdown")
    _data_flags(w, "runs/walkforward")
    w.add_argument("--random", type=int, default=None)
    w.set_defaults(fn=cmd_walkforward)

    a = sp.add_parser("ask", help="send one state to Jev (or the surrogate) and print the typed answers")
    a.add_argument("state", nargs="?", help="a state JSON file (default: build one from data)")
    a.add_argument("--at", help="ET timestamp, e.g. '2023-06-01 15:30'")
    _data_flags(a, "runs/ask")
    a.set_defaults(fn=cmd_ask)

    s = sp.add_parser("structure", help="render the SMC structure chart (visual validation, no Jev)")
    _data_flags(s, "runs/structure")
    s.set_defaults(fn=cmd_structure)

    pp = sp.add_parser("paper", help="hourly paper-trading loop (paper only; no broker connectivity exists)")
    _data_flags(pp, "paper_state", window=False)
    pp.add_argument("--state-dir", default="paper_state", help="directory to store state and logs (default: paper_state)")
    pp.add_argument("--ticks", type=int, default=None, help="stop after N hourly ticks")
    pp.add_argument("-v", "--verbose", action="store_true", default=True, help="print detailed decision logs every tick")
    pp.set_defaults(fn=cmd_paper)


    r = sp.add_parser("replay", help="open an HTML replay dashboard in your web browser")
    r.add_argument("target", nargs="?", default="runs/diagnostics_750d", help="path to replay.html or directory containing it")
    r.set_defaults(fn=cmd_replay)

    lv = sp.add_parser("live", help="open the real-time live TradingView and SMC AI dashboard in your web browser")
    lv.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    lv.set_defaults(fn=cmd_live)

    tv = sp.add_parser("tv-listener", help="listen for live TradingView webhook alerts on SPY 1h candle closes")
    tv.add_argument("--fixture", action="store_true", help="use the offline fixture market (not evidence)")
    tv.add_argument("--yahoo", action="store_true", default=True, help="use free Yahoo Finance market data (default: True)")
    tv.add_argument("--spreads", action="store_true", default=True, help="trade defined-risk credit spreads only (default: True)")
    tv.add_argument("--ob-near", action="store_true", help="require price to be inside or within 1 ATR of order block")
    tv.add_argument("--exit-noul", type=float, default=0.85, help="exit_noul threshold (default: 0.85)")
    tv.add_argument("--host", default="0.0.0.0", help="host to bind (default: 0.0.0.0)")
    tv.add_argument("--port", type=int, default=8767, help="port to bind (default: 8767)")
    tv.add_argument("--state-dir", default="paper_state")
    tv.add_argument("-v", "--verbose", action="store_true", default=True)
    tv.set_defaults(fn=cmd_tv_listener)


def cmd_tv_listener(args) -> None:
    import dataclasses
    from .cli import make_jev
    from .webhook import start_tradingview_listener
    md = market_data(args)
    jev = make_jev(args)
    cfg = DEFAULT
    ex_noul = getattr(args, "exit_noul", None)
    if getattr(args, "spreads", False):
        spread_exit = ex_noul if ex_noul is not None else 0.85
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, credit_spreads_only=True, exit_noul=spread_exit),
            options=dataclasses.replace(cfg.options, short_delta=0.25)
        )
    elif ex_noul is not None:
        cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy, exit_noul=ex_noul))

    if getattr(args, "ob_near", False):
        cfg = dataclasses.replace(
            cfg,
            policy=dataclasses.replace(cfg.policy, require_ob_near=True)
        )
    start_tradingview_listener(
        md=md,
        jev=jev,
        state_dir=Path(args.state_dir),
        cfg=cfg,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

