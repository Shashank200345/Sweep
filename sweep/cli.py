"""Command line: ``sweep <command>`` (scaffolding adapted from jev_bot/cli.py, MIT).

Global flags: ``--model surrogate | jev-latest | jev-1.13.0`` (default: the
offline surrogate when TYPESAFE_API_KEY is unset), ``--cache FILE.jsonl``
records every Jev call, ``--replay-cache FILE.jsonl`` answers only from a
recorded cache (no key needed).

Paper trading only: there is no broker connectivity anywhere in Sweep.
"""
from __future__ import annotations

import argparse
import os
import sys

from .typesafe import JevClient, JevError, ResponseCache


class UserFacingError(RuntimeError):
    """An expected, explainable failure (missing key, bad arguments)."""


def make_jev(args) -> object:
    name = args.model or ("jev-latest" if os.environ.get("TYPESAFE_API_KEY") else "surrogate")
    if name == "surrogate":
        from .surrogate import SweepSurrogate
        return SweepSurrogate()
    cache_path = args.cache or args.replay_cache
    cache = ResponseCache(cache_path) if cache_path else None
    return JevClient(model=name, cache=cache, cache_only=bool(args.replay_cache))


def cmd_models(args) -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise UserFacingError("TYPESAFE_API_KEY is not set, so there is no model list to fetch. "
                              "Everything else runs offline on the surrogate (--model surrogate).")
    for mdl in JevClient().models():
        print(f"{mdl.get('name', '?'):16} {mdl.get('release_date', ''):12} {mdl.get('description', '')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sweep", description="SMC + options research bot judged by Jev. Paper trading only.")
    p.add_argument("--model", help="surrogate | jev-latest | jev-1.13.0 (default: jev-latest if TYPESAFE_API_KEY is set)")
    p.add_argument("--cache", help="record Jev responses to this JSONL file")
    p.add_argument("--replay-cache", help="answer only from this JSONL cache (no key needed)")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("models", help="list the Jev models your key can use").set_defaults(fn=cmd_models)
    _register_commands(sp)
    return p


def _register_commands(sp) -> None:
    """Commands from later modules register here (kept in one place)."""
    from . import commands
    commands.register(sp)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252; never crash on output
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (JevError, UserFacingError) as e:
        sys.exit(f"sweep: {e}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
