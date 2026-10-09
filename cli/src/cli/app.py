"""
argparse entrypoint for the agentic-fn demo CLI.

Examples
--------
List demos::

    uv run cli list

Run one demo::

    uv run cli run seq

Run everything::

    uv run cli run --all
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from cli import __version__
from cli.demos import DEMOS, names, run, run_all


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser and subcommands."""

    parser = argparse.ArgumentParser(
        prog="cli",
        description=(
            "Demo / playground CLI for the local agentic-fn framework. "
            "Each subcommand runs a focused pipeline example with English "
            "comments in the source under src/cli/demos/."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cli {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- list -----------------------------------------------------------------
    list_parser = sub.add_parser("list", help="List available demos")
    list_parser.set_defaults(func=_cmd_list)

    # --- run ------------------------------------------------------------------
    run_parser = sub.add_parser(
        "run",
        help="Run one demo by name, or all demos with --all",
    )
    run_parser.add_argument(
        "demo",
        nargs="?",
        choices=names(),
        help="Demo name (see ``cli list``)",
    )
    run_parser.add_argument(
        "--all",
        action="store_true",
        help="Run every registered demo in order",
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def _cmd_list(_: argparse.Namespace) -> int:
    """Print demo names and short descriptions."""

    print("Available demos:\n")
    width = max(len(name) for name in DEMOS)
    for name, (_fn, help_text) in DEMOS.items():
        print(f"  {name.ljust(width)}  {help_text}")
    print("\nRun one:  uv run cli run <name>")
    print("Run all:  uv run cli run --all")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Dispatch to a single demo or the full suite."""

    if args.all:
        run_all()
        return 0
    if not args.demo:
        print("error: provide a demo name, or pass --all", file=sys.stderr)
        print(f"hint: choose from: {', '.join(names())}", file=sys.stderr)
        return 2
    try:
        run(args.demo)
    except KeyError:
        print(f"error: unknown demo {args.demo!r}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI main.

    Returns a process exit code (0 = success) so ``python -m cli`` and the
    ``cli`` console script behave the same.
    """

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
