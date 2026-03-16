"""Unified CLI entrypoint for lukit."""

import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, List

from ..bin.leaderboard import main as leaderboard_main
from ..bin.visualize import main as visualize_main
from .main_handler import main as eval_main

_HELP_TEXT = """usage: lukit <command> [options]

Commands:
  list-methods          List all registered uncertainty methods
  eval                  Run batch uncertainty evaluation
  leaderboard           Show a leaderboard from a metrics JSON file
  visualize             Generate LaTeX tables and plots from metrics

Compatibility:
  lukit --list-methods
  lukit [eval options...]
"""


@contextmanager
def _patched_argv(argv: Iterable[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


def _dispatch_eval(argv: List[str]) -> None:
    with _patched_argv(argv):
        eval_main()


def _dispatch_tool(handler, prog: str, args: List[str]) -> None:
    with _patched_argv([prog] + args):
        handler()


def main() -> None:
    argv = sys.argv[1:]
    prog = sys.argv[0] if sys.argv else "lukit"

    if not argv or argv[0] in {"-h", "--help"}:
        print(_HELP_TEXT)
        return

    if argv[0] == "--list-methods":
        _dispatch_eval([prog, "--list-methods"])
        return

    command, rest = argv[0], argv[1:]
    if command == "list-methods":
        _dispatch_eval([f"{prog} list-methods", "--list-methods"])
        return
    if command == "eval":
        _dispatch_eval([f"{prog} eval"] + rest)
        return
    if command == "leaderboard":
        _dispatch_tool(leaderboard_main, f"{prog} leaderboard", rest)
        return
    if command == "visualize":
        _dispatch_tool(visualize_main, f"{prog} visualize", rest)
        return

    # Backward compatibility: treat unknown leading flags/args as direct eval args.
    _dispatch_eval([prog] + argv)
