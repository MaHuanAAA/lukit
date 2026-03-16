#!/usr/bin/env python3
"""Batch evaluation CLI for LUKIT."""

from ..cli.main_handler import main as eval_main


def main() -> None:
    """Delegate to the batch evaluation handler used by the unified CLI."""
    eval_main()


if __name__ == "__main__":
    main()
