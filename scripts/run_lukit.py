#!/usr/bin/env python3
import os
import sys


def _ensure_import_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent = os.path.dirname(repo_root)
    if repo_parent not in sys.path:
        sys.path.insert(0, repo_parent)


def main() -> None:
    _ensure_import_path()
    from lukit.cli.main_handler import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
