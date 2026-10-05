#!/usr/bin/env python3
"""Run a child command after importing compute-node core_config.

Usage:
  python scripts/with_core_config.py git ls-remote ...
  python scripts/with_core_config.py python -m pip install ...
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: with_core_config.py <command> [args...]", file=sys.stderr)
        return 2

    try:
        import core_config  # noqa: F401
    except Exception as exc:
        print(f"ERROR: unable to import core_config: {exc}", file=sys.stderr)
        return 1

    return subprocess.run(sys.argv[1:], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
