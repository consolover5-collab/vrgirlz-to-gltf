"""`python -m vrscan.launcher` — direct entry point for the GUI, for
development. End users get it via `vrscan gui` (see cli.py)."""

from __future__ import annotations

import argparse
import sys

from vrscan.launcher.app import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vrscan.launcher")
    parser.add_argument("--fake", action="store_true", help="drive a fake convert process instead of `vrscan convert`")
    parser.add_argument("--out", default=None, help="output directory (default: <data_dir>-vrscan-out)")
    args = parser.parse_args(argv)
    return run(fake=args.fake, out_override=args.out)


if __name__ == "__main__":
    sys.exit(main())
