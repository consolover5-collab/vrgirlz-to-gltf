"""Fake `vrscan convert` (and, with `--list`, `vrscan scan --json`) for
`vrscan gui --fake`: the launcher's screens without a real data folder.
Speaks the same JSONL progress protocol as docs/manifest.md's "Progress
protocol" (`start` / `step` / `scanDone` / `done` / `error`). Dev/test only
— not part of the packaged launcher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STEPS = 20
TOTAL_SECONDS = 6.0
FAKE_POSES = ["79", "86", "91"]


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vrscan.launcher.fake_convert")
    parser.add_argument("--list", action="store_true", help="print a `scan --json` report of fake poses and exit")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--scans", default="91", help="comma-separated scan ids")
    parser.add_argument("--tiers", default="4k", help="comma-separated tiers")
    parser.add_argument(
        "--fail-at",
        type=float,
        default=None,
        metavar="PCT",
        help="emit an `error` event and exit 1 once progress reaches PCT, for testing Cancel/error handling",
    )
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps({"dataset": {"set": "fake", "support": "supported", "note": "fake poses"},
                          "scans": [{"id": f"fake-{pose}"} for pose in FAKE_POSES], "errors": []}))
        return 0
    if not args.out:
        parser.error("--out is required")

    scans = args.scans.split(",")
    tiers = args.tiers.split(",")
    out_dir = Path(args.out)
    scan = scans[0]

    emit({"event": "start", "scans": scans, "tiers": tiers, "out": str(out_dir)})

    step_delay = TOTAL_SECONDS / STEPS
    for i in range(1, STEPS + 1):
        pct = i * 100 / STEPS
        if args.fail_at is not None and pct >= args.fail_at:
            emit(
                {
                    "event": "error",
                    "scan": scan,
                    "message": "fake failure requested via --fail-at",
                    "detail": f"pct={pct:.1f}",
                }
            )
            return 1
        time.sleep(step_delay)
        emit({"event": "step", "scan": scan, "step": f"fake-step-{i}", "pct": pct, "message": f"fake step {i}/{STEPS}"})

    emit({"event": "scanDone", "scan": scan, "dir": f"scans/{scan}", "seconds": TOTAL_SECONDS, "bytes": 0})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    emit({"event": "done", "seconds": TOTAL_SECONDS, "manifest": "manifest.json", "ok": 1, "failed": 0})
    return 0


if __name__ == "__main__":
    sys.exit(main())
