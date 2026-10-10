"""Entry point of the packaged exe (`packaging/vrscan.spec`). No argument:
the launcher window. `--worker <vrscan command ...>`: that one `vrscan`
command in this process — the launcher's scan/convert/serve workers. In
the exe `sys.executable` is the exe itself, so the `-m vrscan.cli` the
launcher uses from source can't start a worker there.

The exe is windowed (no console). A worker's stdout and stderr are the pipe
and the log file the launcher hands it; started any other way it has
neither, so stdout goes nowhere and stderr to a log of its own. Nothing
raises out of a worker into PyInstaller's error dialog: its traceback goes
to stderr, i.e. the log.
"""

from __future__ import annotations

import os
import sys
import traceback

WORKER_FLAG = "--worker"


def _worker(args: list[str]) -> int:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 -- lives as long as the process
    if sys.stderr is None:
        from vrscan.launcher.flow import log_dir

        logs = log_dir()
        logs.mkdir(parents=True, exist_ok=True)
        sys.stderr = open(logs / "worker.log", "a", encoding="utf-8", errors="backslashreplace")  # noqa: SIM115
    try:
        from vrscan.cli import main

        return main(args)
    except SystemExit as exc:  # argparse's usage errors
        return exc.code if isinstance(exc.code, int) else 1
    except BaseException:  # noqa: BLE001 -- a worker reports, it never raises into a dialog
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == [WORKER_FLAG]:
        return _worker(args[1:])
    from vrscan.launcher.app import run

    return run()
