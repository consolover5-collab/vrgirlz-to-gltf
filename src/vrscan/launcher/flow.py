"""The launcher's logic without a window (docs/LAUNCHER.md): which worker
process runs, what its JSONL events mean for the screen, and which
processes get stopped — only the ones started here, each with its whole
tree, never one found by its port. `app.LauncherApp` draws what this
decides; `tests/test_launcher.py` drives it with fake workers, no display
and no real process needed.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vrscan.frozen import WORKER_FLAG

CLI_MODULE = "vrscan.cli"  # `sys.executable -m vrscan.cli <cmd>` — cli.py has an
# `if __name__ == "__main__"` guard, so this works without relying on the
# `vrscan` console-script being on PATH.

SERVE_PORT = 8443  # asked for first; `vrscan serve` takes a free one after it if another program has it

EXIT = "__exit__"  # the pseudo-event a worker queues last, once its process is gone

_PRIVATE_LAN_NETS = [ipaddress.ip_network(net) for net in ("192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12")]
_TAILNET = ipaddress.ip_network("100.64.0.0/10")

logger = logging.getLogger("vrscan.launcher")


# --------------------------------------------------------------------------
# commands and paths
# --------------------------------------------------------------------------


def log_dir() -> Path:
    """Where the launcher and its workers log: per user, never next to the
    data or the output — either can be read-only."""

    local = os.environ.get("LOCALAPPDATA")
    root = Path(local) / "VRScan" if local else Path.home() / ".local" / "state" / "vrscan"
    return root / "logs"


def worker_command(*args: str) -> list[str]:
    """`vrscan <args>` as a process of its own: from source, Python running
    the CLI module; in the packaged exe, the exe itself in worker mode
    (`vrscan.frozen`) — there `sys.executable` is the exe."""

    if getattr(sys, "frozen", False):
        return [sys.executable, WORKER_FLAG, *args]
    return [sys.executable, "-m", CLI_MODULE, *args]


def _fake_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "vrscan.launcher.fake_convert", *args]


def build_scan_command(*, fake: bool, data_dir: Path) -> list[str]:
    if fake:
        return _fake_command("--list")
    return worker_command("scan", "--json", str(data_dir))


def build_convert_command(*, fake: bool, data_dir: Path, out_dir: Path, scan_ids: list[str], tier: str) -> list[str]:
    if fake:
        return _fake_command("--out", str(out_dir), "--scans", ",".join(scan_ids), "--tiers", tier)
    return worker_command(
        "convert", str(data_dir), "--out", str(out_dir), "--scans", ",".join(scan_ids), "--tiers", tier,
        "--progress", "jsonl",
    )


def build_serve_command(*, out_dir: Path, port: int = SERVE_PORT) -> list[str]:
    return worker_command("serve", str(out_dir), "--https", "--port", str(port), "--progress", "jsonl")


def default_out_dir(data_dir: Path) -> Path:
    """One output folder per set, next to it — the window lets the user
    choose another (writing next to the set can be refused)."""

    return data_dir.parent / f"{data_dir.name}-vrscan-out"


# --------------------------------------------------------------------------
# reading what the workers say
# --------------------------------------------------------------------------


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Parse one line of the JSONL progress protocol. Returns None for
    blank lines and anything that isn't a `{"event": ...}` object — the
    protocol says human-readable progress goes to stderr regardless, but a
    stray non-JSON stdout line shouldn't crash the reader thread."""

    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or "event" not in event:
        return None
    return event


@dataclass(frozen=True)
class ScanList:
    """What `vrscan scan --json` found in a folder: the poses to offer, or
    why there are none. There is no fallback list — it would offer another
    set's poses for this one (design review 22.09)."""

    ids: list[str] = field(default_factory=list)
    set_id: str = ""
    note: str = ""  # how the release was recognised, when that is only assumed
    skipped: list[str] = field(default_factory=list)  # poses that can't be converted, one message each
    error: str | None = None

    def label(self, scan_id: str) -> str:
        """A pose as the user knows it: its id without the set's prefix."""

        return scan_id.removeprefix(f"{self.set_id}-") if self.set_id else scan_id


def parse_scan_report(stdout: str, returncode: int | None, data_dir: Path) -> ScanList:
    """`vrscan scan --json`'s report -> `ScanList`. A folder the reader
    refuses (not a `*_Data` folder, an unsupported release, a missing file)
    comes back as its own one-line `error`; anything else unreadable is a
    crash, in the scan log."""

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        return ScanList(error=f"Couldn't read {data_dir.name}: the reader stopped (exit code {returncode}).")
    if report.get("error"):
        return ScanList(error=str(report["error"]))
    ids = [str(scan["id"]) for scan in report.get("scans", []) if isinstance(scan, dict) and "id" in scan]
    skipped = [str(error.get("message", "")) for error in report.get("errors", []) if isinstance(error, dict)]
    if not ids:
        reason = f": {skipped[0]}" if skipped else ""
        return ScanList(skipped=skipped, error=f"No pose in {data_dir.name} can be converted{reason}")
    dataset = report.get("dataset") or {}
    note = str(dataset.get("note", "")) if dataset.get("support") not in (None, "supported") else ""
    return ScanList(ids=ids, set_id=str(dataset.get("set", "")), note=note, skipped=skipped)


@dataclass(frozen=True)
class ConvertOutcome:
    ok: int  # poses converted into the output folder
    failed: list[str]  # one message per pose that wasn't
    error: str | None  # why nothing was, when that's the case

    def warning(self) -> str:
        """The line that makes a partial success visible, "" otherwise."""

        if not self.failed:
            return ""
        shown = "; ".join(self.failed[:3]) + ("; …" if len(self.failed) > 3 else "")
        return f"Converted {self.ok} of {self.ok + len(self.failed)} poses. Not converted: {shown}"


def convert_outcome(events: list[dict[str, Any]], returncode: int | None) -> ConvertOutcome:
    """How a convert run ended (docs/manifest.md, progress protocol). A run
    that reached `done` converted `ok` poses, and each failed one has its
    own `error` event with a `scan`; a run that stopped before `done`
    converted nothing and says why in its last `error` — a folder it can't
    read or write — else only its exit code does."""

    failed = []
    for event in events:
        if event.get("event") == "error" and event.get("scan"):
            scan, message = str(event["scan"]), str(event.get("message", ""))
            failed.append(message if scan in message else f"{scan}: {message}")
    done = next((e for e in reversed(events) if e.get("event") == "done"), None)
    if done is not None:
        ok = done.get("ok")
        if not isinstance(ok, int):
            ok = sum(1 for e in events if e.get("event") == "scanDone")
        if ok:
            return ConvertOutcome(ok, failed, None)
        return ConvertOutcome(0, failed, "Nothing was converted" + (f": {failed[0]}" if failed else "."))
    errors = [str(e.get("message", "")) for e in events if e.get("event") == "error"]
    if errors:
        return ConvertOutcome(0, failed, errors[-1])
    return ConvertOutcome(0, failed, f"The converter stopped unexpectedly (exit code {returncode}).")


def pick_headset_address(addrs: list[str]) -> str | None:
    """Pick the best LAN address for a headset to reach this host: prefer
    plain private ranges (192.168.*/10.*/172.16-31.*), tailnet
    (100.64.0.0/10) last, loopback never."""

    def rank(addr: str) -> int:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return 3
        if ip.is_loopback:
            return 4
        if ip in _TAILNET:
            return 2
        if any(ip in net for net in _PRIVATE_LAN_NETS):
            return 0
        return 1

    candidates = sorted((a for a in addrs if rank(a) < 4), key=rank)
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------
# worker processes
# --------------------------------------------------------------------------


def _own_tree_options() -> dict[str, Any]:
    """Popen options that keep a worker's descendants findable as its tree
    (and keep a console window from flashing up on Windows)."""

    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}  # the worker leads a process group of its own


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Stop `proc` and every process it started. Only ever called on a
    process this launcher started itself: on Windows `taskkill /T` walks
    its tree; elsewhere a worker leads its own process group, signalled as
    a whole (a process that doesn't is signalled alone)."""

    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        result = subprocess.run(
            [str(taskkill), "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0 and proc.poll() is None:
            proc.kill()
        return
    try:
        leads_group = os.getpgid(proc.pid) == proc.pid
    except ProcessLookupError:
        return
    if leads_group:
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if leads_group:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()


class Worker:
    """One process the launcher started. Its stdout is read on a thread —
    JSONL events, or with `collect` the whole text (`scan --json`) — and
    queued, the `EXIT` pseudo-event last; its stderr (human progress, any
    traceback) goes to `log_path`, overwritten per run."""

    def __init__(self, cmd: list[str], log_path: Path, *, collect: bool = False) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as log:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **_own_tree_options(),
            )
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._read, args=(collect,), daemon=True).start()

    def _read(self, collect: bool) -> None:
        assert self.proc.stdout is not None
        lines = []
        for line in self.proc.stdout:
            if collect:
                lines.append(line)
            elif (event := parse_event_line(line)) is not None:
                self._events.put(event)
        self.proc.wait()
        self._events.put({"event": EXIT, "returncode": self.proc.returncode, "stdout": "".join(lines)})

    def drain(self) -> list[dict[str, Any]]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def stop(self) -> None:
        terminate_process_tree(self.proc)


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------


class View(Protocol):
    """What the flow asks the window to show (`app.LauncherApp`)."""

    def show_choose(self, error: str = "") -> None: ...
    def show_loading(self, data_dir: Path) -> None: ...
    def show_options(self, scans: ScanList, out_dir: Path, error: str = "") -> None: ...
    def show_converting(self) -> None: ...
    def show_progress(self, pct: float | None, status: str) -> None: ...
    def show_starting(self) -> None: ...
    def show_ready(self, screen_url: str, headset_url: str | None, warning: str) -> None: ...


class LauncherFlow:
    """Choose a folder → poses, quality and output folder → convert → serve
    → URL and QR. One output folder per set: next to it by default, any
    other one the user chooses. At most one worker of each kind runs; the
    window calls `pump()` on its own thread to hand their events on."""

    def __init__(
        self, view: View, *, fake: bool = False, out_override: Path | None = None, logs: Path | None = None,
        start_worker: Callable[..., Any] = Worker,
    ) -> None:
        self.view = view
        self.fake = fake
        self.out_override = out_override
        self.logs = logs or log_dir()
        self._start_worker = start_worker
        self._workers: dict[str, Any] = {}
        self._handlers = {"scan": self._on_scan, "convert": self._on_convert, "serve": self._on_serve}

        self.data_dir: Path | None = None
        self.out_dir: Path | None = None
        self.scans = ScanList()
        self._convert_events: list[dict[str, Any]] = []
        self._warning = ""
        self._serve_ready = False
        self._serve_error = ""

    # ---- what the window calls ----

    def choose_folder(self, data_dir: Path) -> None:
        logger.info("data folder: %s", data_dir)
        self.data_dir = data_dir
        self.out_dir = self.out_override or default_out_dir(data_dir)
        self.scans = ScanList()
        self._start("scan", build_scan_command(fake=self.fake, data_dir=data_dir), collect=True)
        self.view.show_loading(data_dir)

    def change_out_dir(self, out_dir: Path) -> None:
        logger.info("output folder: %s", out_dir)
        self.out_dir = out_dir
        self.view.show_options(self.scans, out_dir)

    def convert(self, scan_ids: list[str], tier: str) -> None:
        assert self.data_dir is not None and self.out_dir is not None
        if not scan_ids:
            self.view.show_options(self.scans, self.out_dir, error="Choose at least one pose.")
            return
        self._convert_events = []
        cmd = build_convert_command(
            fake=self.fake, data_dir=self.data_dir, out_dir=self.out_dir, scan_ids=scan_ids, tier=tier
        )
        self._start("convert", cmd)
        self.view.show_converting()

    def cancel(self) -> None:
        logger.info("convert cancelled by the user")
        self._stop("convert")
        assert self.out_dir is not None
        self.view.show_options(self.scans, self.out_dir)

    def back(self) -> None:
        """Back to choosing a folder — from the options, or from the ready
        screen, whose server stops (the next set gets its own)."""

        self._stop("scan")
        self._stop("serve")
        self.view.show_choose()

    def close(self) -> None:
        for kind in list(self._workers):
            self._stop(kind)

    def pump(self) -> None:
        for kind, worker in list(self._workers.items()):
            for event in worker.drain():
                if self._workers.get(kind) is not worker:
                    break  # stopped or replaced meanwhile: whatever else it said is stale
                self._handlers[kind](event)

    # ---- workers ----

    def _start(self, kind: str, cmd: list[str], *, collect: bool = False) -> None:
        self._stop(kind)
        logger.info("%s: %s", kind, cmd)
        self._workers[kind] = self._start_worker(cmd, self.logs / f"{kind}.log", collect=collect)

    def _stop(self, kind: str) -> None:
        worker = self._workers.pop(kind, None)
        if worker is not None:
            worker.stop()

    def _with_log(self, message: str, kind: str) -> str:
        return f"{message}\nLog: {self.logs / f'{kind}.log'}"

    # ---- events ----

    def _on_scan(self, event: dict[str, Any]) -> None:
        if event["event"] != EXIT:
            return
        del self._workers["scan"]
        assert self.data_dir is not None and self.out_dir is not None
        self.scans = parse_scan_report(event.get("stdout", ""), event.get("returncode"), self.data_dir)
        for message in self.scans.skipped:
            logger.warning("scan: %s", message)
        if self.scans.error:
            logger.error("scan: %s", self.scans.error)
            self.view.show_choose(error=self._with_log(self.scans.error, "scan"))
            return
        self.view.show_options(self.scans, self.out_dir)

    def _on_convert(self, event: dict[str, Any]) -> None:
        kind = event["event"]
        if kind != EXIT:
            self._convert_events.append(event)
        if kind in ("step", "scanDone"):
            pct = event.get("pct")
            status = str(event.get("message") or f"{event.get('scan')} done")
            self.view.show_progress(float(pct) if isinstance(pct, (int, float)) else None, status)
        elif kind == "error":
            logger.error("convert: %s%s", event.get("message"), f"\n{event['detail']}" if event.get("detail") else "")
        elif kind == EXIT:
            del self._workers["convert"]
            assert self.out_dir is not None
            outcome = convert_outcome(self._convert_events, event.get("returncode"))
            if outcome.error:
                self.view.show_options(self.scans, self.out_dir, error=self._with_log(outcome.error, "convert"))
                return
            logger.info("converted %d, failed %d", outcome.ok, len(outcome.failed))
            self._warning = outcome.warning()
            self._serve_ready, self._serve_error = False, ""
            self._start("serve", build_serve_command(out_dir=self.out_dir))
            self.view.show_starting()

    def _on_serve(self, event: dict[str, Any]) -> None:
        kind = event["event"]
        if kind == "ready":
            # The server bound its port itself and says which it got: the
            # URL and QR can only be drawn now, never from a guessed port.
            self._serve_ready = True
            scheme, port = event.get("scheme", "https"), event.get("port")
            addr = pick_headset_address([str(a) for a in event.get("addresses", [])])
            logger.info("serving on port %s, headset address %s", port, addr)
            self.view.show_ready(
                f"{scheme}://localhost:{port}/", f"{scheme}://{addr}:{port}/" if addr else None, self._warning
            )
        elif kind == "error":
            self._serve_error = str(event.get("message", ""))
            logger.error("serve: %s", self._serve_error)
        elif kind == EXIT:
            del self._workers["serve"]
            assert self.out_dir is not None
            what = "stopped" if self._serve_ready else "didn't start"
            reason = self._serve_error or f"exit code {event.get('returncode')}"
            self.view.show_options(
                self.scans, self.out_dir, error=self._with_log(f"The viewer server {what}: {reason}", "serve")
            )
