"""Launcher tests. The flow (`launcher/flow.py`) is driven with fake workers
and a recording view — no display, no real process; the pieces that do
touch processes (`Worker`, `terminate_process_tree`, `vrscan serve`'s bind)
run real ones. The tkinter smoke test needs a display and is spawned under
`xvfb-run` (skipped if unavailable — this VM has no X server for plain
`uv run pytest`)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vrscan.cli import _bind_server, _make_handler
from vrscan.launcher.flow import (
    EXIT,
    LauncherFlow,
    Worker,
    build_convert_command,
    build_scan_command,
    build_serve_command,
    convert_outcome,
    default_out_dir,
    parse_event_line,
    parse_scan_report,
    pick_headset_address,
    terminate_process_tree,
)

# --------------------------------------------------------------------------
# parse_event_line
# --------------------------------------------------------------------------


def test_parse_event_line_valid():
    assert parse_event_line('{"event": "step", "pct": 12.5}') == {"event": "step", "pct": 12.5}


@pytest.mark.parametrize("line", ["", "   ", "not json", "[1, 2]", '"just a string"', '{"no_event_key": 1}'])
def test_parse_event_line_rejects_garbage(line):
    assert parse_event_line(line) is None


# --------------------------------------------------------------------------
# pick_headset_address
# --------------------------------------------------------------------------


def test_pick_headset_address_prefers_lan_over_tailnet():
    assert pick_headset_address(["127.0.0.1", "100.64.0.7", "192.168.1.5"]) == "192.168.1.5"


def test_pick_headset_address_falls_back_to_tailnet():
    assert pick_headset_address(["127.0.0.1", "100.64.0.7"]) == "100.64.0.7"


def test_pick_headset_address_none_when_only_loopback():
    assert pick_headset_address(["127.0.0.1"]) is None


def test_pick_headset_address_empty():
    assert pick_headset_address([]) is None


# --------------------------------------------------------------------------
# command building
# --------------------------------------------------------------------------


def test_build_scan_command():
    assert build_scan_command(fake=False, data_dir=Path("/data")) == [
        sys.executable, "-m", "vrscan.cli", "scan", "--json", "/data",
    ]
    assert build_scan_command(fake=True, data_dir=Path("/data")) == [
        sys.executable, "-m", "vrscan.launcher.fake_convert", "--list",
    ]


def test_build_convert_command_fake():
    cmd = build_convert_command(fake=True, data_dir=Path("/data"), out_dir=Path("/out"), scan_ids=["91", "86"], tier="4k")
    assert cmd == [
        sys.executable, "-m", "vrscan.launcher.fake_convert", "--out", "/out", "--scans", "91,86", "--tiers", "4k",
    ]


def test_build_convert_command_real():
    cmd = build_convert_command(fake=False, data_dir=Path("/data"), out_dir=Path("/out"), scan_ids=["91"], tier="2k")
    assert cmd == [
        sys.executable, "-m", "vrscan.cli", "convert", "/data", "--out", "/out", "--scans", "91", "--tiers", "2k",
        "--progress", "jsonl",
    ]


def test_build_serve_command():
    assert build_serve_command(out_dir=Path("/out"), port=8443) == [
        sys.executable, "-m", "vrscan.cli", "serve", "/out", "--https", "--port", "8443", "--progress", "jsonl",
    ]


def test_default_out_dir():
    assert default_out_dir(Path("/x/VRGirlz_Data")) == Path("/x/VRGirlz_Data-vrscan-out")


# --------------------------------------------------------------------------
# parse_scan_report — what the reader found, no fallback
# --------------------------------------------------------------------------


def _report(**fields) -> str:
    return json.dumps({"dataset": {"set": "vrgirlz-18-nat", "support": "supported", "note": "Unity 5.3.4p1"},
                       "scans": [], "errors": [], **fields})


def test_a_readable_set_offers_its_poses_labelled_without_the_set_prefix():
    scans = parse_scan_report(_report(scans=[{"id": "vrgirlz-18-nat-91"}, {"id": "vrgirlz-18-nat-93"}]), 0, Path("x"))
    assert scans.error is None and scans.ids == ["vrgirlz-18-nat-91", "vrgirlz-18-nat-93"]
    assert [scans.label(i) for i in scans.ids] == ["91", "93"]
    assert scans.note == ""  # a supported release needs no note


def test_an_assumed_release_is_offered_with_its_note():
    report = _report(scans=[{"id": "91"}], dataset={"set": "s", "support": "assumed", "note": "check the result"})
    assert parse_scan_report(report, 0, Path("x")).note == "check the result"


def test_a_folder_the_reader_refuses_is_its_message_and_no_poses():
    report = json.dumps({"error": "Nat has no level0 — choose the game's *_Data folder", "scans": []})
    scans = parse_scan_report(report, 1, Path("Nat"))
    assert scans.ids == [] and scans.error == "Nat has no level0 — choose the game's *_Data folder"


def test_a_crashed_reader_is_an_error_not_a_list():
    scans = parse_scan_report("Traceback (most recent call last):\n", 1, Path("Nat_Data"))
    assert scans.ids == [] and scans.error.startswith("Couldn't read Nat_Data") and "exit code 1" in scans.error


def test_a_set_whose_poses_all_fail_says_why():
    scans = parse_scan_report(_report(errors=[{"message": "scan '21': part body"}]), 1, Path("Nat_Data"))
    assert scans.ids == [] and scans.error == "No pose in Nat_Data can be converted: scan '21': part body"


# --------------------------------------------------------------------------
# convert_outcome — partial success is visible
# --------------------------------------------------------------------------


def test_a_full_success_has_no_warning():
    outcome = convert_outcome([{"event": "scanDone", "scan": "91"}, {"event": "done", "ok": 1, "failed": 0}], 0)
    assert (outcome.ok, outcome.failed, outcome.error, outcome.warning()) == (1, [], None, "")


def test_a_partial_success_names_what_failed():
    events = [
        {"event": "scanDone", "scan": "91"},
        {"event": "error", "scan": "86", "message": "boom", "detail": "Traceback"},
        {"event": "scanDone", "scan": "79"},
        {"event": "done", "ok": 2, "failed": 1},
    ]
    outcome = convert_outcome(events, 1)
    assert outcome.error is None and outcome.failed == ["86: boom"]
    assert outcome.warning() == "Converted 2 of 3 poses. Not converted: 86: boom"


def test_a_run_that_stopped_before_done_converted_nothing_and_says_why():
    events = [{"event": "start"}, {"event": "error", "message": "can't write to the output folder X"}]
    assert convert_outcome(events, 1).error == "can't write to the output folder X"
    assert convert_outcome([{"event": "start"}], 3).error == "The converter stopped unexpectedly (exit code 3)."
    done_empty = convert_outcome([{"event": "error", "scan": "86", "message": "x"}, {"event": "done", "ok": 0}], 1)
    assert done_empty.error == "Nothing was converted: 86: x"


# --------------------------------------------------------------------------
# LauncherFlow with fake workers
# --------------------------------------------------------------------------


class FakeWorker:
    """A worker whose events the test scripts; `stop()` only records."""

    def __init__(self, cmd, log_path, *, collect=False):
        self.cmd, self.log_path, self.collect = cmd, log_path, collect
        self.pending: list[dict] = []
        self.stopped = False

    def push(self, *events):
        self.pending.extend(events)

    def exit(self, returncode, stdout=""):
        self.pending.append({"event": EXIT, "returncode": returncode, "stdout": stdout})

    def drain(self):
        events, self.pending = self.pending, []
        return events

    def stop(self):
        self.stopped = True


class RecordingView:
    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        if not name.startswith("show_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.calls.append((name, *args, *kwargs.values()))

    def names(self):
        return [call[0] for call in self.calls]

    def last(self, name):
        return next(call for call in reversed(self.calls) if call[0] == name)


@pytest.fixture
def flow(tmp_path):
    view, started = RecordingView(), []

    def start_worker(cmd, log_path, *, collect=False):
        worker = FakeWorker(cmd, log_path, collect=collect)
        started.append(worker)
        return worker

    flow = LauncherFlow(view, logs=tmp_path / "logs", start_worker=start_worker)
    flow.started = started
    return flow


DATA = Path("/games/VRGirlz-18-Nat_Data")
READY = {"event": "ready", "scheme": "https", "port": 8444, "addresses": ["127.0.0.1", "100.64.0.7", "192.168.1.5"]}


def _to_options(flow, ids=("vrgirlz-18-nat-91", "vrgirlz-18-nat-93")):
    flow.choose_folder(DATA)
    scan = flow.started[-1]
    scan.exit(0, _report(scans=[{"id": i} for i in ids]))
    flow.pump()
    return scan


def _converted(flow, *events, returncode=0):
    flow.convert(["vrgirlz-18-nat-91"], "4k")
    convert = flow.started[-1]
    convert.push(*events)
    convert.exit(returncode)
    flow.pump()
    return convert


def test_the_steps_in_order_and_the_qr_only_once_the_server_is_listening(flow):
    scan = _to_options(flow)
    assert scan.collect and scan.cmd[-3:] == ["scan", "--json", str(DATA)]
    assert flow.view.names() == ["show_loading", "show_options"]
    _name, scans, out_dir = flow.view.last("show_options")
    assert scans.ids == ["vrgirlz-18-nat-91", "vrgirlz-18-nat-93"] and out_dir == default_out_dir(DATA)

    flow.convert(["vrgirlz-18-nat-91"], "4k")
    convert = flow.started[-1]
    assert convert.cmd[convert.cmd.index("--out") + 1] == str(default_out_dir(DATA))
    convert.push({"event": "step", "scan": "vrgirlz-18-nat-91", "pct": 0.0, "message": "vrgirlz-18-nat-91: textures"})
    flow.pump()
    assert flow.view.last("show_progress") == ("show_progress", 0.0, "vrgirlz-18-nat-91: textures")

    convert.push({"event": "scanDone", "scan": "vrgirlz-18-nat-91", "pct": 100.0}, {"event": "done", "ok": 1, "failed": 0})
    convert.exit(0)
    flow.pump()
    serve = flow.started[-1]
    assert serve.cmd[3:] == ["serve", str(default_out_dir(DATA)), "--https", "--port", "8443", "--progress", "jsonl"]
    assert flow.view.names()[-1] == "show_starting"
    flow.pump()
    assert "show_ready" not in flow.view.names()  # no URL or QR from a port the server hasn't confirmed

    serve.push(READY)
    flow.pump()
    assert flow.view.last("show_ready") == ("show_ready", "https://localhost:8444/", "https://192.168.1.5:8444/", "")
    assert not any(worker.stopped for worker in flow.started)


def test_a_partial_success_is_served_with_its_warning(flow):
    _to_options(flow)
    _converted(flow, {"event": "error", "scan": "vrgirlz-18-nat-93", "message": "set 'vrgirlz-18-nat' scan "
                      "'vrgirlz-18-nat-93': part body"}, {"event": "done", "ok": 1, "failed": 1}, returncode=1)
    flow.started[-1].push(READY)
    flow.pump()
    warning = flow.view.last("show_ready")[3]
    assert warning.startswith("Converted 1 of 2 poses. Not converted: set 'vrgirlz-18-nat' scan 'vrgirlz-18-nat-93'")


def test_a_failed_convert_returns_to_the_options_with_its_message_and_log(flow, tmp_path):
    _to_options(flow)
    _converted(flow, {"event": "error", "message": "can't write to the output folder X (Permission denied)"},
               returncode=1)
    assert [w.cmd[3] for w in flow.started] == ["scan", "convert"]  # no server for nothing
    error = flow.view.last("show_options")[3]
    assert error == f"can't write to the output folder X (Permission denied)\nLog: {tmp_path / 'logs' / 'convert.log'}"


def test_the_output_folder_can_be_changed(flow):
    _to_options(flow)
    flow.change_out_dir(Path("/elsewhere/out"))
    assert flow.view.last("show_options")[2] == Path("/elsewhere/out")
    flow.convert(["vrgirlz-18-nat-91"], "2k")
    cmd = flow.started[-1].cmd
    assert cmd[cmd.index("--out") + 1] == "/elsewhere/out"


def test_cancel_stops_the_convert_and_ignores_what_it_said_after(flow):
    _to_options(flow)
    flow.convert(["vrgirlz-18-nat-91"], "4k")
    convert = flow.started[-1]
    flow.cancel()
    assert convert.stopped
    convert.push({"event": "error", "message": "killed"})
    convert.exit(1)
    flow.pump()
    assert flow.view.names()[-1] == "show_options" and len(flow.view.last("show_options")) == 3  # no error
    assert len(flow.started) == 2  # and no server


def test_a_server_that_exits_before_it_is_ready_is_an_error(flow):
    _to_options(flow)
    _converted(flow, {"event": "done", "ok": 1, "failed": 0})
    serve = flow.started[-1]
    serve.push({"event": "error", "message": "can't listen on 0.0.0.0 port 8443 or any other"})
    serve.exit(1)
    flow.pump()
    assert "show_ready" not in flow.view.names()
    assert flow.view.last("show_options")[3].startswith(
        "The viewer server didn't start: can't listen on 0.0.0.0 port 8443 or any other\nLog: "
    )


def test_a_folder_the_reader_refuses_goes_back_to_choosing_with_the_reason(flow):
    flow.choose_folder(Path("/games/VRGirlz-18-Nat"))
    flow.started[-1].exit(1, json.dumps({"error": "VRGirlz-18-Nat has no level0", "scans": []}))
    flow.pump()
    assert flow.view.last("show_choose")[1].startswith("VRGirlz-18-Nat has no level0\nLog: ")


def test_no_pose_chosen_is_said_on_the_options(flow):
    _to_options(flow)
    flow.convert([], "4k")
    assert flow.view.last("show_options")[3] == "Choose at least one pose."
    assert len(flow.started) == 1


def test_a_second_set_gets_its_own_output_and_server_and_close_stops_only_ours(flow):
    _to_options(flow)
    _converted(flow, {"event": "done", "ok": 1, "failed": 0})
    first_serve = flow.started[-1]
    first_serve.push(READY)
    flow.pump()

    flow.back()
    assert first_serve.stopped and flow.view.names()[-1] == "show_choose"
    second = Path("/games/VRGirlz-22-Chloe_Data")
    flow.choose_folder(second)
    flow.started[-1].exit(0, _report(scans=[{"id": "vrgirlz-22-chloe-01"}]))
    flow.pump()
    flow.convert(["vrgirlz-22-chloe-01"], "4k")
    convert = flow.started[-1]
    assert convert.cmd[convert.cmd.index("--out") + 1] == str(default_out_dir(second))
    convert.push({"event": "done", "ok": 1, "failed": 0})
    convert.exit(0)
    flow.pump()
    second_serve = flow.started[-1]
    assert second_serve.cmd[second_serve.cmd.index("serve") + 1] == str(default_out_dir(second))

    flow.close()
    assert second_serve.stopped
    # Everything else had already exited: close stops what is still running
    # of the processes this flow started, and there is nothing else it can stop.
    assert [w.stopped for w in flow.started] == [False, False, True, False, False, True]


# --------------------------------------------------------------------------
# Worker and terminate_process_tree — real processes
# --------------------------------------------------------------------------


def _drain_until_exit(worker, timeout=30.0):
    events, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(worker.drain())
        if events and events[-1]["event"] == EXIT:
            return events
        time.sleep(0.05)
    raise AssertionError(f"no exit within {timeout}s: {events}")


def test_a_worker_reads_jsonl_events_and_logs_stderr(tmp_path):
    script = ("import json, sys; print(json.dumps({'event': 'ready', 'port': 1}), flush=True);"
              " print('noise'); sys.stderr.write('human line\\n'); sys.exit(2)")
    cmd = [sys.executable, "-c", script]
    events = _drain_until_exit(Worker(cmd, tmp_path / "logs" / "serve.log"))
    assert events == [{"event": "ready", "port": 1}, {"event": EXIT, "returncode": 2, "stdout": ""}]
    assert (tmp_path / "logs" / "serve.log").read_text() == "human line\n"


def test_a_collecting_worker_hands_over_the_whole_stdout(tmp_path):
    cmd = [sys.executable, "-m", "vrscan.launcher.fake_convert", "--list"]
    (exit_event,) = _drain_until_exit(Worker(cmd, tmp_path / "scan.log", collect=True))
    scans = parse_scan_report(exit_event["stdout"], exit_event["returncode"], Path("fake"))
    assert scans.error is None and scans.ids and scans.label(scans.ids[0]).isdigit()


def test_terminate_process_tree_kills_running_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.2)
    assert proc.poll() is None
    terminate_process_tree(proc)
    assert proc.poll() is not None


def test_terminate_process_tree_already_exited_is_a_noop():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    terminate_process_tree(proc)  # must not raise


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="the Windows tree kill (taskkill /T) is checked on a real-data machine")
def test_stopping_a_worker_stops_its_whole_tree(tmp_path):
    grandchild = "import time; time.sleep(60)"
    child = (f"import json, subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {grandchild!r}]);"
             " print(json.dumps({'event': 'child', 'pid': p.pid}), flush=True); time.sleep(60)")
    worker = Worker([sys.executable, "-c", child], tmp_path / "convert.log")
    deadline = time.monotonic() + 10
    events = []
    while not events and time.monotonic() < deadline:
        events = worker.drain()
        time.sleep(0.05)
    pid = events[0]["pid"]
    assert _alive(pid)

    worker.stop()
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)
    assert _drain_until_exit(worker)[-1]["event"] == EXIT


# --------------------------------------------------------------------------
# vrscan serve — binds itself, never shares a port another program has
# --------------------------------------------------------------------------


def test_a_taken_port_is_left_alone_and_the_next_free_one_used(tmp_path):
    foreign = socket.socket()
    foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like an HTTPServer
    foreign.bind(("127.0.0.1", 0))
    foreign.listen()
    taken = foreign.getsockname()[1]
    try:
        server = _bind_server("127.0.0.1", taken, _make_handler(tmp_path, tmp_path))
        try:
            assert server.server_address[1] != taken
            with socket.create_connection(("127.0.0.1", taken), timeout=2):
                pass  # the foreign server still gets its connections
        finally:
            server.server_close()
    finally:
        foreign.close()


def test_serve_reports_the_port_it_got_as_a_ready_event(tmp_path):
    foreign = socket.socket()
    foreign.bind(("127.0.0.1", 0))
    foreign.listen()
    taken = foreign.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "vrscan.cli", "serve", str(tmp_path), "--host", "127.0.0.1", "--port", str(taken),
         "--progress", "jsonl"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["event"] == "ready" and ready["scheme"] == "http" and ready["addresses"] == ["127.0.0.1"]
        assert ready["port"] != taken
        with socket.create_connection(("127.0.0.1", ready["port"]), timeout=2) as conn:
            conn.sendall(b"GET /out/manifest.json HTTP/1.0\r\n\r\n")
            assert conn.recv(64).startswith(b"HTTP/1.0 404")  # it's serving; the folder is empty
    finally:
        terminate_process_tree(proc)
        proc.communicate()
        foreign.close()


def test_serve_of_a_missing_folder_is_an_error_event(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "vrscan.cli", "serve", str(tmp_path / "nope"), "--progress", "jsonl"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"event": "error", "message": f"{(tmp_path / 'nope').resolve()} does not exist"}


# --------------------------------------------------------------------------
# fake_convert — the JSONL protocol end to end
# --------------------------------------------------------------------------


def test_fake_convert_emits_full_protocol(tmp_path):
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "vrscan.launcher.fake_convert", "--out", str(out_dir), "--scans", "91"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events[0]["event"] == "start"
    assert events[-1] == {"event": "done", "seconds": 6.0, "manifest": "manifest.json", "ok": 1, "failed": 0}
    assert [e["event"] for e in events].count("step") == 20
    assert any(e["event"] == "scanDone" for e in events)
    assert (out_dir / "manifest.json").exists()


def test_fake_convert_fail_at_emits_error_and_exits_1(tmp_path):
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "vrscan.launcher.fake_convert", "--out", str(out_dir), "--fail-at", "10"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events[-1]["event"] == "error"
    assert convert_outcome(events, result.returncode).error == "fake failure requested via --fail-at"
    assert not out_dir.exists()  # only `done` creates the output folder + manifest


# --------------------------------------------------------------------------
# tkinter smoke test — needs a display; run under xvfb-run if present
# --------------------------------------------------------------------------


def test_tkinter_window_creates_under_xvfb():
    if shutil.which("xvfb-run") is None:
        pytest.skip("no xvfb-run on this machine — can't create a Tk() window headless")
    result = subprocess.run(
        ["xvfb-run", "-a", sys.executable, "-c", "import tkinter as tk; r = tk.Tk(); r.update(); r.destroy()"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
