"""The packaged exe's dispatch (`vrscan.frozen`), run from source: in the
exe `sys.executable` is the exe itself, so the launcher starts its workers
as `<exe> --worker <vrscan command>`; a windowed exe may have no stdout or
stderr at all. The build itself is checked on a real-data machine (docs/LAUNCHER.md)."""

from __future__ import annotations

import sys
from pathlib import Path

from vrscan import cli, frozen
from vrscan.launcher.flow import worker_command


def test_the_launcher_starts_workers_through_the_exe_when_frozen(monkeypatch):
    assert worker_command("scan", "--json", "x") == [sys.executable, "-m", "vrscan.cli", "scan", "--json", "x"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert worker_command("scan", "--json", "x") == [sys.executable, "--worker", "scan", "--json", "x"]


def test_worker_mode_runs_one_cli_command(capsys):
    assert frozen.main(["--worker", "doctor"]) == 0
    assert "UnityPy:" in capsys.readouterr().out


def test_a_worker_without_stdout_or_stderr_logs_its_crash_and_raises_nothing(monkeypatch, tmp_path):
    def crash(_args):
        raise RuntimeError("boom")

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(cli, "main", crash)
    try:
        assert frozen.main(["--worker", "scan", "x"]) == 1
    finally:
        sys.stderr.close()
        sys.stdout.close()
    log = (tmp_path / "VRScan" / "logs" / "worker.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "RuntimeError: boom" in log


def test_a_worker_usage_error_is_its_exit_code(capsys):
    assert frozen.main(["--worker", "no-such-command"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_serve_finds_the_viewer_inside_the_bundle(monkeypatch, tmp_path):
    assert cli._viewer_dir() == Path(cli.__file__).resolve().parents[2] / "viewer"
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert cli._viewer_dir() == tmp_path / "viewer"
