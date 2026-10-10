"""`vrscan scan`'s pipeline, exercised against the fixture through
`FixtureBackend` — the real CLI always uses `UnityPyBackend` (there is no
vendor `.assets` file on the VM to point it at), but `load_and_assemble` and
the report builder are backend-agnostic, which is the point of the
`ReaderBackend` protocol."""

import json
import shutil

from vrscan.cli import _collect_report, _print_report, load_and_assemble, run_scan
from vrscan.reader.fixture_backend import FixtureBackend


def test_scan_report_end_to_end(fixture_dir, capsys):
    backend = FixtureBackend()
    poses = load_and_assemble(backend, fixture_dir)
    report = _collect_report([p.scan for p in poses], backend)

    assert len(report["scans"]) == 1
    scan = report["scans"][0]
    assert scan["id"] == "91"

    body = next(p for p in scan["parts"] if p["id"] == "body")
    tex_by_role = {t["role"]: t for t in body["textures"]}
    for role in ("specular", "specularWet"):
        hist = tex_by_role[role]["histogram16"]
        assert len(hist) == 16
        assert sum(hist) == 256 * 256  # one bin count per pixel, luma-binned

    _print_report(report)  # must not raise
    out = capsys.readouterr().out
    assert "91-150K" in out
    assert "hist16=" in out


def test_scan_json_says_which_release_and_profile_it_read(fixture_dir, capsys):
    assert run_scan(FixtureBackend(), fixture_dir, as_json=True) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    report = json.loads(out)
    assert report["dataset"] == {
        "set": "synthetic", "unity": "4.6.3f1", "profile": "vrgirlz-2015", "support": "supported",
        "note": "Unity 4.6.3f1, profile vrgirlz-2015",
    }
    assert [s["id"] for s in report["scans"]] == ["91"]


def test_scan_json_of_an_unreadable_dataset_is_one_error_and_no_scans(fixture_dir, tmp_path, capsys):
    data_dir = tmp_path / "VRGirlz-99-Test_Data"
    shutil.copytree(fixture_dir, data_dir)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["unity_version"] = "2019.4.1f1"
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    assert run_scan(FixtureBackend(), data_dir, as_json=True) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["scans"] == [] and "not supported" in report["error"]


def test_scan_json_ids_are_the_ids_convert_takes(lucid_fixture_dir, tmp_path, capsys):
    """The launcher offers `scan --json`'s ids and passes the chosen ones to
    `convert --scans`: a profile that prefixes scan ids with the set's must
    get the same prefix in both (it didn't — `vrgirlz-2016` sets converted
    nothing from the launcher)."""

    from vrscan.cli import run_convert

    assert run_scan(FixtureBackend(), lucid_fixture_dir, as_json=True) == 0
    ids = [s["id"] for s in json.loads(capsys.readouterr().out)["scans"]]
    assert ids and all(i.startswith("synthetic-lucid-") for i in ids)

    assert run_convert(FixtureBackend(), lucid_fixture_dir, tmp_path, ["2k"], set(ids), "auto", "jsonl") == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(e["scan"] for e in events if e["event"] == "scanDone") == sorted(ids)
