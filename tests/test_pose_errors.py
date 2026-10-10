"""One broken pose among three: `vrscan scan` and `vrscan convert` walk the
same per-pose results (`assembler.assemble_poses`), so both show the two
good poses, report the broken one with its set/pose/part/LOD and no
traceback, and exit non-zero (design review 22.09 A3)."""

from __future__ import annotations

import dataclasses
import json

from vrscan import cli
from vrscan.ir import Transform
from vrscan.reader.backend import GameObjectRec, LodGroupRec, LodLevelRec
from vrscan.reader.fixture_backend import FixtureBackend


class ThreePosesMiddleBroken(FixtureBackend):
    """The instances fixture (poses 21 and 45) plus a pose 33 — a copy of 45
    in a second "file" whose LOD0 merges two renderers with different
    materials, which one glTF primitive can't carry."""

    def load(self, data_dir):
        scene = super().load(data_dir)
        by_name = {go.name: go for go in scene.game_objects.values()}
        identity = Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        root, part0, part1 = (1, 12), (1, 13), (1, 14)
        body = by_name["45_LOD0_MeshPart0"]
        scene.game_objects[root] = GameObjectRec(
            node_id=root, name="33-150K", transform=identity, parent=by_name["Scans"].node_id,
            children=[part0, part1],
            lod_group=LodGroupRec(levels=(LodLevelRec(screen_height=0.5, renderer_game_objects=(part0, part1)),)),
        )
        scene.game_objects[part0] = dataclasses.replace(
            body, node_id=part0, name="33_LOD0_MeshPart0", parent=root, children=[]
        )
        scene.game_objects[part1] = dataclasses.replace(
            body, node_id=part1, name="33_LOD0_MeshPart1", parent=root, children=[], materials=["Eyelashes"]
        )
        by_name["Scans"].children.append(root)
        return scene


def test_scan_text_shows_the_good_poses_and_names_the_broken_one(instances_fixture_dir, capsys):
    rc = cli.run_scan(ThreePosesMiddleBroken(), instances_fixture_dir, as_json=False)
    out = capsys.readouterr().out
    assert rc == 1
    # The ids `convert` gives these poses (the set's prefix: the instances
    # fixture is a `vrgirlz-2016` set).
    assert "[synthetic-instances-21] root=21-150K" in out and "[synthetic-instances-45] root=45-150K" in out
    assert "[synthetic-instances-33] root=" not in out
    error_line = next(line for line in out.splitlines() if line.startswith("[synthetic-instances-33] ERROR:"))
    assert "scan 'synthetic-instances-33'" in error_line and "'body'" in error_line and "LOD0" in error_line
    assert "Poses: 2 ok, 1 failed" in out
    assert "Traceback" not in out


def test_scan_json_reports_the_error_and_the_summary(instances_fixture_dir, capsys):
    rc = cli.run_scan(ThreePosesMiddleBroken(), instances_fixture_dir, as_json=True)
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert [s["id"] for s in report["scans"]] == ["synthetic-instances-21", "synthetic-instances-45"]
    (error,) = report["errors"]
    assert (error["scan"], error["part"], error["levels"]) == ("synthetic-instances-33", "body", [0])
    assert error["set"] in error["message"]
    assert report["summary"] == {"ok": 2, "failed": 1}


def test_convert_jsonl_reports_the_same_error_and_converts_the_rest(instances_fixture_dir, tmp_path, capsys):
    rc = cli.run_convert(ThreePosesMiddleBroken(), instances_fixture_dir, tmp_path, ["2k"], None, "auto", "jsonl")
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert rc == 1
    (error,) = [e for e in events if e["event"] == "error"]
    assert error["scan"].endswith("33") and (error["part"], error["levels"]) == ("body", [0])
    assert "detail" not in error  # an expected error: no traceback
    done = [e["scan"] for e in events if e["event"] == "scanDone"]
    assert [scan_id[-2:] for scan_id in done] == ["21", "45"]
    assert (events[-1]["event"], events[-1]["ok"], events[-1]["failed"]) == ("done", 2, 1)
    assert "2 ok, 1 failed" in captured.err and "Traceback" not in captured.err
