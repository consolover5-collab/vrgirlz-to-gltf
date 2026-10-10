"""`vrscan convert` end-to-end against the `vrgirlz-2016`-shaped fixture —
exercises `gltf/writer.py`/`textures.py` unmodified against the second
dataset's shapes (shared LOD renderer, single-piece sofa, no per-pose
normal/occlusion/specular), and that merging Nat + Lucid scans into one
`out/` keeps both (`docs/manifest.md`'s per-scan `source`)."""

from __future__ import annotations

import json

from vrscan import cli
from vrscan.manifest import Manifest
from vrscan.reader.fixture_backend import FixtureBackend


def test_convert_lucid_fixture_end_to_end(lucid_fixture_dir, tmp_path):
    rc = cli.run_convert(FixtureBackend(), lucid_fixture_dir, tmp_path, ["2k"], None, "auto", None)
    assert rc == 0

    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    ids = {s.id for s in manifest.scans}
    # ids are prefixed with a slug of the *dataset* (lucid_fixture_dir.name
    # == "synthetic_lucid"), not the shared profile id — see cli.py's
    # _dataset_set_id; the profile only opts Lucid-style datasets into
    # prefixing at all (DatasetProfile.prefix_scan_ids).
    assert ids == {"synthetic-lucid-21", "synthetic-lucid-nat-77"}
    for scan in manifest.scans:
        assert (tmp_path / scan.dir / "scan.gltf").is_file()
        assert scan.source is not None
        assert scan.source.unity == "5.3.4p1"
        assert scan.set == "synthetic-lucid"
    assert set(manifest.sets) == {"synthetic-lucid"}
    assert manifest.sets["synthetic-lucid"].profile == "vrgirlz-2016"

    lash = next(
        p for p in next(s for s in manifest.scans if s.id == "synthetic-lucid-21").parts if p.id == "eyelashes"
    )
    assert lash.lod is True


def test_merge_keeps_scans_from_both_datasets(fixture_dir, lucid_fixture_dir, tmp_path):
    """Nat's `run_convert` then Lucid's `run_convert` into the same `out/`
    must not drop either dataset's scans — the exact scenario the batch
    conversion on a real-data machine exercises for real."""

    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, "auto", None)
    assert rc == 0
    rc = cli.run_convert(FixtureBackend(), lucid_fixture_dir, tmp_path, ["4k"], None, "auto", None)
    assert rc == 0

    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {s.id for s in manifest.scans} == {"91", "synthetic-lucid-21", "synthetic-lucid-nat-77"}
    assert set(manifest.sets) == {"synthetic", "synthetic-lucid"}
    by_id = {s.id: s.set for s in manifest.scans}
    assert by_id["91"] == "synthetic"
    assert by_id["synthetic-lucid-21"] == by_id["synthetic-lucid-nat-77"] == "synthetic-lucid"


def test_pose_without_static_parts_writes_no_empty_buffer(lucid_fixture_dir, tmp_path):
    """synthetic-lucid-nat-77 has no static parts (like four Nat-18 poses):
    its static glTFs carry no buffer — glTF requires `byteLength >= 1`, the
    Khronos validator rejected the empty one — and in scan.gltf LOD0's
    geometry is buffer 0."""

    assert cli.run_convert(FixtureBackend(), lucid_fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    scan_dir = tmp_path / "scans" / "synthetic-lucid-nat-77"
    assert (scan_dir / "geometry.static.bin").stat().st_size == 0

    static = json.loads((scan_dir / "static.2k.gltf").read_text(encoding="utf-8"))
    assert static.get("buffers", []) == []
    scan = json.loads((scan_dir / "scan.gltf").read_text(encoding="utf-8"))
    assert [b["uri"] for b in scan["buffers"]] == ["geometry.lod0.bin"]
    assert {view["buffer"] for view in scan["bufferViews"]} == {0}
