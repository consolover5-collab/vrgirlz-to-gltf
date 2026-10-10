"""`vrscan convert`'s orchestration (`run_convert`/`cmd_convert`) against the
synthetic fixture through `FixtureBackend` — real CLI runs use `UnityPyBackend`
(no vendor `.assets` file on the VM), same pattern as `test_cli_scan.py`."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from vrscan import cli
from vrscan.manifest import Manifest, ScanEntry, SetEntry, Source
from vrscan.presets import PRESETS, TIERS
from vrscan.profiles import DEFAULT_PROFILE, NormalRule
from vrscan.reader.fixture_backend import FixtureBackend

_MINIMAL_SCAN_FIELDS = {
    "pivot": [0, 0, 0],
    "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
    "bodyBounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
    "heightM": 1.0,
    "tiers": [],
    "static": {"gltf": "static.{tier}.gltf", "bin": "geometry.static.bin", "bytes": 0, "triangles": 0},
    "lods": [],
    "parts": [],
    "materials": {},
    "textures": {},
    "budget": {},
    "provenance": {"sceneRoot": "x"},
}


def _minimal_scan_entry(scan_id: str, set_id: str | None) -> ScanEntry:
    """A `ScanEntry` with every required field but no real geometry — for
    tests that only care about `_merge_manifest`'s bookkeeping (set-id
    collisions, missing-file checks), not an actual glTF conversion."""

    return ScanEntry.model_validate(
        {"id": scan_id, "dir": f"scans/{scan_id}", "set": set_id, **_MINIMAL_SCAN_FIELDS}
    )


def test_convert_end_to_end_writes_valid_manifest(fixture_dir, tmp_path, capsys):
    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k", "4k"], None, "auto", "jsonl")
    assert rc == 0

    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [s.id for s in manifest.scans] == ["91"]
    assert manifest.scans[0].dir == "scans/91"
    assert (tmp_path / "scans" / "91" / "scan.gltf").is_file()
    assert not (tmp_path / "scans" / "91.tmp").exists()  # renamed into place
    # set id is a slug of the *dataset* (fixture_dir.name == "synthetic"),
    # not the profile id — see cli.py's _dataset_set_id.
    assert manifest.scans[0].set == "synthetic"
    assert set(manifest.sets) == {"synthetic"}
    assert manifest.sets["synthetic"].profile == "vrgirlz-2015"

    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "scanDone" in kinds
    assert all(e.get("scan") in (None, "91") for e in events)


def test_convert_scans_filter_skips_unrequested(fixture_dir, tmp_path):
    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], {"12"}, "auto", None)
    assert rc == 1  # requested scan "12" doesn't exist in the fixture
    assert not (tmp_path / "manifest.json").exists()


def test_convert_merge_preserves_other_scans(fixture_dir, tmp_path):
    """Re-running convert for one scan must not drop a previously-converted
    scan's manifest entry (docs/manifest.md's atomic-merge requirement)."""

    tmp_path.mkdir(exist_ok=True)
    existing = {
        "schema": "vrscan-manifest/1",
        "generator": {"name": "vrscan", "version": "0.0.1"},
        "source": {"app": "VRGirlz-21-Nat-AVR", "unity": "4.6.3f1", "files": []},
        "tiers": {"4k": {"label": "Quest 3 / PC VR", "baseColor": 4096, "normal": 4096, "orm": 2048,
                          "format": "jpeg", "maxLod": 0}},
        "presets": {"quest3": "4k"},
        "scans": [{
            "id": "12", "dir": "scans/12", "pivot": [0, 0, 0],
            "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}, "bodyBounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "heightM": 1.0, "tiers": ["4k"],
            "static": {"gltf": "static.{tier}.gltf", "bin": "geometry.static.bin", "bytes": 1, "triangles": 1},
            "lods": [], "parts": [], "materials": {}, "textures": {}, "budget": {}, "provenance": {"sceneRoot": "x"},
        }],
        "environments": [{"id": "room", "type": "builtin"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(existing), encoding="utf-8")

    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], {"91"}, "auto", None)
    assert rc == 0

    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {s.id for s in manifest.scans} == {"12", "91"}


def test_convert_atomicity_no_finished_dir_on_failure(fixture_dir, tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash mid-scan")

    monkeypatch.setattr(cli, "write_scan", _boom)
    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, "auto", None)
    assert rc == 1
    assert not (tmp_path / "scans" / "91").exists()  # never renamed into place
    assert not (tmp_path / "manifest.json").exists()  # nothing to merge


def test_manifest_stays_valid_if_merge_crashes_after_reexport(fixture_dir, tmp_path, monkeypatch):
    """A second `convert` run that manages to rewrite a scan's files but
    then crashes during the manifest merge itself must leave the PREVIOUS
    manifest.json exactly as it was (docs/reviews/2026-09-22-the reviewer-mvp.md:
    "прерывание между записью скана и манифеста" must not corrupt the prior
    manifest) — `_merge_manifest`'s tmp+replace write already guarantees
    this structurally; this exercises it end to end rather than trusting it
    by inspection."""

    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, "auto", None)
    assert rc == 0
    before = (tmp_path / "manifest.json").read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash during manifest merge")

    monkeypatch.setattr(cli, "_merge_manifest", _boom)
    with pytest.raises(RuntimeError, match="simulated crash during manifest merge"):
        cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, "auto", None)

    after = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert after == before
    Manifest.model_validate_json(after)  # still parses


def test_merge_manifest_rejects_scan_id_collision_across_sets(tmp_path):
    """Two different datasets producing the same scan id (broken prefix,
    hand-edited manifest, ...) must not silently overwrite one dataset's
    scan with another's — (set, scanId) is what's actually unique, not
    scanId alone."""

    tmp_path.mkdir(exist_ok=True)
    existing_entry = _minimal_scan_entry("12", "dataset-a").model_dump(mode="json", by_alias=True, exclude_none=True)
    existing = {
        "schema": "vrscan-manifest/1",
        "generator": {"name": "vrscan", "version": "0.0.1"},
        "source": {"app": "A", "unity": "4.6.3f1", "files": []},
        "tiers": {}, "presets": {},
        "scans": [existing_entry],
        "environments": [{"id": "room", "type": "builtin"}],
        "sets": {"dataset-a": {"label": "A", "profile": "vrgirlz-2015"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(existing), encoding="utf-8")

    colliding = {"12": _minimal_scan_entry("12", "dataset-b")}
    with pytest.raises(ValueError, match="collision|already exists"):
        cli._merge_manifest(
            tmp_path, Source(app="B", unity="4.6.3f1", files=[]), [], colliding, "dataset-b",
            SetEntry(label="B", profile="vrgirlz-2015"),
        )

    # the pre-existing manifest.json must be untouched by the failed merge
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.scans[0].set == "dataset-a"


def test_merge_manifest_allows_reconvert_within_same_set(tmp_path):
    """Re-exporting a scan under the SAME set id (the normal case: `vrscan
    convert` run twice against the same dataset) must not be treated as a
    collision — only a DIFFERENT set claiming the same scan id is."""

    tmp_path.mkdir(exist_ok=True)
    existing_entry = _minimal_scan_entry("12", "dataset-a").model_dump(mode="json", by_alias=True, exclude_none=True)
    existing = {
        "schema": "vrscan-manifest/1",
        "generator": {"name": "vrscan", "version": "0.0.1"},
        "source": {"app": "A", "unity": "4.6.3f1", "files": []},
        "tiers": {}, "presets": {},
        "scans": [existing_entry],
        "environments": [{"id": "room", "type": "builtin"}],
        "sets": {"dataset-a": {"label": "A", "profile": "vrgirlz-2015"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(existing), encoding="utf-8")
    scan_dir = tmp_path / "scans" / "12"
    scan_dir.mkdir(parents=True)
    (scan_dir / "geometry.static.bin").write_bytes(b"x")  # only file _minimal_scan_entry references

    cli._merge_manifest(
        tmp_path, Source(app="A", unity="4.6.3f1", files=[]), [], {"12": _minimal_scan_entry("12", "dataset-a")},
        "dataset-a", SetEntry(label="A", profile="vrgirlz-2015"),
    )
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.scans[0].set == "dataset-a"


def test_merge_manifest_rejects_reference_to_missing_file(tmp_path):
    """A `ScanEntry` referencing files that don't actually exist under
    `out_dir` (e.g. `write_scan` wrote fewer tiers than the entry claims)
    must fail the merge instead of shipping a manifest the viewer 404s on;
    the failed write must not touch/create manifest.json."""

    tmp_path.mkdir(exist_ok=True)
    entry = _minimal_scan_entry("12", "dataset-a")  # entry.dir's files were never written to tmp_path
    with pytest.raises(ValueError, match="missing"):
        cli._merge_manifest(
            tmp_path, Source(app="A", unity="4.6.3f1", files=[]), [], {"12": entry}, "dataset-a",
            SetEntry(label="A", profile="vrgirlz-2015"),
        )
    assert not (tmp_path / "manifest.json").exists()


def test_cmd_convert_rejects_unknown_tier(fixture_dir, tmp_path, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["convert", str(fixture_dir), "--out", str(tmp_path), "--tiers", "bogus"])
    rc = cli.cmd_convert(args)
    assert rc == 1
    assert "unknown tier" in capsys.readouterr().err


@pytest.mark.parametrize("swizzle", ["auto", "ag", "rg", "none"])
def test_normal_swizzle_flag_accepted(fixture_dir, tmp_path, swizzle):
    rc = cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, swizzle, None)
    assert rc == 0


def test_normal_swizzle_none_flag_reads_blue_like_the_string_path():
    """`--normal-swizzle none` must decode exactly like the bare "none" of
    `decode_normal_vectors`: Z from blue, not rebuilt from X/Y (review 22.09
    P1 — the flag gave 0.29/0/0.96 for a stored (0.3, 0, 0.5) where the
    string path gave 0.51/0/0.86), and one cache key for the same bytes."""

    from PIL import Image

    from vrscan.textures import decode_normal_vectors

    flag_rule = cli._resolve_normal_rules("none", DEFAULT_PROFILE).decide("any").rule
    assert flag_rule.z == "stored"
    assert flag_rule.key() == NormalRule.for_swizzle("none").key()

    for rgb in [(166, 128, 191), (128, 128, 0)]:  # stored (0.3, 0, 0.5); stored (0, 0, -1)
        img = Image.new("RGBA", (4, 4), (*rgb, 255))
        from_flag = decode_normal_vectors(img, flag_rule)
        from_string = decode_normal_vectors(img, "none")
        assert np.allclose(from_flag, from_string)
    assert from_flag[0, 0, 2] < -0.99  # z=-1 stays -1, not flipped to +1


def test_normal_rule_rejects_none_with_reconstructed_z():
    with pytest.raises(ValueError, match="stored"):
        NormalRule(swizzle="none")


def test_cache_hit_replaces_a_stale_output_file(fixture_dir, tmp_path):
    """A second convert must materialise the result the cache entry stands
    for, not keep whatever happens to sit at the output path — the old "skip
    if the file exists" check left a stale file in place for good (the reviewer
    review 22.09 point 2)."""

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    texture = tmp_path / "scans" / "91" / "tex" / "2k" / "body_CLR.jpg"
    good = texture.read_bytes()
    texture.write_bytes(b"stale bytes from another run")

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    assert texture.read_bytes() == good


def test_second_convert_reuses_the_cache_instead_of_re_encoding(fixture_dir, tmp_path, monkeypatch):
    """The cache remembers where a texture ends up *after* the staging dir is
    renamed, so a repeat run actually hits it — entries pointing at the
    `.tmp` staging path could never match anything again."""

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0

    from vrscan.gltf import writer

    calls = []
    real_encode = writer._encode

    def counting_encode(*args, **kwargs):
        calls.append(args[1])
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(writer, "_encode", counting_encode)
    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    assert calls == []


def _normal_maps(out_dir):
    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest.scans[0].provenance.normalMaps


def test_one_source_gets_one_resolved_rule_across_tiers_and_cache_hits(fixture_dir, tmp_path):
    """design review 22.09 point 1: "Один источник должен получать одинаковый
    resolved swizzle во всех тирах и при cache hit". Two tiers mean two
    encodes of the same source, and a second run means no encode at all —
    all three must report the same decision and evidence, so the manifest
    can't claim one decoding while the shipped bytes hold another."""

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k", "4k"], None, "auto", None) == 0
    first = _normal_maps(tmp_path)
    assert list(first) == ["body_NM"]  # one entry per texture key, not per tier
    assert first["body_NM"].evidence

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k", "4k"], None, "auto", None) == 0
    assert _normal_maps(tmp_path) == first


def test_wet_variant_is_dropped_when_its_source_equals_the_dry_one(fixture_dir, tmp_path):
    """A "wet" material pointing at the same specular map as the dry one
    encodes to an identical ORM — the manifest must then carry no wet
    variant at all, so the viewer greys the switch out instead of offering a
    toggle that changes nothing (design review 22.09 point 2)."""

    data_dir = tmp_path / "same-wet"
    shutil.copytree(fixture_dir, data_dir)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["materials"]["91Wet"]["textures"]["_SpecTex"] = "91_SPC"
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    out_dir = tmp_path / "out"
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.scans[0].materials["body"].variants == {}
    assert not list((out_dir / "scans" / "91" / "tex" / "2k").glob("*_wet.*"))


def test_scan_id_taken_by_another_set_fails_before_its_directory_is_replaced(fixture_dir, tmp_path):
    """Nat-21 and Saph-11 both keep bare pose ids (same profile). A pose
    number they shared would make the second dataset's convert collide with
    the first's `scans/<id>` — and the files go there before the manifest is
    merged, so the old merge-time check left the other set's entry pointing
    at this set's files. Now the run refuses before writing anything."""

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["scans"][0]["set"] = "vrgirlz-21-nat"
    raw["sets"]["vrgirlz-21-nat"] = {"label": "Nat-21", "profile": "vrgirlz-2015"}
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    before_manifest = manifest_path.read_text(encoding="utf-8")
    scan_dir = tmp_path / "scans" / "91"
    before_files = {p.relative_to(scan_dir): p.read_bytes() for p in scan_dir.rglob("*") if p.is_file()}
    (scan_dir / "marker").write_text("belongs to vrgirlz-21-nat", encoding="utf-8")

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 1
    assert manifest_path.read_text(encoding="utf-8") == before_manifest
    assert (scan_dir / "marker").is_file()
    after_files = {p.relative_to(scan_dir): p.read_bytes() for p in scan_dir.rglob("*") if p.is_file() and p.name != "marker"}
    assert after_files == before_files
    assert not (tmp_path / "scans" / "91.tmp").exists()


def test_wet_variant_is_dropped_when_it_differs_only_outside_the_luma(fixture_dir, tmp_path):
    """Nat-21's WET_SPC maps equal the dry SPC in RGB and differ only in
    alpha, which the ORM's roughness heuristic never reads: the "wet" ORM is
    byte-identical to the dry one, so there must be no variant. A wet map
    whose luma really differs (the fixture as shipped) keeps its variant."""

    from PIL import Image

    out_dir = tmp_path / "out-as-shipped"
    assert cli.run_convert(FixtureBackend(), fixture_dir, out_dir, ["2k"], None, "auto", None) == 0
    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "wet" in manifest.scans[0].materials["body"].variants

    data_dir = tmp_path / "alpha-only-wet"
    shutil.copytree(fixture_dir, data_dir)
    dry = Image.open(data_dir / "textures" / "91_SPC.png").convert("RGBA")
    alpha = Image.new("L", dry.size, 23)
    Image.merge("RGBA", (*dry.split()[:3], alpha)).save(data_dir / "textures" / "91WET_SPC.png")

    out_dir = tmp_path / "out"
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.scans[0].materials["body"].variants == {}
    assert not list((out_dir / "scans" / "91" / "tex" / "2k").glob("*_wet.*"))


def test_one_pose_with_a_cross_level_material_mismatch_fails_alone(instances_fixture_dir, tmp_path, capsys):
    """A pose whose LOD levels export different materials is that scan's
    error — a JSONL `error` event naming set, scan, part and levels — and
    the other poses of the set still convert; the exit code says something
    failed (independent review 22.09, P2: it used to be a traceback out of
    `assemble()` that took the whole set down)."""

    class MismatchedLodBackend(FixtureBackend):
        def load(self, data_dir):
            scene = super().load(data_dir)
            next(go for go in scene.game_objects.values() if go.name == "21_LOD1").materials = ["Eyelashes"]
            return scene

    rc = cli.run_convert(MismatchedLodBackend(), instances_fixture_dir, tmp_path, ["2k"], None, "auto", "jsonl")
    assert rc == 1

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    error = errors[0]
    assert error["scan"].endswith("21") and error["part"] == "body" and error["levels"] == [0, 1]
    assert error["set"] and error["set"] in error["message"] and "LOD1" in error["message"]
    done = [e["scan"] for e in events if e["event"] == "scanDone"]
    assert len(done) == 1 and done[0].endswith("45")
    assert events[-1]["event"] == "done"

    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [s.id for s in manifest.scans] == done


def test_a_cyrillic_path_with_spaces_converts_and_reports_in_ascii(fixture_dir, tmp_path, capsys):
    """A dataset and an output folder whose paths carry Cyrillic and spaces
    convert like any other; the JSONL stream is ASCII-only (a pipe on
    Windows is in the ANSI code page) and still round-trips the paths."""

    data_dir = tmp_path / "Мои сканы" / "VRGirlz 21 Нат_Data"
    shutil.copytree(fixture_dir, data_dir)
    out_dir = tmp_path / "Вывод с пробелами"

    rc = cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", "jsonl")
    assert rc == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert all(line.isascii() for line in lines)
    events = [json.loads(line) for line in lines]
    assert events[0]["out"] == str(out_dir)
    (dataset,) = [e for e in events if e["event"] == "dataset"]
    assert dataset["set"] == "vrgirlz-21" and dataset["support"] == "supported"
    manifest = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest.sets) == {"vrgirlz-21"}


def test_a_name_with_no_latin_letter_still_gets_a_stable_set_id():
    set_id = cli._dataset_set_id(Path("Нат_Data"))
    assert set_id.startswith("set-") and len(set_id) == len("set-") + 8
    assert set_id == cli._dataset_set_id(Path("/elsewhere/Нат_Data"))
    assert set_id != cli._dataset_set_id(Path("Саф_Data"))


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs POSIX permissions, not root")
@pytest.mark.parametrize("inside", [False, True], ids=["read-only folder", "inside a read-only folder"])
def test_an_output_folder_that_cant_be_written_is_the_error(fixture_dir, tmp_path, capsys, inside):
    """Refused before the dataset is read, as one `error` event naming the
    folder — not a traceback after the whole load."""

    locked = tmp_path / "locked"
    locked.mkdir()
    out_dir = locked / "out" if inside else locked
    locked.chmod(0o500)
    try:
        rc = cli.run_convert(FixtureBackend(), fixture_dir, out_dir, ["2k"], None, "auto", "jsonl")
    finally:
        locked.chmod(0o700)
    assert rc == 1

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [e["event"] for e in events] == ["start", "error"]
    assert events[1]["message"].startswith(f"can't write to the output folder {out_dir}")
    assert list(locked.iterdir()) == []


def test_presets_name_only_tiers_the_manifest_has(fixture_dir, tmp_path):
    assert "ultra" not in PRESETS and set(PRESETS.values()) <= set(TIERS)

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["2k"], None, "auto", None) == 0
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.presets == {"quest2": "2k"}

    assert cli.run_convert(FixtureBackend(), fixture_dir, tmp_path, ["4k"], None, "auto", None) == 0
    manifest = Manifest.model_validate_json((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.presets == {"quest2": "2k", "quest3": "4k", "pcvr": "4k"}
    assert set(manifest.presets.values()) <= set(manifest.tiers)


def _fixture_of_release(fixture_dir: Path, tmp_path: Path, unity_version: str) -> Path:
    data_dir = tmp_path / "VRGirlz-99-Test_Data"
    shutil.copytree(fixture_dir, data_dir)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["unity_version"] = unity_version
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return data_dir


def test_an_unsupported_release_is_the_run_error(fixture_dir, tmp_path, capsys):
    data_dir = _fixture_of_release(fixture_dir, tmp_path, "2019.4.1f1")
    rc = cli.run_convert(FixtureBackend(), data_dir, tmp_path / "out", ["2k"], None, "auto", "jsonl")
    assert rc == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [e["event"] for e in events] == ["start", "error"]
    assert events[1]["message"].startswith("Unity 2019.4.1f1: not supported (supported releases: 4.6.3f1, 5.3.4p1)")
    assert "detail" not in events[1]  # an expected refusal, not a crash


def test_an_assumed_release_converts_and_says_so(fixture_dir, tmp_path, capsys):
    data_dir = _fixture_of_release(fixture_dir, tmp_path, "4.6.9f1")
    assert cli.run_convert(FixtureBackend(), data_dir, tmp_path / "out", ["2k"], None, "auto", "jsonl") == 0
    captured = capsys.readouterr()
    (dataset,) = [e for e in map(json.loads, captured.out.splitlines()) if e["event"] == "dataset"]
    assert dataset["support"] == "assumed" and dataset["profile"] == "vrgirlz-2015"
    assert "recognised tentatively as vrgirlz-2015" in dataset["note"]
    assert "warning: Unity 4.6.9f1: recognised tentatively" in captured.err
