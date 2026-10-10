"""assembler.py on the `vrgirlz-2016`-shaped synthetic fixture
(`tests/fixtures/synthetic_lucid/`, see `scripts/make-fixture-lucid.py`) —
everything that differs from Nat's fixture: auto-detected profile, a pose
root whose name doesn't match `NN-150K`, a renderer shared across LOD
levels, and the cross-pose-contamination guard.
"""

from __future__ import annotations

from vrscan.assembler import assemble
from vrscan.profiles import VRGIRLZ_2016
from vrscan.reader.fixture_backend import FixtureBackend


def _scans(lucid_fixture_dir):
    backend = FixtureBackend()
    scene = backend.load(lucid_fixture_dir)
    assert backend.profile is VRGIRLZ_2016  # auto-detected from scene.json's unity_version
    # "ld2-" stands in for cli.py's per-dataset slug (_dataset_set_id) — any
    # string works here, this only exercises that assemble() forwards it to
    # profile.scan_id() when the profile opts into prefixing at all.
    return assemble(scene, backend.profile, scan_id_prefix="ld2-", fingerprint=backend.get_texture_fingerprint)


def test_finds_both_poses_with_correct_ids(lucid_fixture_dir):
    scans = _scans(lucid_fixture_dir)
    ids = {s.id for s in scans}
    assert ids == {"ld2-21", "ld2-nat-77"}  # scan_id_prefix disambiguates from Nat's un-prefixed ids


def test_nonconforming_root_name_still_found(lucid_fixture_dir):
    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-nat-77")
    assert scan.root_name == "Nat-77"  # found via its own LODGroup, not a "NN-150K" name match


def test_eyelashes_shared_across_lod_levels(lucid_fixture_dir):
    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-21")
    by_id = {p.id: p for p in scan.parts}
    lash = by_id["eyelashes"]
    # one renderer, referenced by both LODGroup levels -> two LodMesh entries
    assert [lm.level for lm in lash.lods] == [0, 1]
    assert lash.lods[0].mesh_names == lash.lods[1].mesh_names == ("Eyelashes",)


def test_eyes_clear_naming_recognized(lucid_fixture_dir):
    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-21")
    by_id = {p.id: p for p in scan.parts}
    assert {"eyeL", "eyeR"} <= set(by_id)
    assert [lm.level for lm in by_id["eyeL"].lods] == [0, 1]


def test_cross_pose_body_contamination_is_skipped(lucid_fixture_dir):
    """`77_LOD1_MeshPart0` is nested under pose `21-150K`'s tree (decoy) but
    is not referenced by that pose's own LODGroup — must not appear in
    scan `ld2-21`'s body meshes, and pose `ld2-nat-77` must not gain a
    spurious LOD1 from it either (it belongs to neither pose's LODGroup)."""

    scans = {s.id: s for s in _scans(lucid_fixture_dir)}
    body21 = next(p for p in scans["ld2-21"].parts if p.id == "body")
    all_mesh_names_21 = {name for lm in body21.lods for name in lm.mesh_names}
    assert "77_LOD1_MeshPart0" not in all_mesh_names_21

    body77 = next(p for p in scans["ld2-nat-77"].parts if p.id == "body")
    all_mesh_names_77 = {name for lm in body77.lods for name in lm.mesh_names}
    assert "77_LOD1_MeshPart0" not in all_mesh_names_77  # not in pose 77's LODGroup either


def test_single_piece_sofa_naming(lucid_fixture_dir):
    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-21")
    by_id = {p.id: p for p in scan.parts}
    assert "sofa" in by_id  # not "sofa-C1" — Lucid has no multi-piece split
    assert by_id["sofa"].kind == "sofa"
    assert by_id["sofa"].has_lod_chain is False


def test_bare_shadow_name_recognized(lucid_fixture_dir):
    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-21")
    by_id = {p.id: p for p in scan.parts}
    assert "shadow" in by_id
    assert by_id["shadow"].name == "Shad"


def test_shadow_prefix_mismatch_is_skipped(lucid_fixture_dir):
    """`21shad` is nested under pose `Nat-77`'s tree (decoy, wrong pose
    prefix) — pose 77's own id is "77", so it must be rejected, not
    misattributed as pose 77's shadow."""

    scan = next(s for s in _scans(lucid_fixture_dir) if s.id == "ld2-nat-77")
    assert "shadow" not in {p.id for p in scan.parts}
