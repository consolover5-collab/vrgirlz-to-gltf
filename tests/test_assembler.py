"""assembler.py on the synthetic fixture — no vendor asset file involved."""

import dataclasses

import pytest

from vrscan.assembler import assemble, audit_lod_renderers, classify
from vrscan.profiles import DEFAULT_PROFILE
from vrscan.reader.fixture_backend import FixtureBackend


def _scans(fixture_dir):
    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    return assemble(scene, fingerprint=backend.get_texture_fingerprint)


def test_classify_body_matches_both_meshpart_and_bare_lod_names():
    """Found live on pose 91 (22.09): LOD0/1 split into `_MeshPartJ`, but
    LOD2+ ship as a single renderer named plainly `NN_LODk` — both forms
    must classify as "body" at the right level, or those LODs silently lose
    the body (the bug this test pins)."""

    assert classify("91_LOD0_MeshPart0") == ("body", "body", 0)
    assert classify("91_LOD2") == ("body", "body", 2)
    assert classify("91_LOD4") == ("body", "body", 4)


def test_finds_one_pose(fixture_dir):
    scans = _scans(fixture_dir)
    assert len(scans) == 1
    scan = scans[0]
    assert scan.id == "91"
    assert scan.root_name == "91-150K"
    assert scan.pivot == (0.416, 0.658, 0.510)


def test_lod_levels_from_group(fixture_dir):
    scan = _scans(fixture_dir)[0]
    assert [lvl.level for lvl in scan.lods] == [0, 1]
    assert scan.lods[0].screen_height == 0.25
    assert scan.lods[1].screen_height == 0.125
    # both LOD0 body MeshParts merged into the "body" part's triangle count
    assert scan.lods[0].triangle_count > 0


def test_expected_parts(fixture_dir):
    scan = _scans(fixture_dir)[0]
    by_id = {p.id: p for p in scan.parts}
    assert set(by_id) == {"body", "eyeL", "eyeR", "eyelashes", "sofa-feet", "shadow", "sofa-C1"}

    body = by_id["body"]
    assert body.kind == "body"
    assert body.has_lod_chain is True
    assert [lm.level for lm in body.lods] == [0, 1]
    # 91_LOD0_MeshPart0 + 91_LOD0_MeshPart1 merged at level 0
    assert set(body.lods[0].mesh_names) == {"91_LOD0_MeshPart0", "91_LOD0_MeshPart1"}
    assert body.material is not None
    assert body.material.name == "91_dry"
    roles = {tex.role for tex in body.material.textures}
    assert {"baseColor", "normal", "specular"} <= roles
    assert "_MicroMixMap" in {t.property_name for t in body.material.textures if t.role == "unknown"}
    assert body.material.colors["_Color"] == (0.647, 0.607, 0.571, 1.0)

    # standalone "91Wet" material, found by name (not attached to any renderer)
    assert body.wet_material is not None
    assert body.wet_material.name == "91Wet"
    assert {t.role for t in body.wet_material.textures} == {"specular"}
    assert by_id["eyeL"].wet_material is None  # only body looks this up

    for fixed_id in ("sofa-feet", "shadow", "sofa-C1"):
        part = by_id[fixed_id]
        assert part.has_lod_chain is False
        assert [lm.level for lm in part.lods] == [0]

    # sofa pieces, feet, and the shadow plane are all hideable (docs/manifest.md)
    assert by_id["sofa-C1"].hideable is True
    assert by_id["sofa-feet"].hideable is True
    assert by_id["shadow"].hideable is True
    assert by_id["shadow"].kind == "shadow"
    # ... but only start hidden by default if HIDDEN_BY_DEFAULT_KINDS says so
    # (currently none of these three) — see test_eyes_are_hideable_and_hidden_by_default.
    assert by_id["sofa-C1"].hidden is False
    assert by_id["sofa-feet"].hidden is False
    assert by_id["shadow"].hidden is False


def test_eyes_are_hideable_and_hidden_by_default(fixture_dir):
    """Every dataset, no profile involved — eye spheres are a rough shader
    approximation, not scene dressing (design review 22.09 point 6, "Сферы
    скрыть по умолчанию")."""

    scan = _scans(fixture_dir)[0]
    by_id = {p.id: p for p in scan.parts}
    assert by_id["eyeL"].hideable is True
    assert by_id["eyeL"].hidden is True
    assert by_id["eyeR"].hideable is True
    assert by_id["eyeR"].hidden is True
    # eyelashes are a normal, always-visible part UNLESS a profile's
    # `hidden_parts` names this specific scan (see the next test) — the
    # fixture's own profile doesn't, so pose "91" stays untouched.
    assert by_id["eyelashes"].hideable is False
    assert by_id["eyelashes"].hidden is False


def test_profile_hidden_parts_forces_hideable_and_hidden_for_one_scan(fixture_dir):
    """`DatasetProfile.hidden_parts` (e.g. VRGIRLZ_2015's Nat-90 eyelashes
    rule) must flip both `hideable` and `hidden` for the named (scan, kind)
    pair, through the same mechanism the fixture's pose "91" is used here
    with a synthetic override, since the fixture itself is fixed at "91"."""

    profile = dataclasses.replace(DEFAULT_PROFILE, hidden_parts={"91": ("eyelashes",)})
    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    scan = assemble(scene, profile, fingerprint=backend.get_texture_fingerprint)[0]
    by_id = {p.id: p for p in scan.parts}
    assert by_id["eyelashes"].hideable is True
    assert by_id["eyelashes"].hidden is True
    # unrelated kinds/parts in the same scan must be untouched
    assert by_id["sofa-C1"].hidden is False


def test_no_scans_root_falls_back_to_global_scan(fixture_dir):
    """If a source ever lacks the `Scans` grouping node, pose roots should
    still be found by scanning for `NN-150K` names directly."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    del scene.game_objects[(0, 1)]  # path_id 1 is the "Scans" node in the fixture
    scans = assemble(scene, fingerprint=backend.get_texture_fingerprint)
    assert len(scans) == 1
    assert scans[0].id == "91"


def test_merged_meshes_with_different_materials_raise(fixture_dir):
    """`91_LOD0_MeshPart0/1` are merged into one glTF primitive, which can
    carry only one material — the representative's. If the source ever gave
    them different materials that assumption would silently mis-paint half
    the body (the reviewer geometry review 22.09), so it's an error."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    part1 = next(go for go in scene.game_objects.values() if go.name == "91_LOD0_MeshPart1")
    part1.materials = ["Eyelashes"]

    with pytest.raises(ValueError, match="different materials"):
        assemble(scene, fingerprint=backend.get_texture_fingerprint)


def test_same_material_under_a_different_name_passes(fixture_dir):
    """Different Unity material names (or PPtrs) across LOD levels are fine
    as long as the *effective export signature* matches — the package they
    produce is identical (design review 22.09 point 5)."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    scene.materials["91_dry_copy"] = dataclasses.replace(scene.materials["91_dry"], name="91_dry_copy")
    lod1 = next(go for go in scene.game_objects.values() if go.name == "91_LOD1_MeshPart0")
    lod1.materials = ["91_dry_copy"]

    scan = assemble(scene, fingerprint=backend.get_texture_fingerprint)[0]
    assert next(p for p in scan.parts if p.id == "body").material.name == "91_dry"


def test_different_texture_between_lod_levels_raises(fixture_dir):
    """One part carries one material, so a level whose material binds a
    different texture cannot be published under the representative's — that
    is a silent substitution, not a warning."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    lod1 = next(go for go in scene.game_objects.values() if go.name == "91_LOD1_MeshPart0")
    lod1.materials = ["Eyelashes"]

    with pytest.raises(ValueError, match="LOD1 exports a different material"):
        assemble(scene, fingerprint=backend.get_texture_fingerprint)


def test_different_base_color_factor_between_lod_levels_raises(fixture_dir):
    """Same textures, different `_Color` — the coefficient is exported as
    `baseColorFactor`, so it counts as part of the signature."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    scene.materials["91_dry_tinted"] = dataclasses.replace(
        scene.materials["91_dry"], name="91_dry_tinted", colors={"_Color": (1.0, 0.2, 0.2, 1.0)}
    )
    lod1 = next(go for go in scene.game_objects.values() if go.name == "91_LOD1_MeshPart0")
    lod1.materials = ["91_dry_tinted"]

    with pytest.raises(ValueError, match="baseColorFactor"):
        assemble(scene, fingerprint=backend.get_texture_fingerprint)


def test_audit_lod_renderers_accounts_for_every_reference(fixture_dir):
    """Every LODGroup renderer reference is either exported or carries an
    explicit reason (design review 22.09 point 6) — on a healthy fixture that
    means every row says "exported", naming the part and the levels it went
    out at."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    rows = audit_lod_renderers(scene, assemble(scene, fingerprint=backend.get_texture_fingerprint))
    assert rows
    assert {row["status"] for row in rows} == {"exported"}
    assert {row["scan"] for row in rows} == {"91"}
    body_rows = [row for row in rows if row["part"] == "body"]
    assert sorted({row["level"] for row in body_rows}) == [0, 1]


def test_audit_lod_renderers_names_the_reason_a_reference_is_dropped(fixture_dir):
    """A renderer reference whose GameObject isn't in the scene at all (a
    dangling PPtr) must show up as "missing", not vanish from the report."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    root = scene.game_objects[assemble(scene, fingerprint=backend.get_texture_fingerprint)[0].root_node_id]
    dropped = root.lod_group.levels[0].renderer_game_objects[0]
    del scene.game_objects[dropped]

    rows = audit_lod_renderers(scene, assemble(scene, fingerprint=backend.get_texture_fingerprint))
    missing = [row for row in rows if row["status"] == "missing"]
    assert len(missing) == 1
    assert missing[0]["nodeId"] == f"{dropped[0]}:{dropped[1]}"
