"""Instances and roots — the reviewer geometry review 22.09, test 3.

Exercises `tests/fixtures/synthetic_instances/` (see
`scripts/make-fixture-instances.py`): one mesh shared by two pose
hierarchies, a LOD renderer that lives outside its pose root's subtree, the
live `Shad` case (listed as a child of the pose root, parented elsewhere),
and a nested `Pivot`. Baked coordinates are checked against an independent
`F · inv(W_root) · W_node · p` computed straight from `scene.json`, not
against `compose_world_matrix`'s own output.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vrscan.assembler import assemble
from vrscan.gltf.writer import bake_lod_mesh
from vrscan.reader.fixture_backend import FixtureBackend

FLIP_X = np.diag([-1.0, 1.0, 1.0])


def _raw_scene(fixture_dir):
    return json.loads((fixture_dir / "scene.json").read_text(encoding="utf-8"))


def _local_matrix(raw_go: dict) -> np.ndarray:
    """4x4 TRS straight from the fixture JSON — an independent second
    implementation, deliberately not importing `vrscan.transforms`."""

    x, y, z, w = raw_go["transform"]["r"]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot @ np.diag(raw_go["transform"]["s"])
    m[:3, 3] = raw_go["transform"]["t"]
    return m


def _world_matrix(raw, name: str) -> np.ndarray:
    """World matrix (to the scene root) of the uniquely-named node `name`."""

    by_id = {g["path_id"]: g for g in raw["game_objects"]}
    matches = [g for g in raw["game_objects"] if g["name"] == name]
    assert len(matches) == 1, f"{name!r} is not unique in the fixture"
    go = matches[0]
    matrix = np.eye(4)
    chain = []
    while go is not None:
        chain.append(go)
        go = by_id.get(go["parent"]) if go["parent"] is not None else None
    for node in reversed(chain):
        matrix = matrix @ _local_matrix(node)
    return matrix


def _expected_positions(raw, node_name: str, root_name: str, positions: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(_world_matrix(raw, root_name)) @ _world_matrix(raw, node_name)
    homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
    return (homogeneous @ relative.T)[:, :3] @ FLIP_X


@pytest.fixture
def instances(instances_fixture_dir):
    backend = FixtureBackend()
    scene = backend.load(instances_fixture_dir)
    scans = {s.id: s for s in assemble(scene, backend.profile, fingerprint=backend.get_texture_fingerprint)}
    return backend, scene, scans, _raw_scene(instances_fixture_dir)


def test_lod_chain_is_complete_including_the_renderer_outside_the_subtree(instances):
    """`21_LOD1` hangs off `Scans`, not off the pose root — only the pose's
    own LODGroup names it. A plain subtree walk loses LOD1 entirely (the
    defect the reviewer called "`_build_parts` не увидит renderer вне поддерева")."""

    _backend, _scene, scans, _raw = instances
    body = next(p for p in scans["21"].parts if p.id == "body")
    assert [lm.level for lm in body.lods] == [0, 1]
    assert body.lods[1].mesh_names == ("shared_tri",)
    # the whole LOD chain, not just the body: eyelashes are shared by both levels
    lash = next(p for p in scans["21"].parts if p.id == "eyelashes")
    assert [lm.level for lm in lash.lods] == [0, 1]
    assert {p.id for p in scans["21"].parts} == {"body", "eyeL", "eyeR", "eyelashes", "sofa", "shadow"}


def test_outside_subtree_renderer_bakes_at_the_exact_expected_coordinates(instances):
    backend, scene, scans, raw = instances
    scan = scans["21"]
    body = next(p for p in scan.parts if p.id == "body")
    baked = bake_lod_mesh(backend, scene, body.lods[1], scan.root_node_id)

    source = backend.get_mesh_geometry("shared_tri")
    expected = _expected_positions(raw, "21_LOD1", "21-150K", source.positions.astype(np.float64))
    np.testing.assert_allclose(baked.positions, expected, atol=1e-6)


def test_shadow_parented_outside_the_pose_root_still_converts(instances):
    """The live lucid-21/ariel-45 failure: `Shad` is listed among the pose
    root's children but its own `parent` is `Scans`, so walking up from it
    never reaches the pose root. Before the fix this raised
    `ancestor chain of node 'Shad' never reaches pose root`; now it resolves
    through the common scene root."""

    backend, scene, scans, raw = instances
    scan = scans["21"]
    shadow = next(p for p in scan.parts if p.id == "shadow")
    baked = bake_lod_mesh(backend, scene, shadow.lods[0], scan.root_node_id)

    source = backend.get_mesh_geometry("shadow_plane")
    expected = _expected_positions(raw, "Shad", "21-150K", source.positions.astype(np.float64))
    np.testing.assert_allclose(baked.positions, expected, atol=1e-5)


def test_one_shared_mesh_in_two_hierarchies_bakes_to_two_different_places(instances):
    """`shared_tri` is rendered by pose 21 and pose 45 through different
    transforms (pose 45's also has a non-uniform scale) — each scan must get
    its own coordinates, not the other's."""

    backend, scene, scans, raw = instances
    source = backend.get_mesh_geometry("shared_tri").positions.astype(np.float64)

    baked21 = bake_lod_mesh(
        backend, scene, next(p for p in scans["21"].parts if p.id == "body").lods[0], scans["21"].root_node_id
    )
    baked45 = bake_lod_mesh(
        backend, scene, next(p for p in scans["45"].parts if p.id == "body").lods[0], scans["45"].root_node_id
    )

    np.testing.assert_allclose(
        baked21.positions, _expected_positions(raw, "21_LOD0_MeshPart0", "21-150K", source), atol=1e-6
    )
    np.testing.assert_allclose(
        baked45.positions, _expected_positions(raw, "45_LOD0_MeshPart0", "45-150K", source), atol=1e-6
    )
    assert not np.allclose(baked21.positions, baked45.positions)


def test_nested_pivot_is_expressed_in_pose_root_space(instances):
    """Pose 21's `Pivot` sits under a rotated, translated `PivotHolder`, so
    its bare local translation (0.4, 0.6, 0.5) is in the wrong space — the
    scan's pivot must be `inv(W_root) @ W_pivot` (design review 22.09)."""

    _backend, _scene, scans, raw = instances
    holder = next(g for g in raw["game_objects"] if g["name"] == "PivotHolder")
    pivot = next(g for g in raw["game_objects"] if g["name"] == "Pivot" and g["parent"] == holder["path_id"])
    relative = np.linalg.inv(_world_matrix(raw, "21-150K")) @ _world_matrix(raw, "PivotHolder")
    expected = (relative @ np.array([*pivot["transform"]["t"], 1.0]))[:3]

    np.testing.assert_allclose(scans["21"].pivot, expected, atol=1e-6)
    assert not np.allclose(scans["21"].pivot, (0.4, 0.6, 0.5))


def test_pivot_search_is_scoped_to_one_pose_root(instances):
    """Three nodes in this scene are named `Pivot` (pose 21's nested one,
    pose 45's direct child, and a decoy under `Scans`) — each scan must find
    its own, never the first one by name."""

    _backend, _scene, scans, _raw = instances
    np.testing.assert_allclose(scans["45"].pivot, (0.7, 0.1, 0.2), atol=1e-6)
    assert not np.allclose(scans["21"].pivot, (9.0, 9.0, 9.0))
    assert not np.allclose(scans["45"].pivot, (9.0, 9.0, 9.0))


def test_direct_child_pivot_keeps_its_plain_translation(fixture_dir):
    """Nat's shape — pose root at the origin, `Pivot` a direct child — must
    produce exactly the same numbers as before this change, so the existing
    manifests' `scans[].pivot` don't move."""

    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    scan = assemble(scene, fingerprint=backend.get_texture_fingerprint)[0]
    assert scan.pivot == (0.416, 0.658, 0.510)
