"""`gltf/writer.py` end to end, against the synthetic fixture (no vendor
asset file) — bake/transform correctness, glTF structural validity via
`pygltflib`/`trimesh`, and bounds/heightM sanity."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest
import trimesh

from vrscan.assembler import assemble
from vrscan.gltf.materials import MaterialPlan, MaterialRecipe
from vrscan.gltf.writer import bake_lod_mesh, build_tier_materials, compose_world_matrix, write_scan
from vrscan.ir import LodMesh, MaterialDef, Transform
from vrscan.profiles import NormalRules
from vrscan.reader.backend import GameObjectRec, MeshGeometryRec, SceneRec
from vrscan.reader.fixture_backend import FixtureBackend
from vrscan.textures import TextureCache
from vrscan.transforms import quat_to_rotation_matrix

_IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


class _OneTriangleBackend:
    """Minimal `ReaderBackend` stand-in for `bake_lod_mesh` tests below —
    one hand-built triangle, no fixture files, so a node's transform (scale,
    determinant) can be set exactly instead of relying on what's in
    `tests/fixtures/synthetic/`."""

    def __init__(self, positions: np.ndarray, normals: np.ndarray, indices: np.ndarray) -> None:
        self._geo = MeshGeometryRec(
            positions=positions.astype(np.float32),
            normals=normals.astype(np.float32),
            uvs=np.zeros((len(positions), 2), dtype=np.float32),
            indices=indices.astype(np.uint32),
        )

    def get_mesh_geometry(self, name: str) -> MeshGeometryRec | None:
        return self._geo if name == "tri" else None


def _load(fixture_dir):
    backend = FixtureBackend()
    scene = backend.load(fixture_dir)
    scans = assemble(scene, fingerprint=backend.get_texture_fingerprint)
    return backend, scene, scans[0]


def _write(fixture_dir, out_dir, tiers=("4k",)):
    """`write_scan` itself only populates `scans/<id>.tmp/` (the staging dir,
    see its docstring) — replicate the tmp-dir-then-rename step `cli.py
    cmd_convert` does once a scan finishes, so tests see the same finished
    layout `docs/manifest.md` describes."""

    backend, scene, scan = _load(fixture_dir)
    cache = TextureCache(out_dir)
    result = write_scan(backend, scene, scan, out_dir, list(tiers), NormalRules(), cache, lambda _msg: None)
    (out_dir / f"scans/{scan.id}.tmp").rename(out_dir / result.dir_name)
    return backend, scene, scan, result


class TestBakeLodMesh:
    def test_uv_v_flip_is_exact(self, fixture_dir):
        """docs/manifest.md: `v = 1 - v`. Sofa-Feet has a single source mesh
        (no MeshPart merge), so its baked UVs compare 1:1 against the
        backend's raw geometry."""

        backend, scene, scan = _load(fixture_dir)
        part = next(p for p in scan.parts if p.id == "sofa-feet")
        lod_mesh = part.lods[0]
        assert lod_mesh.mesh_names == ("Sofa-Feet",)

        baked = bake_lod_mesh(backend, scene, lod_mesh, scan.root_node_id)
        source = backend.get_mesh_geometry("Sofa-Feet")
        np.testing.assert_allclose(baked.uvs[:, 0], source.uvs[:, 0])
        np.testing.assert_allclose(baked.uvs[:, 1], 1.0 - source.uvs[:, 1])

    def test_x_flip_and_winding_reversed(self, fixture_dir):
        backend, scene, scan = _load(fixture_dir)
        part = next(p for p in scan.parts if p.id == "sofa-feet")
        lod_mesh = part.lods[0]
        baked = bake_lod_mesh(backend, scene, lod_mesh, scan.root_node_id)

        source = backend.get_mesh_geometry("Sofa-Feet")
        # x = -x on positions — Sofa-Feet's own -90deg-X rotation (its parent
        # "91sofa" is identity) never mixes x into y/z, and there's no
        # translation/scale in this chain, so this holds exactly here.
        assert baked.positions.shape == source.positions.shape
        np.testing.assert_allclose(baked.positions[:, 0], -source.positions[:, 0], atol=1e-6)
        # Winding: each source triangle (a,b,c) becomes (a,c,b) once merged.
        src_tris = source.indices.reshape(-1, 3)
        baked_tris = baked.indices.reshape(-1, 3)
        assert len(src_tris) == len(baked_tris)
        assert np.array_equal(baked_tris[:, 0], src_tris[:, 0])
        assert np.array_equal(baked_tris[:, 1], src_tris[:, 2])
        assert np.array_equal(baked_tris[:, 2], src_tris[:, 1])

    def test_merges_multiple_mesh_parts_into_one_primitive(self, fixture_dir):
        """91_LOD0_MeshPart0 + 91_LOD0_MeshPart1 -> one BakedMesh (the 16-bit
        index limit split is an artifact, not semantics, per ir.py)."""

        backend, scene, scan = _load(fixture_dir)
        body = next(p for p in scan.parts if p.id == "body")
        lod0 = next(lm for lm in body.lods if lm.level == 0)
        assert set(lod0.mesh_names) == {"91_LOD0_MeshPart0", "91_LOD0_MeshPart1"}

        baked = bake_lod_mesh(backend, scene, lod0, scan.root_node_id)
        part0 = backend.get_mesh_geometry("91_LOD0_MeshPart0")
        part1 = backend.get_mesh_geometry("91_LOD0_MeshPart1")
        assert len(baked.positions) == len(part0.positions) + len(part1.positions)
        # part1's indices must be offset past part0's vertex count.
        assert baked.indices.max() == len(baked.positions) - 1


class TestWriteScanGltfOutputs:
    def test_files_and_gltf_validity(self, fixture_dir, tmp_path):
        _backend, _scene, scan, _result = _write(fixture_dir, tmp_path, tiers=("2k", "4k"))
        scan_dir = tmp_path / "scans" / scan.id

        for name in ("static.2k.gltf", "static.4k.gltf", "lod0.4k.gltf", "lod1.4k.gltf",
                     "lod0.2k.gltf", "lod1.2k.gltf", "scan.gltf"):
            assert (scan_dir / name).is_file(), name
        for name in ("geometry.static.bin", "geometry.lod0.bin", "geometry.lod1.bin"):
            assert (scan_dir / name).is_file(), name

        for name in ("static.4k.gltf", "lod0.4k.gltf", "scan.gltf"):
            doc = pygltflib.GLTF2().load(str(scan_dir / name))
            assert doc.meshes, name
            assert doc.accessors, name
            pos_accessors = [a for a in doc.accessors if a.type == "VEC3" and a.min is not None]
            assert pos_accessors, f"{name}: no POSITION accessor with min/max"

    def test_trimesh_can_load_scan_gltf(self, fixture_dir, tmp_path):
        _backend, _scene, scan, _result = _write(fixture_dir, tmp_path, tiers=("4k",))
        scan_dir = tmp_path / "scans" / scan.id
        loaded = trimesh.load(str(scan_dir / "scan.gltf"))
        # a Scene (multiple named parts) or a single Trimesh, either way it
        # must parse without raising and contain actual geometry.
        assert loaded is not None
        total_vertices = (
            sum(len(g.vertices) for g in loaded.geometry.values())
            if isinstance(loaded, trimesh.Scene)
            else len(loaded.vertices)
        )
        assert total_vertices > 0

    def test_no_tangent_attribute(self, fixture_dir, tmp_path):
        _backend, _scene, scan, _result = _write(fixture_dir, tmp_path, tiers=("4k",))
        doc = pygltflib.GLTF2().load(str(tmp_path / "scans" / scan.id / "lod0.4k.gltf"))
        for mesh in doc.meshes:
            assert mesh.primitives[0].attributes.TANGENT is None
            assert mesh.extras["vrscan"]["tangents"] == "none"

    def test_bounds_and_height(self, fixture_dir, tmp_path):
        _backend, _scene, _scan, result = _write(fixture_dir, tmp_path, tiers=("4k",))
        assert result.height_m > 0
        bmin, bmax = np.array(result.bounds_min), np.array(result.bounds_max)
        assert np.all(bmax >= bmin)
        body_min, body_max = np.array(result.body_bounds_min), np.array(result.body_bounds_max)
        # bodyBounds (LOD group only) must sit within the overall bounds
        # (static+lod0 union) up to floating point slop.
        assert np.all(body_min >= bmin - 1e-3)
        assert np.all(body_max <= bmax + 1e-3)
        assert abs(result.height_m - (body_max[1] - body_min[1])) < 1e-6

    def test_material_plans_cover_every_part_kind(self, fixture_dir, tmp_path):
        _backend, _scene, _scan, result = _write(fixture_dir, tmp_path, tiers=("4k",))
        expected_keys = {"body", "eyes", "eyelashes", "shadow", "sofa-C1", "sofa-feet"}
        assert set(result.material_plans) == expected_keys
        body_plan = result.material_plans["body"]
        assert body_plan.wet_specular is not None
        assert body_plan.base_color_factor() == (0.647, 0.607, 0.571, 1.0)


class TestComposeWorldMatrixErrors:
    """`compose_world_matrix` must fail loudly on a broken hierarchy instead
    of silently returning a partial matrix (P1, design review 22.09)."""

    def test_missing_parent_raises(self):
        scene = SceneRec(
            game_objects={
                (0, 10): GameObjectRec(node_id=(0, 10), name="node10",
                                   transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)),
                                   parent=(0, 20)),
                # path_id 20 (node10's own parent) is not in game_objects.
            },
            meshes={}, textures={}, materials={},
        )
        with pytest.raises(ValueError, match="node10|0:10"):
            compose_world_matrix(scene, go_node_id=(0, 10), root_node_id=(0, 999))

    def test_cycle_raises(self):
        scene = SceneRec(
            game_objects={
                (0, 10): GameObjectRec(node_id=(0, 10), name="A", transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)), parent=(0, 20)),
                (0, 20): GameObjectRec(node_id=(0, 20), name="B", transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)), parent=(0, 10)),
            },
            meshes={}, textures={}, materials={},
        )
        with pytest.raises(ValueError, match="cycle"):
            compose_world_matrix(scene, go_node_id=(0, 10), root_node_id=(0, 999))

    def test_disjoint_scene_roots_raise(self):
        """A node and the pose root that share no scene root at all have no
        common space — unlike a node merely parented outside the pose root's
        subtree, which is resolved through that shared root (see
        `tests/test_geometry_instances.py`)."""

        scene = SceneRec(
            game_objects={
                (0, 10): GameObjectRec(node_id=(0, 10), name="orphan",
                                   transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)), parent=None),
                (0, 20): GameObjectRec(node_id=(0, 20), name="pose-root",
                                   transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)), parent=None),
            },
            meshes={}, textures={}, materials={},
        )
        with pytest.raises(ValueError, match="different scene roots"):
            compose_world_matrix(scene, go_node_id=(0, 10), root_node_id=(0, 20))

    def test_missing_pose_root_raises(self):
        scene = SceneRec(
            game_objects={
                (0, 10): GameObjectRec(node_id=(0, 10), name="orphan",
                                   transform=Transform((0, 0, 0), _IDENTITY_ROTATION, (1, 1, 1)), parent=None),
            },
            meshes={}, textures={}, materials={},
        )
        with pytest.raises(ValueError, match="missing from the scene"):
            compose_world_matrix(scene, go_node_id=(0, 10), root_node_id=(0, 999))


class TestBakeLodMeshTransformCorrectness:
    """Synthetic single-triangle scenes (no fixture files) that pin down
    exact transforms `tests/fixtures/synthetic/` doesn't happen to exercise:
    non-uniform scale and a locally mirrored node."""

    @staticmethod
    def _scene_with_one_node(scale: tuple[float, float, float]) -> tuple[SceneRec, LodMesh]:
        node = GameObjectRec(
            node_id=(0, 2), name="tri_node",
            transform=Transform(translation=(0.0, 0.0, 0.0), rotation=_IDENTITY_ROTATION, scale=scale),
            parent=(0, 1),  # root_node_id — never dereferenced, just the walk's stop condition
        )
        scene = SceneRec(game_objects={(0, 2): node}, meshes={}, textures={}, materials={})
        lod_mesh = LodMesh(level=0, vertex_count=3, triangle_count=1, mesh_names=("tri",), node_ids=((0, 2),))
        return scene, lod_mesh

    def test_normals_use_inverse_transpose_not_rot_scale(self):
        """Node with scale (1, 2, 1): an off-axis normal must be transformed
        by inverse(rot_scale), not rot_scale.T — the two disagree exactly
        when scale is non-uniform, which is the P1 this fixes."""

        scene, lod_mesh = self._scene_with_one_node(scale=(1.0, 2.0, 1.0))
        n_local = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normals = np.tile(n_local, (3, 1))
        indices = np.array([0, 1, 2])
        backend = _OneTriangleBackend(positions, normals, indices)

        baked = bake_lod_mesh(backend, scene, lod_mesh, root_node_id=(0, 1))

        rot_scale = np.diag([1.0, 2.0, 1.0])
        correct = n_local @ np.linalg.inv(rot_scale)
        correct = correct / np.linalg.norm(correct)
        correct[0] *= -1.0  # Unity -> glTF x flip
        buggy = n_local @ rot_scale.T  # the old (wrong) formula
        buggy = buggy / np.linalg.norm(buggy)
        buggy[0] *= -1.0

        for row in baked.normals:
            np.testing.assert_allclose(row, correct, atol=1e-6)
        assert not np.allclose(baked.normals[0], buggy, atol=1e-3)

    def test_winding_follows_full_transform_determinant(self):
        """Node with scale (-1, 1, 1) (a local mirror): its own rot_scale
        already has a negative determinant, which cancels the Unity->glTF
        x-flip instead of stacking with it — winding must NOT be reversed,
        unlike the always-reverse behaviour this replaces."""

        scene, lod_mesh = self._scene_with_one_node(scale=(-1.0, 1.0, 1.0))
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
        indices = np.array([0, 1, 2])
        backend = _OneTriangleBackend(positions, normals, indices)

        baked = bake_lod_mesh(backend, scene, lod_mesh, root_node_id=(0, 1))

        assert list(baked.indices) == [0, 1, 2]


class TestBuildTierMaterialsRoughness:
    """`roughness_factor or 1.0` coerced an intentional 0.0 to 1.0 (P1,
    design review 22.09) — `build_tier_materials` must tell "not set" (None)
    apart from "set to zero"."""

    @staticmethod
    def _plan(roughness_factor: float | None) -> MaterialPlan:
        recipe = MaterialRecipe(roughness_factor=roughness_factor)
        source = MaterialDef(name="mat", shader="shader", textures=())
        return MaterialPlan(key="mat", kind="body", recipe=recipe, source=source)

    def test_zero_roughness_factor_is_kept(self):
        plans = {"mat": self._plan(0.0)}
        tier_materials = build_tier_materials(plans, texture_index_by_key={})
        material = tier_materials.materials[tier_materials.index_by_key["mat"]]
        assert material.pbrMetallicRoughness.roughnessFactor == 0.0

    def test_unset_roughness_factor_still_defaults_to_one(self):
        plans = {"mat": self._plan(None)}
        tier_materials = build_tier_materials(plans, texture_index_by_key={})
        material = tier_materials.materials[tier_materials.index_by_key["mat"]]
        assert material.pbrMetallicRoughness.roughnessFactor == 1.0


class TestMatrixAndGeometryHygiene:
    """Explicit errors instead of quietly-wrong buffers (the reviewer geometry
    review 22.09, point 1)."""

    @staticmethod
    def _scene_with_one_node(transform: Transform) -> tuple[SceneRec, LodMesh]:
        node = GameObjectRec(node_id=(0, 2), name="tri_node", transform=transform, parent=(0, 1))
        scene = SceneRec(game_objects={(0, 2): node}, meshes={}, textures={}, materials={})
        return scene, LodMesh(level=0, vertex_count=3, triangle_count=1, mesh_names=("tri",), node_ids=((0, 2),))

    @staticmethod
    def _backend(positions=None, normals=None, indices=None, uvs=None):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if positions is None else positions
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1)) if normals is None else normals
        indices = np.array([0, 1, 2]) if indices is None else indices
        backend = _OneTriangleBackend(positions, normals, indices)
        if uvs is not None:
            backend._geo = MeshGeometryRec(
                positions=backend._geo.positions, normals=backend._geo.normals,
                uvs=uvs.astype(np.float32), indices=backend._geo.indices,
            )
        return backend

    def test_rounded_quaternion_is_normalized(self):
        """Unity scenes carry rounded quaternions (0.707, not 0.70710678) —
        built as-is the matrix isn't orthogonal and the error compounds
        along a chain."""

        rotation = quat_to_rotation_matrix((0.707, 0.0, 0.0, 0.707))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="degenerate rotation quaternion"):
            quat_to_rotation_matrix((0.0, 0.0, 0.0, 0.0), node="tri_node")

    def test_nan_quaternion_raises(self):
        with pytest.raises(ValueError, match="degenerate rotation quaternion"):
            quat_to_rotation_matrix((float("nan"), 0.0, 0.0, 1.0))

    def test_degenerate_scale_raises_naming_the_node(self):
        """A flattened node (zero scale on one axis) has no invertible
        linear part, so its normals can't be transformed at all."""

        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(0, 0, 0), rotation=_IDENTITY_ROTATION, scale=(1.0, 0.0, 1.0))
        )
        with pytest.raises(ValueError, match="tri_node.*degenerate"):
            bake_lod_mesh(self._backend(), scene, lod_mesh, root_node_id=(0, 1))

    def test_uniformly_tiny_scale_is_accepted(self):
        """Well-conditioned but tiny (millimetre-unit) transforms have a
        minuscule determinant — the check is on singular-value ratio, not
        `det`, so these must still bake."""

        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(0, 0, 0), rotation=_IDENTITY_ROTATION, scale=(1e-4,) * 3)
        )
        baked = bake_lod_mesh(self._backend(), scene, lod_mesh, root_node_id=(0, 1))
        assert len(baked.positions) == 3

    def test_non_finite_transform_raises(self):
        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(float("inf"), 0, 0), rotation=_IDENTITY_ROTATION, scale=(1, 1, 1))
        )
        with pytest.raises(ValueError, match="not finite"):
            bake_lod_mesh(self._backend(), scene, lod_mesh, root_node_id=(0, 1))

    def test_index_count_must_be_a_multiple_of_three(self):
        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(0, 0, 0), rotation=_IDENTITY_ROTATION, scale=(1, 1, 1))
        )
        with pytest.raises(ValueError, match="not a whole number of triangles"):
            bake_lod_mesh(self._backend(indices=np.array([0, 1, 2, 0])), scene, lod_mesh, root_node_id=(0, 1))

    def test_index_out_of_range_raises(self):
        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(0, 0, 0), rotation=_IDENTITY_ROTATION, scale=(1, 1, 1))
        )
        with pytest.raises(ValueError, match="outside its 3 vertices"):
            bake_lod_mesh(self._backend(indices=np.array([0, 1, 7])), scene, lod_mesh, root_node_id=(0, 1))

    def test_attribute_length_mismatch_raises(self):
        scene, lod_mesh = self._scene_with_one_node(
            Transform(translation=(0, 0, 0), rotation=_IDENTITY_ROTATION, scale=(1, 1, 1))
        )
        with pytest.raises(ValueError, match="uvs has 2 elements"):
            bake_lod_mesh(self._backend(uvs=np.zeros((2, 2))), scene, lod_mesh, root_node_id=(0, 1))
