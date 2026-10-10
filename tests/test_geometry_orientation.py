"""Orientation, normals and winding — the reviewer geometry review 22.09, tests 1
and 2.

Test 1 is an asymmetric "axis tripod" (one vertex per axis, each addressing
its own quadrant of a four-colour texture): it pins both the baked
coordinates and which *texel* each vertex ends up addressing, so a stray
flip in either the x-flip or the `v = 1 - v` can't cancel out unnoticed.

Test 2 is a tilted triangle under nested rotations with non-uniform scale
and zero/one/two reflections: normals must stay unit-length and
perpendicular to the transformed edges, and the final index order must wind
consistently with them (`dot(cross(e1, e2), n) > 0`).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vrscan.gltf.writer import bake_lod_mesh
from vrscan.ir import LodMesh, Transform
from vrscan.reader.backend import GameObjectRec, MeshGeometryRec, SceneRec
from vrscan.textures import encode_rgba_png

IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
FLIP_X = np.diag([-1.0, 1.0, 1.0])
ROOT_ID = (0, 1)


class _StubBackend:
    def __init__(self, geo: MeshGeometryRec) -> None:
        self._geo = geo

    def get_mesh_geometry(self, name: str) -> MeshGeometryRec | None:
        return self._geo if name == "mesh" else None


def _quat_matrix(rotation) -> np.ndarray:
    """Independent quaternion -> matrix, so the expectations below don't go
    through the same code they check."""

    x, y, z, w = np.asarray(rotation, dtype=np.float64) / np.linalg.norm(rotation)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _matrix(t: Transform) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(t.rotation) @ np.diag(t.scale)
    m[:3, 3] = t.translation
    return m


def _chain_scene(transforms: list[Transform]) -> tuple[SceneRec, LodMesh, np.ndarray]:
    """A chain root(ROOT_ID) -> ... -> renderer, one node per transform, plus
    the independently composed relative matrix of the renderer."""

    game_objects = {}
    parent = ROOT_ID
    relative = np.eye(4)
    for index, transform in enumerate(transforms):
        node_id = (0, ROOT_ID[1] + 1 + index)
        game_objects[node_id] = GameObjectRec(
            node_id=node_id, name=f"node{index}", transform=transform, parent=parent
        )
        relative = relative @ _matrix(transform)
        parent = node_id
    scene = SceneRec(game_objects=game_objects, meshes={}, textures={}, materials={})
    lod_mesh = LodMesh(level=0, vertex_count=0, triangle_count=0, mesh_names=("mesh",), node_ids=(parent,))
    return scene, lod_mesh, relative


# --------------------------------------------------------------------------
# test 1: orientation — axes and UV quadrants
# --------------------------------------------------------------------------

# One vertex per axis (plus the origin), each with a UV in the middle of its
# own quadrant of the texture below. Unity UV convention (v grows upward).
TRIPOD_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.2]])
TRIPOD_UVS = np.array([[0.25, 0.75], [0.75, 0.75], [0.25, 0.25], [0.75, 0.25]])
QUADRANT_COLORS = {  # image quadrant (top/bottom, left/right) -> RGB
    ("top", "left"): (220, 30, 30),
    ("top", "right"): (30, 200, 30),
    ("bottom", "left"): (30, 60, 220),
    ("bottom", "right"): (230, 220, 40),
}


def _four_colour_texture(size: int = 64) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 3] = 255
    half = size // 2
    arr[:half, :half, :3] = QUADRANT_COLORS[("top", "left")]
    arr[:half, half:, :3] = QUADRANT_COLORS[("top", "right")]
    arr[half:, :half, :3] = QUADRANT_COLORS[("bottom", "left")]
    arr[half:, half:, :3] = QUADRANT_COLORS[("bottom", "right")]
    return Image.fromarray(arr, mode="RGBA")


def _sample(img: Image.Image, u: float, v_from_top: float) -> tuple[int, int, int]:
    width, height = img.size
    x = min(width - 1, int(u * width))
    y = min(height - 1, int(v_from_top * height))
    return tuple(np.asarray(img.convert("RGB"))[y, x].tolist())


def test_axis_tripod_lands_at_the_expected_coordinates():
    """Only X flips: an asymmetric model must come out mirrored in x and
    untouched in y/z, so a stray extra flip anywhere is visible."""

    geo = MeshGeometryRec(
        positions=TRIPOD_POSITIONS.astype(np.float32),
        normals=np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32),
        uvs=TRIPOD_UVS.astype(np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    )
    scene, lod_mesh, _relative = _chain_scene([Transform((0, 0, 0), IDENTITY_ROT, (1, 1, 1))])

    baked = bake_lod_mesh(_StubBackend(geo), scene, lod_mesh, ROOT_ID)

    np.testing.assert_allclose(baked.positions, TRIPOD_POSITIONS @ FLIP_X, atol=1e-6)
    # the +X arm is the only one that moved, and it moved to -X
    assert baked.positions[1].tolist() == [-0.4000000059604645, 0.0, 0.0]


def test_each_vertex_still_addresses_its_own_texel():
    """Unity UVs address a texture from the bottom-left, glTF from the
    top-left; the pixels are written unflipped (docs/manifest.md), so the
    `v = 1 - v` on the UVs has to land every vertex on the same colour it
    started on. Comparing flips alone would miss a double flip — this
    compares the actual texel."""

    source = _four_colour_texture()
    encoded = Image.open(BytesIO(encode_rgba_png(source, (64, 64))))

    geo = MeshGeometryRec(
        positions=TRIPOD_POSITIONS.astype(np.float32),
        normals=np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32),
        uvs=TRIPOD_UVS.astype(np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
    )
    scene, lod_mesh, _relative = _chain_scene([Transform((0, 0, 0), IDENTITY_ROT, (1, 1, 1))])
    baked = bake_lod_mesh(_StubBackend(geo), scene, lod_mesh, ROOT_ID)

    for vertex, (u, v_unity) in enumerate(TRIPOD_UVS):
        unity_colour = _sample(source, u, 1.0 - v_unity)  # Unity: v grows upward
        u_gltf, v_gltf = baked.uvs[vertex]
        gltf_colour = _sample(encoded, float(u_gltf), float(v_gltf))  # glTF: v grows downward
        assert gltf_colour == unity_colour, f"vertex {vertex} changed quadrant"

    # ... and the four vertices really do land on four different colours,
    # or the check above would pass on a uniform texture.
    sampled = {_sample(encoded, float(u), float(v)) for u, v in baked.uvs}
    assert len(sampled) == 4


# --------------------------------------------------------------------------
# test 2: normals + winding under nested rotations, scale and reflections
# --------------------------------------------------------------------------

TILTED_POSITIONS = np.array([[0.1, -0.2, 0.3], [0.6, 0.1, 0.2], [0.2, 0.5, -0.1]])
PARENT_ROT = (0.2, 0.4, 0.1, 0.885)  # arbitrary, normalized inside the composer
CHILD_ROT = (-0.35, 0.15, 0.5, 0.775)


@pytest.mark.parametrize(
    ("child_scale", "reflections"),
    [((1.0, 2.0, 0.5), 0), ((-1.0, 2.0, 0.5), 1), ((-1.0, -2.0, 0.5), 2)],
)
def test_tilted_triangle_normals_and_winding(child_scale, reflections):
    edge1 = TILTED_POSITIONS[1] - TILTED_POSITIONS[0]
    edge2 = TILTED_POSITIONS[2] - TILTED_POSITIONS[0]
    source_normal = np.cross(edge1, edge2)
    source_normal /= np.linalg.norm(source_normal)

    geo = MeshGeometryRec(
        positions=TILTED_POSITIONS.astype(np.float32),
        normals=np.tile(source_normal, (3, 1)).astype(np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
    )
    scene, lod_mesh, relative = _chain_scene([
        Transform((0.3, 0.0, -0.1), PARENT_ROT, (1.5, 1.5, 1.5)),
        Transform((0.0, 0.2, 0.0), CHILD_ROT, child_scale),
    ])

    baked = bake_lod_mesh(_StubBackend(geo), scene, lod_mesh, ROOT_ID)

    expected = (np.hstack([TILTED_POSITIONS, np.ones((3, 1))]) @ relative.T)[:, :3] @ FLIP_X
    np.testing.assert_allclose(baked.positions, expected, atol=1e-6)

    np.testing.assert_allclose(np.linalg.norm(baked.normals, axis=1), 1.0, atol=1e-6)

    tri = baked.indices.reshape(3)
    p0, p1, p2 = baked.positions[tri[0]], baked.positions[tri[1]], baked.positions[tri[2]]
    baked_edge1, baked_edge2 = p1 - p0, p2 - p0
    for normal in baked.normals:
        # perpendicular to the transformed surface, whatever the scale did
        assert abs(np.dot(normal, baked_edge1)) < 1e-6
        assert abs(np.dot(normal, baked_edge2)) < 1e-6
    # ... and the index order winds the same way the normals point
    assert np.dot(np.cross(baked_edge1, baked_edge2), baked.normals[0]) > 0

    # An odd number of reflections cancels the Unity->glTF x-flip, so the
    # triangle keeps its source order; an even number reverses it.
    reversed_order = list(tri) == [0, 2, 1]
    assert reversed_order == (reflections % 2 == 0)
