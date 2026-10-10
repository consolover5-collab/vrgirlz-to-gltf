"""Backend-agnostic scene records + the `ReaderBackend` protocol.

`assembler.py` only ever sees these types — never UnityPy classes. That's
what lets `fixture_backend.py` stand in for `unitypy_backend.py` in tests
that run on the VM, which has no vendor asset file to read.

Everything here is *metadata*: names, counts, sizes. No vertex/index buffers,
no decoded pixels (Ф1 concern) — the one exception is `get_texture_image`,
which backends may decode lazily on request (used by `cli.py` for the
SPC/WET_SPC brightness histograms, where the plan explicitly allows decode
in Ф0 because the source textures are small: "здесь декод разрешён, 1024²,
дёшево").
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from vrscan.ir import Transform

if TYPE_CHECKING:
    from PIL.Image import Image

# How a GameObject is identified across the whole scene: the load-order index
# of the serialized file it came from, plus its Unity pathID. A pathID is only
# unique *within* one file and a Lucid-style dataset loads several, so the pair
# is the smallest thing that is actually unique (design review 22.09 point 6:
# a `file_index << 40 | path_id` packing is only safe for
# `0 <= path_id < 2**40`, and Unity pathIDs are signed 64-bit). Backends with a
# single source file (`fixture_backend.py`) use file index 0 throughout. These
# ids never leave Python — the manifest/viewer address parts by `Part.id`, not
# by node id — so nothing here is ever handed to JS as a Number.
NodeId = tuple[int, int]


def node_id_str(node_id: NodeId) -> str:
    """`<file index>:<pathID>` — the form for log/error messages (and the only
    form a node id should ever be written out in). Texture ids use it too."""

    file_index, path_id = node_id
    return f"{file_index}:{path_id}"


# How a Texture2D is identified: the same `(file index, pathID)` pair as a
# GameObject. Texture names are NOT unique in these files any more than mesh
# or material names are, and a registry keyed by name let the last texture of
# a name stand in for every other one — pixels, fingerprint and size alike
# (design review 22.09 «что осталось», step 1). Records, readers, fingerprints
# and material references are all keyed by this id; a texture's name is only
# a label for logs, provenance and audits.
TextureId = tuple[int, int]


@dataclass(frozen=True)
class MeshRec:
    name: str
    vertex_count: int
    triangle_count: int


@dataclass(frozen=True)
class MeshGeometryRec:
    """Actual vertex/index buffers for one mesh, Unity local space (left-handed,
    Y-up), decoded on demand by `ReaderBackend.get_mesh_geometry`.

    `uvs` keep Unity's convention (same origin as the source pixels); the
    `v = 1 - v` flip to glTF/image convention happens in `gltf/writer.py`,
    not here.
    """

    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32
    indices: np.ndarray  # (M,) uint32, M % 3 == 0, Unity winding


@dataclass(frozen=True)
class TextureRec:
    id: TextureId
    name: str  # label only, see `TextureId`
    width: int
    height: int
    format: str  # Unity TextureFormat name
    byte_size: int


@dataclass(frozen=True)
class MaterialRec:
    name: str
    shader: str
    textures: dict[str, TextureId]  # shader property name (e.g. "_MainTex") -> key into `SceneRec.textures`
    floats: dict[str, float] = field(default_factory=dict)  # e.g. "_Cutoff" -> 0.554
    colors: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)  # e.g. "_Color"


@dataclass(frozen=True)
class LodLevelRec:
    """One entry of a Unity LODGroup: a screen-height threshold plus the
    GameObjects (by `NodeId`) of the renderers visible at that level."""

    screen_height: float
    renderer_game_objects: tuple[NodeId, ...]


@dataclass(frozen=True)
class LodGroupRec:
    levels: tuple[LodLevelRec, ...]


@dataclass
class GameObjectRec:
    node_id: NodeId
    name: str
    transform: Transform
    parent: NodeId | None  # the parent GameObject, None for a scene root
    children: list[NodeId] = field(default_factory=list)
    mesh: str | None = None  # name of the Mesh this node renders, if any
    materials: list[str] = field(default_factory=list)  # material names, renderer order
    lod_group: LodGroupRec | None = None


@dataclass
class SceneRec:
    game_objects: dict[NodeId, GameObjectRec]
    meshes: dict[str, MeshRec]
    textures: dict[TextureId, TextureRec]  # in the order the backend resolved them
    materials: dict[str, MaterialRec]

    def find_by_name(self, name: str) -> GameObjectRec | None:
        for go in self.game_objects.values():
            if go.name == name:
                return go
        return None


# texture id -> `ReaderBackend.get_texture_fingerprint` of it; what
# `assembler.py` compares materials by, without holding the backend itself.
TextureFingerprint = Callable[[TextureId], str | None]


class ReaderBackend(Protocol):
    """What `assembler.py` (and, in Ф1, `gltf/writer.py`/`textures.py`) needs
    from a data source."""

    def load(self, data_dir: Path) -> SceneRec:
        """Read the whole scene graph + mesh/texture/material metadata."""
        ...

    def get_texture_image(self, texture_id: TextureId) -> Image | None:
        """Decode a texture to a Pillow image on demand, or None if unknown.

        Callers are expected to decode one texture at a time and drop the
        reference (`del` + let it be collected) before decoding the next —
        see `textures.py` module docstring for the memory budget this keeps.
        """
        ...

    def get_mesh_geometry(self, name: str) -> MeshGeometryRec | None:
        """Decode a mesh's vertex/index buffers on demand, or None if unknown."""
        ...

    def get_texture_fingerprint(self, texture_id: TextureId) -> str | None:
        """What the texture's pixels are, independent of where they are
        stored or what they are called: `texture_fingerprint` over the
        encoded image payload (inline, or the range of a `.resS`/`.resource`
        companion) and the parameters its decode actually uses. None if the
        id is unknown; `TexturePayloadError` if the texture is known but
        its payload can't be read. Cheap — no DXT/pixel decode. It is
        `textures.py`'s cache key, the normal map pin key (`normal_pins.py`)
        and the material signature's texture identity."""
        ...


class TexturePayloadError(ValueError):
    """A texture's encoded payload can't be read (a missing streaming
    companion, an empty image) — raised instead of fingerprinting its
    metadata, which would stay the same whatever the pixels are."""


FINGERPRINT_SCHEME = b"vrscan-texture-fingerprint/1"


def texture_fingerprint(payload: bytes, params: list[tuple[str, object]]) -> str:
    """sha256 over the encoded image payload and the canonical decode
    parameters (design review 22.09 A2). Every field is written as its
    length and then its bytes, so no two different inputs can concatenate
    to the same stream. Names, file paths and offsets are deliberately not
    parameters: a renamed or relocated texture with the same bytes is the
    same image."""

    digest = hashlib.sha256()

    def field_bytes(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    field_bytes(FINGERPRINT_SCHEME)
    for key, value in params:
        field_bytes(key.encode("utf-8"))
        field_bytes(repr(value).encode("utf-8"))
    field_bytes(b"payload")
    field_bytes(payload)
    return digest.hexdigest()
