"""Reads the JSON+NPZ synthetic fixture format produced by
`scripts/make-fixture.py`. Stands in for `unitypy_backend.py` in tests that
run on the VM, which has no copy of the vendor `.assets` file.

Fixture directory layout (see `tests/fixtures/synthetic/`):
    scene.json    — game objects, LOD groups, materials, texture metadata
    meshes.npz    — one `<mesh_name>__vertices` (Nx3 float32) and
                    `<mesh_name>__indices` (Mx3 int32) array pair per mesh
    textures/*.png — actual pixel data for textures the tests need to decode
                     (SPC/WET_SPC, for the brightness histogram)

`scene.json` schema (informal):
    {
      "game_objects": [
        {"path_id": int, "name": str, "parent": int|null, "children": [int, ...],
         "transform": {"t": [x,y,z], "r": [x,y,z,w], "s": [x,y,z]},
         "mesh": str|null, "materials": [str, ...],
         "lod_group": {"levels": [{"screen_height": float, "renderers": [path_id, ...]}]}|null}
      ],
      (a fixture is a single source file, so every `path_id` here becomes the
      `NodeId` `(0, path_id)` — see `reader/backend.py`.)
      "materials": {name: {"shader": str, "textures": {prop_name: texture_key}}},
      "textures": {texture_key: {"width": int, "height": int, "format": str,
                                  "byte_size": int, "image": "textures/foo.png"|null,
                                  "name": str (optional, default: texture_key)}}
    }

A texture's key is what materials reference it by, its `name` only its label
— two entries can carry the same name, like two Texture2D objects in a real
file. Its `TextureId` is `(1, <position in "textures">)`: textures get a file
index of their own, so a texture id never reads like a GameObject's. A
material slot naming a key that isn't in "textures" is dropped, the way the
real reader drops a slot it can't resolve to a Texture2D.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vrscan.ir import Transform
from vrscan.profiles import DatasetProfile, SupportStatus, match_profile
from vrscan.reader.backend import (
    GameObjectRec,
    LodGroupRec,
    LodLevelRec,
    MaterialRec,
    MeshGeometryRec,
    MeshRec,
    NodeId,
    SceneRec,
    TextureId,
    TexturePayloadError,
    TextureRec,
    texture_fingerprint,
)

if TYPE_CHECKING:
    from PIL.Image import Image


def _node_id(path_id: int) -> NodeId:
    """A fixture is one "file", so its scene.json path_ids all live in file 0."""

    return (0, int(path_id))


_TEXTURE_FILE_INDEX = 1  # see the module docstring


class FixtureBackend:
    """`ReaderBackend` implementation backed by a synthetic JSON+NPZ fixture.

    `profile`, `unity_version`, `source_files` mirror `UnityPyBackend`'s
    post-`load()` attributes (`cli.py` reads them the same way regardless
    of backend) — `scene.json`'s optional top-level `"unity_version"` key
    drives auto-detection here the same way a real `level0`'s header does
    for `UnityPyBackend`; omitted, it defaults to "4.6.3f1" (the original
    Nat fixture's implicit version, `vrgirlz-2015`), so every existing
    fixture/test keeps its un-prefixed scan ids."""

    def __init__(self, profile: DatasetProfile | None = None) -> None:
        self._profile_override = profile
        self.support: SupportStatus | None = None
        self.support_note = ""
        self._fixture_dir: Path | None = None
        self._texture_images: dict[TextureId, str] = {}  # texture id -> relative path to PNG
        self._mesh_arrays: dict[str, np.ndarray] = {}
        self._textures: dict[TextureId, TextureRec] = {}
        self.profile = profile
        self.unity_version = ""
        self.source_files: list[Path] = []

    def load(self, data_dir: Path) -> SceneRec:
        self._fixture_dir = data_dir
        scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
        mesh_arrays = np.load(data_dir / "meshes.npz") if (data_dir / "meshes.npz").exists() else {}
        self._mesh_arrays = dict(mesh_arrays)

        self.unity_version = scene.get("unity_version", "4.6.3f1")
        match = match_profile(self.unity_version, self._profile_override)
        self.profile, self.support, self.support_note = match.profile, match.support, match.note
        self.source_files = [data_dir / "scene.json"]

        game_objects: dict[NodeId, GameObjectRec] = {}
        for raw in scene["game_objects"]:
            t = raw["transform"]
            lod_group = None
            if raw.get("lod_group"):
                levels = tuple(
                    LodLevelRec(
                        screen_height=level["screen_height"],
                        renderer_game_objects=tuple(_node_id(r) for r in level["renderers"]),
                    )
                    for level in raw["lod_group"]["levels"]
                )
                lod_group = LodGroupRec(levels=levels)
            game_objects[_node_id(raw["path_id"])] = GameObjectRec(
                node_id=_node_id(raw["path_id"]),
                name=raw["name"],
                transform=Transform(
                    translation=tuple(t["t"]),
                    rotation=tuple(t["r"]),
                    scale=tuple(t["s"]),
                ),
                parent=_node_id(raw["parent"]) if raw.get("parent") is not None else None,
                children=[_node_id(c) for c in raw.get("children", [])],
                mesh=raw.get("mesh"),
                materials=list(raw.get("materials", [])),
                lod_group=lod_group,
            )

        meshes: dict[str, MeshRec] = {}
        mesh_names = {go.mesh for go in game_objects.values() if go.mesh}
        for name in mesh_names:
            vkey, ikey = f"{name}__vertices", f"{name}__indices"
            if vkey not in mesh_arrays or ikey not in mesh_arrays:
                raise KeyError(f"meshes.npz missing '{vkey}'/'{ikey}' for mesh referenced in scene.json")
            vertex_count = len(mesh_arrays[vkey])
            triangle_count = len(mesh_arrays[ikey])
            meshes[name] = MeshRec(name=name, vertex_count=vertex_count, triangle_count=triangle_count)

        textures: dict[TextureId, TextureRec] = {}
        ids_by_key: dict[str, TextureId] = {}
        self._texture_images = {}
        for index, (key, raw) in enumerate(scene.get("textures", {}).items()):
            tex_id = (_TEXTURE_FILE_INDEX, index)
            ids_by_key[key] = tex_id
            textures[tex_id] = TextureRec(
                id=tex_id,
                name=raw.get("name", key),
                width=raw["width"],
                height=raw["height"],
                format=raw["format"],
                byte_size=raw["byte_size"],
            )
            if raw.get("image"):
                self._texture_images[tex_id] = raw["image"]

        materials: dict[str, MaterialRec] = {
            name: MaterialRec(
                name=name,
                shader=raw["shader"],
                textures={prop: ids_by_key[key] for prop, key in raw["textures"].items() if key in ids_by_key},
                floats={k: float(v) for k, v in raw.get("floats", {}).items()},
                colors={k: tuple(v) for k, v in raw.get("colors", {}).items()},
            )
            for name, raw in scene.get("materials", {}).items()
        }

        self._textures = textures
        return SceneRec(game_objects=game_objects, meshes=meshes, textures=textures, materials=materials)

    def get_texture_image(self, texture_id: TextureId) -> Image | None:
        if self._fixture_dir is None or texture_id not in self._texture_images:
            return None
        from PIL import Image

        img = Image.open(self._fixture_dir / self._texture_images[texture_id])
        return img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")

    def get_texture_fingerprint(self, texture_id: TextureId) -> str | None:
        """The fixture's "encoded payload" is its PNG file, decoded with the
        width/height/format scene.json declares — the same shape of identity
        as `UnityPyBackend`'s, so neither the texture's name nor its file's
        name is part of it."""

        if self._fixture_dir is None or texture_id not in self._texture_images:
            return None
        rec = self._textures[texture_id]
        path = self._fixture_dir / self._texture_images[texture_id]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise TexturePayloadError(f"texture {rec.name!r}: can't read its image payload ({exc})") from exc
        return texture_fingerprint(payload, [("width", rec.width), ("height", rec.height), ("format", rec.format)])

    def get_mesh_geometry(self, name: str) -> MeshGeometryRec | None:
        vkey, nkey, ukey, ikey = (f"{name}__{part}" for part in ("vertices", "normals", "uvs", "indices"))
        if vkey not in self._mesh_arrays or ikey not in self._mesh_arrays:
            return None
        positions = self._mesh_arrays[vkey].astype(np.float32)
        normals = self._mesh_arrays[nkey].astype(np.float32) if nkey in self._mesh_arrays else np.zeros_like(positions)
        uvs = self._mesh_arrays[ukey].astype(np.float32) if ukey in self._mesh_arrays else np.zeros((len(positions), 2), dtype=np.float32)
        indices = self._mesh_arrays[ikey].astype(np.uint32).reshape(-1)
        return MeshGeometryRec(positions=positions, normals=normals, uvs=uvs, indices=indices)
