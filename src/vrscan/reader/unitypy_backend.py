"""The only module that knows about UnityPy / `.assets` typetrees.

Loads exactly the files `profiles.DatasetProfile.source_file_patterns` says
to, resolved by `profiles.discover_source_files` — for Nat that's still
exactly `<Data>/level0` + `<Data>/sharedassets1.assets` (the original plan's
two files, unchanged); a second dataset can need more (`docs/DATASETS.md`:
Lucid Dreams II's per-pose hierarchies live in `level1`, not `level0`).
`resources.assets` (945 unrelated UI textures) is never opened, on either
dataset. `.resS`/`.resource` streaming companions are never named
explicitly — UnityPy resolves them itself from the primary file's
`StreamingInfo` once it's loaded by path (verified live on Lucid's 632 MB
`sharedassets1.assets` + separate 182 MB `.resS`/20 MB `.resource`
companions, 22.09: texture pixel decode and mesh vertex data both come back
correct without ever passing those paths to `UnityPy.load()`).

If no `--profile` override is given, the profile is auto-detected from the
Unity version string embedded in `level0`'s header — read via a cheap
throwaway `UnityPy.load()` of that one (always-present, always small) file,
*before* deciding the rest of the file list (see `discover()`). A folder that
isn't a supported dataset — no `level0`, a Unity release no profile covers, a
missing source file or streaming companion — is a `profiles.DatasetError`
naming what is wrong, raised before anything is converted.

Memory discipline (Ф0 risk #1, budget: RSS < 3 GB for a 1.8 GB source file):
- `UnityPy.load(path, ...)` is given plain path strings, which UnityPy turns
  into a buffered *file stream* (`EndianBinaryReader_Streamable_LocalFile`),
  not a fully-materialized in-memory buffer — so opening the file does not by
  itself pull 1.8 GB into RAM.
- Structural objects (GameObject/Transform/*Renderer/MeshFilter/LODGroup) are
  cheap and read eagerly to rebuild the scene graph.
- Mesh/Material/Texture2D objects are only resolved lazily, on first
  reference from a GameObject we actually visit, and cached by global id so a
  shared mesh/material/texture (e.g. `Eyes-Left_LOD0`, `leather_SPC`) is only
  parsed once.
- Texture2D metadata is read without ever calling `.image` (pixel
  decode/DXT decompression) — see `TextureRec`. `.read()` on a Texture2D does
  transiently materialize the raw (still-compressed) `image_data` bytes, but
  we never keep the parsed object around: only `width`/`height`/`format` and
  `object_reader.byte_size` (the on-disk size, free — no parse needed) end up
  in `TextureRec`. The parsed object is dropped immediately, so peak RSS is
  bounded by the single largest texture in flight, not the sum of all of
  them. Actual pixel decode only happens in `get_texture_image()`, called by
  `cli.py` for the SPC/WET_SPC brightness histograms — a couple of small
  (1024²) textures per pose.
"""

from __future__ import annotations

import ntpath
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import UnityPy
from UnityPy.enums import TextureFormat as UnityTextureFormat
from UnityPy.helpers.MeshHelper import MeshHandler

from vrscan.ir import Transform
from vrscan.profiles import (
    DatasetError,
    DatasetProfile,
    SupportStatus,
    discover_source_files,
    match_profile,
)
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

_RENDERER_TYPES = {"MeshRenderer", "SkinnedMeshRenderer"}

# Auxiliary "wet" material for a pose's body, e.g. "91Wet" — never attached to
# any renderer (the original .exe swaps `_SpecTex` at runtime instead of
# swapping materials), so it's invisible to the renderer-driven resolve_material
# traversal below. Found by a dedicated name-scan pass, see `load()`.
WET_MATERIAL_RE = re.compile(r"^\d{2}Wet$")

# Global object ids. A `path_id` is unique only *within* one serialized file,
# and Lucid-style datasets load several (`level0`, `level1`,
# `sharedassets1.assets`, ...). Keying the scene graph by bare path_id let
# objects from different files overwrite each other — found live 22.09 on
# Lucid: `game_objects[21]` was pose 41's `Shad` node (one file) sitting where
# pose 21's LODGroup expected its own renderer (another file), so lucid-21's
# shadow part merged materials '21shad' and '41shad' and the convert refused.
# Every GameObject-keyed structure below therefore uses `_gid()`, which returns
# the `(file index, path_id)` pair itself (`backend.NodeId`) rather than
# packing it into one int: Unity pathIDs are signed 64-bit and an asset-bundle
# file can legitimately carry negative or >40-bit ones, which no bit packing
# survives without either colliding or refusing to convert (design review 22.09
# point 6).


def _gid(file_index: int, path_id: int) -> NodeId:
    return (file_index, path_id)


def companion_candidates(basename: str) -> list[str]:
    """The file names UnityPy tries for a payload streamed from `basename`
    (`UnityPy.helpers.ResourceReader.get_resource_data`), in its order."""

    name, _ext = ntpath.splitext(basename)
    return [basename, f"{name}.resource", f"{name}.assets.resS", f"{name}.resS"]


class UnityPyBackend:
    """`ReaderBackend` implementation backed by UnityPy.

    `profile`, if given, skips auto-detection (`--profile` override);
    `None` (default) detects from `level0`'s Unity version on `load()`.
    After `discover()` (the first thing `load()` does) returns,
    `self.profile`/`self.unity_version`/`self.source_files` describe what
    is actually used — `cli.py` reads these to fill in `manifest.json`'s
    per-scan `source` — and `self.support`/`self.support_note` how
    established that profile is for this release (`profiles.match_profile`)."""

    def __init__(self, profile: DatasetProfile | None = None) -> None:
        self._profile_override = profile
        self.support: SupportStatus | None = None
        self.support_note = ""
        self._texture_readers: dict[TextureId, Any] = {}  # texture id -> ObjectReader (unparsed, cheap)
        self._mesh_readers: dict[str, Any] = {}  # mesh name -> ObjectReader (unparsed, cheap)
        self._fingerprints: dict[TextureId, str] = {}  # texture id -> fingerprint, once per run
        # The loaded UnityPy environment — alive anyway through the readers
        # above; kept so audits can look past what materials reference.
        self._env: Any = None
        self.profile = profile
        self.unity_version = ""
        self.source_files: list[Path] = []

    def discover(self, data_dir: Path) -> None:
        """What `load()` will read, decided from the folder alone: the Unity
        release (`level0`'s header), the profile and how established it is,
        the source files. Raises `DatasetError` naming the problem for a
        folder that isn't a supported dataset."""

        if not data_dir.is_dir():
            raise DatasetError(f"{data_dir} is not a folder")
        level0 = data_dir / "level0"
        if not level0.is_file():
            raise DatasetError(
                f"{data_dir.name} has no level0 — choose the game's *_Data folder (the one next to its .exe)"
            )
        self.unity_version = self._peek_unity_version(level0)
        if not self.unity_version:
            raise DatasetError(f"{data_dir.name}/level0 can't be read as a Unity file — choose the game's *_Data folder")
        match = match_profile(self.unity_version, self._profile_override)
        self.profile, self.support, self.support_note = match.profile, match.support, match.note
        self.source_files = discover_source_files(data_dir, self.profile)

    def load(self, data_dir: Path) -> SceneRec:
        self.discover(data_dir)
        env = UnityPy.load(*(str(p) for p in self.source_files))
        self._env = env

        file_index: dict[int, int] = {}  # id(SerializedFile) -> load-order index, see `_gid`

        def gid_of_reader(reader: Any) -> NodeId:
            idx = file_index.setdefault(id(reader.assets_file), len(file_index))
            return _gid(idx, reader.path_id)

        def gid_of_ptr(ptr: Any) -> NodeId | None:
            """Global id of the object a PPtr points at, or None when it points
            nowhere / into a file that was never loaded (built-in resources)."""
            if not ptr or not ptr.path_id:
                return None
            try:
                reader = ptr.deref()
            except (FileNotFoundError, ValueError, AttributeError, KeyError):
                return None
            return gid_of_reader(reader)

        game_objects: dict[NodeId, GameObjectRec] = {}
        transforms: dict[NodeId, Any] = {}
        renderer_by_go: dict[NodeId, Any] = {}
        meshfilter_by_go: dict[NodeId, Any] = {}
        lodgroup_by_go: dict[NodeId, Any] = {}

        # Pass 1: read every structural (cheap) object, bucket by kind.
        for obj in env.objects:
            type_name = obj.type.name
            if type_name == "GameObject":
                data = obj.read()
                gid = gid_of_reader(obj)
                game_objects[gid] = GameObjectRec(
                    node_id=gid,
                    name=data.m_Name,
                    transform=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
                    parent=None,
                )
            elif type_name == "Transform":
                transforms[gid_of_reader(obj)] = obj.read()
            elif type_name in _RENDERER_TYPES:
                data = obj.read()
                go_gid = gid_of_ptr(data.m_GameObject)
                if go_gid is not None:
                    renderer_by_go[go_gid] = data
            elif type_name == "MeshFilter":
                data = obj.read()
                go_gid = gid_of_ptr(data.m_GameObject)
                if go_gid is not None:
                    meshfilter_by_go[go_gid] = data
            elif type_name == "LODGroup":
                data = obj.read()
                go_gid = gid_of_ptr(data.m_GameObject)
                if go_gid is not None:
                    lodgroup_by_go[go_gid] = data
            # Mesh / Material / Texture2D: resolved lazily in pass 3.

        # Pass 2: wire up transform (position/rotation/scale, parent, children)
        # from the Transform components onto the GameObjectRecs.
        for t in transforms.values():
            go_gid = gid_of_ptr(t.m_GameObject)
            go = game_objects.get(go_gid) if go_gid is not None else None
            if go is None:
                continue
            go.transform = Transform(
                translation=(t.m_LocalPosition.x, t.m_LocalPosition.y, t.m_LocalPosition.z),
                rotation=(
                    t.m_LocalRotation.x,
                    t.m_LocalRotation.y,
                    t.m_LocalRotation.z,
                    t.m_LocalRotation.w,
                ),
                scale=(t.m_LocalScale.x, t.m_LocalScale.y, t.m_LocalScale.z),
            )
            father_gid = gid_of_ptr(t.m_Father)
            father_t = transforms.get(father_gid) if father_gid is not None else None
            if father_t is not None:
                parent_gid = gid_of_ptr(father_t.m_GameObject)
                if parent_gid is not None:
                    go.parent = parent_gid
            for child_ptr in t.m_Children:
                child_gid = gid_of_ptr(child_ptr)
                child_t = transforms.get(child_gid) if child_gid is not None else None
                if child_t is None:
                    continue
                child_go_gid = gid_of_ptr(child_t.m_GameObject)
                if child_go_gid is not None:
                    go.children.append(child_go_gid)

        # Pass 3: resolve mesh/material/texture references lazily, caching by path_id.
        meshes: dict[str, MeshRec] = {}
        materials: dict[str, MaterialRec] = {}
        # Keyed by the texture's own global id — the registry *is* the cache.
        textures: dict[TextureId, TextureRec] = {}
        # Keyed by global id (`_gid`, via the dereferenced reader): a
        # PPtr's own `file_id` is relative to the *referencing* file's
        # externals table, so the same shared mesh/material would get a
        # different (file_id, path_id) from every file that points at it.
        mesh_cache: dict[NodeId, str] = {}
        material_cache: dict[NodeId, str] = {}
        # Files the resolved textures/meshes stream their payload from — a
        # companion is only opened when a payload is read, too late to say
        # clearly which file is missing, so they are checked after the walk.
        streamed_from: set[str] = set()

        def note_stream(data: Any) -> None:
            stream = getattr(data, "m_StreamData", None)
            if stream is not None and getattr(stream, "path", "") and getattr(stream, "size", 0):
                streamed_from.add(ntpath.basename(stream.path))

        def unique_key(name: str, taken: dict[str, Any], reader: Any) -> str:
            """Mesh/material names are NOT unique in these files; first name
            wins, later same-named objects get an `@path_id` suffix (and a
            `.file` suffix on top if even that collides across files)."""
            if name not in taken:
                return name
            key = f"{name}@{reader.path_id}"
            if key in taken:
                key = f"{key}.{file_index[id(reader.assets_file)]}"
            return key

        def resolve_texture(ptr: Any) -> TextureId | None:
            """The texture's global id — never its name: two Texture2D
            objects can share a name, and keying by it handed every material
            that used either one the pixels of whichever was resolved last
            (`backend.TextureId`)."""

            if not ptr or not ptr.path_id:
                return None
            obj_reader = ptr.deref()
            gid = gid_of_reader(obj_reader)
            if gid in textures:
                return gid
            if obj_reader.type.name != "Texture2D":
                # A shader texture slot can point at a RenderTexture (e.g. a
                # reflection probe) instead of a plain Texture2D — found live
                # on the real asset file, not in the established facts.
                # Nothing to export for those in Ф0/Ф1, skip.
                return None
            tex = obj_reader.read()
            note_stream(tex)
            try:
                fmt = UnityTextureFormat(tex.m_TextureFormat).name
            except ValueError:
                fmt = f"unknown({tex.m_TextureFormat})"
            textures[gid] = TextureRec(
                id=gid,
                name=tex.m_Name,
                width=tex.m_Width,
                height=tex.m_Height,
                format=fmt,
                byte_size=obj_reader.byte_size,
            )
            self._texture_readers[gid] = obj_reader  # kept for lazy pixel decode, no big buffer held
            return gid

        def resolve_material(ptr: Any) -> str | None:
            if not ptr or not ptr.path_id:
                return None
            obj_reader = ptr.deref()
            gid = gid_of_reader(obj_reader)
            if gid in material_cache:
                return material_cache[gid]
            mat = obj_reader.read()
            name = mat.m_Name
            shader_name = "unknown"
            if mat.m_Shader and mat.m_Shader.path_id:
                # A built-in shader may live in a Unity resources file we never
                # loaded (only level0 + sharedassets1.assets) — deref can fail
                # in several ways depending on where UnityPy gives up.
                try:
                    shader_name = mat.m_Shader.deref().read().m_Name
                except (FileNotFoundError, ValueError, AttributeError, KeyError):
                    shader_name = "unknown"
            tex_map: dict[str, TextureId] = {}
            for prop_name, texenv in mat.m_SavedProperties.m_TexEnvs or []:
                # Unity 4.x stores shader property names as `FastPropertyName`
                # (a wrapper with a `.name` str), not a plain str directly —
                # unwrap it so `_MainTex` etc. compare equal downstream
                # (assembler.py's TEXTURE_ROLE_BY_PROPERTY).
                prop_name = getattr(prop_name, "name", prop_name)
                tex_id = resolve_texture(texenv.m_Texture)
                if tex_id is not None:
                    tex_map[prop_name] = tex_id
            float_map: dict[str, float] = {}
            for prop_name, value in mat.m_SavedProperties.m_Floats or []:
                float_map[getattr(prop_name, "name", prop_name)] = float(value)
            color_map: dict[str, tuple[float, float, float, float]] = {}
            for prop_name, value in mat.m_SavedProperties.m_Colors or []:
                color_map[getattr(prop_name, "name", prop_name)] = (value.r, value.g, value.b, value.a)
            # Material names are NOT unique in these files (e.g. two different
            # "Eyes" materials: the real eye one and an unrelated IBL test one
            # found live on a real-data machine 22.09) — key by name, but on a collision
            # suffix with the path_id so a later material never overwrites an
            # earlier one. `material_cache` (by global id) keeps the key stable.
            key = unique_key(name, materials, obj_reader)
            materials[key] = MaterialRec(
                name=name, shader=shader_name, textures=tex_map, floats=float_map, colors=color_map
            )
            material_cache[gid] = key
            return key

        def resolve_mesh(ptr: Any) -> str | None:
            if not ptr or not ptr.path_id:
                return None
            obj_reader = ptr.deref()
            gid = gid_of_reader(obj_reader)
            if gid in mesh_cache:
                return mesh_cache[gid]
            mesh = obj_reader.read()
            note_stream(mesh)
            name = mesh.m_Name
            vertex_count = sum(sm.vertexCount for sm in mesh.m_SubMeshes)
            triangle_count = sum(sm.indexCount for sm in mesh.m_SubMeshes) // 3
            # Mesh names are NOT unique: Lucid has seven distinct per-pose
            # meshes all called `Eyelashes` (found live 22.09 — pose 21 got
            # another pose's lashes, floating 6 cm off the face). Same fix as
            # materials: first name wins, later same-named meshes get `@path_id`.
            key = unique_key(name, meshes, obj_reader)
            meshes[key] = MeshRec(name=key, vertex_count=vertex_count, triangle_count=triangle_count)
            # Kept for lazy geometry decode as the reader, not as `mesh`: a
            # parsed Mesh holds its raw vertex data — Nat-18's 281, held for
            # the whole run, were 392 MiB (tracemalloc on a real-data machine).
            self._mesh_readers[key] = obj_reader
            mesh_cache[gid] = key
            return key

        for go in game_objects.values():
            renderer = renderer_by_go.get(go.node_id)
            if renderer is not None:
                mesh_ptr = getattr(renderer, "m_Mesh", None)  # SkinnedMeshRenderer only
                if mesh_ptr is None:
                    meshfilter = meshfilter_by_go.get(go.node_id)
                    mesh_ptr = meshfilter.m_Mesh if meshfilter is not None else None
                if mesh_ptr is not None:
                    go.mesh = resolve_mesh(mesh_ptr)
                go.materials = [
                    name for name in (resolve_material(mp) for mp in renderer.m_Materials) if name
                ]

            lodgroup = lodgroup_by_go.get(go.node_id)
            if lodgroup is not None:
                levels = []
                for lod in lodgroup.m_LODs:
                    renderer_gos: list[NodeId] = []
                    for lod_renderer in lod.renderers:
                        rptr = lod_renderer.renderer
                        if not rptr or not rptr.path_id:
                            continue
                        try:
                            robj = rptr.deref().read()
                        except (FileNotFoundError, ValueError):
                            continue
                        rgo_gid = gid_of_ptr(getattr(robj, "m_GameObject", None))
                        if rgo_gid is not None:
                            renderer_gos.append(rgo_gid)
                    levels.append(
                        LodLevelRec(
                            screen_height=lod.screenRelativeHeight,
                            renderer_game_objects=tuple(renderer_gos),
                        )
                    )
                go.lod_group = LodGroupRec(levels=tuple(levels))

        # Pass 4: standalone "wet" body materials (e.g. "91Wet") are never
        # referenced by any renderer — the original .exe swaps `_SpecTex` at
        # runtime rather than swapping materials — so pass 3's renderer-driven
        # traversal never visits them. Find them directly by name.
        for obj in env.objects:
            if obj.type.name != "Material":
                continue
            data = obj.read()
            if not WET_MATERIAL_RE.match(data.m_Name) or data.m_Name in materials:
                continue
            resolve_material(_FakePPtr(obj))

        present = {p.name.lower() for p in data_dir.iterdir()}
        missing = sorted(
            name for name in streamed_from
            if not any(candidate.lower() in present for candidate in companion_candidates(name))
        )
        if missing:
            raise DatasetError(
                f"{data_dir.name} is incomplete — missing: {', '.join(missing)} "
                "(textures/meshes are stored there; copy the whole *_Data folder)"
            )

        return SceneRec(game_objects=game_objects, meshes=meshes, textures=textures, materials=materials)

    @staticmethod
    def _peek_unity_version(level0: Path) -> str:
        """`SerializedFile.unity_version` (e.g. "4.6.3f1") from `level0`
        alone — cheap (small file, no mesh/texture typetree parsing) and
        always present, so it can run *before* `profiles.discover_source_files`
        decides the rest of the file list, which itself depends on the
        version-derived profile."""

        peek_env = UnityPy.load(str(level0))
        for obj in peek_env.objects:
            return obj.assets_file.unity_version
        return ""

    def get_texture_image(self, texture_id: TextureId) -> Image | None:
        obj_reader = self._texture_readers.get(texture_id)
        if obj_reader is None:
            return None
        return obj_reader.read().image

    def get_texture_fingerprint(self, texture_id: TextureId) -> str | None:
        obj_reader = self._texture_readers.get(texture_id)
        if obj_reader is None:
            return None
        # Once per resource: the same Texture2D is asked for once per tier
        # and per material that uses it.
        cached = self._fingerprints.get(texture_id)
        if cached is None:
            tex = obj_reader.read()
            payload, params = texture_payload(tex, tex.m_Name)
            cached = texture_fingerprint(payload, params)
            del tex, payload
            self._fingerprints[texture_id] = cached
        return cached

    def get_mesh_geometry(self, name: str) -> MeshGeometryRec | None:
        obj_reader = self._mesh_readers.get(name)
        if obj_reader is None:
            return None
        handler = MeshHandler(obj_reader.read())
        handler.process()
        positions = np.asarray(handler.m_Vertices, dtype=np.float32).reshape(-1, 3)
        if handler.m_Normals:
            normals = np.asarray(handler.m_Normals, dtype=np.float32).reshape(-1, 3)[:, :3]
        else:
            normals = np.zeros_like(positions)
        if handler.m_UV0:
            uvs = np.asarray(handler.m_UV0, dtype=np.float32).reshape(-1, 2)
        else:
            uvs = np.zeros((len(positions), 2), dtype=np.float32)
        triangles_per_submesh = handler.get_triangles() or []
        if triangles_per_submesh:
            indices = np.concatenate(
                [np.asarray(tris, dtype=np.uint32).reshape(-1) for tris in triangles_per_submesh if tris]
            )
        else:
            indices = np.zeros(0, dtype=np.uint32)
        return MeshGeometryRec(positions=positions, normals=normals, uvs=uvs, indices=indices)


def texture_payload(tex: Any, name: str) -> tuple[bytes, list[tuple[str, object]]]:
    """(encoded image payload, canonical decode parameters) of a parsed
    Texture2D — the input of `texture_fingerprint`.

    The payload is what UnityPy's own decode reads (`get_image_data`): the
    inline `image data` if there is one, else `m_StreamData`'s byte range of
    the `.resS`/`.resource` companion. Hashing the serialized object instead
    (the previous key) covered only the stream's path/offset/size, so
    different pixels at the same offset kept the same key, and the texture's
    name was part of it (design review 22.09 A2).

    The parameters are the ones `Texture2DConverter.parse_image_data`
    actually uses: size, format, platform (+ its blob), and the Unity
    version only for crunched formats, the one place it changes the
    decoder. Mip count and image count shape the payload layout, so they
    are part of the identity too."""

    try:
        payload = tex.get_image_data()
    except (FileNotFoundError, ValueError) as exc:
        raise TexturePayloadError(f"texture {name!r}: can't read its image payload ({exc})") from exc
    if not payload:
        raise TexturePayloadError(f"texture {name!r}: its image payload is empty")
    reader = getattr(tex, "object_reader", None)
    fmt = int(tex.m_TextureFormat)
    try:
        crunched = "Crunched" in UnityTextureFormat(fmt).name
    except ValueError:
        crunched = False
    version = tuple(getattr(reader, "version", ())) if crunched else ()
    # Unity 5+ stores a count; Unity 4 only says whether there is a chain.
    mip_count = getattr(tex, "m_MipCount", None)
    if mip_count is None:
        mip_count = "chain" if getattr(tex, "m_MipMap", False) else 1
    params: list[tuple[str, object]] = [
        ("width", int(tex.m_Width)),
        ("height", int(tex.m_Height)),
        ("format", fmt),
        ("mipCount", mip_count),
        ("imageCount", int(getattr(tex, "m_ImageCount", 1) or 1)),
        ("platform", int(getattr(reader, "platform", 0) or 0)),
        ("platformBlob", tuple(getattr(tex, "m_PlatformBlob", None) or ())),
        ("crunchVersion", version),
    ]
    return payload, params


class _FakePPtr:
    """Adapts an already-resolved `ObjectReader` to the tiny PPtr interface
    `resolve_material`/`resolve_texture` need (`.path_id`, `.deref()`), for
    the standalone-material pass above where we start from the object
    itself, not from a renderer's `PPtr[Material]`."""

    # The resolvers key their caches by the *dereferenced* reader's global
    # id (`_gid`), so `deref()` returning the reader itself is all they need;
    # pass 4 additionally skips materials already keyed by name, so a
    # standalone material is never processed twice.

    def __init__(self, obj_reader: Any) -> None:
        self._obj = obj_reader
        self.path_id = obj_reader.path_id

    def deref(self) -> Any:
        return self._obj
