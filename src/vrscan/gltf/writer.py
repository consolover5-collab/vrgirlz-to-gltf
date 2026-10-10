"""Bakes a `Scan` (IR) + its `SceneRec`/`ReaderBackend` into the glTF package
`docs/manifest.md` describes: `static.<tier>.gltf`, `lod<K>.<tier>.gltf`,
`scan.gltf`, their shared `.bin` buffers, and `tex/<tier>/*`.

Coordinate conversion (docs/manifest.md): every Unity node TRS from the
renderer up to (not including) the pose root is baked into the vertices;
`x = -x` on positions/normals; triangle winding reversed; `v = 1 - v`. No
TANGENT attribute — `extras.vrscan.tangents = "none"` on every mesh.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pygltflib as gltf

from vrscan.gltf.materials import MaterialPlan, build_material_plans, material_key_for_part
from vrscan.ir import LodMesh, Part, Scan, TextureRef
from vrscan.presets import TIERS
from vrscan.profiles import NormalDecision, NormalRule, NormalRules
from vrscan.reader.backend import NodeId, ReaderBackend, SceneRec, node_id_str
from vrscan.textures import (
    CacheHit,
    TextureCache,
    encode_base_color,
    encode_normal,
    encode_orm,
    encode_rgba_png,
    fit_size,
    gpu_bytes,
    rgb_copy_evidence,
    write_bytes_with_gc,
)
from vrscan.transforms import check_invertible, compose_world_matrix

logger = logging.getLogger("vrscan.gltf.writer")

_FLOAT = gltf.FLOAT
_UINT = gltf.UNSIGNED_INT
_ARRAY_BUFFER = gltf.ARRAY_BUFFER
_ELEMENT_ARRAY_BUFFER = gltf.ELEMENT_ARRAY_BUFFER
_TRIANGLES = gltf.TRIANGLES


# --------------------------------------------------------------------------
# Unity -> glTF space: compose ancestor TRS, then flip.
# --------------------------------------------------------------------------


@dataclass
class BakedMesh:
    positions: np.ndarray  # (N, 3) float32, final glTF space
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32
    indices: np.ndarray  # (M,) uint32

    @property
    def is_empty(self) -> bool:
        return len(self.positions) == 0


def _check_geometry(name: str, geo: Any) -> None:
    """Attribute lengths and index sanity, before anything is written into a
    buffer (the reviewer geometry review 22.09, point 1). glTF requires NORMAL/
    TEXCOORD_0 to have exactly as many elements as POSITION and indices to
    address existing vertices in whole triangles — a backend that gets this
    wrong produces a file that loads but renders as garbage, so say so here
    instead."""

    count = len(geo.positions)
    for attribute, data in (("normals", geo.normals), ("uvs", geo.uvs)):
        if len(data) != count:
            raise ValueError(
                f"mesh {name!r}: {attribute} has {len(data)} elements, POSITION has {count} — "
                f"glTF requires every attribute to match POSITION"
            )
    indices = np.asarray(geo.indices)
    if indices.size % 3 != 0:
        raise ValueError(f"mesh {name!r}: {indices.size} indices is not a whole number of triangles")
    if indices.size and (int(indices.max()) >= count or int(indices.min()) < 0):
        raise ValueError(
            f"mesh {name!r}: index range [{int(indices.min())}, {int(indices.max())}] "
            f"is outside its {count} vertices"
        )


def bake_lod_mesh(
    backend: ReaderBackend, scene: SceneRec, lod_mesh: LodMesh, root_node_id: NodeId
) -> BakedMesh:
    """Fetch + transform + merge every source mesh a part contributes at one
    LOD level into a single primitive's worth of buffers, already in glTF
    space (see module docstring)."""

    pos_chunks: list[np.ndarray] = []
    norm_chunks: list[np.ndarray] = []
    uv_chunks: list[np.ndarray] = []
    idx_chunks: list[np.ndarray] = []
    vertex_offset = 0
    for name, node_id in zip(lod_mesh.mesh_names, lod_mesh.node_ids, strict=True):
        geo = backend.get_mesh_geometry(name)
        if geo is None or len(geo.positions) == 0:
            logger.warning("mesh %r: no geometry from backend, skipped", name)
            continue
        node = scene.game_objects.get(node_id)
        node_name = node.name if node is not None else node_id_str(node_id)
        _check_geometry(name, geo)
        matrix = check_invertible(compose_world_matrix(scene, node_id, root_node_id), node_name)
        rot_scale = matrix[:3, :3]
        pos = np.asarray(geo.positions, dtype=np.float64)
        world_pos = pos @ rot_scale.T + matrix[:3, 3]
        # Normals need the inverse-transpose of rot_scale, not rot_scale
        # itself — `rot_scale.T` is only correct under uniform scale (P1,
        # design review 22.09). In column-vector form that's
        # `n' = inverse(rot_scale).T @ n`; the row-vector convention already
        # used for positions above (`pos @ rot_scale.T`) contributes one
        # transpose on its own, so it collapses to a plain `inverse(rot_scale)`
        # here.
        world_norm = np.asarray(geo.normals, dtype=np.float64) @ np.linalg.inv(rot_scale)
        lengths = np.linalg.norm(world_norm, axis=1, keepdims=True)
        lengths[lengths < 1e-12] = 1.0
        world_norm = world_norm / lengths

        world_pos[:, 0] *= -1.0  # Unity -> glTF: x = -x
        world_norm[:, 0] *= -1.0

        uv = np.array(geo.uvs, dtype=np.float32, copy=True)
        if uv.size:
            uv[:, 1] = 1.0 - uv[:, 1]

        idx = np.asarray(geo.indices, dtype=np.uint32).reshape(-1, 3).copy()
        # Winding flips iff the FULL transform (this node's rot_scale,
        # composed with the x=-1 flip above) has a negative determinant —
        # not unconditionally (P1, design review 22.09). `det(full) =
        # det(diag(-1,1,1)) * det(rot_scale) = -det(rot_scale)`, so a plain
        # rotation+positive-scale node (det(rot_scale) > 0, the common case)
        # still always flips, but a node whose own transform already
        # contains a reflection (e.g. a mirrored body part, det < 0) cancels
        # the x-flip instead of stacking with it.
        if np.linalg.det(rot_scale) > 0:
            idx = idx[:, [0, 2, 1]]
        idx = (idx.reshape(-1) + vertex_offset).astype(np.uint32)

        pos_chunks.append(world_pos.astype(np.float32))
        norm_chunks.append(world_norm.astype(np.float32))
        uv_chunks.append(uv)
        idx_chunks.append(idx)
        vertex_offset += len(pos)

    if not pos_chunks:
        return BakedMesh(
            np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
            np.zeros((0, 2), np.float32), np.zeros((0,), np.uint32),
        )
    return BakedMesh(
        positions=np.concatenate(pos_chunks),
        normals=np.concatenate(norm_chunks),
        uvs=np.concatenate(uv_chunks),
        indices=np.concatenate(idx_chunks),
    )


# --------------------------------------------------------------------------
# geometry -> accessors/bufferViews/nodes/meshes, one "group" (static, or one
# LOD level) at a time, self-contained (buffer index 0 within its own bin).
# --------------------------------------------------------------------------


@dataclass
class GeometryGroup:
    nodes: list[gltf.Node] = field(default_factory=list)
    meshes: list[gltf.Mesh] = field(default_factory=list)
    accessors: list[gltf.Accessor] = field(default_factory=list)
    buffer_views: list[gltf.BufferView] = field(default_factory=list)
    buffer_bytes: bytearray = field(default_factory=bytearray)
    material_keys: list[str] = field(default_factory=list)  # parallel to meshes/nodes
    triangle_count: int = 0
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None


def _pad4(buf: bytearray) -> None:
    while len(buf) % 4 != 0:
        buf.append(0)


def _append_accessor(
    buf: bytearray, buffer_views: list[gltf.BufferView], accessors: list[gltf.Accessor],
    data: np.ndarray, component_type: int, type_str: str, target: int, with_minmax: bool = False,
) -> int:
    _pad4(buf)
    byte_offset = len(buf)
    raw = data.tobytes()
    buf.extend(raw)
    bv_index = len(buffer_views)
    buffer_views.append(gltf.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw), target=target))
    accessor = gltf.Accessor(bufferView=bv_index, byteOffset=0, componentType=component_type,
                              count=data.shape[0], type=type_str)
    if with_minmax and data.shape[0]:
        accessor.min = data.min(axis=0).tolist()
        accessor.max = data.max(axis=0).tolist()
    accessors.append(accessor)
    return len(accessors) - 1


def build_geometry_group(backend: ReaderBackend, scene: SceneRec, parts: list[Part], level: int | None,
                          root_node_id: NodeId) -> GeometryGroup:
    """`level=None` for static parts (single fixed mesh, `lods[0]`); an int
    for one LOD level of the LOD-group parts."""

    group = GeometryGroup()
    for part in parts:
        lod_mesh = next((lm for lm in part.lods if lm.level == (level if level is not None else 0)), None)
        if lod_mesh is None or not lod_mesh.mesh_names:
            continue
        baked = bake_lod_mesh(backend, scene, lod_mesh, root_node_id)
        if baked.is_empty:
            continue
        pos_idx = _append_accessor(group.buffer_bytes, group.buffer_views, group.accessors,
                                    baked.positions, _FLOAT, "VEC3", _ARRAY_BUFFER, with_minmax=True)
        norm_idx = _append_accessor(group.buffer_bytes, group.buffer_views, group.accessors,
                                     baked.normals, _FLOAT, "VEC3", _ARRAY_BUFFER)
        uv_idx = _append_accessor(group.buffer_bytes, group.buffer_views, group.accessors,
                                   baked.uvs, _FLOAT, "VEC2", _ARRAY_BUFFER)
        idx_idx = _append_accessor(group.buffer_bytes, group.buffer_views, group.accessors,
                                    baked.indices, _UINT, "SCALAR", _ELEMENT_ARRAY_BUFFER)
        primitive = gltf.Primitive(
            attributes=gltf.Attributes(POSITION=pos_idx, NORMAL=norm_idx, TEXCOORD_0=uv_idx),
            indices=idx_idx, mode=_TRIANGLES,
        )
        mesh = gltf.Mesh(primitives=[primitive], name=part.id, extras={"vrscan": {"tangents": "none"}})
        group.meshes.append(mesh)
        group.nodes.append(gltf.Node(mesh=len(group.meshes) - 1, name=part.id))
        group.material_keys.append(material_key_for_part(part))
        group.triangle_count += len(baked.indices) // 3
        pmin, pmax = baked.positions.min(axis=0), baked.positions.max(axis=0)
        group.bounds_min = pmin if group.bounds_min is None else np.minimum(group.bounds_min, pmin)
        group.bounds_max = pmax if group.bounds_max is None else np.maximum(group.bounds_max, pmax)
    return group


def _reindex_group(group: GeometryGroup, accessor_offset: int, bufferview_offset: int,
                    buffer_index: int) -> tuple[list[gltf.Accessor], list[gltf.BufferView], list[gltf.Mesh]]:
    """Clone a group's accessors/bufferViews/meshes for reuse in a document
    that concatenates it after other groups (`scan.gltf`) — offsets indices,
    repoints bufferViews at a different physical `Buffer`."""

    new_buffer_views = [
        gltf.BufferView(buffer=buffer_index, byteOffset=bv.byteOffset, byteLength=bv.byteLength, target=bv.target)
        for bv in group.buffer_views
    ]
    new_accessors = [
        gltf.Accessor(bufferView=acc.bufferView + bufferview_offset, byteOffset=acc.byteOffset,
                       componentType=acc.componentType, count=acc.count, type=acc.type, min=acc.min, max=acc.max)
        for acc in group.accessors
    ]
    new_meshes = []
    for mesh in group.meshes:
        prim = mesh.primitives[0]
        attrs = prim.attributes
        new_attrs = gltf.Attributes(
            POSITION=attrs.POSITION + accessor_offset if attrs.POSITION is not None else None,
            NORMAL=attrs.NORMAL + accessor_offset if attrs.NORMAL is not None else None,
            TEXCOORD_0=attrs.TEXCOORD_0 + accessor_offset if attrs.TEXCOORD_0 is not None else None,
        )
        new_prim = gltf.Primitive(attributes=new_attrs, indices=prim.indices + accessor_offset, mode=prim.mode)
        new_meshes.append(gltf.Mesh(primitives=[new_prim], name=mesh.name, extras=mesh.extras))
    return new_accessors, new_buffer_views, new_meshes


# --------------------------------------------------------------------------
# materials -> glTF Material/Image/Texture, per tier.
# --------------------------------------------------------------------------


_EXT_BY_ROLE_FORMAT = {"jpeg": "jpg", "png": "png"}


@dataclass
class TierMaterials:
    materials: list[gltf.Material]
    images: list[gltf.Image]
    textures: list[gltf.Texture]
    samplers: list[gltf.Sampler]
    index_by_key: dict[str, int]  # material key -> index into `materials`
    uses_unlit: bool


def build_tier_materials(plans: dict[str, MaterialPlan], texture_index_by_key: dict[str, int]) -> TierMaterials:
    materials: list[gltf.Material] = []
    index_by_key: dict[str, int] = {}
    uses_unlit = False
    for key, plan in plans.items():
        recipe = plan.recipe
        # `or 1.0` would coerce an intentional roughness_factor=0.0 to 1.0
        # (P1, design review 22.09) — only a genuinely unset (None) recipe
        # value should fall back to the default.
        roughness_factor = recipe.roughness_factor if recipe.roughness_factor is not None else 1.0
        pbr = gltf.PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=roughness_factor)
        base_color_idx = texture_index_by_key.get(f"{plan.prefix}_CLR")
        if base_color_idx is not None:
            pbr.baseColorTexture = gltf.TextureInfo(index=base_color_idx)
        bcf = plan.base_color_factor()
        if bcf is not None:
            pbr.baseColorFactor = list(bcf)
        material = gltf.Material(
            name=key, alphaMode=recipe.alpha_mode, doubleSided=recipe.double_sided, pbrMetallicRoughness=pbr,
        )
        if recipe.alpha_cutoff is not None:
            material.alphaCutoff = recipe.alpha_cutoff
        normal_idx = texture_index_by_key.get(f"{plan.prefix}_NM")
        if normal_idx is not None:
            material.normalTexture = gltf.NormalMaterialTexture(index=normal_idx)
        orm_idx = texture_index_by_key.get(f"{plan.prefix}_ORM")
        if orm_idx is not None:
            material.occlusionTexture = gltf.OcclusionTextureInfo(index=orm_idx)
            pbr.metallicRoughnessTexture = gltf.TextureInfo(index=orm_idx)
        if recipe.type == "unlit":
            material.extensions = {"KHR_materials_unlit": {}}
            uses_unlit = True
        if recipe.blend != "normal":
            material.extras = {"vrscan_blend": recipe.blend}  # viewer hint; plain BLEND for other consumers
        index_by_key[key] = len(materials)
        materials.append(material)
    return TierMaterials(materials=materials, images=[], textures=[], samplers=[],
                          index_by_key=index_by_key, uses_unlit=uses_unlit)


# --------------------------------------------------------------------------
# top level: write one scan's whole package.
# --------------------------------------------------------------------------


@dataclass
class TextureExport:
    key: str
    file: str  # relative to scan dir, e.g. "tex/4k/body_CLR.jpg"
    size: tuple[int, int]
    channels: str  # "rgb" | "rgba"
    gpu_bytes: int
    source_name: str  # provenance: original Unity texture name(s), "+"-joined


@dataclass
class NormalMapNote:
    """One normal map's decision for the whole scan (every tier, hit or
    miss) — `provenance.normalMaps[<key>]`."""

    source: str
    fingerprint: str | None
    decision: NormalDecision
    evidence: str | None = None  # `textures.rgb_copy_evidence`, when the map was exported


@dataclass
class GeometryFileResult:
    bytes: int
    triangles: int


@dataclass
class ScanWriteResult:
    dir_name: str
    pivot: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    body_bounds_min: tuple[float, float, float]
    body_bounds_max: tuple[float, float, float]
    height_m: float
    tiers: list[str]
    static: GeometryFileResult
    lods: list[tuple[int, float, GeometryFileResult]]  # (level, screenHeight, result)
    parts: list[Part]
    material_plans: dict[str, MaterialPlan]
    textures_by_tier: dict[str, dict[str, TextureExport]]
    normal_maps: dict[str, NormalMapNote] = field(default_factory=dict)  # texture key -> note


def _texture_targets_for_plan(plan: MaterialPlan) -> list[tuple[str, str]]:
    """(export role, output key) pairs a material plan needs — "orm_wet" is
    body's wet-variant override, exported alongside the dry ORM."""

    targets = [("baseColor", f"{plan.prefix}_CLR")] if plan.recipe.use_base_color_texture else []
    if plan.recipe.use_normal and plan.texture("normal") is not None:
        targets.append(("normal", f"{plan.prefix}_NM"))
    if plan.recipe.use_orm and (plan.texture("occlusion") is not None or plan.texture("specular") is not None):
        targets.append(("orm", f"{plan.prefix}_ORM"))
        if plan.wet_specular is not None:
            targets.append(("orm_wet", f"{plan.prefix}_ORM_wet"))
    return targets


def export_textures(
    backend: ReaderBackend, plans: dict[str, MaterialPlan], tier: str, out_dir: Path, scan_dir_rel: str,
    scan_dir_final_rel: str, cache: TextureCache, normal_rules: NormalRules,
    normal_maps: dict[str, NormalMapNote], log_step: Any,
) -> dict[str, TextureExport]:
    """Encode (or take from the cache) every texture `plans` need at `tier`.
    `normal_maps` is shared by every tier of the scan: each normal map is
    decided once, on the first tier that meets it."""

    tier_cfg = TIERS[tier]
    result: dict[str, TextureExport] = {}
    for plan in plans.values():
        for role, key in _texture_targets_for_plan(plan):
            if role == "baseColor":
                refs = [plan.texture("baseColor")] if plan.texture("baseColor") else []
            elif role == "normal":
                refs = [plan.texture("normal")]
            elif role == "orm":
                refs = [ref for ref in (plan.texture("occlusion"), plan.texture("specular")) if ref]
            else:  # orm_wet
                refs = [ref for ref in (plan.texture("occlusion"), plan.wet_specular) if ref]
            if not refs:
                continue
            # Sources are looked up by id; their names are only labels, for
            # the log and the provenance (two textures can share a name).
            source_names = [ref.name for ref in refs]

            tier_size = tier_cfg["normal"] if role == "normal" else tier_cfg["orm"] if role.startswith("orm") else tier_cfg["baseColor"]
            # The tier size caps the longer side; the source's aspect is kept
            # and it is never upscaled (22.09: 4096px outputs were found baked
            # from 2048px sources). orm/orm_wet pack two sources into one
            # image, sized from the larger of them in each dimension.
            size = fit_size(max(ref.width for ref in refs), max(ref.height for ref in refs), tier_size)

            if role == "normal" or (role == "baseColor" and plan.recipe.base_color_format == "png_rgba"):
                fmt = "png"
            else:
                fmt = "jpeg"
            ext = _EXT_BY_ROLE_FORMAT[fmt]
            rel_file = f"tex/{tier}/{key}.{ext}"
            out_path = out_dir / scan_dir_rel / rel_file

            # A source with no fingerprint can't be cached on: keying on its
            # *name* instead (the old fallback) would hand out a stale result
            # for changed data under an unchanged name. Skip the cache for it
            # entirely rather than key it on something that isn't content
            # (design review 22.09 point 2). A known texture whose payload
            # can't be read raises instead (`TexturePayloadError`).
            fingerprints = [backend.get_texture_fingerprint(ref.id) for ref in refs]
            rule = None
            if role == "normal":
                note = normal_maps.get(key)
                if note is None:
                    note = normal_maps[key] = NormalMapNote(
                        source_names[0], fingerprints[0], normal_rules.decide(fingerprints[0])
                    )
                    _warn_normal_decision(note, log_step)
                if note.decision.rule is None:
                    continue
                # The rule decides the output bytes, so it is part of the key.
                rule = note.decision.rule
            cache_key = (
                TextureCache.key(
                    source_hash="+".join(h for h in fingerprints if h is not None),
                    role=role, swizzle=rule.key() if rule is not None else "auto", size=size, fmt=fmt,
                )
                if all(h is not None for h in fingerprints)
                else None
            )
            cached = cache.lookup(cache_key, out_dir) if cache_key is not None else None
            # A hit is only usable once the exact bytes it stands for are in
            # this run's staging dir — the output path can hold another
            # run's/version's file, and the cached file itself can have been
            # replaced since. Anything that doesn't verify falls through to a
            # re-encode instead of leaving the wrong file in place.
            if cached is not None and _materialize(out_path, cached):
                evidence = cached.evidence
            else:
                log_step(f"{'+'.join(source_names)} -> {tier} {key}.{ext}")
                data, evidence = _encode(backend, role, refs, plan, size, rule, log_step)
                digest = hashlib.sha256(data).hexdigest()
                write_bytes_with_gc(out_path, data)
                if cache_key is not None:
                    # The path a *later* run will find this at: the staging
                    # dir is renamed to `scans/<id>` as soon as this scan
                    # finishes (cli.py), so remembering the `.tmp` name would
                    # make every entry a guaranteed miss next time.
                    cache.remember(cache_key, f"{scan_dir_final_rel}/{rel_file}", digest, evidence)
            if role == "normal" and normal_maps[key].evidence is None:
                normal_maps[key].evidence = evidence

            with_alpha = fmt == "png" and role != "normal"
            result[key] = TextureExport(
                key=key, file=rel_file, size=size, channels="rgba" if with_alpha else "rgb",
                gpu_bytes=gpu_bytes(*size), source_name="+".join(source_names),
            )
    return result


def _warn_normal_decision(note: NormalMapNote, log_step: Any) -> None:
    """A map without an audited fingerprint says so in the log, once per
    scan — its rule is the profile's guess, or there is none at all."""

    decision = note.decision
    if decision.status == "unverified":
        log_step(f"warning: normal map {note.source}: {decision.basis} — {decision.rule.key()} applied, unverified")
    elif decision.status == "skipped":
        log_step(f"warning: normal map {note.source}: {decision.basis} — skipped (--normal-swizzle includes it)")


def _drop_degenerate_wet(
    backend: ReaderBackend, plans: dict[str, MaterialPlan], log_step: Any
) -> dict[str, MaterialPlan]:
    """Drop a wet variant whose specular source is the same content as the
    dry one: it would encode to a byte-identical ORM, i.e. a switch in the
    viewer that does nothing (design review 22.09 point 2, seen on Nat-18-95).
    Better no variant at all — the viewer then greys the switch out instead
    of promising an effect.

    "Same content" means what `encode_orm` reads — the luma — not the whole
    texture: Nat-21's `NNWET_SPC` maps equal `NN_SPC` in R, G and B and
    differ only in alpha (mean 0.80 dry vs 0.09 wet, audit-wet-pair 22.09),
    so all four of its wet ORMs came out byte-identical to the dry ones.
    What that alpha means to the original shader is not established, so
    nothing is derived from it here."""

    result: dict[str, MaterialPlan] = {}
    for key, plan in plans.items():
        dry = plan.texture("specular")
        if plan.wet_specular is not None and dry is not None:
            wet_hash = backend.get_texture_fingerprint(plan.wet_specular.id)
            dry_hash = backend.get_texture_fingerprint(dry.id)
            same_source = plan.wet_specular.id == dry.id or (
                wet_hash is not None and wet_hash == dry_hash
            )
            if same_source:
                log_step(
                    f"{key}: no wet variant — its specular source "
                    f"({plan.wet_specular.name}) is identical to the dry one ({dry.name})"
                )
                plan = dataclasses.replace(plan, wet_specular=None)
            elif _same_luma(backend, dry, plan.wet_specular, log_step):
                log_step(
                    f"{key}: no wet variant — {plan.wet_specular.name} has the same luma as {dry.name} "
                    "(they differ only in channels the ORM does not read)"
                )
                plan = dataclasses.replace(plan, wet_specular=None)
        result[key] = plan
    return result


def _luma_identity(backend: ReaderBackend, ref: TextureRef) -> tuple[int, int, str] | None:
    """(width, height, sha256 of the 8-bit luma) of one texture, or None if
    the backend has no image for it. The decoded image and its luma are
    released before this returns, so a caller comparing two textures holds
    at most one decoded copy at a time (design review 22.09 B3) — the backend
    keeps none either: `get_texture_image` decodes afresh on every call."""

    img = backend.get_texture_image(ref.id)
    if img is None:
        return None
    width, height = img.size
    luma = img.convert("L")
    del img
    digest = hashlib.sha256(luma.tobytes()).hexdigest()
    del luma
    return width, height, digest


def _same_luma(backend: ReaderBackend, dry: TextureRef, wet: TextureRef, log_step: Any) -> bool:
    identities = []
    for ref in (dry, wet):
        identity = _luma_identity(backend, ref)
        if identity is None:
            log_step(f"warning: specular {ref.name} has no readable image — luma of {dry.name} and {wet.name} "
                     "not compared, wet variant kept")
            return False
        identities.append(identity)
    return identities[0] == identities[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize(out_path: Path, cached: CacheHit) -> bool:
    """Make `out_path` hold exactly the bytes `cached` stands for, copying
    them from the cached file if needed. False = the cache entry can't be
    honoured (the cached file's content no longer matches its recorded hash),
    and the caller must re-encode."""

    if out_path.is_file() and _sha256_file(out_path) == cached.sha256:
        return True
    if _sha256_file(cached.path) != cached.sha256:
        logger.warning("cache: %s no longer matches its recorded hash, re-encoding", cached.path)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cached.path.read_bytes())
    return True


def _encode(backend: ReaderBackend, role: str, refs: list[TextureRef], plan: MaterialPlan, size: tuple[int, int],
            rule: NormalRule | None, log_step: Any) -> tuple[bytes, str | None]:
    """(encoded bytes, R≈G≈B evidence for a normal map / None otherwise)."""

    evidence = None
    if role == "baseColor":
        img = backend.get_texture_image(refs[0].id)
        data = (encode_rgba_png if plan.recipe.base_color_format == "png_rgba" else encode_base_color)(img, size)
        del img
    elif role == "normal":
        img = backend.get_texture_image(refs[0].id)
        data = encode_normal(img, size, rule)
        evidence = rgb_copy_evidence(img)
        log_step(f"normal {rule.key()}; {evidence}")
        del img
    elif role == "orm":
        occ = backend.get_texture_image(plan.texture("occlusion").id) if plan.texture("occlusion") else None
        spc = backend.get_texture_image(plan.texture("specular").id) if plan.texture("specular") else None
        data = encode_orm(occ, spc, size)
        del occ, spc
    else:  # orm_wet
        occ = backend.get_texture_image(plan.texture("occlusion").id) if plan.texture("occlusion") else None
        spc = backend.get_texture_image(plan.wet_specular.id)
        data = encode_orm(occ, spc, size)
        del occ, spc
    return data, evidence


def _attach_materials(group: GeometryGroup, material_index_by_key: dict[str, int]) -> None:
    for mesh, key in zip(group.meshes, group.material_keys, strict=True):
        mesh.primitives[0].material = material_index_by_key[key]


def _build_texture_objects(
    textures_by_key: dict[str, TextureExport],
) -> tuple[list[gltf.Image], list[gltf.Texture], list[gltf.Sampler], dict[str, int]]:
    images: list[gltf.Image] = []
    textures: list[gltf.Texture] = []
    index_by_key: dict[str, int] = {}
    for key, texp in textures_by_key.items():
        mime = gltf.IMAGEJPEG if texp.file.endswith(".jpg") else gltf.IMAGEPNG
        images.append(gltf.Image(uri=texp.file, mimeType=mime))
        textures.append(gltf.Texture(source=len(images) - 1, sampler=0))
        index_by_key[key] = len(textures) - 1
    samplers = (
        [gltf.Sampler(magFilter=gltf.LINEAR, minFilter=gltf.LINEAR_MIPMAP_LINEAR, wrapS=gltf.REPEAT, wrapT=gltf.REPEAT)]
        if images
        else []
    )
    return images, textures, samplers, index_by_key


def _save_gltf(
    nodes: list[gltf.Node], meshes: list[gltf.Mesh], accessors: list[gltf.Accessor],
    buffer_views: list[gltf.BufferView], buffers: list[gltf.Buffer], tier_materials: TierMaterials,
    out_path: Path,
) -> int:
    document = gltf.GLTF2(
        asset=gltf.Asset(version="2.0", generator="vrscan"),
        scene=0,
        scenes=[gltf.Scene(nodes=list(range(len(nodes))))],
        nodes=nodes, meshes=meshes, accessors=accessors, bufferViews=buffer_views, buffers=buffers,
        materials=tier_materials.materials, images=tier_materials.images,
        textures=tier_materials.textures, samplers=tier_materials.samplers,
    )
    if tier_materials.uses_unlit:
        document.extensionsUsed = ["KHR_materials_unlit"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document.save_json(str(out_path))
    return out_path.stat().st_size


def _buffers(uri: str, group: GeometryGroup) -> list[gltf.Buffer]:
    """The group's buffer, or none for a group without geometry (a pose
    with no static parts): glTF requires `byteLength >= 1`, and no
    bufferView would point at it."""

    return [gltf.Buffer(uri=uri, byteLength=len(group.buffer_bytes))] if group.buffer_bytes else []


def _vec3(a: np.ndarray) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


def write_scan(
    backend: ReaderBackend, scene: SceneRec, scan: Scan, out_dir: Path, tiers: list[str],
    normal_rules: NormalRules, cache: TextureCache, log_step: Any,
) -> ScanWriteResult:
    """Build the full `scans/<id>/` package for one scan. `out_dir` is the
    package root (i.e. `out/`); this writes under `out_dir / scan_dir_rel`
    where `scan_dir_rel` is the `.tmp` staging dir the caller picked — see
    `cli.py cmd_convert` for the tmp-dir-then-rename step."""

    scan_dir_rel = f"scans/{scan.id}.tmp"
    scan_dir = out_dir / scan_dir_rel
    # One decision per normal map, shared by every tier (design review 22.09
    # point 1: "Один источник должен получать одинаковый resolved swizzle
    # во всех тирах и при cache hit").
    normal_maps: dict[str, NormalMapNote] = {}

    plans = _drop_degenerate_wet(backend, build_material_plans(scan), log_step)
    material_index_by_key = {key: i for i, key in enumerate(plans.keys())}

    static_parts = [p for p in scan.parts if not p.has_lod_chain]
    lod_parts = [p for p in scan.parts if p.has_lod_chain]

    static_group = build_geometry_group(backend, scene, static_parts, None, scan.root_node_id)
    _attach_materials(static_group, material_index_by_key)
    (scan_dir / "geometry.static.bin").parent.mkdir(parents=True, exist_ok=True)
    (scan_dir / "geometry.static.bin").write_bytes(bytes(static_group.buffer_bytes))
    log_step(f"static: {len(static_group.meshes)} parts, {static_group.triangle_count} triangles")

    lod_groups: dict[int, GeometryGroup] = {}
    for lvl in scan.lods:
        group = build_geometry_group(backend, scene, lod_parts, lvl.level, scan.root_node_id)
        _attach_materials(group, material_index_by_key)
        (scan_dir / f"geometry.lod{lvl.level}.bin").write_bytes(bytes(group.buffer_bytes))
        log_step(f"lod{lvl.level}: {len(group.meshes)} parts, {group.triangle_count} triangles")
        lod_groups[lvl.level] = group

    # bounds: static+lod0 union; bodyBounds: lod0 (finest LOD-group parts) only
    all_mins = [g.bounds_min for g in [static_group, *lod_groups.values()] if g.bounds_min is not None]
    all_maxs = [g.bounds_max for g in [static_group, *lod_groups.values()] if g.bounds_max is not None]
    bounds_min = np.minimum.reduce(all_mins) if all_mins else np.zeros(3)
    bounds_max = np.maximum.reduce(all_maxs) if all_maxs else np.zeros(3)
    lod0 = lod_groups.get(0)
    body_min = lod0.bounds_min if lod0 and lod0.bounds_min is not None else bounds_min
    body_max = lod0.bounds_max if lod0 and lod0.bounds_max is not None else bounds_max
    height_m = float(body_max[1] - body_min[1])

    textures_by_tier: dict[str, dict[str, TextureExport]] = {}
    tier_materials_by_tier: dict[str, TierMaterials] = {}
    for tier in tiers:
        textures_by_tier[tier] = export_textures(
            backend, plans, tier, out_dir, scan_dir_rel, f"scans/{scan.id}", cache, normal_rules, normal_maps,
            lambda msg, tier=tier: log_step(f"[{tier}] {msg}"),
        )
        images, tex_objs, samplers, texture_index_by_key = _build_texture_objects(textures_by_tier[tier])
        tier_mats = build_tier_materials(plans, texture_index_by_key)
        tier_mats.images, tier_mats.textures, tier_mats.samplers = images, tex_objs, samplers
        tier_materials_by_tier[tier] = tier_mats

        _save_gltf(
            static_group.nodes, static_group.meshes, static_group.accessors, static_group.buffer_views,
            _buffers("geometry.static.bin", static_group), tier_mats, scan_dir / f"static.{tier}.gltf",
        )
        for level, group in lod_groups.items():
            _save_gltf(
                group.nodes, group.meshes, group.accessors, group.buffer_views,
                _buffers(f"geometry.lod{level}.bin", group), tier_mats, scan_dir / f"lod{level}.{tier}.gltf",
            )

    # scan.gltf: static + LOD0 of the best (highest baseColor resolution) available tier.
    best_tier = max(tiers, key=lambda t: TIERS[t]["baseColor"])
    lod0_group = lod_groups.get(0)
    if lod0_group is not None:
        static_buffers = _buffers("geometry.static.bin", static_group)
        accessor_offset = len(static_group.accessors)
        bufferview_offset = len(static_group.buffer_views)
        lod0_accessors, lod0_buffer_views, lod0_meshes = _reindex_group(
            lod0_group, accessor_offset, bufferview_offset, buffer_index=len(static_buffers)
        )
        combined_nodes = list(static_group.nodes) + [
            gltf.Node(mesh=len(static_group.meshes) + i, name=n.name) for i, n in enumerate(lod0_group.nodes)
        ]
        _save_gltf(
            combined_nodes,
            list(static_group.meshes) + lod0_meshes,
            list(static_group.accessors) + lod0_accessors,
            list(static_group.buffer_views) + lod0_buffer_views,
            static_buffers + _buffers("geometry.lod0.bin", lod0_group),
            tier_materials_by_tier[best_tier], scan_dir / "scan.gltf",
        )

    lods_result = [
        (lvl.level, lvl.screen_height,
         GeometryFileResult(bytes=(scan_dir / f"geometry.lod{lvl.level}.bin").stat().st_size,
                             triangles=lod_groups[lvl.level].triangle_count))
        for lvl in scan.lods if lvl.level in lod_groups
    ]

    return ScanWriteResult(
        dir_name=f"scans/{scan.id}",
        pivot=(-scan.pivot[0], scan.pivot[1], scan.pivot[2]),
        bounds_min=_vec3(bounds_min), bounds_max=_vec3(bounds_max),
        body_bounds_min=_vec3(body_min), body_bounds_max=_vec3(body_max),
        height_m=height_m, tiers=list(tiers),
        static=GeometryFileResult(bytes=len(static_group.buffer_bytes), triangles=static_group.triangle_count),
        lods=lods_result, parts=scan.parts, material_plans=plans, textures_by_tier=textures_by_tier,
        normal_maps=normal_maps,
    )
