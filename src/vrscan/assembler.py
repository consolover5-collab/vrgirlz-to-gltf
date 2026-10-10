"""Turns a backend-agnostic `SceneRec` into `Scan` objects.

Scene shape (established facts, `~/.claude/plans/partitioned-exploring-pearl.md`,
plus `docs/DATASETS.md` for the second dataset): a root node named `Scans`
usually has one child per pose, named `NN-150K` (`NN` = 2-digit pose id).
Each pose root carries a `LODGroup` component whose levels reference
renderer GameObjects by screen-height threshold. Renderer names encode both
role and (for body/eyes/eyelashes) LOD level: `NN_LODk_MeshPartJ` (body, up
to 3 parts merged per level in Ф1) — found live on pose 91 (`~/vrscan/out/
manifest.json` review, 22.09): only LOD0/LOD1 actually split into
`_MeshPartJ`; LOD2+ ship as one renderer named plainly `NN_LODk` (no
MeshPart suffix, still merged into one primitive same as a MeshPart group of
size 1) — `profiles.DatasetProfile.body_re` matches both forms. Exact name
patterns (eyes/eyelashes/sofa/feet/shadow, and whether every pose root even
matches `NN-150K`) vary by source and live in `profiles.py`, not here — this
module only knows the *shape* (a LODGroup, renderer names classify into
kinds), not any one dataset's exact spelling of it.

This module does NOT bake transforms into geometry or merge MeshParts into
one primitive — both are Ф1 (`gltf/writer.py`). Here we only classify nodes
and group them into `Part`/`LodMesh` records with counts, for `vrscan scan`
and the manifest.

Robustness notes:
- If no `Scans` grouping node exists in the source, pose roots are found by
  scanning the whole scene instead — see `find_pose_roots`.
- A pose root is any GameObject with its own `LODGroup` that references at
  least one body-shaped renderer (by `profile.body_re`) — not just a name
  matching `NN-150K`. Found live on the second dataset (22.09): one demo
  scan's root is named `Nat-42` (literally reused from another product),
  which no `NN-150K` regex would ever match, but its `LODGroup` looks
  exactly like every other pose's.
- A pose's own `LODGroup` is the only thing trusted to attribute a
  body/eyes/eyelashes renderer to *this* pose — a same-named renderer found
  elsewhere in a full subtree walk (found live: pose 33's own
  `33_LOD1_MeshPart0` sits, for no discernible reason, several hops deep
  under pose 21's tree in the messier of the two datasets) is skipped
  rather than guessed into this scan's LOD chain by name alone. sofa/feet/
  shadow have no LODGroup to check against, so their pose-prefixed name
  patterns are matched against `pose_id` directly instead (`_prefix_ok`).
- A pose's LODGroup can also reference a renderer that is NOT in the pose
  root's subtree (found live on Unity 5.3 sets). The LODGroup is the
  authority on what belongs to the pose, so such a renderer is still
  assembled into this scan — `_part_candidates` below — and `gltf/writer.py`
  resolves its matrix through the common scene root.
- A single renderer can appear in more than one of a LODGroup's levels
  (found live: pose 21's `Eyelashes` renderer is referenced by both L0 and
  L1 — one shared mesh, not swapped per level) — `_level_lookup` maps a
  renderer to *every* level it's used at, not just one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from vrscan.gltf.materials import material_export_signature, signature_textures
from vrscan.ir import LodLevel, LodMesh, MaterialDef, Part, Scan, TextureRef
from vrscan.profiles import DEFAULT_PROFILE, DatasetProfile
from vrscan.reader.backend import (
    GameObjectRec,
    LodGroupRec,
    MaterialRec,
    NodeId,
    SceneRec,
    TextureFingerprint,
    TexturePayloadError,
    node_id_str,
)
from vrscan.transforms import compose_world_matrix

logger = logging.getLogger("vrscan.assembler")

SCANS_ROOT_NAME = "Scans"
PIVOT_NAME = "Pivot"

# Parts that switch geometry across the pose's LODGroup levels. Everything
# else (sofa/feet/shadow) is a single fixed mesh, always rendered.
HAS_LOD_CHAIN = {"body", "eyes", "eyelashes"}

# Parts the viewer lets the user hide (docs/manifest.md's `parts[].hideable`
# — sofa pieces, feet, and the shadow plane are all optional set dressing).
HIDEABLE_KINDS = {"sofa", "feet", "shadow"}

# Kinds that additionally start hidden by default (`parts[].hidden`, viewer's
# initial partVisibility) — every dataset, no profile involved: the eye
# spheres are a rough approximation of a shader effect (design review 22.09
# point 6, "Сферы скрыть по умолчанию"), not scene dressing the owner would
# normally want off, so unlike HIDEABLE_KINDS they're hideable *because*
# they default to hidden, not the other way round. The viewer names the
# re-enable toggle "Corneal highlight", not "eyes".
HIDDEN_BY_DEFAULT_KINDS = {"eyes"}

# Shader property name -> manifest texture role, per the material dump in
# established facts (`IR Deform PreIntegrated Skin Wet Light Override`).
TEXTURE_ROLE_BY_PROPERTY = {
    "_MainTex": "baseColor",
    "_BumpTex": "normal",
    "_BumpMap": "normal",  # Marmoset shaders (eyes) use the stock Unity name
    "_OcclusionTex": "occlusion",
    "_SpecTex": "specular",
    "_WetSpecTex": "specularWet",
}


def _prefix_ok(pose_group: str | None, pose_id: str | None) -> bool:
    """sofa/feet/shadow name patterns optionally capture a 2-digit pose
    prefix (group "pose"); when present it must match the scan actually
    being assembled, or this is another pose's node reached through messy
    nesting (see module docstring). No prefix in the pattern (e.g. the
    shared `Sofa-Feet`, or bare `Shad`) means the name is kind-generic —
    always accepted. `pose_id=None` (direct `classify()` calls, e.g. tests)
    skips the check entirely, matching the pre-profile behaviour."""

    return pose_group is None or pose_id is None or pose_group == pose_id


def classify(
    name: str, profile: DatasetProfile = DEFAULT_PROFILE, pose_id: str | None = None
) -> tuple[str, str, int | None] | None:
    """Name -> (kind, part id, LOD level from the name, if any), or None if
    `name` isn't a part we recognize (group/helper nodes like `91sofa`)."""

    m = profile.body_re.match(name)
    if m:
        return ("body", "body", int(m.group("lod")))
    m = profile.eyes_re.match(name)
    if m:
        side = "L" if m.group("side") == "Left" else "R"
        return ("eyes", f"eye{side}", int(m.group("lod")))
    m = profile.eyelashes_re.match(name)
    if m:
        lod = m.groupdict().get("lod")
        return ("eyelashes", "eyelashes", int(lod) if lod is not None else None)
    m = profile.sofa_re.match(name)
    if m:
        groups = m.groupdict()
        if not _prefix_ok(groups.get("pose"), pose_id):
            return None
        part = groups.get("part")
        return ("sofa", f"sofa-{part}" if part else "sofa", None)
    m = profile.feet_re.match(name)
    if m:
        return ("feet", "sofa-feet", None)
    m = profile.shadow_re.match(name)
    if m:
        if not _prefix_ok(m.groupdict().get("pose"), pose_id):
            return None
        return ("shadow", "shadow", None)
    return None


def _lod_group_has_body(scene: SceneRec, lod_group: LodGroupRec, profile: DatasetProfile) -> bool:
    for level in lod_group.levels:
        for go_id in level.renderer_game_objects:
            go = scene.game_objects.get(go_id)
            if go is not None and profile.body_re.match(go.name):
                return True
    return False


def find_pose_roots(scene: SceneRec, profile: DatasetProfile = DEFAULT_PROFILE) -> list[GameObjectRec]:
    """A pose root is a GameObject with its own `LODGroup` referencing at
    least one body-shaped renderer — not necessarily one whose *own* name
    matches `profile.pose_root_re` (see module docstring, `Nat-42`)."""

    scans_node = scene.find_by_name(SCANS_ROOT_NAME)

    def matches(pool: list[GameObjectRec]) -> list[GameObjectRec]:
        return [go for go in pool if go.lod_group is not None and _lod_group_has_body(scene, go.lod_group, profile)]

    candidates: list[GameObjectRec] = []
    if scans_node is not None:
        pool = [go for child_id in scans_node.children if (go := scene.game_objects.get(child_id)) is not None]
        candidates = matches(pool)
    if not candidates:
        # No "Scans" node (or it had no matching children) — fall back to a
        # global scan so assembly still works.
        candidates = matches(list(scene.game_objects.values()))
    return sorted(candidates, key=lambda go: go.name)


def _descendants(scene: SceneRec, root: GameObjectRec) -> list[GameObjectRec]:
    out: list[GameObjectRec] = []
    stack = list(root.children)
    seen: set[NodeId] = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        go = scene.game_objects.get(node_id)
        if go is None:
            continue
        out.append(go)
        stack.extend(go.children)
    return out


def _find_lod_group(root: GameObjectRec, descendants: list[GameObjectRec]) -> LodGroupRec | None:
    if root.lod_group is not None:
        return root.lod_group
    for go in descendants:
        if go.lod_group is not None:
            return go.lod_group
    return None


def _level_lookup(lod_group: LodGroupRec | None) -> dict[NodeId, list[int]]:
    """Renderer GameObject id -> every LOD level it's used at (usually one;
    see module docstring for the shared-renderer case)."""

    lookup: dict[NodeId, list[int]] = defaultdict(list)
    if lod_group is None:
        return lookup
    for level_idx, level in enumerate(lod_group.levels):
        for go_id in level.renderer_game_objects:
            lookup[go_id].append(level_idx)
    return lookup


def _part_candidates(
    scene: SceneRec, descendants: list[GameObjectRec], level_lookup: dict[NodeId, list[int]]
) -> list[GameObjectRec]:
    """Every node this pose can contribute parts from: its subtree, plus any
    renderer its own LODGroup references from *outside* that subtree.

    The LODGroup is the pose's own authoritative list of renderers, so a
    renderer it names belongs to this pose even when the hierarchy puts the
    node elsewhere (found live on Unity 5.3 sets: lucid-21/ariel-45 carry
    nodes that are siblings of the pose root rather than descendants). A
    subtree walk alone drops those parts before `gltf/writer.py` ever gets a
    chance to place them (design review 22.09, "Дефект: `_build_parts` не
    увидит renderer вне поддерева") — `compose_world_matrix` resolves their
    matrix through the common scene root.

    Static parts (sofa/feet/shadow) have no LODGroup to be named by and are
    still taken from the subtree only, by the profile's name rules."""

    out = list(descendants)
    seen = {go.node_id for go in descendants}
    for go_id in level_lookup:
        if go_id in seen:
            continue
        go = scene.game_objects.get(go_id)
        if go is None:
            logger.warning("LODGroup references GameObject %s that is missing from the scene", node_id_str(go_id))
            continue
        seen.add(go_id)
        out.append(go)
    return out


def _resolve_material_def(scene: SceneRec, material: MaterialRec) -> MaterialDef:
    textures = []
    for prop_name, tex_id in material.textures.items():
        tex = scene.textures.get(tex_id)
        if tex is None:
            continue
        role = TEXTURE_ROLE_BY_PROPERTY.get(prop_name, "unknown")
        textures.append(
            TextureRef(role=role, id=tex.id, name=tex.name, width=tex.width, height=tex.height,
                       format=tex.format, byte_size=tex.byte_size, property_name=prop_name)
        )
    return MaterialDef(
        name=material.name, shader=material.shader, textures=tuple(textures),
        floats=dict(material.floats), colors=dict(material.colors),
    )


def _find_wet_material(scene: SceneRec, pose_id: str) -> MaterialDef | None:
    """The body's standalone "wet" material (e.g. "91Wet"), see
    `reader/unitypy_backend.py`'s dedicated name-scan pass — not attached to
    any renderer, so not reachable through `_build_parts`'s renderer walk."""

    mat = scene.materials.get(f"{pose_id}Wet")
    return _resolve_material_def(scene, mat) if mat is not None else None


class PartMaterialError(ValueError):
    """One part's materials cannot be exported as the single material it
    carries — raised by the checks below. Holds what `cli.py` reports per
    scan (independent review 22.09, P2: "понятная ошибка с набором, частью и
    уровнями") so one bad pose fails alone instead of the whole dataset."""

    def __init__(self, message: str, part_id: str, levels: list[int]) -> None:
        super().__init__(message)
        self.part_id = part_id
        self.levels = levels


def _check_one_material(part_id: str, level: int, mesh_gos: list[GameObjectRec]) -> None:
    """Meshes merged into one primitive must share a material.

    `gltf/writer.py` merges every mesh of a part at one LOD level into a
    single glTF primitive, which can carry exactly one material — the
    assembled `Part` takes the representative renderer's first material and
    the rest are silently assumed to match (design review 22.09,
    "`representative.materials[0]` предполагает один материал"). If they
    don't, the output would be geometrically correct with the wrong
    material on part of it; say so instead."""

    materials = {go.materials[0] for go in mesh_gos if go.materials}
    if len(materials) > 1:
        raise PartMaterialError(
            f"part {part_id!r} LOD{level}: renderers merged into one primitive use different materials "
            f"({sorted(materials)}) — one primitive can only carry one material",
            part_id, [level],
        )


def _level_materials(
    scene: SceneRec, by_level: dict[int, list[GameObjectRec]]
) -> dict[int, MaterialDef | None]:
    """LOD level -> the material its renderers carry, taken from the level's
    first renderer — `_check_one_material` has already established that the
    renderers merged into one level agree."""

    materials: dict[int, MaterialDef | None] = {}
    for level, gos in sorted(by_level.items()):
        representative = gos[0]
        mat_rec = scene.materials.get(representative.materials[0]) if representative.materials else None
        materials[level] = _resolve_material_def(scene, mat_rec) if mat_rec is not None else None
    return materials


def _texture_names(kind: str, material: MaterialDef | None) -> dict[str, str | None] | None:
    """Role -> source texture name of what `material_export_signature`
    compares by fingerprint — for messages and audits only."""

    if material is None:
        return None
    return {role: tex and tex.name for role, tex in signature_textures(kind, material).items()}


def _check_cross_level_material_signature(
    scene: SceneRec, pose_id: str, part_id: str, kind: str, by_level: dict[int, list[GameObjectRec]],
    fingerprint: TextureFingerprint,
) -> None:
    """A part's material is taken once, from the representative renderer, and
    reused for every LOD level. Different *names* or PPtrs across levels are
    fine; a different effective export signature is not — switching LOD would
    then change the look in a way nothing in the package describes, and a
    warning is not enough to ship on (design review 22.09 point 5: "Не
    публиковать пакет с молчаливой подменой").

    Levels that all carry the one same material are equal without looking
    at a texture, so only a part whose levels name different materials
    reads texture payloads for their fingerprints."""

    if len({gos[0].materials[0] if gos[0].materials else None for gos in by_level.values()}) == 1:
        return
    materials = _level_materials(scene, by_level)
    try:
        signatures = {level: material_export_signature(kind, m, fingerprint) for level, m in materials.items()}
    except TexturePayloadError as exc:
        # An unreadable texture is this pose's error, not the whole set's.
        raise PartMaterialError(f"pose {pose_id} part {part_id!r}: {exc}", part_id, sorted(by_level)) from exc
    baseline_level = min(signatures)
    baseline = signatures[baseline_level]
    differing = [level for level, sig in signatures.items() if sig != baseline]
    if not differing:
        return
    other = signatures[differing[0]]
    fields = sorted({k for k in set(baseline) | set(other) if baseline.get(k) != other.get(k)})
    names = ""
    if "textures" in fields:
        names = (f"; textures {_texture_names(kind, materials[differing[0]])} vs "
                 f"{_texture_names(kind, materials[baseline_level])}")
    raise PartMaterialError(
        f"pose {pose_id} part {part_id!r}: LOD{differing[0]} exports a different material than "
        f"LOD{baseline_level} ({', '.join(fields)} differ: {[other.get(f) for f in fields]} vs "
        f"{[baseline.get(f) for f in fields]}{names}) — one part carries a single material, so this would "
        f"silently ship the wrong one for some levels",
        part_id, [baseline_level, *differing],
    )


def _group_part_levels(
    scene: SceneRec,
    descendants: list[GameObjectRec],
    level_lookup: dict[NodeId, list[int]],
    pose_id: str,
    profile: DatasetProfile,
    lod_group_present: bool,
) -> dict[tuple[str, str], dict[int, list[GameObjectRec]]]:
    """(kind, part id) -> LOD level -> the renderers that make up that level.

    The one place that decides which renderer belongs to which part at which
    level; `_build_parts` turns the result into `Part`s and
    `audit_lod_materials` reports on it without building anything."""

    groups: dict[tuple[str, str], dict[int, list[GameObjectRec]]] = defaultdict(lambda: defaultdict(list))
    for go in _part_candidates(scene, descendants, level_lookup):
        classified = classify(go.name, profile, pose_id)
        if classified is None or go.mesh is None:
            continue
        kind, part_id, name_level = classified
        levels = level_lookup.get(go.node_id)
        if levels is None:
            if kind in HAS_LOD_CHAIN and lod_group_present:
                # Matches a body/eyes/eyelashes name pattern but isn't
                # actually referenced by *this* pose's own LODGroup — almost
                # certainly another pose's renderer reached through messy
                # nesting (see module docstring). Skip rather than guess.
                continue
            levels = [name_level if name_level is not None else 0]
        for level in levels:
            groups[(kind, part_id)][level].append(go)
    return groups


def _build_parts(
    scene: SceneRec,
    descendants: list[GameObjectRec],
    level_lookup: dict[NodeId, list[int]],
    pose_id: str,
    profile: DatasetProfile,
    lod_group_present: bool,
    fingerprint: TextureFingerprint,
) -> list[Part]:
    groups = _group_part_levels(scene, descendants, level_lookup, pose_id, profile, lod_group_present)
    profile_hidden_kinds = profile.hidden_parts.get(pose_id, ())

    parts: list[Part] = []
    for (kind, part_id), by_level in groups.items():
        lods = []
        for level in sorted(by_level):
            gos = by_level[level]
            mesh_gos = [go for go in gos if go.mesh]
            mesh_names = tuple(go.mesh for go in mesh_gos)
            node_ids = tuple(go.node_id for go in mesh_gos)
            _check_one_material(part_id, level, mesh_gos)
            vertex_count = sum(scene.meshes[m].vertex_count for m in mesh_names if m in scene.meshes)
            triangle_count = sum(scene.meshes[m].triangle_count for m in mesh_names if m in scene.meshes)
            lods.append(
                LodMesh(level=level, vertex_count=vertex_count, triangle_count=triangle_count,
                        mesh_names=mesh_names, node_ids=node_ids)
            )

        # The representative supplies the part's material (and its manifest
        # name) for every level, so it must be a deterministic choice: the
        # first renderer of the finest level this part has, not whichever
        # node the subtree walk happened to reach first.
        representative = by_level[min(by_level)][0]
        _check_cross_level_material_signature(scene, pose_id, part_id, kind, by_level, fingerprint)
        material = None
        if representative.materials:
            mat_rec = scene.materials.get(representative.materials[0])
            if mat_rec is not None:
                material = _resolve_material_def(scene, mat_rec)

        hidden_by_default = kind in HIDDEN_BY_DEFAULT_KINDS or kind in profile_hidden_kinds
        parts.append(
            Part(
                id=part_id,
                kind=kind,
                name=representative.name,
                transform=representative.transform,
                lods=lods,
                material=material,
                hideable=(kind in HIDEABLE_KINDS or hidden_by_default),
                hidden=hidden_by_default,
                has_lod_chain=(kind in HAS_LOD_CHAIN),
                wet_material=_find_wet_material(scene, pose_id) if kind == "body" else None,
            )
        )
    return parts


def _build_scan_lods(lod_group: LodGroupRec | None, parts: list[Part]) -> list[LodLevel]:
    if lod_group is None:
        return []
    result = []
    for level_idx, level in enumerate(lod_group.levels):
        triangle_count = sum(
            lod_mesh.triangle_count
            for part in parts
            if part.has_lod_chain
            for lod_mesh in part.lods
            if lod_mesh.level == level_idx
        )
        result.append(LodLevel(level=level_idx, screen_height=level.screen_height, triangle_count=triangle_count))
    return result


def _find_pivot(scene: SceneRec, root: GameObjectRec, descendants: list[GameObjectRec]) -> tuple[float, float, float]:
    """The pose's `Pivot` node in pose-root space (Unity axes — the x-flip
    into glTF space happens once, in `gltf/writer.py`, alongside the one
    applied to vertices).

    A bare `transform.translation` is only the right answer for a Pivot that
    is a direct child of the pose root; nested under any intermediate node
    it is in the wrong space entirely (design review 22.09). Searching stays
    scoped to this root's own descendants, so a same-named node belonging to
    another pose can never be picked up."""

    for go in descendants:
        if go.name == PIVOT_NAME:
            matrix = compose_world_matrix(scene, go.node_id, root.node_id)
            return (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
    return (0.0, 0.0, 0.0)


@dataclass
class PoseResult:
    """One pose root's outcome: its `Scan`, or the expected error that kept
    it out. `vrscan scan` and `vrscan convert` walk the same list, so both
    report a broken pose the same way and carry on with the rest (the reviewer
    review 22.09 A3)."""

    scan_id: str
    root_name: str
    scan: Scan | None = None
    error: PartMaterialError | None = None


def assemble_poses(
    scene: SceneRec, profile: DatasetProfile = DEFAULT_PROFILE, scan_id_prefix: str = "", *,
    fingerprint: TextureFingerprint,
) -> list[PoseResult]:
    """Every pose found in `scene`, in pose-id order — assembled, or with the
    `PartMaterialError` its parts failed. Anything else raised is a bug and
    propagates. `scan_id_prefix` is per-*dataset* (`cli.py`'s slug of
    `_infer_app_name(data_dir)`), not per-profile — `profile.scan_id` only
    applies it when the profile opted into prefixing at all
    (`DatasetProfile.prefix_scan_ids`). `fingerprint` is the scene's
    backend's `get_texture_fingerprint`, what LOD levels' materials are
    compared by."""

    results = []
    for root in find_pose_roots(scene, profile):
        scan_id = profile.scan_id(root.name, scan_id_prefix)
        local_id = profile.local_id(root.name)
        descendants = _descendants(scene, root)
        lod_group = _find_lod_group(root, descendants)
        level_lookup = _level_lookup(lod_group)
        try:
            parts = _build_parts(
                scene, descendants, level_lookup, local_id, profile, lod_group is not None, fingerprint
            )
        except PartMaterialError as exc:
            results.append(PoseResult(scan_id=scan_id, root_name=root.name, error=exc))
            continue
        parts.sort(key=lambda p: (p.kind, p.id))
        scan = Scan(
            id=scan_id,
            root_name=root.name,
            root_node_id=root.node_id,
            pivot=_find_pivot(scene, root, descendants),
            parts=parts,
            lods=_build_scan_lods(lod_group, parts),
        )
        results.append(PoseResult(scan_id=scan_id, root_name=root.name, scan=scan))
    return results


def assemble(
    scene: SceneRec, profile: DatasetProfile = DEFAULT_PROFILE, scan_id_prefix: str = "", *,
    fingerprint: TextureFingerprint,
    on_error: Callable[[str, PartMaterialError], None] | None = None,
) -> list[Scan]:
    """The assembled scans of `assemble_poses`. A pose that failed raises its
    `PartMaterialError`; with `on_error` it is handed (scan id, error)
    instead and left out, so the other poses still assemble."""

    scans = []
    for result in assemble_poses(scene, profile, scan_id_prefix, fingerprint=fingerprint):
        if result.error is not None:
            if on_error is None:
                raise result.error
            on_error(result.scan_id, result.error)
            continue
        scans.append(result.scan)
    return scans


def audit_lod_materials(
    scene: SceneRec, profile: DatasetProfile = DEFAULT_PROFILE, scan_id_prefix: str = "", *,
    fingerprint: TextureFingerprint,
) -> list[dict[str, object]]:
    """Every part of every pose with the material each of its LOD levels
    would export, and whether those agree — the report form of
    `_check_cross_level_material_signature`, which raises instead (the reviewer
    review 22.09 point 5: "аудит всех частей 37 сканов"). Read-only and
    never raises, so it can be run on a dataset the converter now refuses.
    `scripts/audit-lod-materials.py` dumps it as JSON."""

    rows: list[dict[str, object]] = []
    for root in find_pose_roots(scene, profile):
        local_id = profile.local_id(root.name)
        descendants = _descendants(scene, root)
        lod_group = _find_lod_group(root, descendants)
        level_lookup = _level_lookup(lod_group)
        groups = _group_part_levels(
            scene, descendants, level_lookup, local_id, profile, lod_group is not None
        )
        for (kind, part_id), by_level in sorted(groups.items()):
            materials = _level_materials(scene, by_level)
            signatures = {level: material_export_signature(kind, m, fingerprint) for level, m in materials.items()}
            baseline = signatures[min(signatures)]
            rows.append({
                "scan": profile.scan_id(root.name, scan_id_prefix),
                "part": part_id,
                "kind": kind,
                "levels": sorted(by_level),
                "materialNames": {
                    level: (gos[0].materials[0] if gos[0].materials else None)
                    for level, gos in sorted(by_level.items())
                },
                "textureNames": {level: _texture_names(kind, m) for level, m in materials.items()},
                "consistent": all(sig == baseline for sig in signatures.values()),
                "signatures": {level: sig for level, sig in sorted(signatures.items())},
            })
    return rows


def audit_lod_renderers(
    scene: SceneRec, scans: list[Scan], profile: DatasetProfile = DEFAULT_PROFILE
) -> list[dict[str, object]]:
    """Every renderer reference in every pose's LODGroup, paired with what
    the assembled `scans` actually export for it — one row per reference,
    `status` either "exported" or the explicit reason it isn't (design review
    22.09 point 6: "каждая ссылка либо экспортирована, либо имеет явную
    причину исключения"). Read-only; `scripts/audit-lod-renderers.py` dumps
    it as JSON.

    Reasons mirror `_build_parts`' own skips exactly: `missing` (the
    referenced GameObject isn't in the scene at all), `no-mesh` (a renderer
    GameObject with no mesh — e.g. a group node), `unclassified` (a name no
    profile pattern recognizes as a part)."""

    scans_by_root = {scan.root_node_id: scan for scan in scans}
    rows: list[dict[str, object]] = []
    for root in find_pose_roots(scene, profile):
        scan = scans_by_root.get(root.node_id)
        lod_group = _find_lod_group(root, _descendants(scene, root))
        if lod_group is None:
            continue
        exported: dict[NodeId, tuple[str, list[int]]] = {}
        for part in scan.parts if scan is not None else []:
            for lod_mesh in part.lods:
                for nid in lod_mesh.node_ids:
                    exported.setdefault(nid, (part.id, []))[1].append(lod_mesh.level)
        for level_idx, level in enumerate(lod_group.levels):
            for node_id in level.renderer_game_objects:
                go = scene.game_objects.get(node_id)
                if node_id in exported:
                    part_id, levels = exported[node_id]
                    status, detail = "exported", {"part": part_id, "exportedLevels": sorted(set(levels))}
                elif go is None:
                    status, detail = "missing", {}
                elif go.mesh is None:
                    status, detail = "no-mesh", {}
                elif classify(go.name, profile, profile.local_id(root.name)) is None:
                    status, detail = "unclassified", {}
                else:
                    # Classified, has a mesh, in the scene — but no part
                    # carries it. Nothing in `_build_parts` should produce
                    # this; report it rather than let it pass as expected.
                    status, detail = "not-exported", {}
                rows.append({
                    "scan": scan.id if scan is not None else None,
                    "root": root.name,
                    "level": level_idx,
                    "node": go.name if go is not None else None,
                    "nodeId": node_id_str(node_id),
                    "mesh": go.mesh if go is not None else None,
                    "status": status,
                    **detail,
                })
    return rows
