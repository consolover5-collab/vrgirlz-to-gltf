#!/usr/bin/env python3
"""Generates `tests/fixtures/synthetic_instances/` — a third synthetic scene
in the same JSON+NPZ shape as `make-fixture.py`, built around the node/root
questions the geometry review raised (design review 22.09, point 2 and test
3 "Экземпляры и корни"). Everything in it is deliberate:

- Two pose roots (`21-150K`, `45-150K`) whose bodies render the SAME mesh
  (`shared_tri`) through different transforms — one shared mesh in two
  hierarchies must come out at two different sets of coordinates.
- Pose 21's LOD1 body renderer (`21_LOD1`) is NOT in the pose root's
  subtree at all: it hangs off `Scans` and is only reachable through the
  pose's own LODGroup. A subtree walk drops it (the defect fixed in
  `assembler.py`'s `_part_candidates`); its matrix comes from the common
  scene root, `inv(W_root) @ W_node`.
- `Shad` reproduces the live lucid-21/ariel-45 failure: the pose root lists
  it among its `children`, but the node's own `parent` points at `Scans`,
  so walking *up* from it never reaches the pose root
  (`compose_world_matrix: ancestor chain of node 'Shad' never reaches pose
  root`, the error that stopped those two poses converting on 22.09).
- `Pivot` sits two levels down, under a rotated+translated `PivotHolder`,
  so a pivot read as a bare local `translation` is measurably wrong; a
  decoy `Pivot` under `Scans` and another under pose 45 pin that the search
  is scoped to one pose root.
- The pose roots themselves carry a non-identity TRS (translation + 90 deg
  yaw), so `inv(W_root)` actually does something.

Run: `uv run python scripts/make-fixture-instances.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_instances"

IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
MESH_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)  # Z-up source -> Y-up
YAW_90 = (0.0, 0.70710678, 0.0, 0.70710678)  # +90 deg about Y


def transform(t=(0.0, 0.0, 0.0), r=IDENTITY_ROT, s=(1.0, 1.0, 1.0)) -> dict:
    return {"t": list(t), "r": list(r), "s": list(s)}


def triangle_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One flat triangle with exact, hand-checkable coordinates — no RNG, so
    a test can assert baked positions to the last float."""

    vertices = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=np.float32)
    normals = np.array([[0.0, 1.0, 0.0]] * 3, dtype=np.float32)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, normals, uvs, indices


def main() -> None:
    if OUT_DIR.exists():
        import shutil

        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "textures").mkdir(parents=True)

    mesh_arrays: dict[str, np.ndarray] = {}
    game_objects: list[dict] = []
    next_id = [1]

    def go_by_id(path_id: int) -> dict:
        return next(g for g in game_objects if g["path_id"] == path_id)

    def add_go(name, parent, xform, **kw):
        pid = next_id[0]
        next_id[0] += 1
        go = {"path_id": pid, "name": name, "parent": parent, "children": [], "transform": xform}
        go.update(kw)
        game_objects.append(go)
        if parent is not None:
            go_by_id(parent)["children"].append(pid)
        return pid

    def add_mesh(name):
        vertices, normals, uvs, indices = triangle_mesh()
        mesh_arrays[f"{name}__vertices"] = vertices
        mesh_arrays[f"{name}__normals"] = normals
        mesh_arrays[f"{name}__uvs"] = uvs
        mesh_arrays[f"{name}__indices"] = indices
        return name

    shared = add_mesh("shared_tri")
    scans_root = add_go("Scans", None, transform())

    # --- pose 21: rotated/translated root, LOD1 renderer outside its subtree ---
    pose21 = add_go(
        "21-150K", scans_root, transform((1.0, 0.5, -2.0), r=YAW_90),
        lod_group={"levels": [{"screen_height": 0.25, "renderers": []}, {"screen_height": 0.125, "renderers": []}]},
    )
    body21_l0 = add_go("21_LOD0_MeshPart0", pose21, transform((0.05, 0.0, 0.0), r=MESH_ROT),
                        mesh=shared, materials=["21"])
    # LOD1 body: parented to `Scans`, never listed under the pose root —
    # only the pose's own LODGroup names it.
    body21_l1 = add_go("21_LOD1", scans_root, transform((0.7, 0.4, -1.5), r=MESH_ROT),
                        mesh=shared, materials=["21"])
    eye_l = add_go("Eyes-Left_LOD0", pose21, transform((0.0, 0.1, 0.0), r=MESH_ROT, s=(0.9,) * 3),
                    mesh=add_mesh("eye_l"), materials=["Eyes"])
    eye_r = add_go("Eyes-Right_LOD0", pose21, transform((0.02, 0.1, 0.0), r=MESH_ROT, s=(0.9,) * 3),
                    mesh=add_mesh("eye_r"), materials=["Eyes"])
    lash = add_go("Eyelashes", pose21, transform((0.0, 0.12, 0.0), r=MESH_ROT, s=(0.9,) * 3),
                   mesh=add_mesh("lash"), materials=["Eyelashes"])
    pose21_go = go_by_id(pose21)
    pose21_go["lod_group"]["levels"][0]["renderers"] = [body21_l0, eye_l, eye_r, lash]
    pose21_go["lod_group"]["levels"][1]["renderers"] = [body21_l1, lash]  # lash shared across levels

    add_go("21sofa", pose21, transform(r=MESH_ROT), mesh=add_mesh("sofa"), materials=["21sofa"])
    # `Shad`: listed among the pose root's children, but its own parent is
    # `Scans` — the live lucid-21/ariel-45 shape.
    shad = add_go("Shad", scans_root, transform((0.9, 0.0, -1.9), r=MESH_ROT, s=(50.0,) * 3),
                   mesh=add_mesh("shadow_plane"), materials=["21shad"])
    pose21_go["children"].append(shad)

    holder = add_go("PivotHolder", pose21, transform((0.1, 0.2, 0.3), r=YAW_90))
    add_go("Pivot", holder, transform((0.4, 0.6, 0.5)))

    # --- pose 45: same mesh, different hierarchy, its own Pivot ---
    pose45 = add_go(
        "45-150K", scans_root, transform((-1.0, 0.0, 3.0)),
        lod_group={"levels": [{"screen_height": 0.25, "renderers": []}]},
    )
    body45 = add_go("45_LOD0_MeshPart0", pose45, transform((0.0, 0.3, 0.0), r=MESH_ROT, s=(2.0, 1.0, 1.0)),
                     mesh=shared, materials=["45"])
    go_by_id(pose45)["lod_group"]["levels"][0]["renderers"] = [body45]
    add_go("Pivot", pose45, transform((0.7, 0.1, 0.2)))

    # decoy: a `Pivot` belonging to no pose at all.
    add_go("Pivot", scans_root, transform((9.0, 9.0, 9.0)))

    textures = {}

    def add_texture(name, size=32, color=(128, 128, 128), fmt="DXT1", alpha=False):
        mode = "RGBA" if alpha else "RGB"
        fill = (*color, 200) if alpha else color
        Image.new(mode, (size, size), fill).save(OUT_DIR / "textures" / f"{name}.png")
        textures[name] = {"width": size, "height": size, "format": fmt,
                           "byte_size": size * size * (4 if alpha else 3), "image": f"textures/{name}.png"}

    add_texture("21_CLR", color=(200, 150, 120))
    add_texture("45_CLR", color=(190, 145, 115))
    add_texture("Eye-T", color=(240, 240, 240), fmt="RGB24")
    add_texture("Eyelash", color=(20, 20, 20), fmt="DXT5", alpha=True)
    add_texture("21shad", color=(0, 0, 0), fmt="DXT5", alpha=True)
    add_texture("21sofa_CLR", color=(90, 60, 40))

    materials = {
        "21": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "21_CLR"}},
        "45": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "45_CLR"}},
        "Eyes": {"shader": "Marmoset Specular IBL", "textures": {"_MainTex": "Eye-T"}},
        "Eyelashes": {"shader": "IR Hair", "textures": {"_MainTex": "Eyelash"}, "floats": {"_Cutoff": 1.0}},
        "21shad": {"shader": "Marmoset Diffuse IBL", "textures": {"_MainTex": "21shad"}, "floats": {"_Cutoff": 0.5}},
        "21sofa": {"shader": "IR Deform PreIntegrated Skin", "textures": {"_MainTex": "21sofa_CLR"}},
    }

    scene = {"unity_version": "5.3.4p1", "game_objects": game_objects,
             "materials": materials, "textures": textures}
    (OUT_DIR / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    np.savez(OUT_DIR / "meshes.npz", **mesh_arrays)

    print(f"wrote fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
