#!/usr/bin/env python3
"""Generates `tests/fixtures/synthetic/` — a tiny synthetic scene in the
`fixture_backend.py` JSON+NPZ format, standing in for a real (vendor)
`.assets` file so tests can run on the VM with zero vendor bytes.

Mirrors the real naming/TRS/shader patterns from the plan for one pose
(`91`): `91-150K` root + LODGroup (2 levels, not the real 5 — enough to
exercise LOD grouping) -> `91_LOD0_MeshPart0/1`, `Eyes-Left/Right_LOD0`,
`Eyelashes_LOD0` at level 0, just `91_LOD1_MeshPart0`/`Eyes-Left_LOD1` at
level 1; a `91sofa` group with `91-C1`, `Sofa-Feet`, `91shad` (no LOD chain);
a `Pivot` node; a standalone `91Wet` material (unattached to any renderer,
see `reader/unitypy_backend.py`). All meshes are tiny procedural strips
authored Z-up (height along local Z, matching the real established fact
that mesh nodes carry a -90 deg X rotation converting Z-up source data to
Y-up) with real normals/UVs, not just vertex/triangle counts. Textures are
256x256 checkers with a distinguishing letter; `91_SPC`/`91WET_SPC` get real
(small) pixel data so `vrscan scan`'s brightness histogram has something to
decode.

Run: `uv run python scripts/make-fixture.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"

# Established TRS pattern: pose root at (0, 0.001, 0); mesh nodes rotated
# -90 degrees around X (Z-up source -> Y-up), quaternion (x,y,z,w).
ROOT_POS = (0.0, 0.001, 0.0)
MESH_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)
IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
PIVOT_POS = (0.416, 0.658, 0.510)


def identity_transform(t=(0.0, 0.0, 0.0), r=IDENTITY_ROT, s=(1.0, 1.0, 1.0)) -> dict:
    return {"t": list(t), "r": list(r), "s": list(s)}


def make_strip_mesh(seed: int, n_quads: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A tiny procedural quad strip, authored Z-up (height along local Z —
    the -90 deg X rotation on the renderer node stands it up into Y) ->
    (vertices Nx3 float32, normals Nx3 float32, uvs Nx2 float32, triangle
    indices Mx3 int32)."""

    rng = np.random.default_rng(seed)
    verts, normals, uvs = [], [], []
    for i in range(n_quads + 1):
        x = i * 0.1
        z_top = 0.2 + 0.01 * rng.standard_normal()
        verts.append((x, 0.0, 0.0))
        verts.append((x, 0.0, z_top))
        normals.append((0.0, 1.0, 0.0))
        normals.append((0.0, 1.0, 0.0))
        u = i / n_quads
        uvs.append((u, 0.0))
        uvs.append((u, 1.0))
    vertices = np.array(verts, dtype=np.float32)
    normals_arr = np.array(normals, dtype=np.float32)
    uvs_arr = np.array(uvs, dtype=np.float32)
    indices = []
    for i in range(n_quads):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        indices.append((a, b, c))
        indices.append((b, d, c))
    return vertices, normals_arr, uvs_arr, np.array(indices, dtype=np.int32)


def checker_with_letter(size: int, color_a: tuple, color_b: tuple, letter: str = "F", alpha: bool = False) -> Image.Image:
    mode = "RGBA" if alpha else "RGB"
    fill_a = (*color_a, 255) if alpha else color_a
    fill_b = (*color_b, 255) if alpha else color_b
    img = Image.new(mode, (size, size))
    tile = size // 8
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = fill_a if ((x // tile) + (y // tile)) % 2 == 0 else fill_b
    draw = ImageDraw.Draw(img)
    draw.text((size // 3, size // 3), letter, fill=(255, 0, 0, 255) if alpha else (255, 0, 0))
    if alpha:
        # a gradient alpha ramp (left=opaque, right=transparent) so shadow's
        # BLEND alpha channel actually carries information, not just 255.
        arr = np.asarray(img).copy()
        arr[:, :, 3] = np.linspace(255, 0, size, dtype=np.uint8)[None, :]
        img = Image.fromarray(arr, mode="RGBA")
    return img


def main() -> None:
    if OUT_DIR.exists():
        import shutil

        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "textures").mkdir(parents=True)

    mesh_arrays: dict[str, np.ndarray] = {}
    game_objects = []
    next_id = [1]

    def add_go(name, parent, transform, **kw):
        pid = next_id[0]
        next_id[0] += 1
        go = {"path_id": pid, "name": name, "parent": parent, "children": [], "transform": transform}
        go.update(kw)
        game_objects.append(go)
        if parent is not None:
            for g in game_objects:
                if g["path_id"] == parent:
                    g["children"].append(pid)
                    break
        return pid

    def add_mesh(name, seed, n_quads=4):
        vertices, normals, uvs, indices = make_strip_mesh(seed, n_quads)
        mesh_arrays[f"{name}__vertices"] = vertices
        mesh_arrays[f"{name}__normals"] = normals
        mesh_arrays[f"{name}__uvs"] = uvs
        mesh_arrays[f"{name}__indices"] = indices
        return name

    scans_root = add_go("Scans", None, identity_transform())
    pose_root = add_go(
        "91-150K",
        scans_root,
        identity_transform(ROOT_POS),
        lod_group={
            "levels": [
                {"screen_height": 0.25, "renderers": []},  # filled in below
                {"screen_height": 0.125, "renderers": []},
            ]
        },
    )

    # LOD0: 2 body MeshParts + eyes + eyelashes.
    body0_a = add_go("91_LOD0_MeshPart0", pose_root, identity_transform(r=MESH_ROT),
                      mesh=add_mesh("91_LOD0_MeshPart0", seed=1, n_quads=6), materials=["91_dry"])
    body0_b = add_go("91_LOD0_MeshPart1", pose_root, identity_transform(r=MESH_ROT),
                      mesh=add_mesh("91_LOD0_MeshPart1", seed=2, n_quads=6), materials=["91_dry"])
    eye_l0 = add_go("Eyes-Left_LOD0", pose_root, identity_transform(r=MESH_ROT, s=(0.919,) * 3),
                     mesh=add_mesh("Eyes-Left_LOD0", seed=3, n_quads=2), materials=["Eyes"])
    eye_r0 = add_go("Eyes-Right_LOD0", pose_root, identity_transform(r=MESH_ROT, s=(0.919,) * 3),
                     mesh=add_mesh("Eyes-Right_LOD0", seed=4, n_quads=2), materials=["Eyes"])
    lash0 = add_go("Eyelashes_LOD0", pose_root, identity_transform(r=MESH_ROT, s=(0.919,) * 3),
                    mesh=add_mesh("Eyelashes_LOD0", seed=5, n_quads=2), materials=["Eyelashes"])

    # LOD1: fewer body parts, coarser.
    body1_a = add_go("91_LOD1_MeshPart0", pose_root, identity_transform(r=MESH_ROT),
                      mesh=add_mesh("91_LOD1_MeshPart0", seed=6, n_quads=3), materials=["91_dry"])
    eye_l1 = add_go("Eyes-Left_LOD1", pose_root, identity_transform(r=MESH_ROT, s=(0.919,) * 3),
                     mesh=add_mesh("Eyes-Left_LOD1", seed=7, n_quads=1), materials=["Eyes"])

    pose_root_go = next(g for g in game_objects if g["path_id"] == pose_root)
    pose_root_go["lod_group"]["levels"][0]["renderers"] = [body0_a, body0_b, eye_l0, eye_r0, lash0]
    pose_root_go["lod_group"]["levels"][1]["renderers"] = [body1_a, eye_l1]

    # Sofa group: no LOD chain, single fixed meshes. C1 and Feet get their
    # own materials/textures (established fact: "Sofa-Feet_CLR/_NM/_AO" is a
    # distinct texture set from the leather pieces).
    sofa_group = add_go("91sofa", pose_root, identity_transform())
    add_go("91-C1", sofa_group, identity_transform(r=MESH_ROT),
           mesh=add_mesh("91-C1", seed=8, n_quads=5), materials=["91_sofa_C1"])
    add_go("Sofa-Feet", sofa_group, identity_transform(r=MESH_ROT),
           mesh=add_mesh("Sofa-Feet", seed=9, n_quads=2), materials=["91_feet"])
    add_go("91shad", sofa_group, identity_transform(r=MESH_ROT, s=(167.4, 167.4, 167.4)),
           mesh=add_mesh("91shad", seed=10, n_quads=1), materials=["Shadow"])

    add_go("Pivot", pose_root, identity_transform(PIVOT_POS))

    # Textures: checkers with a letter, small enough to be cheap to decode.
    textures = {}

    def add_texture(name: str, size: int, color_a: tuple, color_b: tuple, letter: str, fmt: str, alpha: bool = False) -> None:
        img = checker_with_letter(size, color_a, color_b, letter=letter, alpha=alpha)
        img.save(OUT_DIR / "textures" / f"{name}.png")
        byte_size = size * size * (4 if alpha else 3) // (2 if fmt == "DXT1" else 1)
        textures[name] = {"width": size, "height": size, "format": fmt, "byte_size": byte_size,
                           "image": f"textures/{name}.png"}

    add_texture("91_CLR", 256, (200, 150, 120), (120, 90, 70), "F", "DXT1")
    # DXT5nm-ish flat-blue standard normal map: B near-constant and high (~255)
    # -> swizzle detector should resolve this to "none".
    add_texture("91_NM", 256, (128, 128, 255), (140, 140, 255), "N", "DXT5")
    add_texture("91_SPC", 256, (40, 40, 40), (200, 200, 200), "S", "DXT5")
    add_texture("91WET_SPC", 256, (220, 220, 220), (250, 250, 250), "W", "DXT5")
    add_texture("91_RGB01", 32, (60, 60, 60), (90, 90, 90), "R", "RGB24")  # unmapped _MicroMixMap slot
    add_texture("Eye-T", 64, (255, 255, 255), (230, 230, 230), "E", "RGB24")
    add_texture("Eyelash", 64, (10, 10, 10), (30, 30, 30), "L", "DXT5", alpha=True)
    add_texture("91shad", 128, (0, 0, 0), (20, 20, 20), "D", "DXT5", alpha=True)
    add_texture("91-C1_CLR", 128, (90, 60, 40), (70, 45, 30), "C", "DXT1")
    add_texture("Sofa-Feet_CLR", 128, (50, 35, 25), (65, 45, 30), "T", "DXT1")

    materials = {
        "91_dry": {
            "shader": "IR Deform PreIntegrated Skin Wet Light Override",
            "textures": {"_MainTex": "91_CLR", "_BumpTex": "91_NM", "_SpecTex": "91_SPC",
                         "_MicroMixMap": "91_RGB01"},
            "colors": {"_Color": [0.647, 0.607, 0.571, 1.0]},
        },
        # Standalone wet material, per established fact never attached to a
        # renderer (the real .exe swaps `_SpecTex` at runtime) — found by
        # `reader/unitypy_backend.py`'s name-scan pass / `assembler.py`'s
        # `_find_wet_material`, both keyed on the fixed "<id>Wet" pattern.
        "91Wet": {"shader": "IR Deform PreIntegrated Skin Wet Light Override",
                  "textures": {"_SpecTex": "91WET_SPC"}},
        "Eyes": {"shader": "Marmoset Specular IBL", "textures": {"_MainTex": "Eye-T"}},
        "Eyelashes": {"shader": "IR Hair", "textures": {"_MainTex": "Eyelash"}, "floats": {"_Cutoff": 1.0}},
        "91_sofa_C1": {"shader": "IR Deform PreIntegrated Skin", "textures": {"_MainTex": "91-C1_CLR"}},
        "91_feet": {"shader": "IR Deform PreIntegrated Skin", "textures": {"_MainTex": "Sofa-Feet_CLR"}},
        "Shadow": {"shader": "Marmoset Diffuse IBL", "textures": {"_MainTex": "91shad"}, "floats": {"_Cutoff": 0.554}},
    }

    scene = {"game_objects": game_objects, "materials": materials, "textures": textures}
    (OUT_DIR / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    np.savez(OUT_DIR / "meshes.npz", **mesh_arrays)

    print(f"wrote fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
