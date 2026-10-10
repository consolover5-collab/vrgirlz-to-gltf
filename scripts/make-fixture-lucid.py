#!/usr/bin/env python3
"""Generates `tests/fixtures/synthetic_lucid/` — a second synthetic scene,
in the same JSON+NPZ shape as `make-fixture.py`, but shaped like the
*second* real dataset (`vrgirlz-2016` profile, see `docs/DATASETS.md`)
instead of Nat's (`vrgirlz-2015`). Exercises exactly what differs:

- `unity_version: "5.3.4p1"` in `scene.json` -> `profiles.detect_profile`
  picks `vrgirlz-2016`, not the Nat default.
- Two pose roots: `"21-150K"` (matches `pose_root_re`, ordinary case) and
  `"Nat-77"` (does not — found live on the real second dataset: one demo
  scan is literally named after a different product; `find_pose_roots`
  must still find it via its own LODGroup, and `scan_id`/`local_id` fall
  back to a slug, `ld2-nat-77`).
- `21-150K`'s `Eyelashes` renderer is referenced by *both* LODGroup levels
  (one shared mesh, not one renderer per level — found live on the real
  pose `21-150K`) — exercises `_level_lookup`'s multi-level support.
- Eyes use the `Eyes-Clear-Left/Right_LODk` naming (the `vrgirlz-2016`
  profile's looser `eyes_re`).
- A decoy `77_LOD1_MeshPart0` (pose 77's own body meshpart) is nested three
  hops deep under `21-150K`'s tree, NOT referenced by `21-150K`'s own
  LODGroup — pins the cross-pose-contamination guard in `_build_parts`
  (found live: a same-shaped defect on the real data, see `assembler.py`
  module docstring).
- `21sofa`/`77sofa` are single-piece sofas (not the Nat-style C1/L1/M1/SR1
  split); `Shad` (pose 21, bare name, no numeric prefix) and `77shad`
  (pose 77, prefixed) exercise both `shadow_re` branches; a decoy `21shad`
  nested under pose 77's tree pins the sofa/shadow prefix-mismatch guard
  (`_prefix_ok`).

Run: `uv run python scripts/make-fixture-lucid.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_lucid"

MESH_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)  # Z-up source -> Y-up, same convention as Nat
IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)


def identity_transform(t=(0.0, 0.0, 0.0), r=IDENTITY_ROT, s=(1.0, 1.0, 1.0)) -> dict:
    return {"t": list(t), "r": list(r), "s": list(s)}


def make_strip_mesh(seed: int, n_quads: int = 3):
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


def checker(size: int, color_a: tuple, color_b: tuple, letter: str, alpha: bool = False) -> Image.Image:
    mode = "RGBA" if alpha else "RGB"
    fill_a = (*color_a, 255) if alpha else color_a
    fill_b = (*color_b, 255) if alpha else color_b
    img = Image.new(mode, (size, size))
    tile = size // 8
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = fill_a if ((x // tile) + (y // tile)) % 2 == 0 else fill_b
    ImageDraw.Draw(img).text((size // 3, size // 3), letter, fill=(255, 0, 0, 255) if alpha else (255, 0, 0))
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

    def add_mesh(name, seed, n_quads=3):
        vertices, normals, uvs, indices = make_strip_mesh(seed, n_quads)
        mesh_arrays[f"{name}__vertices"] = vertices
        mesh_arrays[f"{name}__normals"] = normals
        mesh_arrays[f"{name}__uvs"] = uvs
        mesh_arrays[f"{name}__indices"] = indices
        return name

    scans_root = add_go("Scans", None, identity_transform())

    # --- pose "21-150K": ordinary name, messy nesting, shared eyelashes ---
    pose21 = add_go(
        "21-150K", scans_root, identity_transform((0.1, 0.6, 0.2)),
        lod_group={"levels": [{"screen_height": 0.25, "renderers": []}, {"screen_height": 0.125, "renderers": []}]},
    )
    body21_l0 = add_go("21_LOD0_MeshPart0", pose21, identity_transform(r=MESH_ROT),
                        mesh=add_mesh("21_LOD0_MeshPart0", seed=1), materials=["21"])
    # decoy: pose 77's own LOD1 meshpart, nested deep under pose 21's tree,
    # NOT part of pose 21's LODGroup — must not leak into pose 21's "body".
    add_go("77_LOD1_MeshPart0", body21_l0, identity_transform(r=MESH_ROT),
           mesh=add_mesh("77_LOD1_MeshPart0", seed=99), materials=["77"])
    eye_l0_left = add_go("Eyes-Clear-Left_LOD0", pose21, identity_transform(r=MESH_ROT, s=(0.9,) * 3),
                          mesh=add_mesh("Eyes-Clear-Left_LOD0", seed=2, n_quads=1), materials=["Eyes"])
    eye_l0_right = add_go("Eyes-Clear-Right_LOD0", pose21, identity_transform(r=MESH_ROT, s=(0.9,) * 3),
                           mesh=add_mesh("Eyes-Clear-Right_LOD0", seed=3, n_quads=1), materials=["Eyes"])
    # ONE shared "Eyelashes" renderer, referenced by BOTH LOD levels below.
    lash21 = add_go("Eyelashes", pose21, identity_transform(r=MESH_ROT, s=(0.9,) * 3),
                     mesh=add_mesh("Eyelashes", seed=4, n_quads=1), materials=["Eyelashes"])

    body21_l1 = add_go("21_LOD1_MeshPart0", pose21, identity_transform(r=MESH_ROT),
                        mesh=add_mesh("21_LOD1_MeshPart0", seed=5, n_quads=2), materials=["21"])
    eye_l1_left = add_go("Eyes-Clear-Left_LOD1", pose21, identity_transform(r=MESH_ROT, s=(0.9,) * 3),
                          mesh=add_mesh("Eyes-Clear-Left_LOD1", seed=6, n_quads=1), materials=["Eyes"])

    pose21_go = next(g for g in game_objects if g["path_id"] == pose21)
    pose21_go["lod_group"]["levels"][0]["renderers"] = [body21_l0, eye_l0_left, eye_l0_right, lash21]
    pose21_go["lod_group"]["levels"][1]["renderers"] = [body21_l1, eye_l1_left, lash21]  # lash21 shared!

    # single-piece sofa (no C1/L1/M1/SR1 split) + bare-named shadow, both direct children.
    add_go("21sofa", pose21, identity_transform(r=MESH_ROT), mesh=add_mesh("21sofa", seed=7, n_quads=2),
           materials=["21sofa"])
    add_go("Shad", pose21, identity_transform(r=MESH_ROT, s=(50.0,) * 3), mesh=add_mesh("21shad", seed=8, n_quads=1),
           materials=["21shad"])
    add_go("Pivot", pose21, identity_transform((0.2, 0.5, 0.3)))

    # --- pose "Nat-77": root name doesn't match "NN-150K" at all ---
    pose77 = add_go(
        "Nat-77", scans_root, identity_transform((0.4, 0.6, 0.1)),
        lod_group={"levels": [{"screen_height": 0.25, "renderers": []}]},
    )
    body77 = add_go("77_LOD0_MeshPart0", pose77, identity_transform(r=MESH_ROT),
                     mesh=add_mesh("77_LOD0_MeshPart0", seed=9), materials=["77"])
    pose77_go = next(g for g in game_objects if g["path_id"] == pose77)
    pose77_go["lod_group"]["levels"][0]["renderers"] = [body77]

    add_go("77sofa", pose77, identity_transform(r=MESH_ROT), mesh=add_mesh("77sofa", seed=10, n_quads=2),
           materials=["77sofa"])
    # decoy: pose 21's shadow name, wrongly nested under pose 77 — the "21"
    # prefix must not match pose_id "77", so this gets skipped, not renamed.
    add_go("21shad", pose77, identity_transform(r=MESH_ROT), mesh=add_mesh("21shad-decoy", seed=11, n_quads=1),
           materials=["21shad"])
    add_go("Pivot", pose77, identity_transform((0.5, 0.5, 0.2)))

    # Textures: `_MainTex` only per material (matches the real Lucid body
    # shader, no per-pose normal/occlusion/specular) + Eyes gets a normal.
    textures = {}

    def add_texture(name, size, color_a, color_b, letter, fmt, alpha=False):
        img = checker(size, color_a, color_b, letter, alpha=alpha)
        img.save(OUT_DIR / "textures" / f"{name}.png")
        byte_size = size * size * (4 if alpha else 3) // (2 if fmt == "DXT1" else 1)
        textures[name] = {"width": size, "height": size, "format": fmt, "byte_size": byte_size,
                           "image": f"textures/{name}.png"}

    add_texture("21_CLR", 128, (200, 150, 120), (120, 90, 70), "2", "DXT1")
    add_texture("77_CLR", 128, (180, 140, 110), (100, 80, 60), "7", "DXT1")
    add_texture("Eye-T", 32, (0, 0, 0), (0, 0, 0), "E", "RGB24")  # empty, like the real vendor asset
    add_texture("Eye_NM", 64, (128, 128, 255), (140, 140, 255), "N", "DXT5")
    add_texture("Eyelash", 64, (10, 10, 10), (30, 30, 30), "L", "DXT5", alpha=True)
    add_texture("21shad", 64, (0, 0, 0), (20, 20, 20), "D", "DXT5", alpha=True)
    add_texture("21sofa_CLR", 64, (90, 60, 40), (70, 45, 30), "S", "DXT1")
    add_texture("77sofa_CLR", 64, (95, 65, 45), (75, 50, 35), "S", "DXT1")

    materials = {
        "21": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "21_CLR", "_microTex1": "skin02_DMM"},
               "colors": {"_Color": (0.985, 0.985, 0.985, 0.0)}},
        "77": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "77_CLR", "_microTex1": "skin02_DMM"},
               "colors": {"_Color": (0.985, 0.985, 0.985, 0.0)}},
        "Eyes": {"shader": "Marmoset Bumped Specular IBL",
                 "textures": {"_MainTex": "Eye-T", "_BumpMap": "Eye_NM"}},
        "Eyelashes": {"shader": "Marmoset Hair IBL", "textures": {"_MainTex": "Eyelash"},
                      "colors": {"_Color": (0.21, 0.16, 0.13, 1.0)}, "floats": {"_Cutoff": 1.0}},
        "21shad": {"shader": "Unlit-Alpha", "textures": {"_MainTex": "21shad"}, "floats": {"_Cutoff": 0.55}},
        "21sofa": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "21sofa_CLR"}},
        "77sofa": {"shader": "IR Skin Unlit", "textures": {"_MainTex": "77sofa_CLR"}},
    }
    # `skin02_DMM` isn't declared as its own texture entry (mirrors the real
    # shared micro-detail texture, deliberately left unmapped/absent from
    # `scene.json["textures"]` — `_resolve_material_def` already skips a
    # texture name with no `TextureRec`, same as an unknown source would).

    scene = {
        "unity_version": "5.3.4p1",
        "game_objects": game_objects,
        "materials": materials,
        "textures": textures,
    }
    (OUT_DIR / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    np.savez(OUT_DIR / "meshes.npz", **mesh_arrays)

    print(f"wrote fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
