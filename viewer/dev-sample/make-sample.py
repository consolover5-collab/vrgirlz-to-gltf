#!/usr/bin/env python3
"""Generates viewer/dev-sample/ — a synthetic vrscan output package that
matches docs/manifest.md (contract v1) exactly, so the viewer can be built
and tested before the real converter produces a real package. Pure Python +
numpy + Pillow — no UnityPy, no vendor assets.

One scan "s1": a low-poly humanoid silhouette (body ellipsoid + two eye
spheres + an eyelash quad, all LOD-group parts) sitting in front of a cube
"sofa" and a soft alpha-blend "shadow" plane (both static, hideable). Two
tiers (2k/4k), each tier's textures half the pixel size of the one above.

Run: uv run python viewer/dev-sample/make-sample.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
SCAN_ID = "s1"
SCAN_DIR = SCRIPT_DIR / "scans" / SCAN_ID
TIERS = ["4k", "2k"]
SET_ID = "dev-sample"

# Deliberately tiny compared to real presets.py (2048/4096) — this is a
# synthetic sample, not a stand-in for real texture budgets; keeps the whole
# package well under the 3 MB the brief asks for.
TEX_SIZES = {
    "4k": {"baseColor": 512, "normal": 512, "orm": 256},
    "2k": {"baseColor": 256, "normal": 256, "orm": 128},
}

COMPONENT_FLOAT = 5126
COMPONENT_UINT = 5125
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

BODY_CENTER = (0.0, 0.8, 0.0)
BODY_RADII = (0.22, 0.8, 0.22)  # ellipsoid standing in for a capsule, y 0..1.6
EYE_RADIUS = 0.022
EYE_X = 0.045
EYE_Y = 1.5
EYE_Z = 0.12
LASH_Y = 1.5
LASH_Z = 0.13
LASH_SIZE = (0.11, 0.03)

SOFA_SIZE = (0.6, 0.4, 0.6)
SOFA_CENTER = (0.0, 0.2, -0.8)
SHADOW_SIZE = (1.0, 1.0)
SHADOW_CENTER = (0.0, 0.001, 0.0)

PIVOT = (0.0, 0.9, 0.0)

# rings, segments per LOD level (all divisible by 2/4 so pole + equator +
# the four axis-aligned meridians are always exact samples — bounds don't
# drift between LOD levels even though triangle count does).
LOD_SPHERE_PARAMS = {
    0: {"body": (16, 24), "eye": (8, 12)},
    1: {"body": (8, 12), "eye": (5, 8)},
    2: {"body": (4, 6), "eye": (3, 4)},
}
LOD_SCREEN_HEIGHT = {0: 0.25, 1: 0.125, 2: 0.062}

MATERIALS = {
    "body": {
        "type": "pbr", "alphaMode": "OPAQUE", "doubleSided": False,
        "textures": {"baseColor": "body_CLR", "normal": "body_NM", "orm": "body_ORM"},
        "variants": {"wet": {"textures": {"orm": "body_ORM_wet"}}},
    },
    "eyes": {
        "type": "pbr", "alphaMode": "OPAQUE",
        "textures": {"baseColor": "eyes_CLR"}, "roughnessFactor": 0.15,
    },
    "eyelashes": {
        "type": "pbr", "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True,
        "textures": {"baseColor": "eyelash_CLR"},
    },
    "shadow": {
        "type": "unlit", "alphaMode": "BLEND",
        "textures": {"baseColor": "shadow_CLR"},
    },
    "sofa-C1": {
        "type": "pbr", "alphaMode": "OPAQUE",
        "textures": {"baseColor": "sofa-C1_CLR", "normal": "sofa-C1_NM", "orm": "sofa-C1_ORM"},
    },
}


# --------------------------------------------------------------------------
# geometry — analytic, in glTF space (Y-up, right-handed), already baked in
# world position (identity node transforms, per docs/manifest.md).
# --------------------------------------------------------------------------


def uv_sphere(radii, rings, segments, center):
    """Ellipsoid: every vertex lies exactly on the surface (parametrised by
    trig, not a subdivided base mesh), so bounds are identical at every LOD
    level as long as rings/segments are chosen to hit the poles + equator +
    the four axis-aligned meridians exactly (see LOD_SPHERE_PARAMS)."""
    rx, ry, rz = radii
    positions, normals, uvs = [], [], []
    for i in range(rings + 1):
        theta = i / rings * np.pi
        v = i / rings
        for j in range(segments + 1):
            phi = j / segments * 2 * np.pi
            u = j / segments
            ux, uy, uz = np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)
            positions.append((center[0] + rx * ux, center[1] + ry * uy, center[2] + rz * uz))
            n = np.array([ux / rx, uy / ry, uz / rz], dtype=np.float64)
            n = n / np.linalg.norm(n)
            normals.append(tuple(n))
            uvs.append((u, 1.0 - v))
    indices = []
    stride = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j
            b = a + stride
            c = a + 1
            d = b + 1
            indices += [a, b, c, b, d, c]
    return dict(
        positions=np.array(positions, dtype=np.float32),
        normals=np.array(normals, dtype=np.float32),
        uvs=np.array(uvs, dtype=np.float32),
        indices=np.array(indices, dtype=np.uint32),
    )


def box(size, center):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    faces = [
        ((0, 0, 1), [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]),
        ((0, 0, -1), [(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz)]),
        ((0, 1, 0), [(-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz), (-sx, sy, -sz)]),
        ((0, -1, 0), [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)]),
        ((1, 0, 0), [(sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz)]),
        ((-1, 0, 0), [(-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz)]),
    ]
    positions, normals, uvs, indices = [], [], [], []
    for normal, corners in faces:
        base = len(positions)
        for k, corner in enumerate(corners):
            positions.append((cx + corner[0], cy + corner[1], cz + corner[2]))
            normals.append(normal)
            uvs.append([(0, 1), (1, 1), (1, 0), (0, 0)][k])
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return dict(
        positions=np.array(positions, dtype=np.float32),
        normals=np.array(normals, dtype=np.float32),
        uvs=np.array(uvs, dtype=np.float32),
        indices=np.array(indices, dtype=np.uint32),
    )


def plane(size, center, axis="y"):
    """A single quad. axis='y' -> flat in XZ, normal +Y (floor/shadow).
    axis='z' -> upright in XY, normal +Z (facing the camera, eyelashes)."""
    sx, sy = (s / 2 for s in size)
    cx, cy, cz = center
    if axis == "y":
        corners = [(-sx, 0, sy), (sx, 0, sy), (sx, 0, -sy), (-sx, 0, -sy)]
        normal = (0, 1, 0)
    else:
        corners = [(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)]
        normal = (0, 0, 1)
    positions = [(cx + x, cy + y, cz + z) for x, y, z in corners]
    normals = [normal] * 4
    uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]
    indices = [0, 1, 2, 0, 2, 3]
    return dict(
        positions=np.array(positions, dtype=np.float32),
        normals=np.array(normals, dtype=np.float32),
        uvs=np.array(uvs, dtype=np.float32),
        indices=np.array(indices, dtype=np.uint32),
    )


def make_lod_parts(level):
    rings_b, segs_b = LOD_SPHERE_PARAMS[level]["body"]
    rings_e, segs_e = LOD_SPHERE_PARAMS[level]["eye"]
    body = uv_sphere(BODY_RADII, rings_b, segs_b, BODY_CENTER)
    eye_l = uv_sphere((EYE_RADIUS,) * 3, rings_e, segs_e, (-EYE_X, EYE_Y, EYE_Z))
    eye_r = uv_sphere((EYE_RADIUS,) * 3, rings_e, segs_e, (EYE_X, EYE_Y, EYE_Z))
    lash = plane(LASH_SIZE, (0.0, LASH_Y, LASH_Z), axis="z")
    return [
        ("body", body, "body"),
        ("eyeL", eye_l, "eyes"),
        ("eyeR", eye_r, "eyes"),
        ("eyelashes", lash, "eyelashes"),
    ]


def make_static_parts():
    sofa = box(SOFA_SIZE, SOFA_CENTER)
    shadow = plane(SHADOW_SIZE, SHADOW_CENTER, axis="y")
    return [("sofa-C1", sofa, "sofa-C1"), ("shadow", shadow, "shadow")]


# --------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------


def draw_letter_f(draw, box_, color):
    x0, y0, x1, y1 = box_
    w, h = x1 - x0, y1 - y0
    draw.rectangle([x0, y0, x0 + w * 0.22, y1], fill=color)  # stem
    draw.rectangle([x0, y0, x1, y0 + h * 0.22], fill=color)  # top bar
    mid_y0 = y0 + h * 0.42
    draw.rectangle([x0, mid_y0, x0 + w * 0.7, mid_y0 + h * 0.2], fill=color)  # middle bar


def make_checker_f(size, bg, fg, letter):
    """8x8 checkerboard + a big asymmetric 'F' — the QA checklist in the
    plan uses this letter to catch mirrored UVs / flipped V at a glance."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    n, cell = 8, size / 8
    for y in range(size):
        for x in range(size):
            px[x, y] = fg if (int(x / cell) + int(y / cell)) % 2 else bg
    draw_letter_f(ImageDraw.Draw(img), (size * 0.2, size * 0.15, size * 0.75, size * 0.85), letter)
    return img


def make_flat_normal(size):
    return Image.new("RGB", (size, size), (128, 128, 255))


def make_orm(size, occlusion, roughness, metalness=0):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = occlusion
    arr[..., 1] = roughness
    arr[..., 2] = metalness
    return Image.fromarray(arr, "RGB")


def make_eyelash_mask(size):
    img = Image.new("RGBA", (size, size), (20, 15, 10, 0))
    draw = ImageDraw.Draw(img)
    stripe = size / 10
    for i in range(0, 10, 2):
        draw.rectangle([i * stripe, size * 0.2, i * stripe + stripe * 0.6, size * 0.8], fill=(20, 15, 10, 255))
    return img


def make_shadow_blob(size):
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    alpha = np.clip(1.0 - dist, 0, 1) ** 1.5
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 3] = (alpha * 180).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


TEXTURES: dict[str, dict[str, dict]] = {tier: {} for tier in TIERS}


def save_texture(tier, key, img, ext, channels):
    tex_dir = SCAN_DIR / "tex" / tier
    tex_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{key}.{ext}"
    path = tex_dir / filename
    if ext == "jpg":
        img.convert("RGB").save(path, "JPEG", quality=88)
    else:
        img.save(path, "PNG")
    w, h = img.size
    # docs/manifest.md rule: gpuBytes = w*h*4*1.333 (RGBA8 + mipmaps).
    gpu_bytes = int(w * h * 4 * 1.333)
    TEXTURES[tier][key] = {"file": f"tex/{tier}/{filename}", "size": [w, h], "channels": channels, "gpuBytes": gpu_bytes}


def generate_textures():
    for tier in TIERS:
        s = TEX_SIZES[tier]
        save_texture(tier, "body_CLR", make_checker_f(s["baseColor"], (235, 235, 235), (60, 60, 120), (200, 30, 30)), "jpg", "rgb")
        save_texture(tier, "body_NM", make_flat_normal(s["normal"]), "png", "rgb")
        save_texture(tier, "body_ORM", make_orm(s["orm"], occlusion=200, roughness=190), "jpg", "rgb")
        save_texture(tier, "body_ORM_wet", make_orm(s["orm"], occlusion=200, roughness=45), "jpg", "rgb")
        save_texture(tier, "eyes_CLR", make_checker_f(s["baseColor"], (230, 230, 250), (40, 40, 90), (10, 10, 10)), "jpg", "rgb")
        save_texture(tier, "eyelash_CLR", make_eyelash_mask(s["baseColor"]), "png", "rgba")
        save_texture(tier, "sofa-C1_CLR", make_checker_f(s["baseColor"], (200, 160, 120), (120, 80, 50), (30, 20, 10)), "jpg", "rgb")
        save_texture(tier, "sofa-C1_NM", make_flat_normal(s["normal"]), "png", "rgb")
        save_texture(tier, "sofa-C1_ORM", make_orm(s["orm"], occlusion=200, roughness=160), "jpg", "rgb")
        save_texture(tier, "shadow_CLR", make_shadow_blob(s["baseColor"]), "png", "rgba")


# --------------------------------------------------------------------------
# glTF assembly — one buffer per file (static / lod<K>), shared byte-for-byte
# across tiers; only the baked material/image JSON differs per tier (the
# viewer ignores these materials anyway and rebuilds its own MaterialSet,
# but self-sufficient files are part of the contract for Blender/validators).
# --------------------------------------------------------------------------


class Buffer:
    def __init__(self):
        self.data = bytearray()
        self.buffer_views = []
        self.accessors = []

    def _align(self):
        while len(self.data) % 4:
            self.data.append(0)

    def _add_accessor(self, array, target, component_type, accessor_type, with_bounds=False):
        self._align()
        offset = len(self.data)
        raw = array.tobytes()
        self.data += raw
        view_index = len(self.buffer_views)
        self.buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        accessor = {"bufferView": view_index, "componentType": component_type, "count": len(array), "type": accessor_type}
        if with_bounds:
            flat = array.reshape(len(array), -1)
            accessor["min"] = [float(v) for v in flat.min(axis=0)]
            accessor["max"] = [float(v) for v in flat.max(axis=0)]
        idx = len(self.accessors)
        self.accessors.append(accessor)
        return idx

    def add_part(self, geom):
        pos = self._add_accessor(geom["positions"], TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, "VEC3", with_bounds=True)
        norm = self._add_accessor(geom["normals"], TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, "VEC3")
        uv = self._add_accessor(geom["uvs"], TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, "VEC2")
        idx = self._add_accessor(geom["indices"], TARGET_ELEMENT_ARRAY_BUFFER, COMPONENT_UINT, "SCALAR")
        return pos, norm, uv, idx


def build_file(file_stem, parts):
    """Writes geometry.<file_stem>.bin once, and <file_stem>.<tier>.gltf for
    every tier. Returns (bytes, triangles) for the manifest."""
    buf = Buffer()
    mesh_specs = []
    for part_id, geom, material_name in parts:
        pos, norm, uv, idx = buf.add_part(geom)
        prim = {"attributes": {"POSITION": pos, "NORMAL": norm, "TEXCOORD_0": uv}, "indices": idx, "mode": 4}
        mesh_specs.append((part_id, material_name, prim))

    bin_name = f"geometry.{file_stem}.bin"
    (SCAN_DIR / bin_name).write_bytes(bytes(buf.data))
    byte_length = len(buf.data)
    triangles = sum(len(geom["indices"]) // 3 for _, geom, _ in parts)

    for tier in TIERS:
        textures_registry = TEXTURES[tier]
        gltf = {
            "asset": {"version": "2.0", "generator": "vrscan dev-sample generator"},
            "buffers": [{"uri": bin_name, "byteLength": byte_length}],
            "bufferViews": buf.buffer_views,
            "accessors": buf.accessors,
            "materials": [],
            "images": [],
            "textures": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
            "meshes": [],
            "nodes": [],
            "scenes": [{"nodes": []}],
            "scene": 0,
            "extensionsUsed": [],
        }
        texture_index: dict[str, int] = {}
        material_index: dict[str, int] = {}

        def get_texture_ref(tex_key, texture_index=texture_index, gltf=gltf, textures_registry=textures_registry):
            if tex_key in texture_index:
                return texture_index[tex_key]
            rec = textures_registry[tex_key]
            img_i = len(gltf["images"])
            gltf["images"].append({"uri": rec["file"]})
            tex_i = len(gltf["textures"])
            gltf["textures"].append({"source": img_i, "sampler": 0})
            texture_index[tex_key] = tex_i
            return tex_i

        def get_material(material_name, material_index=material_index, gltf=gltf):
            if material_name in material_index:
                return material_index[material_name]
            mdef = MATERIALS[material_name]
            textures = mdef["textures"]
            mat = {"name": material_name, "doubleSided": mdef.get("doubleSided", False)}
            alpha_mode = mdef.get("alphaMode", "OPAQUE")
            if alpha_mode != "OPAQUE":
                mat["alphaMode"] = alpha_mode
                if alpha_mode == "MASK":
                    mat["alphaCutoff"] = mdef.get("alphaCutoff", 0.5)
            pbr = {"baseColorFactor": [1, 1, 1, 1]}
            if "baseColor" in textures:
                pbr["baseColorTexture"] = {"index": get_texture_ref(textures["baseColor"])}
            if "orm" in textures:
                pbr["metallicRoughnessTexture"] = {"index": get_texture_ref(textures["orm"])}
                mat["occlusionTexture"] = {"index": get_texture_ref(textures["orm"])}
                pbr["roughnessFactor"] = 1.0
                pbr["metallicFactor"] = 0.0
            else:
                pbr["roughnessFactor"] = mdef.get("roughnessFactor", 1.0)
                pbr["metallicFactor"] = 0.0
            mat["pbrMetallicRoughness"] = pbr
            if "normal" in textures:
                mat["normalTexture"] = {"index": get_texture_ref(textures["normal"])}
            if mdef["type"] == "unlit":
                mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
                if "KHR_materials_unlit" not in gltf["extensionsUsed"]:
                    gltf["extensionsUsed"].append("KHR_materials_unlit")
            idx = len(gltf["materials"])
            gltf["materials"].append(mat)
            material_index[material_name] = idx
            return idx

        for part_id, material_name, prim in mesh_specs:
            prim_with_mat = dict(prim, material=get_material(material_name))
            mesh_i = len(gltf["meshes"])
            gltf["meshes"].append({"name": part_id, "primitives": [prim_with_mat], "extras": {"vrscan": {"tangents": "none"}}})
            node_i = len(gltf["nodes"])
            gltf["nodes"].append({"name": part_id, "mesh": mesh_i})
            gltf["scenes"][0]["nodes"].append(node_i)

        if not gltf["extensionsUsed"]:
            del gltf["extensionsUsed"]
        (SCAN_DIR / f"{file_stem}.{tier}.gltf").write_text(json.dumps(gltf))

    return byte_length, triangles


# --------------------------------------------------------------------------
# manifest.json
# --------------------------------------------------------------------------


def bounds_of(parts):
    positions = np.concatenate([geom["positions"] for _, geom, _ in parts], axis=0)
    return positions.min(axis=0), positions.max(axis=0)


def main():
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    generate_textures()

    static_parts = make_static_parts()
    lod0_parts = make_lod_parts(0)
    lod1_parts = make_lod_parts(1)
    lod2_parts = make_lod_parts(2)

    static_bytes, static_tris = build_file("static", static_parts)
    lod_bytes_tris = {
        0: build_file("lod0", lod0_parts),
        1: build_file("lod1", lod1_parts),
        2: build_file("lod2", lod2_parts),
    }

    body_min, body_max = bounds_of(lod0_parts)
    scene_min, scene_max = bounds_of(lod0_parts + static_parts)
    height_m = float(body_max[1] - body_min[1])

    lods = [
        {
            "level": level, "gltf": f"lod{level}.{{tier}}.gltf", "bin": f"geometry.lod{level}.bin",
            "bytes": lod_bytes_tris[level][0], "triangles": lod_bytes_tris[level][1],
            "screenHeight": LOD_SCREEN_HEIGHT[level],
        }
        for level in (0, 1, 2)
    ]

    tiers = {
        "2k": {"label": "Quest 2 (dev-sample)", "baseColor": TEX_SIZES["2k"]["baseColor"], "normal": TEX_SIZES["2k"]["normal"], "orm": TEX_SIZES["2k"]["orm"], "format": "jpeg", "maxLod": 1},
        "4k": {"label": "Quest 3 / PC VR (dev-sample)", "baseColor": TEX_SIZES["4k"]["baseColor"], "normal": TEX_SIZES["4k"]["normal"], "orm": TEX_SIZES["4k"]["orm"], "format": "jpeg", "maxLod": 0},
    }

    budget = {}
    for tier, tier_def in tiers.items():
        gpu_textures = sum(rec["gpuBytes"] for rec in TEXTURES[tier].values())
        finest_available = tier_def["maxLod"]  # the tier never loads anything finer than its own cap
        budget[tier] = {"gpuBytesTextures": gpu_textures, "gpuBytesLod0": lod_bytes_tris[finest_available][0]}

    scan_entry = {
        "id": SCAN_ID, "dir": f"scans/{SCAN_ID}",
        "pivot": list(PIVOT),
        "bounds": {"min": [float(v) for v in scene_min], "max": [float(v) for v in scene_max]},
        "bodyBounds": {"min": [float(v) for v in body_min], "max": [float(v) for v in body_max]},
        "heightM": height_m,
        "set": SET_ID,
        "tiers": ["2k", "4k"],
        "static": {"gltf": "static.{tier}.gltf", "bin": "geometry.static.bin", "bytes": static_bytes, "triangles": static_tris},
        "lods": lods,
        "parts": [
            {"id": "body", "kind": "body", "material": "body", "lod": True},
            # hideable+hidden, matching assembler.py's HIDDEN_BY_DEFAULT_KINDS
            # for real converts — the viewer's "Corneal highlight" Scene
            # toggle needs a hidden eyes part to actually show anything.
            {"id": "eyeL", "kind": "eyes", "material": "eyes", "lod": True, "hideable": True, "hidden": True},
            {"id": "eyeR", "kind": "eyes", "material": "eyes", "lod": True, "hideable": True, "hidden": True},
            {"id": "eyelashes", "kind": "eyelashes", "material": "eyelashes", "lod": True},
            {"id": "sofa-C1", "kind": "sofa", "material": "sofa-C1", "lod": False, "hideable": True},
            {"id": "shadow", "kind": "shadow", "material": "shadow", "lod": False, "hideable": True},
        ],
        "materials": MATERIALS,
        "textures": {tier: TEXTURES[tier] for tier in TIERS},
        "budget": budget,
        "provenance": {
            "sceneRoot": f"{SCAN_ID}-150K",
            "meshes": ["s1_LOD0_body", "s1_LOD0_eyeL", "s1_LOD0_eyeR", "s1_LOD0_eyelashes", "s1_sofa-C1", "s1_shadow"],
            "materials": {
                "body": {"unity": "s1", "shader": "IR Deform PreIntegrated Skin Wet Light Override (synthetic)", "unmapped": ["_MicroMixMap", "_BSSRDF", "_SSSColor"]},
            },
            "textures": {"body_CLR": "s1_CLR", "body_ORM": "s1_SPC+s1_OCC", "body_ORM_wet": "s1WET_SPC+s1_OCC"},
            "approximations": ["synthetic dev-sample — not real scan data, generated by make-sample.py"],
        },
    }

    manifest = {
        "schema": "vrscan-manifest/1",
        "generator": {"name": "vrscan-dev-sample", "version": "0.1.0"},
        "source": {"app": "synthetic", "unity": "n/a", "files": [{"name": "make-sample.py", "bytes": 0, "sha256": hashlib.sha256(b"vrscan-dev-sample").hexdigest()}]},
        "tiers": tiers,
        "presets": {"quest2": "2k", "quest3": "4k", "pcvr": "4k", "ultra": "4k"},
        "sets": {SET_ID: {"label": "vrscan dev-sample", "profile": SET_ID, "app": "synthetic"}},
        "scans": [scan_entry],
        "environments": [{"id": "room", "type": "builtin"}],
    }

    (SCRIPT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_bytes = sum(f.stat().st_size for f in SCRIPT_DIR.rglob("*") if f.is_file())
    print(f"dev-sample written to {SCRIPT_DIR} — {total_bytes / 1024:.1f} KiB total, heightM={height_m:.3f}")


if __name__ == "__main__":
    main()
