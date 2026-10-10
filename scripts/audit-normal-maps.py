#!/usr/bin/env python
"""What each unique source normal map actually contains (design review 22.09
point 1). Read-only, one decode per source texture — not per tier.

    uv run --no-sync python scripts/audit-normal-maps.py <*_Data> --out audit.json

Reports, per NM: which materials/kinds reference it, its Unity format and
size, its fingerprint (`reader/backend.texture_fingerprint`) and the part
kinds that use it, the sha256 of the decoded RGBA, per-channel
min/max/mean/std, corr(R, A), and — read as AG and as RG, *before* any clamp
or normalisation — the fraction of texels with x^2+y^2 > 1+eps plus the
median/p95 angle between the two candidate decodings.

Two checks on the decoder path rather than on the statistics:
- `rgbCopies`: per texel, how far R and B are from G. Unity's DXT5nm
  packing leaves X only in alpha and fills the colour block from Y; if R
  is a (RGB565-quantised) copy of G, R carries no X at all and reading it
  as X ("rg") is structurally wrong, whatever the dispersion says.
- `reference`: the same DXT5 block data decoded a second time by an
  independent BC3 implementation (texture2ddecoder, AssetStudio's C++
  decoder — UnityPy's own path is Pillow's `bcn` codec), flipped the way
  UnityPy flips it, compared byte for byte with what the backend returned.

None of these numbers picks a swizzle on its own (the whole point of the
review item): a small angle means the candidates are near-equivalent for
that map, a large one means the encoding has to be established from the
decoder path, not from statistics.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from _audit_common import add_common_args, emit, load_scene
from PIL import Image

from vrscan.assembler import TEXTURE_ROLE_BY_PROPERTY, assemble
from vrscan.reader.backend import node_id_str
from vrscan.textures import detect_normal_swizzle

EPS = 0.01  # ~2.5 8-bit steps: below that, "off the unit disc" is quantisation, not encoding


def _channels(img: Image.Image) -> dict[str, np.ndarray]:
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    return dict(zip("rgba", (arr[..., i] for i in range(4))))


def _stats(channel: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(channel.min()), 5), "max": round(float(channel.max()), 5),
        "mean": round(float(channel.mean()), 5), "std": round(float(channel.std()), 5),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 5)


def _candidate(channels: dict[str, np.ndarray], swizzle: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) in [-1, 1], exactly as `decode_normal_vectors` reads them for
    this swizzle and before it clamps anything."""

    x_channel = "a" if swizzle == "ag" else "r"
    return channels[x_channel] * 2.0 - 1.0, channels["g"] * 2.0 - 1.0


def _reconstructed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.sqrt(np.clip(1.0 - np.clip(x * x + y * y, 0.0, 1.0), 0.0, 1.0))
    vectors = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norm == 0, 1.0, norm)


def _angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _copy_of_green(u8: np.ndarray, channel: int) -> dict[str, float]:
    """How far `channel` is from G per texel, in 8-bit steps. RGB565 keeps 5
    bits of R and B against G's 6, so a true copy differs by up to ~8 steps
    (1/32 of the range) from the endpoint rounding alone."""

    diff = np.abs(u8[..., channel].astype(np.int16) - u8[..., 1].astype(np.int16))
    result = {
        "mean": round(float(diff.mean()), 3),
        "p99": float(np.percentile(diff[::4, ::4], 99)),
        "max": int(diff.max()),
    }
    del diff
    a = u8[::4, ::4, channel].astype(np.float32)
    g = u8[::4, ::4, 1].astype(np.float32)
    result["corr"] = _corr(a, g)
    return result


def _reference_check(backend, tex_id: tuple[int, int], u8: np.ndarray) -> dict[str, object]:
    import texture2ddecoder

    tex = backend._texture_readers[tex_id].read()
    if tex.m_TextureFormat != 12:  # TextureFormat.DXT5
        return {"skipped": f"format {tex.m_TextureFormat} is not DXT5"}
    width, height = tex.m_Width, tex.m_Height
    bgra = texture2ddecoder.decode_bc3(bytes(tex.image_data), width, height)
    del tex
    ref = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)[::-1, :, [2, 1, 0, 3]]
    result: dict[str, object] = {"decoder": f"texture2ddecoder {texture2ddecoder.__version__} decode_bc3"}
    max_diff = {}
    differing = 0
    for index, channel in enumerate("rgba"):
        diff = np.abs(ref[..., index].astype(np.int16) - u8[..., index].astype(np.int16))
        max_diff[channel] = int(diff.max())
        differing = max(differing, int((diff > 0).sum()))
        del diff
    result["maxAbsDiff"] = max_diff
    result["texelsDiffering"] = differing
    result["identical"] = all(v == 0 for v in max_diff.values())
    return result


def _normal_users(scene) -> dict[tuple[int, int], list[dict[str, str]]]:
    """source NM id -> the materials that reference it, with the shader
    property it sits on (a pin is per texture or per kind, so both sides of
    that mapping have to be visible in the audit)."""

    users: dict[tuple[int, int], list[dict[str, str]]] = {}
    for material in scene.materials.values():
        for prop, tex_id in material.textures.items():
            if TEXTURE_ROLE_BY_PROPERTY.get(prop) != "normal":
                continue
            users.setdefault(tex_id, []).append(
                {"material": material.name, "shader": material.shader, "property": prop}
            )
    return users


def _normal_kinds(backend, scene, profile, set_id: str) -> dict[tuple[int, int], set[str]]:
    """source NM id -> the part kinds whose exported material uses it —
    the role name `scripts/make-normal-pins.py` records next to each pinned
    fingerprint."""

    kinds: dict[tuple[int, int], set[str]] = {}
    scans = assemble(
        scene, profile, scan_id_prefix=f"{set_id}-", fingerprint=backend.get_texture_fingerprint,
        on_error=lambda *_: None,
    )
    for scan in scans:
        for part in scan.parts:
            for ref in part.material.textures if part.material is not None else ():
                if ref.role == "normal":
                    kinds.setdefault(ref.id, set()).add(part.kind)
    return kinds


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    datasets = []
    for raw in args.data_dir:
        data_dir = Path(raw)
        backend, scene, set_id = load_scene(data_dir, args.profile)
        users = _normal_users(scene)
        kinds = _normal_kinds(backend, scene, backend.profile, set_id)
        rows = []
        for tex_id in sorted(users, key=lambda i: (scene.textures[i].name, i)):
            tex = scene.textures[tex_id]
            img = backend.get_texture_image(tex_id)
            if img is None:
                rows.append({"texture": tex.name, "id": node_id_str(tex_id), "error": "no decoded image",
                             "users": users[tex_id]})
                continue
            u8 = np.asarray(img.convert("RGBA"))
            rgb_copies = {"r": _copy_of_green(u8, 0), "b": _copy_of_green(u8, 2)}
            reference = _reference_check(backend, tex_id, u8)
            del u8
            channels = _channels(img)
            row: dict[str, object] = {
                "texture": tex.name,
                "id": node_id_str(tex_id),
                "users": users[tex_id],
                "width": tex.width,
                "height": tex.height,
                "unityFormat": tex.format,
                "fingerprint": backend.get_texture_fingerprint(tex_id),
                "kinds": sorted(kinds.get(tex_id, ())),
                "decodedRgbaSha256": hashlib.sha256(img.convert("RGBA").tobytes()).hexdigest(),
                "channels": {c: _stats(v) for c, v in channels.items()},
                "corrRA": _corr(channels["r"], channels["a"]),
                "rgbCopies": rgb_copies,
                "reference": reference,
                # Diagnostic only — this is the heuristic under review, not a verdict.
                "detectorGuess": detect_normal_swizzle(img)[0],
            }
            vectors = {}
            for swizzle in ("ag", "rg"):
                x, y = _candidate(channels, swizzle)
                radius2 = x * x + y * y
                row[swizzle] = {
                    "fractionOffUnitDisc": round(float((radius2 > 1.0 + EPS).mean()), 5),
                    "maxRadius2": round(float(radius2.max()), 5),
                    "meanRadius2": round(float(radius2.mean()), 5),
                }
                vectors[swizzle] = _reconstructed(x, y)
            angles = _angles_deg(vectors["ag"], vectors["rg"])
            row["agVsRgDeg"] = {
                "median": round(float(np.median(angles)), 3),
                "p95": round(float(np.percentile(angles, 95)), 3),
                "max": round(float(angles.max()), 3),
            }
            del img, channels, vectors
            rows.append(row)
        datasets.append({"dataset": set_id, "dataDir": str(data_dir), "normalMaps": len(rows), "rows": rows})
    emit({"eps": EPS, "datasets": datasets}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
