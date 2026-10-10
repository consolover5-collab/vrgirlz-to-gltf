"""Texture pipeline: decode one texture at a time from the `ReaderBackend`,
resize to a tier's resolution, encode to the output format, and cache the
result by source hash so a repeat `vrscan convert` only redoes what changed.

Memory discipline (measured live on pose 91: peak RSS 919 MB decoding
91_CLR 8192² DXT1 + 91_NM 4096² DXT5 + 91_OCC 4096² RGB24 back to back,
budget 3 GB): every `encode_*` function here takes exactly one decoded
`PIL.Image` (or two, for ORM which genuinely needs OCC+SPC together) and the
caller (`gltf/writer.py`) is expected to `del` its reference and let `gc`
run between textures — see `CONVERT_VERSION` below for why encode functions
are pure (bytes in, bytes out): easy to unit-test without a live decode.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from vrscan.profiles import NormalRule

logger = logging.getLogger("vrscan.textures")

# Bump when a change here would produce different output bytes for the same
# input — invalidates every cache.json entry keyed on it.
# v2: normal maps are resized as float vectors and renormalized before
# quantization (see `encode_normal`).
# v3: a material with no occlusion map gets a constant R = 1 in its ORM
# instead of its specular map's luma (see `encode_orm`).
# v4: the source key is the texture fingerprint (encoded payload, streamed
# or inline, + decode parameters), not a hash of the serialized object
# (see `reader/backend.texture_fingerprint`).
# v5: a rectangular source keeps its aspect (`fit_size`) instead of being
# stretched to a square; the key's size is "WxH".
CONVERT_VERSION = 5

NormalSwizzle = Literal["auto", "ag", "rg", "none"]


# --------------------------------------------------------------------------
# normal map swizzle detection + Z reconstruction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    mean: dict[str, float]
    std: dict[str, float]
    corr: list[list[float]]  # 4x4, order R,G,B,A


def _channel_stats(img: Image.Image) -> ChannelStats:
    rgba = np.asarray(img.convert("RGBA"), dtype=np.float64) / 255.0
    # Subsample — this is a detection heuristic, not a per-pixel operation;
    # a few thousand pixels are plenty and keep it cheap on 4k+ sources.
    step_y = max(1, rgba.shape[0] // 96)
    step_x = max(1, rgba.shape[1] // 96)
    flat = rgba[::step_y, ::step_x].reshape(-1, 4)
    channels = "rgba"
    mean = {c: float(flat[:, i].mean()) for i, c in enumerate(channels)}
    std = {c: float(flat[:, i].std()) for i, c in enumerate(channels)}
    # A channel with zero variance (flat color, common in synthetic/solid
    # test textures and in real flat alpha) makes corrcoef divide by a
    # zero stddev — expected, not a bug; nan_to_num already handles the
    # result, just do it quietly.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(flat.T)
    corr = np.nan_to_num(corr, nan=0.0).tolist()
    return ChannelStats(mean=mean, std=std, corr=corr)


def detect_normal_swizzle(img: Image.Image) -> tuple[Literal["ag", "rg", "none"], ChannelStats]:
    """Which channels hold X/Y of a tangent-space normal map — the old
    channel-statistics guess. Not used by `vrscan convert` any more (it
    cannot tell AG from RG, see `profiles.NormalRules`); kept for
    `scripts/audit-normal-maps.py`, which reports what it would have said.

    A standard (unswizzled) tangent normal has Z in blue, pointing mostly
    "up" out of the surface — blue stays close to 1.0 with low variance. If
    blue *isn't* behaving like that, the texture has been repacked (DXT5nm:
    X in alpha, Y in green, red/blue left for the compressor) and X/Y must
    be read from whichever of {alpha, red} actually varies, with Z
    reconstructed from X/Y — see `reconstruct_normal` below.
    """

    stats = _channel_stats(img)
    if stats.mean["b"] > 0.75 and stats.std["b"] < 0.12:
        return "none", stats
    return ("ag" if stats.std["a"] >= stats.std["r"] else "rg"), stats


# The normal map math runs over bands of rows, so its float temporaries
# exist for one band at a time: over the whole image, decoding one
# 8192x8192 map (Nat-18's NN_NM) had 5.3 GiB of them alive at once. Every
# step is per texel — the result is the same to the bit.
BAND_ROWS = 256


def _row_bands(height: int) -> list[slice]:
    return [slice(top, top + BAND_ROWS) for top in range(0, height, BAND_ROWS)]


def _rgba_u8(img: Image.Image) -> np.ndarray:
    """`img` as an (H, W, 4) uint8 array: one copy — `convert` would copy
    an image that is RGBA already once more."""

    return np.asarray(img if img.mode == "RGBA" else img.convert("RGBA"))


def decode_normal_vectors(img: Image.Image, rule: NormalRule | str) -> np.ndarray:
    """(H, W, 3) float32 unit normals in standard tangent space (X=R, Y=G,
    Z=B, OpenGL/+Y-up convention — unchanged from Unity's, see
    docs/manifest.md).

    `rule` (`profiles.NormalRule`, or a bare "ag"/"rg"/"none" for its
    default form) says where X and Y are read from, whether either is
    flipped, and whether Z is the stored blue channel or rebuilt as
    sqrt(max(0, 1 - x^2 - y^2)) — DXT5nm leaves blue to the compressor, so
    for a repacked map it must not be read."""

    if isinstance(rule, str):
        rule = NormalRule.for_swizzle(rule)
    rgba = _rgba_u8(img)
    vectors = np.empty((*rgba.shape[:2], 3), dtype=np.float32)
    for rows in _row_bands(rgba.shape[0]):
        channels = rgba[rows].astype(np.float32) / 255.0
        x = channels[..., 3 if rule.swizzle == "ag" else 0] * 2.0 - 1.0
        y = channels[..., 1] * 2.0 - 1.0
        if rule.flip_x:
            x = -x
        if rule.flip_y:
            y = -y
        if rule.z == "stored":
            z = channels[..., 2] * 2.0 - 1.0
        else:
            z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
        vectors[rows] = _normalized_rows(np.stack([x, y, z], axis=-1))
    return vectors


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Unit-length vectors; anything that filtering has cancelled out to
    (near) zero falls back to the flat normal (0, 0, 1) rather than to a
    NaN direction."""

    out = np.empty(vectors.shape, dtype=np.float32)
    for rows in _row_bands(vectors.shape[0]):
        out[rows] = _normalized_rows(vectors[rows])
    return out


def _normalized_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    flat = lengths < 1e-6
    safe = np.where(flat, 1.0, lengths)
    out = vectors / safe
    return np.where(flat, np.array([0.0, 0.0, 1.0], dtype=np.float32), out).astype(np.float32)


def _quantize_normal_vectors(vectors: np.ndarray) -> Image.Image:
    rgb = np.empty(vectors.shape, dtype=np.uint8)
    for rows in _row_bands(vectors.shape[0]):
        encoded = (vectors[rows] + 1.0) * 0.5
        rgb[rows] = np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def reconstruct_normal(img: Image.Image, rule: NormalRule | str) -> Image.Image:
    """`decode_normal_vectors` quantized straight back to an RGB8 image —
    the un-resized form, kept for callers/tests that want to inspect one
    texel."""

    return _quantize_normal_vectors(decode_normal_vectors(img, rule))


# --------------------------------------------------------------------------
# resize + encode
# --------------------------------------------------------------------------


def fit_size(width: int, height: int, limit: int) -> tuple[int, int]:
    """(width, height) of a `width`x`height` source scaled so its longer side
    is at most `limit`: aspect kept, never upscaled — `64x512` stays
    `64x512` under any limit >= 512, `1024x512` becomes `512x256` under 512
    (design review 22.09 «Неквадратные текстуры»). The one size calculation
    every encoder's output, the manifest, `gpuBytes` and the cache key use."""

    scale = min(1.0, limit / max(width, height))
    return max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5))


# `size` below is always (width, height), as `fit_size` returns it.


def resize_lanczos(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def encode_base_color(img: Image.Image, size: tuple[int, int]) -> bytes:
    """CLR -> JPEG q88, RGB (no alpha — body/sofa base color never needs it)."""

    resized = resize_lanczos(img if img.mode == "RGB" else img.convert("RGB"), size)
    from io import BytesIO

    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def resize_normal_vectors(vectors: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos-resize a (H, W, 3) float vector field, channel by channel —
    filtering the *vectors* rather than their RGB8 encoding, so the result
    can be renormalized before it is quantized (the reviewer geometry review
    22.09: "ресайз делать над float-векторами, затем нормализовать и лишь
    после этого квантовать")."""

    if (vectors.shape[1], vectors.shape[0]) == size:
        return vectors
    channels = [
        np.asarray(
            Image.fromarray(vectors[..., i].astype(np.float32), mode="F").resize(
                size, Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        for i in range(3)
    ]
    return np.stack(channels, axis=-1)


def encode_normal(img: Image.Image, size: tuple[int, int], rule: NormalRule) -> bytes:
    """NM -> lossless PNG RGB8: decode to float vectors by `rule`, resize
    those vectors, renormalize, and only then quantize. Resizing the
    quantized RGB8 image instead (the previous order) left every filtered
    texel slightly off the unit sphere, tilting normals wherever the map has
    detail. The rule always comes from `profiles.NormalRules.decide` — there
    is no statistics-based guess on this path any more."""

    vectors = decode_normal_vectors(img, rule)
    resized = resize_normal_vectors(vectors, size)
    del vectors  # the full-size field, unless no resize was needed
    resized = _normalized(resized)
    from io import BytesIO

    buf = BytesIO()
    _quantize_normal_vectors(resized).save(buf, format="PNG")
    return buf.getvalue()


def rgb_copy_evidence(img: Image.Image) -> str:
    """How close R and B are to G, per texel — the R≈G≈B pattern Unity's
    DXT5nm packing leaves behind (X only in alpha). Diagnostic evidence for
    `provenance.normalMaps`, never a decision: similar channels can have
    other causes (design review 22.09 A1)."""

    rgba = _rgba_u8(img)
    step_y, step_x = max(1, rgba.shape[0] // 256), max(1, rgba.shape[1] // 256)
    sample = rgba[::step_y, ::step_x].astype(np.int16)
    diff_r = float(np.abs(sample[..., 0] - sample[..., 1]).mean())
    diff_b = float(np.abs(sample[..., 2] - sample[..., 1]).mean())
    pattern = "R≈G≈B" if diff_r <= 2.0 and diff_b <= 2.0 else "R,B differ from G"
    return f"{pattern} (mean |R-G| {diff_r:.2f}, |B-G| {diff_b:.2f} of 255)"


def encode_orm(occlusion: Image.Image | None, specular: Image.Image | None, size: tuple[int, int]) -> bytes:
    """ORM -> JPEG q85. R = occlusion luma, G = roughness =
    clamp(1 - luma(specular), 0.25, 0.95) (task's roughness heuristic — no
    real roughness data in the source shader), B = 0 (no metalness: the body
    is never metallic). `specular=None` (a part with no spec map) -> flat
    mid-roughness (0.6) so the part isn't accidentally treated as a mirror;
    `occlusion=None` (a material with no AO map) -> constant R = 1, i.e. no
    occlusion at all. Substituting the specular map for the missing AO
    (what the caller used to do) shades the part with its own highlights —
    a different map's data in the AO channel, dry and wet alike (the reviewer
    review 22.09 point 2)."""

    if occlusion is None:
        occ = np.ones((size[1], size[0]), dtype=np.float32)
    else:
        occ = np.asarray(resize_lanczos(occlusion.convert("L"), size), dtype=np.float32) / 255.0
    if specular is not None:
        spc = np.asarray(resize_lanczos(specular.convert("L"), size), dtype=np.float32) / 255.0
        roughness = np.clip(1.0 - spc, 0.25, 0.95)
    else:
        roughness = np.full_like(occ, 0.6)
    orm = np.stack([occ, roughness, np.zeros_like(occ)], axis=-1)
    img = Image.fromarray(np.clip(orm * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def encode_rgba_png(img: Image.Image, size: tuple[int, int]) -> bytes:
    """Eyes/eyelashes/shadow -> lossless PNG, alpha preserved (eyelashes
    MASK cutout, shadow BLEND)."""

    resized = resize_lanczos(img if img.mode == "RGBA" else img.convert("RGBA"), size)
    from io import BytesIO

    buf = BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def gpu_bytes(width: int, height: int) -> int:
    """RGBA8 + mipmaps, per docs/manifest.md's `gpuBytes` rule."""

    return round(width * height * 4 * 1.333)


# --------------------------------------------------------------------------
# on-disk cache: out/.vrscan/cache.json
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheHit:
    path: Path  # where the previously written result actually is, absolute
    sha256: str  # of that result's bytes, as written
    evidence: str | None  # normal maps only: `rgb_copy_evidence` of the source


class TextureCache:
    """Keyed by the source texture fingerprint(s) + role + swizzle + size +
    format + `CONVERT_VERSION` -> the relative-to-`out/` path already written *and the
    sha256 of its bytes*, so a second `vrscan convert` run skips both decode
    and re-encode when nothing changed. Not thread-safe (the converter
    processes textures serially by design, per the "one texture in memory"
    budget).

    The stored hash is what makes a hit safe to reuse: the file at that path
    can have been replaced by a different run/tier/version since, and a hit
    that only checked "does the output path exist" would keep whatever was
    already there (design review 22.09 point 2). An entry written by an older
    vrscan (no hash) is treated as a miss rather than trusted."""

    def __init__(self, out_dir: Path) -> None:
        self._path = out_dir / ".vrscan" / "cache.json"
        self._entries: dict[str, dict[str, str]] = {}
        if self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                loaded = {}
            self._entries = {k: v for k, v in loaded.items() if isinstance(v, dict) and "sha256" in v}

    @staticmethod
    def key(*, source_hash: str, role: str, swizzle: str, size: tuple[int, int], fmt: str) -> str:
        return f"{source_hash}:{role}:{swizzle}:{size[0]}x{size[1]}:{fmt}:v{CONVERT_VERSION}"

    def lookup(self, cache_key: str, out_dir: Path) -> CacheHit | None:
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
        path = out_dir / entry["path"]
        if not path.is_file():
            return None
        return CacheHit(path=path, sha256=entry["sha256"], evidence=entry.get("evidence"))

    def remember(self, cache_key: str, rel_path: str, sha256: str, evidence: str | None = None) -> None:
        """`evidence` rides along so a hit can report it without decoding the
        source again."""

        entry = {"path": rel_path, "sha256": sha256}
        if evidence is not None:
            entry["evidence"] = evidence
        self._entries[cache_key] = entry

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._entries, indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def write_bytes_with_gc(path: Path, data: bytes) -> None:
    """Write `data` and drop it — the module docstring's "one texture in
    memory" discipline: callers `del` the source `Image` and call
    `gc.collect()` after this, not before (the encode step itself needs the
    decoded pixels alive)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    del data
    gc.collect()
