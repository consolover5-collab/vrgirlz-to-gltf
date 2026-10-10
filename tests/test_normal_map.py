"""Analytic normal-map fixture — the reviewer geometry review 22.09, test 4.

A known height field `h(u, v)` gives a known tangent-space normal
`n = normalize(-dh/du, -dh/dv, 1)`. That normal is packed the two ways the
real sources use (DXT5nm "ag": X in alpha, Y in green; "rg": X in red),
each with deliberately misleading content in the channels the decoder must
NOT read, and run through the whole `encode_normal` path (swizzle ->
reconstruct -> resize -> quantize). What's checked is the angle between the
decoded direction and the analytic one — RGB values alone can't tell a
correct decode from a plausible-looking wrong one.

The image's row axis is treated as `v` throughout; the `v = 1 - v` flip
belongs to the glTF UVs (`gltf/writer.py`), not to the pixels, so it plays
no part here.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vrscan.profiles import NormalRule
from vrscan.textures import (
    _quantize_normal_vectors,
    decode_normal_vectors,
    encode_normal,
    resize_lanczos,
)

SOURCE_SIZE = 128
TARGET_SIZE = 64
BORDER = 4  # Lanczos needs a few texels of support; edges aren't what's under test


def _uv_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size) + 0.5) / size
    return np.meshgrid(axis, axis, indexing="xy")  # u varies along columns, v along rows


def _analytic_normals(size: int) -> np.ndarray:
    """h(u, v) = 0.06 sin(2pi u) + 0.02 cos(4pi v) + 0.05 u v — asymmetric in
    u and v, with different amplitudes, so a swapped or sign-flipped channel
    can't accidentally agree."""

    u, v = _uv_grid(size)
    dh_du = 0.06 * 2 * np.pi * np.cos(2 * np.pi * u) + 0.05 * v
    dh_dv = -0.02 * 4 * np.pi * np.sin(4 * np.pi * v) + 0.05 * u
    normals = np.stack([-dh_du, -dh_dv, np.ones_like(u)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _distraction(size: int, seed: int) -> np.ndarray:
    """Content for the channels a correct decoder must ignore — plausible
    image data, not flat filler."""

    rng = np.random.default_rng(seed)
    u, v = _uv_grid(size)
    return np.clip(0.5 + 0.4 * np.sin(6 * np.pi * u + v) + 0.05 * rng.standard_normal((size, size)), 0, 1)


def _pack(normals: np.ndarray, swizzle: str, flip_g: bool = False) -> Image.Image:
    size = normals.shape[0]
    x = (normals[..., 0] + 1.0) * 0.5
    y = (normals[..., 1] + 1.0) * 0.5
    if flip_g:
        y = 1.0 - y
    if swizzle == "ag":
        channels = [_distraction(size, 1), y, _distraction(size, 2), x]
    else:  # "rg"
        channels = [x, y, _distraction(size, 3), _distraction(size, 4)]
    arr = np.stack([np.clip(c, 0, 1) * 255.0 for c in channels], axis=-1)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _decode_png(data: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64) / 255.0
    vectors = rgb * 2.0 - 1.0
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _angular_error_deg(decoded: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.sum(decoded * reference, axis=-1), -1.0, 1.0)
    inner = slice(BORDER, -BORDER)
    return np.degrees(np.arccos(cosine))[inner, inner]


@pytest.mark.parametrize("swizzle", ["ag", "rg"])
def test_full_path_keeps_the_normal_direction_within_two_degrees(swizzle):
    """swizzle -> reconstruct -> resize (128 -> 64) -> quantize, compared
    against the analytic normal at the output texel centres."""

    source = _pack(_analytic_normals(SOURCE_SIZE), swizzle)
    data = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle=swizzle))

    error = _angular_error_deg(_decode_png(data), _analytic_normals(TARGET_SIZE))
    assert error.max() < 2.0, f"max angular error {error.max():.2f} deg"


def test_wrong_swizzle_is_measurably_wrong():
    """An "ag"-packed map read as "rg" takes X from the decoy red channel —
    the result still looks like a normal map (Z is reconstructed, so every
    vector is unit-length and points outward) but is nowhere near the truth.
    This is why the swizzle is pinned per profile instead of trusted to the
    channel-statistics heuristic."""

    source = _pack(_analytic_normals(SOURCE_SIZE), "ag")
    data = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle="rg"))
    error = _angular_error_deg(_decode_png(data), _analytic_normals(TARGET_SIZE))
    assert error.max() > 10.0


def test_flipped_green_sign_is_measurably_wrong():
    """+Y-up vs -Y-up (DirectX-style) green: the same map with G inverted
    must not pass the 2-degree check, or the test couldn't tell the two
    conventions apart."""

    source = _pack(_analytic_normals(SOURCE_SIZE), "ag", flip_g=True)
    data = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle="ag"))
    error = _angular_error_deg(_decode_png(data), _analytic_normals(TARGET_SIZE))
    assert error.max() > 10.0


def test_resizing_quantized_pixels_would_be_worse_than_resizing_vectors():
    """The fix itself: filtering the RGB8 encoding and quantizing the result
    (the previous order) leaves texels off the unit sphere, so the decoded
    direction drifts. Resizing the float vectors and renormalizing must be
    no worse than that, by the same worst-texel measure."""

    source = _pack(_analytic_normals(SOURCE_SIZE), "ag")
    data = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle="ag"))
    vector_resized = _decode_png(data)

    quantized_first = resize_lanczos(_quantize_normal_vectors(decode_normal_vectors(source, "ag")), (TARGET_SIZE, TARGET_SIZE))
    buf = BytesIO()
    quantized_first.save(buf, format="PNG")
    pixel_resized = _decode_png(buf.getvalue())

    reference = _analytic_normals(TARGET_SIZE)
    assert _angular_error_deg(vector_resized, reference).max() <= _angular_error_deg(pixel_resized, reference).max()


def test_unity_dxt5nm_packing_with_y_copied_into_rgb_decodes_as_ag():
    """What the audit found in all 66 real normal maps (22.09): R, G and B
    all carry Y, X is only in alpha. The pinned "ag" rule recovers the
    normal; "rg" would read Y as X and tilt every texel off course."""

    normals = _analytic_normals(SOURCE_SIZE)
    x = (normals[..., 0] + 1.0) * 0.5
    y = (normals[..., 1] + 1.0) * 0.5
    arr = np.stack([np.clip(c, 0, 1) * 255.0 for c in (y, y, y, x)], axis=-1)
    source = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    reference = _analytic_normals(TARGET_SIZE)

    ag = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle="ag"))
    assert _angular_error_deg(_decode_png(ag), reference).max() < 2.0
    rg = encode_normal(source, (TARGET_SIZE, TARGET_SIZE), NormalRule(swizzle="rg"))
    assert _angular_error_deg(_decode_png(rg), reference).max() > 10.0
