"""`textures.py`'s pixel-level pieces — swizzle detection/reconstruction,
ORM channel math, and the on-disk cache — exercised against small synthetic
Pillow images (no vendor asset needed)."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vrscan import textures
from vrscan.profiles import NormalRule
from vrscan.textures import (
    TextureCache,
    detect_normal_swizzle,
    encode_base_color,
    encode_orm,
    gpu_bytes,
    reconstruct_normal,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _solid(size: int, r: int, g: int, b: int, a: int | None = None) -> Image.Image:
    mode = "RGBA" if a is not None else "RGB"
    fill = (r, g, b, a) if a is not None else (r, g, b)
    return Image.new(mode, (size, size), fill)


def _dxt5nm_like(size: int, *, varying_channel: str) -> Image.Image:
    """A synthetic DXT5nm-style repack: `varying_channel` carries a real
    left-to-right ramp (std > 0), the rest are flat and not bright — like the
    real repack the plan's facts describe (X in alpha or red, Y in green)."""

    ramp = np.broadcast_to(np.linspace(0, 255, size, dtype=np.uint8)[None, :], (size, size))
    flat_mid = np.full((size, size), 128, dtype=np.uint8)
    flat_dim = np.full((size, size), 50, dtype=np.uint8)  # dim -> never trips the "none" gate (needs mean > 0.75)
    channels = {"r": flat_mid, "g": flat_mid, "b": flat_dim, "a": flat_mid}
    channels[varying_channel] = ramp
    arr = np.stack([channels["r"], channels["g"], channels["b"], channels["a"]], axis=-1)
    return Image.fromarray(arr, mode="RGBA")


class TestDetectNormalSwizzle:
    def test_standard_tangent_normal_resolves_to_none(self):
        # B near-constant and bright (~255), like a real Z-up-encoded tangent
        # normal map — the "no repack" case.
        img = _solid(64, 128, 128, 250)
        resolved, _stats = detect_normal_swizzle(img)
        assert resolved == "none"

    def test_dxt5nm_style_ag_resolves_to_ag(self):
        # X in alpha — alpha channel has real (image-content) variance while
        # red/blue don't; the detector should prefer "ag" whenever alpha
        # varies at least as much as red.
        img = _dxt5nm_like(64, varying_channel="a")
        resolved, stats = detect_normal_swizzle(img)
        assert resolved == "ag"
        assert stats.std["a"] > stats.std["r"]

    def test_dxt5nm_style_rg_resolves_to_rg(self):
        # X in red, alpha constant (opaque, std=0) — not "none" (blue isn't
        # bright/constant here), and red varies more than alpha.
        img = _dxt5nm_like(64, varying_channel="r")
        resolved, stats = detect_normal_swizzle(img)
        assert resolved == "rg"
        assert stats.std["r"] > stats.std["a"]


class TestReconstructNormal:
    def test_none_swizzle_keeps_the_direction_and_normalizes_it(self):
        """A standard (unswizzled) map is read as-is, but still comes back
        as a unit vector — the whole pipeline works on unit normals now, so
        an authored texel that is slightly off the sphere is pulled onto
        it instead of being passed through verbatim."""

        img = _solid(8, 200, 50, 250)
        out = reconstruct_normal(img, "none")
        assert out.mode == "RGB"

        source = np.array([200, 50, 250], dtype=np.float64) / 255.0 * 2.0 - 1.0
        decoded = np.asarray(out, dtype=np.float64)[0, 0] / 255.0 * 2.0 - 1.0
        assert abs(np.linalg.norm(decoded) - 1.0) < 0.01
        cosine = np.dot(decoded, source) / (np.linalg.norm(decoded) * np.linalg.norm(source))
        assert cosine > 0.9999  # same direction, unit length

    def test_ag_swizzle_reconstructs_z_from_xy(self):
        # x=0, y=0 (mid-gray in both A and G) -> z should reconstruct to 1.0
        # (full blue), the whole point of the Z-reconstruction step.
        img = _solid(8, 0, 128, 0, a=128)
        out = reconstruct_normal(img, "ag")
        b = np.asarray(out)[0, 0, 2]
        assert b >= 250  # (x+1)/2=0.5 (x=0), (y+1)/2=0.5 (y=0) -> z=1 -> 255


class TestEncodeOrm:
    def test_channel_packing(self):
        occlusion = _solid(32, 200, 200, 200)  # luma ~200/255 ~ 0.784
        specular = _solid(32, 60, 60, 60)  # luma ~60/255 ~ 0.235 -> roughness = clamp(1-0.235, .25, .95) = 0.765
        data = encode_orm(occlusion, specular, (32, 32))
        decoded = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64)
        r, g, b = decoded[..., 0].mean(), decoded[..., 1].mean(), decoded[..., 2].mean()
        assert abs(r - 200) < 6  # JPEG q85 tolerance
        assert abs(g - round(0.765 * 255)) < 6
        assert b < 6

    def test_roughness_is_clamped(self):
        occlusion = _solid(16, 255, 255, 255)
        bright_specular = _solid(16, 255, 255, 255)  # roughness would be ~0 -> clamp floor 0.25
        data = encode_orm(occlusion, bright_specular, (16, 16))
        decoded = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64)
        g = decoded[..., 1].mean()
        assert abs(g - round(0.25 * 255)) < 6

    def test_no_occlusion_gives_full_r(self):
        """No AO map -> R = 1 (no occlusion), NOT the specular map's luma —
        that substitution shaded parts with their own highlights (the reviewer
        review 22.09 point 2)."""

        dark_specular = _solid(16, 20, 20, 20)
        data = encode_orm(None, dark_specular, (16, 16))
        decoded = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64)
        assert decoded[..., 0].mean() > 250  # JPEG q85 tolerance on a flat white channel

    def test_no_specular_falls_back_to_fixed_roughness(self):
        occlusion = _solid(16, 128, 128, 128)
        data = encode_orm(occlusion, None, (16, 16))
        decoded = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float64)
        g = decoded[..., 1].mean()
        assert abs(g - round(0.6 * 255)) < 6


def test_gpu_bytes_formula():
    assert gpu_bytes(2048, 2048) == round(2048 * 2048 * 4 * 1.333)


def test_cache_key_changes_with_convert_version(monkeypatch):
    """Bumping `CONVERT_VERSION` (meant for whenever the pixel-encode math
    itself changes, e.g. normal/ORM channel handling) must change every
    cache key, so a stale on-disk cache from before the bump is never
    reused for the new encoding — design review 22.09 asked for this to be
    covered by a test, not just present in the source."""

    kwargs = {"source_hash": "abc", "role": "normal", "swizzle": "ag", "size": (2048, 2048), "fmt": "png"}
    key_current = TextureCache.key(**kwargs)
    monkeypatch.setattr(textures, "CONVERT_VERSION", textures.CONVERT_VERSION + 1)
    key_bumped = TextureCache.key(**kwargs)
    assert key_current != key_bumped
    assert key_current.endswith(f":v{textures.CONVERT_VERSION - 1}")


def test_texture_encode_pixels_are_not_flipped():
    """docs/manifest.md: "Textures are written so that glTF UVs address them
    directly (no extra flip at load time)" — the pixel data itself must not
    be mirrored vertically by the encode step (only glTF UVs get v=1-v)."""

    top_half = _solid(4, 255, 0, 0)
    bottom_half = _solid(4, 0, 0, 255)
    combo = Image.new("RGB", (4, 8))
    combo.paste(top_half, (0, 0))
    combo.paste(bottom_half, (0, 4))

    data = encode_base_color(combo, (8, 8))
    decoded = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    assert decoded[1, 0].tolist()[0] > decoded[1, 0].tolist()[2]  # top rows still red-ish
    assert decoded[6, 0].tolist()[2] > decoded[6, 0].tolist()[0]  # bottom rows still blue-ish


class TestTextureCache:
    def test_remember_and_lookup_round_trip(self, tmp_path):
        cache = TextureCache(tmp_path)
        out_file = tmp_path / "tex" / "body_CLR.jpg"
        out_file.parent.mkdir(parents=True)
        out_file.write_bytes(b"fake-jpeg")

        key = TextureCache.key(source_hash="abc", role="baseColor", swizzle="auto", size=(2048, 2048), fmt="jpeg")
        assert cache.lookup(key, tmp_path) is None

        cache.remember(key, "tex/body_CLR.jpg", _sha256(b"fake-jpeg"))
        found = cache.lookup(key, tmp_path)
        assert found is not None
        assert found.path == out_file
        assert found.sha256 == _sha256(b"fake-jpeg")
        assert found.evidence is None

    def test_remember_with_evidence_round_trips_and_persists(self, tmp_path):
        cache = TextureCache(tmp_path)
        out_file = tmp_path / "tex" / "body_NM.png"
        out_file.parent.mkdir(parents=True)
        out_file.write_bytes(b"fake-png")

        key = TextureCache.key(source_hash="abc", role="normal", swizzle="auto", size=(4096, 4096), fmt="png")
        cache.remember(key, "tex/body_NM.png", _sha256(b"fake-png"), "R≈G≈B")
        cache.save()

        found = TextureCache(tmp_path).lookup(key, tmp_path)
        assert found.path == out_file
        assert found.evidence == "R≈G≈B"

    def test_lookup_misses_if_file_was_removed(self, tmp_path):
        cache = TextureCache(tmp_path)
        out_file = tmp_path / "tex" / "gone.jpg"
        out_file.parent.mkdir(parents=True)
        out_file.write_bytes(b"x")
        key = TextureCache.key(source_hash="x", role="baseColor", swizzle="auto", size=(1, 1), fmt="jpeg")
        cache.remember(key, "tex/gone.jpg", _sha256(b"x"))
        out_file.unlink()
        assert cache.lookup(key, tmp_path) is None

    def test_entry_without_a_hash_is_dropped_on_load(self, tmp_path):
        """A cache.json written by an older vrscan records no output hash, so
        a hit off it couldn't be verified — it is a miss, not a guess (the reviewer
        review 22.09 point 2)."""

        key = TextureCache.key(source_hash="abc", role="baseColor", swizzle="auto", size=(8, 8), fmt="jpeg")
        (tmp_path / ".vrscan").mkdir()
        (tmp_path / ".vrscan" / "cache.json").write_text(json.dumps({key: "tex/old.jpg"}), encoding="utf-8")
        (tmp_path / "tex").mkdir()
        (tmp_path / "tex" / "old.jpg").write_bytes(b"old")

        assert TextureCache(tmp_path).lookup(key, tmp_path) is None


class TestNormalMathInBands:
    """The normal map math runs over bands of `textures.BAND_ROWS` rows to
    keep its float temporaries small; the whole-image computation it
    replaced is kept here as the reference, and the output must match it to
    the bit — pixels, encoded PNG bytes, evidence text."""

    @staticmethod
    def _decode_whole(img: Image.Image, rule: NormalRule) -> np.ndarray:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        x = rgba[..., 3 if rule.swizzle == "ag" else 0] * 2.0 - 1.0
        y = rgba[..., 1] * 2.0 - 1.0
        if rule.flip_x:
            x = -x
        if rule.flip_y:
            y = -y
        if rule.z == "stored":
            z = rgba[..., 2] * 2.0 - 1.0
        else:
            z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
        return TestNormalMathInBands._normalized_whole(np.stack([x, y, z], axis=-1))

    @staticmethod
    def _normalized_whole(vectors: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        flat = lengths < 1e-6
        out = vectors / np.where(flat, 1.0, lengths)
        return np.where(flat, np.array([0.0, 0.0, 1.0], dtype=np.float32), out).astype(np.float32)

    @staticmethod
    def _encode_whole(img: Image.Image, size: tuple[int, int], rule: NormalRule) -> bytes:
        vectors = TestNormalMathInBands._normalized_whole(
            textures.resize_normal_vectors(TestNormalMathInBands._decode_whole(img, rule), size)
        )
        encoded = np.clip((vectors + 1.0) * 0.5 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(encoded, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _image(mode: str) -> Image.Image:
        # Taller than two bands and not a multiple of one; a flat patch
        # exercises the zero-length fallback.
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(2 * textures.BAND_ROWS + 37, 300, 4), dtype=np.uint8)
        pixels[:20, :20] = (128, 128, 128, 128)
        return Image.fromarray(pixels, mode="RGBA").convert(mode)

    @pytest.mark.parametrize("mode", ["RGBA", "RGB"])
    @pytest.mark.parametrize(
        "rule",
        [NormalRule("ag"), NormalRule("rg", flip_y=True), NormalRule("ag", flip_x=True, z="stored"),
         NormalRule("none", z="stored")],
        ids=lambda rule: rule.key(),
    )
    def test_bands_match_the_whole_image_to_the_bit(self, mode, rule):
        img = self._image(mode)

        assert np.array_equal(textures.decode_normal_vectors(img, rule), self._decode_whole(img, rule))
        for size in ((150, 274), img.size):  # resized, and the no-resize path
            assert textures.encode_normal(img, size, rule) == self._encode_whole(img, size, rule)

    def test_evidence_is_unchanged(self):
        img = self._image("RGBA")
        u8 = np.asarray(img.convert("RGBA"), dtype=np.int16)
        sample = u8[:: max(1, u8.shape[0] // 256), :: max(1, u8.shape[1] // 256)]
        diff_r = float(np.abs(sample[..., 0] - sample[..., 1]).mean())
        diff_b = float(np.abs(sample[..., 2] - sample[..., 1]).mean())

        assert f"mean |R-G| {diff_r:.2f}, |B-G| {diff_b:.2f} of 255" in textures.rgb_copy_evidence(img)
