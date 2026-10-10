"""A texture's identity is its encoded payload plus the parameters its decode
uses — not its name, not where the bytes are stored (design review 22.09 A2).

The old key hashed the serialized Texture2D object: for a streamed texture
that is its name plus the `.resS` path/offset/size, so different pixels at
the same offset kept the old key (and a cache hit handed out the old
result), while a renamed texture with identical pixels got a new one."""

from __future__ import annotations

import json
import shutil
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vrscan import cli
from vrscan.reader.backend import TexturePayloadError, texture_fingerprint
from vrscan.reader.fixture_backend import FixtureBackend
from vrscan.reader.unitypy_backend import UnityPyBackend, texture_payload

_DXT5 = 12


class _Texture:
    """A parsed Texture2D as far as `texture_payload` looks at one: inline
    `image_data` or a `m_StreamData` range served by a fake companion file."""

    def __init__(self, *, name="91_NM", inline=b"", stream=None, resS=None, width=4, height=4):
        self.m_Name = name
        self.m_Width, self.m_Height = width, height
        self.m_TextureFormat = _DXT5
        self.m_MipCount = 1
        self.m_ImageCount = 1
        self.m_PlatformBlob = None
        self.image_data = inline
        self.m_StreamData = stream
        self.object_reader = SimpleNamespace(platform=5, version=(5, 3, 4, 1))
        self._resS = resS or {}

    def get_image_data(self):
        # UnityPy's own order: inline bytes first, then the stream range.
        if self.image_data:
            return self.image_data
        if self.m_StreamData is not None:
            if self.m_StreamData.path not in self._resS:
                raise FileNotFoundError(f"Resource file {self.m_StreamData.path} not found")
            data = self._resS[self.m_StreamData.path]
            return data[self.m_StreamData.offset:self.m_StreamData.offset + self.m_StreamData.size]
        raise ValueError("No image data found")


def _stream(path: str, offset: int, size: int) -> SimpleNamespace:
    return SimpleNamespace(path=path, offset=offset, size=size)


def _fingerprint(tex: _Texture) -> str:
    payload, params = texture_payload(tex, tex.m_Name)
    return texture_fingerprint(payload, params)


def test_changed_stream_bytes_under_unchanged_metadata_change_the_fingerprint():
    res = "sharedassets1.assets.resS"
    before = _Texture(stream=_stream(res, 16, 8), resS={res: bytes(16) + b"AAAAAAAA"})
    after = _Texture(stream=_stream(res, 16, 8), resS={res: bytes(16) + b"AAAABAAA"})
    assert _fingerprint(before) != _fingerprint(after)


def test_renamed_or_relocated_payload_keeps_the_fingerprint():
    pixels = b"12345678"
    res1, res0 = "sharedassets1.assets.resS", "sharedassets0.assets.resS"
    streamed = _Texture(stream=_stream(res1, 16, 8), resS={res1: bytes(16) + pixels})
    moved = _Texture(name="renamed_NM", stream=_stream(res0, 0, 8), resS={res0: pixels})
    inline = _Texture(name="other", inline=pixels)
    assert _fingerprint(streamed) == _fingerprint(moved) == _fingerprint(inline)


def test_decode_parameters_are_part_of_the_identity():
    """The same bytes read as a different size or format are a different image."""

    base = _Texture(inline=b"12345678")
    wider = _Texture(inline=b"12345678", width=8, height=2)
    other_format = _Texture(inline=b"12345678")
    other_format.m_TextureFormat = 10  # DXT1
    assert len({_fingerprint(base), _fingerprint(wider), _fingerprint(other_format)}) == 3


def test_fields_are_length_delimited():
    """Moving bytes between two fields must not produce the same stream."""

    assert texture_fingerprint(b"bc", [("k", "a")]) != texture_fingerprint(b"c", [("k", "ab")])


def test_unreadable_payload_is_an_error_naming_the_texture_not_a_metadata_hash():
    missing = _Texture(name="91_CLR", stream=_stream("sharedassets1.assets.resS", 0, 8))
    with pytest.raises(TexturePayloadError, match="91_CLR"):
        texture_payload(missing, missing.m_Name)
    empty = _Texture(name="Eye-T", stream=_stream("x.resS", 0, 0), resS={"x.resS": b""})
    with pytest.raises(TexturePayloadError, match="Eye-T"):
        texture_payload(empty, empty.m_Name)


def test_fingerprint_is_computed_once_per_resource_per_run():
    reads = []
    tex = _Texture(inline=b"12345678")
    reader = SimpleNamespace(assets_file=object(), path_id=7, read=lambda: reads.append(1) or tex)
    backend = UnityPyBackend()
    backend._texture_readers[(0, 7)] = reader
    first = backend.get_texture_fingerprint((0, 7))
    assert backend.get_texture_fingerprint((0, 7)) == first
    assert len(reads) == 1
    assert backend.get_texture_fingerprint((1, 7)) is None


# --------------------------------------------------------------------------
# end to end through the cache, on the fixture
# --------------------------------------------------------------------------


def _cache_keys(out_dir) -> set[str]:
    return set(json.loads((out_dir / ".vrscan" / "cache.json").read_text(encoding="utf-8")))


def test_changed_pixels_with_the_same_metadata_miss_the_cache(fixture_dir, tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    out_dir = tmp_path / "out"
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    keys_before = _cache_keys(out_dir)
    clr = out_dir / "scans" / "91" / "tex" / "2k" / "body_CLR.jpg"
    before = clr.read_bytes()

    source = data_dir / "textures" / "91_CLR.png"
    img = np.asarray(Image.open(source).convert("RGB")).copy()
    img[:, :, 0] = 255 - img[:, :, 0]
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    source.write_bytes(buf.getvalue())  # same name, size, format in scene.json

    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    assert _cache_keys(out_dir) - keys_before  # a new key for the changed source
    assert clr.read_bytes() != before


def test_renamed_texture_with_the_same_bytes_hits_the_cache(fixture_dir, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    out_dir = tmp_path / "out"
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    keys_before = _cache_keys(out_dir)

    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["textures"]["renamed_CLR"] = {**scene["textures"].pop("91_CLR"), "image": "textures/renamed_CLR.png"}
    scene["materials"]["91_dry"]["textures"]["_MainTex"] = "renamed_CLR"
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (data_dir / "textures" / "91_CLR.png").rename(data_dir / "textures" / "renamed_CLR.png")

    from vrscan.gltf import writer

    encoded = []
    real_encode = writer._encode
    monkeypatch.setattr(
        writer, "_encode", lambda *a, **k: encoded.append(a[2]) or real_encode(*a, **k)
    )
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
    assert encoded == []
    assert _cache_keys(out_dir) == keys_before


def test_a_shared_name_changes_no_exported_pixel(fixture_dir, tmp_path):
    """A texture's name is only its label: giving the sofa's texture the
    body's name still exports each material with its own pixels, to its own
    file (keyed by name, both would have got whichever came last)."""

    baseline = tmp_path / "baseline"
    assert cli.run_convert(FixtureBackend(), fixture_dir, baseline, ["2k"], None, "auto", None) == 0

    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["textures"]["91-C1_CLR"]["name"] = "91_CLR"
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0

    tex = "scans/91/tex/2k"
    body, sofa = (f"{tex}/body_CLR.jpg", f"{tex}/sofa-C1_CLR.jpg")
    assert (baseline / body).read_bytes() != (baseline / sofa).read_bytes()
    for rel in (body, sofa):
        assert (out_dir / rel).read_bytes() == (baseline / rel).read_bytes()
