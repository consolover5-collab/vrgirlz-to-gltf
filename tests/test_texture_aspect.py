"""A rectangular source keeps its aspect and is never upscaled; the file,
the manifest's `size`, `gpuBytes` and the cache key all carry the same
actual width and height (design review 22.09 «Неквадратные текстуры»)."""

from __future__ import annotations

import json
import shutil

import pytest
from PIL import Image

from vrscan import cli
from vrscan.manifest import Manifest
from vrscan.presets import TIERS
from vrscan.reader.fixture_backend import FixtureBackend
from vrscan.textures import fit_size, gpu_bytes


@pytest.mark.parametrize(
    ("source", "limit", "expected"),
    [
        ((64, 512), 2048, (64, 512)),  # Nat-21 Eyelash: kept as is, not stretched to a square
        ((64, 512), 512, (64, 512)),
        ((1024, 512), 512, (512, 256)),  # landscape, scaled down
        ((512, 1024), 512, (256, 512)),  # portrait, scaled down
        ((32, 32), 2048, (32, 32)),  # never upscaled
        ((4096, 4096), 2048, (2048, 2048)),
        ((8, 4096), 1024, (2, 1024)),
        ((1, 4096), 1024, (1, 1024)),  # a side never rounds to 0
    ],
)
def test_fit_size(source, limit, expected):
    assert fit_size(*source, limit) == expected


def _resize_source(data_dir, name, size) -> None:
    path = data_dir / "textures" / f"{name}.png"
    Image.open(path).resize(size).save(path)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    scene["textures"][name].update(width=size[0], height=size[1])
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")


def test_rectangular_sources_keep_their_aspect_in_file_manifest_budget_and_cache(fixture_dir, tmp_path, monkeypatch, capsys):
    for role in ("baseColor", "normal", "orm"):
        monkeypatch.setitem(TIERS["2k"], role, 128)
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    _resize_source(data_dir, "91_CLR", (256, 128))  # landscape over the limit
    _resize_source(data_dir, "91_NM", (128, 256))  # portrait over the limit, through the normal-vector path
    _resize_source(data_dir, "Eyelash", (32, 64))  # portrait under the limit

    out_dir = tmp_path / "out"
    expected = {"body_CLR": (128, 64), "body_NM": (64, 128), "eyelash_CLR": (32, 64)}
    written = {}
    for _run in range(2):  # the second run is served from the texture cache
        assert cli.run_convert(FixtureBackend(), data_dir, out_dir, ["2k"], None, "auto", None) == 0
        scan = Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8")).scans[0]
        textures = scan.textures["2k"]
        for key, size in expected.items():
            tex = textures[key]
            assert tuple(tex.size) == size
            path = out_dir / scan.dir / tex.file
            with Image.open(path) as img:
                assert img.size == size
            assert tex.gpuBytes == gpu_bytes(*size)
            written.setdefault(key, []).append(path.read_bytes())
        assert scan.budget["2k"].gpuBytesTextures == sum(t.gpuBytes for t in textures.values())

    assert all(first == second for first, second in written.values())
    cache_keys = json.loads((out_dir / ".vrscan" / "cache.json").read_text(encoding="utf-8"))
    assert any(":128x64:" in key for key in cache_keys) and any(":64x128:" in key for key in cache_keys)
    assert "stretch" not in capsys.readouterr().err
    assert not [n for n in scan.provenance.approximations if "stretch" in n]
