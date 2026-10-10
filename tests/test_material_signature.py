"""A material's export signature knows its textures by content fingerprint,
not by name/size/format (design review 22.09 B1): the same pixels under other
names are the same export, other pixels under the same name/size/format are
not."""

from __future__ import annotations

import json
import shutil
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vrscan.assembler import PartMaterialError, assemble, audit_lod_materials
from vrscan.gltf.materials import material_export_signature
from vrscan.reader.fixture_backend import FixtureBackend


def _flip_red(path) -> None:
    img = np.asarray(Image.open(path).convert("RGB")).copy()
    img[:, :, 0] = 255 - img[:, :, 0]
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _lod1_gets_a_renamed_copy(fixture_dir, data_dir, change_pixels: bool) -> None:
    """LOD1's body renderer gets its own material `91_lod1` whose textures
    are copies of `91_dry`'s under new names — byte-identical, or with the
    base color's pixels changed."""

    shutil.copytree(fixture_dir, data_dir)
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    dry = scene["materials"]["91_dry"]
    copies = {}
    for prop, name in dry["textures"].items():
        copy = f"lod1_{name}"
        shutil.copyfile(data_dir / scene["textures"][name]["image"], data_dir / "textures" / f"{copy}.png")
        scene["textures"][copy] = {**scene["textures"][name], "image": f"textures/{copy}.png"}
        copies[prop] = copy
    if change_pixels:
        _flip_red(data_dir / "textures" / f"{copies['_MainTex']}.png")
    scene["materials"]["91_lod1"] = {**dry, "textures": copies}
    lod1 = next(go for go in scene["game_objects"] if go["name"] == "91_LOD1_MeshPart0")
    lod1["materials"] = ["91_lod1"]
    (data_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")


def test_renamed_textures_with_the_same_content_are_the_same_export(fixture_dir, tmp_path):
    data_dir = tmp_path / "data"
    _lod1_gets_a_renamed_copy(fixture_dir, data_dir, change_pixels=False)
    backend = FixtureBackend()
    scene = backend.load(data_dir)

    (scan,) = assemble(scene, fingerprint=backend.get_texture_fingerprint)
    assert next(p for p in scan.parts if p.id == "body").material.name == "91_dry"
    (row,) = [r for r in audit_lod_materials(scene, fingerprint=backend.get_texture_fingerprint)
              if r["part"] == "body"]
    assert row["consistent"]
    assert row["textureNames"][1]["baseColor"] == "lod1_91_CLR"  # names stay in the diagnostics


def test_renamed_textures_with_other_pixels_fail_the_pose_naming_the_textures(fixture_dir, tmp_path):
    data_dir = tmp_path / "data"
    _lod1_gets_a_renamed_copy(fixture_dir, data_dir, change_pixels=True)
    backend = FixtureBackend()
    scene = backend.load(data_dir)

    with pytest.raises(PartMaterialError, match="LOD1 exports a different material") as err:
        assemble(scene, fingerprint=backend.get_texture_fingerprint)
    assert "textures" in str(err.value) and "lod1_91_CLR" in str(err.value) and "'91_CLR'" in str(err.value)


def test_same_name_size_and_format_with_other_pixels_is_another_export(fixture_dir, tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    _flip_red(data_dir / "textures" / "91_CLR.png")  # scene.json unchanged

    signatures = []
    for source in (fixture_dir, data_dir):
        backend = FixtureBackend()
        scene = backend.load(source)
        (scan,) = assemble(scene, fingerprint=backend.get_texture_fingerprint)
        material = next(p for p in scan.parts if p.id == "body").material
        signatures.append(material_export_signature("body", material, backend.get_texture_fingerprint))
    before, after = signatures
    assert before["textures"]["baseColor"] != after["textures"]["baseColor"]
    assert before["textures"]["normal"] == after["textures"]["normal"]
    assert {k: v for k, v in before.items() if k != "textures"} == {
        k: v for k, v in after.items() if k != "textures"
    }
