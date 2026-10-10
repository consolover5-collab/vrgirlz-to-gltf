"""Which rule a normal map is decoded with, and on what grounds (design review
22.09 A1): explicit `--normal-swizzle` -> the audited rule of its content
fingerprint -> the profile's rule as "unverified" -> skipped. Never the
dataset's folder name, never the channel-statistics autodetect."""

from __future__ import annotations

import dataclasses
import json
import shutil
from io import BytesIO

import numpy as np
from PIL import Image

from vrscan import cli, normal_pins, textures
from vrscan.manifest import Manifest
from vrscan.profiles import DEFAULT_PROFILE, NormalRule
from vrscan.reader.fixture_backend import FixtureBackend


def _fixture_nm_fingerprint(data_dir) -> str:
    backend = FixtureBackend()
    scene = backend.load(data_dir)
    (nm_id,) = [tex_id for tex_id, tex in scene.textures.items() if tex.name == "91_NM"]
    return backend.get_texture_fingerprint(nm_id)


def _pin(monkeypatch, fingerprint: str, key: str = "rg-reconstruct") -> None:
    """Pin the fixture's map to a rule its profile would not pick ("rg"), so
    a verified decision is distinguishable from the profile fallback."""

    monkeypatch.setattr(cli, "verified_rules", lambda: {fingerprint: NormalRule.from_key(key)})


def _convert(data_dir, out_dir, swizzle="auto", backend=None) -> Manifest:
    assert cli.run_convert(backend or FixtureBackend(), data_dir, out_dir, ["2k"], None, swizzle, None) == 0
    return Manifest.model_validate_json((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_pinned_table_is_hashes_and_kind_names_only():
    rules = normal_pins.verified_rules()
    assert rules
    for fingerprint, (rule_key, kinds) in normal_pins.VERIFIED_NORMAL_MAPS.items():
        assert len(fingerprint) == 64 and int(fingerprint, 16) >= 0
        assert NormalRule.from_key(rule_key).key() == rule_key
        assert set(kinds) <= {"body", "eyes", "eyelashes", "sofa", "feet", "shadow"}


def test_renamed_dataset_folder_keeps_the_verified_rule_and_the_pixels(fixture_dir, tmp_path, monkeypatch):
    _pin(monkeypatch, _fixture_nm_fingerprint(fixture_dir))
    renamed = tmp_path / "VRGirlz-99-Renamed_Data"
    shutil.copytree(fixture_dir, renamed)

    first = _convert(fixture_dir, tmp_path / "a").scans[0].provenance.normalMaps["body_NM"]
    second = _convert(renamed, tmp_path / "b").scans[0].provenance.normalMaps["body_NM"]
    assert (first.status, first.rule) == ("verified", "rg-reconstruct")
    assert second == first
    nm = "scans/91/tex/2k/body_NM.png"
    assert (tmp_path / "a" / nm).read_bytes() == (tmp_path / "b" / nm).read_bytes()


def test_changed_pixels_lose_the_verified_status(fixture_dir, tmp_path, monkeypatch):
    _pin(monkeypatch, _fixture_nm_fingerprint(fixture_dir))
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_dir, data_dir)
    source = data_dir / "textures" / "91_NM.png"
    img = np.asarray(Image.open(source)).copy()
    img[0, 0, 1] ^= 1
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    source.write_bytes(buf.getvalue())

    note = _convert(data_dir, tmp_path / "out").scans[0].provenance.normalMaps["body_NM"]
    assert note.status == "unverified"
    assert note.rule == DEFAULT_PROFILE.normal_rule.key() == "ag-reconstruct"
    assert DEFAULT_PROFILE.id in note.basis


def test_unknown_map_gets_the_profile_rule_with_a_warning_not_autodetect(fixture_dir, tmp_path, monkeypatch, capsys):
    def no_autodetect(*_args):
        raise AssertionError("the statistics autodetect must not run on the convert path")

    monkeypatch.setattr(textures, "detect_normal_swizzle", no_autodetect)
    monkeypatch.setattr(cli, "verified_rules", dict)
    note = _convert(fixture_dir, tmp_path).scans[0].provenance.normalMaps["body_NM"]
    assert (note.status, note.rule) == ("unverified", "ag-reconstruct")
    assert note.evidence  # R≈G≈B is recorded as evidence only
    err = capsys.readouterr().err
    assert "warning: normal map 91_NM" in err and "unverified" in err


def test_cli_override_beats_a_verified_fingerprint(fixture_dir, tmp_path, monkeypatch):
    _pin(monkeypatch, _fixture_nm_fingerprint(fixture_dir))
    note = _convert(fixture_dir, tmp_path, swizzle="ag").scans[0].provenance.normalMaps["body_NM"]
    assert (note.status, note.rule, note.basis) == ("override", "ag-reconstruct", "--normal-swizzle")


def test_no_rule_at_all_skips_the_map_but_not_the_scan(fixture_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "verified_rules", dict)
    backend = FixtureBackend(profile=dataclasses.replace(DEFAULT_PROFILE, normal_rule=None))
    scan = _convert(fixture_dir, tmp_path, backend=backend).scans[0]
    note = scan.provenance.normalMaps["body_NM"]
    assert (note.status, note.rule) == ("skipped", None)
    assert "normal" not in scan.materials["body"].textures
    assert not (tmp_path / "scans" / "91" / "tex" / "2k" / "body_NM.png").exists()
    gltf = json.loads((tmp_path / "scans" / "91" / "lod0.2k.gltf").read_text(encoding="utf-8"))
    assert all("normalTexture" not in m for m in gltf["materials"])
    assert "warning: normal map 91_NM" in capsys.readouterr().err
