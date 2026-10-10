"""manifest.py schema v1 against `docs/manifest.md`'s example shape."""

from vrscan.manifest import Manifest, SetEntry

SAMPLE = {
    "schema": "vrscan-manifest/1",
    "generator": {"name": "vrscan", "version": "0.1.0"},
    "source": {
        "app": "VRGirlz-21-Nat-AVR",
        "unity": "4.6.3f1",
        "files": [{"name": "sharedassets1.assets", "bytes": 1_800_000_000, "sha256": "a" * 64}],
    },
    "tiers": {
        "2k": {"label": "Quest 2", "baseColor": 2048, "normal": 2048, "orm": 1024, "format": "jpeg", "maxLod": 1},
        "4k": {"label": "Quest 3 / PC VR", "baseColor": 4096, "normal": 4096, "orm": 2048, "format": "jpeg", "maxLod": 0},
    },
    "presets": {"quest2": "2k", "quest3": "4k"},
    "scans": [
        {
            "id": "91",
            "dir": "scans/91",
            "pivot": [0.416, 0.658, 0.510],
            "bounds": {"min": [-1.0, 0.0, -1.0], "max": [1.0, 2.0, 1.0]},
            "bodyBounds": {"min": [-0.5, 0.0, -0.5], "max": [0.5, 1.7, 0.5]},
            "heightM": 1.7,
            "tiers": ["2k", "4k"],
            "static": {"gltf": "static.{tier}.gltf", "bin": "geometry.static.bin", "bytes": 4096, "triangles": 500},
            "lods": [
                {"level": 0, "gltf": "lod0.{tier}.gltf", "bin": "geometry.lod0.bin",
                 "bytes": 12345, "triangles": 95000, "screenHeight": 0.25},
            ],
            "parts": [
                {"id": "body", "kind": "body", "material": "body", "lod": True},
                {"id": "eyeL", "kind": "eyes", "material": "eyes", "lod": True},
                {"id": "sofa-C1", "kind": "sofa", "material": "sofa-C1", "lod": False, "hideable": True},
                {"id": "shadow", "kind": "shadow", "material": "shadow", "lod": False},
            ],
            "materials": {
                "body": {
                    "type": "pbr", "alphaMode": "OPAQUE", "doubleSided": False,
                    "textures": {"baseColor": "body_CLR", "normal": "body_NM", "orm": "body_ORM"},
                    "variants": {"wet": {"textures": {"orm": "body_ORM_wet"}}},
                    "baseColorFactor": [0.647, 0.607, 0.571, 1.0],
                },
                "eyes": {"type": "pbr", "textures": {"baseColor": "eyes_CLR"}, "roughnessFactor": 0.15},
                "sofa-C1": {"type": "pbr", "textures": {"baseColor": "sofa-C1_CLR"}},
                "shadow": {"type": "unlit", "alphaMode": "BLEND", "textures": {"baseColor": "shadow_CLR"}},
            },
            "textures": {
                "4k": {
                    "body_CLR": {"file": "tex/4k/body_CLR.jpg", "size": [4096, 4096], "channels": "rgb", "gpuBytes": 89478485},
                    "body_NM": {"file": "tex/4k/body_NM.png", "size": [4096, 4096], "channels": "rgb", "gpuBytes": 89478485},
                    "body_ORM": {"file": "tex/4k/body_ORM.jpg", "size": [2048, 2048], "channels": "rgb", "gpuBytes": 22369621},
                },
            },
            "budget": {"4k": {"gpuBytesTextures": 201326591, "gpuBytesLod0": 12345}},
            "provenance": {
                "sceneRoot": "91-150K",
                "meshes": ["91_LOD0_MeshPart0", "91_LOD0_MeshPart1"],
                "materials": {
                    "body": {"unity": "91", "shader": "IR Deform PreIntegrated Skin Wet Light Override",
                              "unmapped": ["_MicroMixMap", "_BSSRDF"]},
                },
                "textures": {"body_CLR": "91_CLR", "body_ORM": "91_SPC+91_OCC"},
                "approximations": ["roughness = clamp(1 - luma(SPC), 0.25, 0.95)"],
            },
        }
    ],
    "environments": [{"id": "room", "type": "builtin"}],
}


def test_sample_manifest_validates():
    manifest = Manifest.model_validate(SAMPLE)
    assert manifest.schema_ == "vrscan-manifest/1"
    assert manifest.scans[0].parts[2].hideable is True
    assert manifest.scans[0].parts[0].hideable is False  # default
    assert manifest.scans[0].parts[0].hidden is False  # default — SAMPLE predates the `hidden` field
    assert manifest.scans[0].materials["body"].variants["wet"].textures["orm"] == "body_ORM_wet"


def test_hidden_field_round_trips_and_defaults_false():
    sample = {**SAMPLE, "scans": [{**SAMPLE["scans"][0], "parts": [
        {"id": "eyeL", "kind": "eyes", "material": "eyes", "lod": True, "hideable": True, "hidden": True},
    ]}]}
    manifest = Manifest.model_validate(sample)
    assert manifest.scans[0].parts[0].hidden is True
    dumped = manifest.model_dump(mode="json", by_alias=True)
    assert dumped["scans"][0]["parts"][0]["hidden"] is True


def test_schema_field_round_trips_through_alias():
    manifest = Manifest.model_validate(SAMPLE)
    dumped = manifest.model_dump(mode="json", by_alias=True)
    assert dumped["schema"] == "vrscan-manifest/1"
    assert "schema_" not in dumped


def test_manifest_without_sets_still_validates():
    """A manifest written before `sets`/`scans[].set` existed (SAMPLE has
    neither) must still validate — both fields are optional."""

    manifest = Manifest.model_validate(SAMPLE)
    assert manifest.sets == {}
    assert manifest.scans[0].set is None


def test_sets_and_scan_set_validate():
    sample = {**SAMPLE, "sets": {"vrgirlz-2015": {"label": "VRGirlz-21-Nat-AVR", "profile": "vrgirlz-2015"}}}
    sample["scans"] = [{**SAMPLE["scans"][0], "set": "vrgirlz-2015"}]
    manifest = Manifest.model_validate(sample)
    assert manifest.sets["vrgirlz-2015"] == SetEntry(label="VRGirlz-21-Nat-AVR", profile="vrgirlz-2015")
    assert manifest.scans[0].set == "vrgirlz-2015"
