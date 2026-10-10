"""The synthetic fixtures, converted end to end, against the Khronos
glTF-Validator (`scripts/gltf-validator`, version pinned there): no errors,
and no warning but the one Ф1 accepts by design. Skipped without node;
real output is validated on a real-data machine (docs/DEV.md, "glTF validation")."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vrscan import cli
from vrscan.reader.fixture_backend import FixtureBackend

VALIDATOR_DIR = Path(__file__).resolve().parents[1] / "scripts" / "gltf-validator"
NODE = shutil.which("node")

# No TANGENT attribute in Ф1, the viewer derives tangents (docs/manifest.md).
ACCEPTED_WARNINGS = {"MESH_PRIMITIVE_GENERATED_TANGENT_SPACE"}

pytestmark = pytest.mark.skipif(NODE is None, reason="node not found")


def _validate(out_dir: Path) -> dict:
    if not (VALIDATOR_DIR / "node_modules" / "gltf-validator").is_dir():
        pytest.fail("glTF-Validator not installed — run `npm ci --prefix scripts/gltf-validator` first")
    result = subprocess.run(
        [NODE, str(VALIDATOR_DIR / "validate.mjs"), str(out_dir)],
        capture_output=True, text=True, timeout=300, check=False,  # exit code 1 = errors found, asserted below
    )
    assert result.returncode in (0, 1), result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("fixture", ["fixture_dir", "lucid_fixture_dir", "instances_fixture_dir"])
def test_converted_fixture_is_valid_gltf(fixture, request, tmp_path):
    data_dir = request.getfixturevalue(fixture)
    assert cli.run_convert(FixtureBackend(), data_dir, tmp_path, ["2k", "4k"], None, "auto", None) == 0

    report = _validate(tmp_path)

    assert report["totals"]["files"] == len(list(tmp_path.rglob("*.gltf")))
    errors = [(f["file"], m["code"], m["pointer"]) for f in report["files"] for m in f["messages"]
              if m["severity"] == "error"]
    assert errors == []
    assert set(report["codes"]) <= ACCEPTED_WARNINGS
