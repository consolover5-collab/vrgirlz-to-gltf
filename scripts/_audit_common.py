"""Shared plumbing for the read-only `scripts/audit-*.py` tools.

They all do the same first two steps — open a vendor `*_Data` folder with the
real reader backend and assemble it exactly the way `vrscan convert` would —
and they all print JSON to stdout so their output can live in `scratch/`
without carrying a single vendor pixel (see scratch/brief-next.md's rule: the
audits' *numbers* come back to the VM, the data never does).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrscan.assembler import assemble
from vrscan.cli import _dataset_set_id
from vrscan.ir import Scan
from vrscan.profiles import PROFILES
from vrscan.reader.backend import SceneRec
from vrscan.reader.unitypy_backend import UnityPyBackend


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("data_dir", nargs="+", help="path(s) to a VRGirlz *_Data folder")
    parser.add_argument("--profile", default="auto", choices=["auto", *sorted(PROFILES)])
    parser.add_argument("--out", default=None, help="write JSON here instead of stdout")
    return parser


def load_scene(data_dir: Path, profile_arg: str) -> tuple[UnityPyBackend, SceneRec, str]:
    """(backend, scene, set_id) — reading the source only, no assembly, for
    audits that must still work on a dataset `assemble()` now refuses."""

    backend = UnityPyBackend(profile=None if profile_arg == "auto" else PROFILES[profile_arg])
    scene = backend.load(data_dir)
    return backend, scene, _dataset_set_id(data_dir)


def load_dataset(data_dir: Path, profile_arg: str) -> tuple[UnityPyBackend, SceneRec, list[Scan], str]:
    """(backend, scene, scans, set_id) — the same four things `run_convert`
    works from, so an audit describes what a convert would actually export."""

    backend, scene, set_id = load_scene(data_dir, profile_arg)
    scans = assemble(
        scene, backend.profile, scan_id_prefix=f"{set_id}-", fingerprint=backend.get_texture_fingerprint
    )
    return backend, scene, scans, set_id


def texture_sources(plan: Any, role: str) -> list[Any]:
    """The `TextureRef`s (or None for an absent slot) one exported texture
    of `plan` is made from — the writer's `export_textures`, role by role."""

    if role == "baseColor":
        return [plan.texture("baseColor")]
    if role == "normal":
        return [plan.texture("normal")]
    refs = [plan.texture("occlusion")]
    refs.append(plan.texture("specular") if role == "orm" else plan.wet_specular)
    return refs


def emit(payload: Any, out: str | None) -> None:
    text = json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=False)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)
