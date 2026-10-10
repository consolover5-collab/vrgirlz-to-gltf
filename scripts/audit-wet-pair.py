#!/usr/bin/env python
"""Where a pose's wet ORM differs from its dry one, end to end (design review
22.09 point 2). Read-only.

    uv run --no-sync python scripts/audit-wet-pair.py <*_Data> --package out --out audit.json

Per pose with a `NNWet` material it reports, in pipeline order:
source ids -> sha256 of the raw source bytes -> sha256 of the decoded RGBA ->
roughness statistics *before* JPEG (mean/std, MAD dry vs wet) -> sha256 of the
two exported files -> the G channel after decoding those files back (mean, MAD,
fraction of differing texels). "The peak moved from bin 6 to bins 8-9" is a
histogram claim; these are the numbers that say whether anything actually
reaches the package.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from _audit_common import add_common_args, emit, load_dataset
from PIL import Image

from vrscan.gltf.materials import build_material_plans
from vrscan.reader.backend import node_id_str


def _luma(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _roughness(img: Image.Image) -> np.ndarray:
    """`textures.encode_orm`'s heuristic, at source resolution."""

    return np.clip(1.0 - _luma(img), 0.25, 0.95)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _green(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)[..., 1] / 255.0


def _exported_pair(package: Path, scan_id: str, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tex_root = package / "scans" / scan_id / "tex"
    if not tex_root.is_dir():
        return rows
    for tier_dir in sorted(p for p in tex_root.iterdir() if p.is_dir()):
        dry = tier_dir / f"{prefix}_ORM.jpg"
        wet = tier_dir / f"{prefix}_ORM_wet.jpg"
        if not dry.is_file() or not wet.is_file():
            rows.append({"tier": tier_dir.name, "dryFile": dry.is_file(), "wetFile": wet.is_file()})
            continue
        g_dry, g_wet = _green(dry), _green(wet)
        diff = np.abs(g_dry - g_wet)
        rows.append({
            "tier": tier_dir.name,
            "drySha256": _sha256_file(dry),
            "wetSha256": _sha256_file(wet),
            "identicalFiles": _sha256_file(dry) == _sha256_file(wet),
            "gMeanDry": round(float(g_dry.mean()), 5),
            "gMeanWet": round(float(g_wet.mean()), 5),
            "gMad": round(float(diff.mean()), 5),
            "gMaxDiff": round(float(diff.max()), 5),
            # One 8-bit step is the smallest difference a JPEG can even carry.
            "fractionDiffering": round(float((diff > 1 / 255).mean()), 5),
        })
    return rows


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--package", default=None, help="converted output dir (out/) to read exported ORMs from")
    args = parser.parse_args()

    package = Path(args.package) if args.package else None
    datasets = []
    for raw in args.data_dir:
        data_dir = Path(raw)
        backend, _scene, scans, set_id = load_dataset(data_dir, args.profile)
        poses = []
        for scan in scans:
            for key, plan in build_material_plans(scan).items():
                if plan.wet_specular is None:
                    continue
                dry = plan.texture("specular")
                dry_img = backend.get_texture_image(dry.id) if dry is not None else None
                wet_img = backend.get_texture_image(plan.wet_specular.id)
                row: dict[str, object] = {
                    "scan": scan.id,
                    "material": key,
                    "drySource": None if dry is None else dry.name,
                    "drySourceId": None if dry is None else node_id_str(dry.id),
                    "wetSource": plan.wet_specular.name,
                    "wetSourceId": node_id_str(plan.wet_specular.id),
                    "drySourceSha256": None if dry is None else backend.get_texture_fingerprint(dry.id),
                    "wetSourceSha256": backend.get_texture_fingerprint(plan.wet_specular.id),
                    "dryRgbaSha256": None if dry_img is None else hashlib.sha256(
                        dry_img.convert("RGBA").tobytes()).hexdigest(),
                    "wetRgbaSha256": hashlib.sha256(wet_img.convert("RGBA").tobytes()).hexdigest(),
                }
                wet_rough = _roughness(wet_img)
                row["roughnessWet"] = {"mean": round(float(wet_rough.mean()), 5),
                                       "std": round(float(wet_rough.std()), 5)}
                if dry_img is not None and dry_img.size == wet_img.size:
                    # Which channels the wet map actually changes — the ORM
                    # heuristic reads luma only, so a difference confined to
                    # alpha never reaches the package.
                    dry_rgba = np.asarray(dry_img.convert("RGBA"), dtype=np.int16)
                    wet_rgba = np.asarray(wet_img.convert("RGBA"), dtype=np.int16)
                    row["sourceChannelMad"] = {
                        c: round(float(np.abs(dry_rgba[..., i] - wet_rgba[..., i]).mean()) / 255.0, 5)
                        for i, c in enumerate("rgba")
                    }
                    row["sourceAlphaMean"] = {"dry": round(float(dry_rgba[..., 3].mean()) / 255.0, 5),
                                              "wet": round(float(wet_rgba[..., 3].mean()) / 255.0, 5)}
                    del dry_rgba, wet_rgba
                    dry_rough = _roughness(dry_img)
                    row["roughnessDry"] = {"mean": round(float(dry_rough.mean()), 5),
                                           "std": round(float(dry_rough.std()), 5)}
                    row["roughnessMad"] = round(float(np.abs(dry_rough - wet_rough).mean()), 5)
                del dry_img, wet_img
                if package is not None:
                    row["exported"] = _exported_pair(package, scan.id, plan.prefix)
                poses.append(row)
        datasets.append({"dataset": set_id, "dataDir": str(data_dir), "wetMaterials": len(poses), "rows": poses})
    emit({"datasets": datasets}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
