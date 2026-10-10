#!/usr/bin/env python
"""Size of every source texture a convert would export (design review 22.09
point 4: "проверить инвентаризацию размеров всех экспортируемых текстур").
Read-only, metadata only — no pixel is decoded.

    uv run --no-sync python scripts/audit-texture-sizes.py <*_Data> --out audit.json

Per dataset: one row per unique source texture that feeds an exported role
(baseColor, normal, the ORM's occlusion/specular, the wet ORM's specular),
with its size, Unity format and the scans/roles using it; non-square ones
are listed separately. The writer keeps a source's aspect and scales it
only down, to the tier's limit on the longer side (`textures.fit_size`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _audit_common import add_common_args, emit, load_dataset, texture_sources

from vrscan.gltf.materials import build_material_plans
from vrscan.gltf.writer import _texture_targets_for_plan
from vrscan.reader.backend import node_id_str


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    datasets = []
    for raw in args.data_dir:
        _backend, _scene, scans, set_id = load_dataset(Path(raw), args.profile)
        rows: dict[tuple[int, int], dict[str, object]] = {}
        for scan in scans:
            for plan in build_material_plans(scan).values():
                for role, _key in _texture_targets_for_plan(plan):
                    for ref in filter(None, texture_sources(plan, role)):
                        row = rows.setdefault(ref.id, {
                            "texture": ref.name, "id": node_id_str(ref.id), "width": ref.width,
                            "height": ref.height, "format": ref.format, "uses": set(),
                        })
                        row["uses"].add(f"{scan.id}:{role}")
        table = [rows[tex_id] for tex_id in sorted(rows, key=lambda i: (rows[i]["texture"], i))]
        for row in table:
            row["uses"] = sorted(row["uses"])
        non_square = [
            {"texture": r["texture"], "id": r["id"], "size": f"{r['width']}x{r['height']}", "scans": len(r["uses"])}
            for r in table if r["width"] != r["height"]
        ]
        datasets.append({
            "dataset": set_id, "dataDir": raw, "scans": len(scans), "textures": len(table),
            "nonSquare": non_square, "rows": table,
        })
    emit({"datasets": datasets}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
