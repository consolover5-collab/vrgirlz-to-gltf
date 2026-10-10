#!/usr/bin/env python
"""Per-part, per-LOD-level material export signatures for every pose (the reviewer
review 22.09 point 5). Read-only, and it never raises — unlike a convert,
which now refuses a part whose levels disagree.

    uv run --no-sync python scripts/audit-lod-materials.py <*_Data> [...] --out audit.json

Output: {"datasets": [{"dataset", "inconsistent": [...], "rows": [...]}]},
one row per (pose, part) with the material name and full export signature of
each level plus `consistent`. `inconsistent` is the short list to look at:
those are the parts that would block a convert.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _audit_common import add_common_args, emit, load_scene

from vrscan.assembler import audit_lod_materials


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    datasets = []
    for raw in args.data_dir:
        data_dir = Path(raw)
        backend, scene, set_id = load_scene(data_dir, args.profile)
        rows = audit_lod_materials(
            scene, backend.profile, scan_id_prefix=f"{set_id}-", fingerprint=backend.get_texture_fingerprint
        )
        datasets.append({
            "dataset": set_id,
            "dataDir": str(data_dir),
            "profile": backend.profile.id,
            "parts": len(rows),
            "inconsistent": [f"{r['scan']}/{r['part']}" for r in rows if not r["consistent"]],
            "rows": rows,
        })
    emit({"datasets": datasets}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
