#!/usr/bin/env python
"""Every LODGroup renderer reference of every pose vs what the converter
actually exports for it (design review 22.09 point 6). Read-only.

    uv run --no-sync python scripts/audit-lod-renderers.py <*_Data> [...] --out audit.json

Output: {"datasets": [{"dataset", "profile", "scans", "byStatus", "rows": [...]}]},
one row per (pose, LOD level, renderer reference) with `status` = exported |
missing | no-mesh | unclassified | not-exported. Anything other than
"exported" is a reference the package silently drops, and needs a named
reason before acceptance — e.g. the Nat-18-95/97 and Chloe poses with no
eyes parts, which must be confirmed from the source rather than assumed to
be "closed eyes".
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _audit_common import add_common_args, emit, load_dataset

from vrscan.assembler import audit_lod_renderers


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    datasets = []
    for raw in args.data_dir:
        data_dir = Path(raw)
        backend, scene, scans, set_id = load_dataset(data_dir, args.profile)
        rows = audit_lod_renderers(scene, scans, backend.profile)
        datasets.append({
            "dataset": set_id,
            "dataDir": str(data_dir),
            "profile": backend.profile.id,
            "unity": backend.unity_version,
            "scans": [s.id for s in scans],
            "partsByScan": {s.id: sorted(p.id for p in s.parts) for s in scans},
            "byStatus": dict(Counter(str(r["status"]) for r in rows)),
            "rows": rows,
        })
    emit({"datasets": datasets}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
