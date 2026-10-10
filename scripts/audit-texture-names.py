#!/usr/bin/env python
"""Texture2D names that stand for more than one resource, and what keying
textures by name did to the export (design review 22.09 «что осталось»,
step 1). Read-only; no pixel is decoded — a resource's content is its
fingerprint (`reader/backend.texture_fingerprint`), a hash of the encoded
payload.

    uv run --no-sync python scripts/audit-texture-names.py <*_Data> [...] --out audit.json

Per dataset:
- `all`: every Texture2D in the files a convert loads (its source files and
  the dependencies UnityPy pulls in to resolve their references) — the names
  carried by more than one resource, and how many of those differ in content;
- `referenced`: the same over the textures materials reference, i.e. what a
  convert can export;
- `exportChanges`: every exported texture (per scan, all tiers) whose source
  the reader's old name-keyed registry swapped for another resource of the
  same name with different content. That registry kept one entry per name,
  the resource resolved *last*, and handed it — pixels, fingerprint, size —
  to every material that referenced any resource of that name. These are the
  files a convert after this change writes differently from the current
  `out`; an empty list means none;
- `wetVariantChanges`: body materials whose wet variant the old keying
  dropped or kept differently (a same-named dry/wet pair counted as one
  source);
- `assemblyChanges`: poses that assemble under one keying and are refused
  under the other (the LOD material check compares fingerprints).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from _audit_common import add_common_args, emit, load_scene, texture_sources
from UnityPy.enums import TextureFormat

from vrscan.assembler import assemble
from vrscan.gltf.materials import build_material_plans
from vrscan.gltf.writer import _texture_targets_for_plan
from vrscan.reader.backend import TexturePayloadError, node_id_str, texture_fingerprint
from vrscan.reader.unitypy_backend import texture_payload


def _resource(reader: Any) -> str:
    """`<file>:<pathID>` — a resource named the way a person can find it."""

    return f"{reader.assets_file.name}:{reader.path_id}"


def _content(reader: Any) -> dict[str, object]:
    tex = reader.read()
    try:
        fmt = TextureFormat(tex.m_TextureFormat).name
    except ValueError:
        fmt = f"unknown({tex.m_TextureFormat})"
    row: dict[str, object] = {"width": tex.m_Width, "height": tex.m_Height, "format": fmt}
    try:
        payload, params = texture_payload(tex, tex.m_Name)
    except TexturePayloadError as exc:
        row["fingerprint"], row["error"] = None, str(exc)
    else:
        row["fingerprint"] = texture_fingerprint(payload, params)
        del payload
    del tex
    return row


def _name_table(groups: dict[str, list[Any]], contents: dict[str, dict[str, object]],
                referenced: set[str]) -> dict[str, object]:
    """name -> readers, reduced to the names with more than one resource."""

    rows = []
    for name in sorted(groups):
        readers = groups[name]
        if len(readers) < 2:
            continue
        resources = []
        for reader in readers:
            key = _resource(reader)
            resources.append({"resource": key, **contents[key], "referenced": key in referenced})
        fingerprints = {r["fingerprint"] for r in resources}
        rows.append({
            "name": name,
            "resources": resources,
            "differentContent": len(fingerprints) > 1 or None in fingerprints,
        })
    return {
        "textures": sum(len(readers) for readers in groups.values()),
        "names": len(groups),
        "namesWithSeveralResources": len(rows),
        "namesWithDifferentContent": sum(1 for r in rows if r["differentContent"]),
        "rows": rows,
    }


def _safe(fingerprint: Any, tex_id: tuple[int, int]) -> str | None:
    try:
        return fingerprint(tex_id)
    except TexturePayloadError:
        return None


def _audit(data_dir: Path, profile_arg: str) -> dict[str, object]:
    backend, scene, set_id = load_scene(data_dir, profile_arg)
    readers = backend._texture_readers
    referenced = {_resource(readers[tex_id]) for tex_id in scene.textures}

    # Every Texture2D the loaded files hold, grouped by name; only the names
    # that occur more than once get their payload read. `env.objects` leaves
    # out the files UnityPy loaded as dependencies (Nat-21's materials
    # reference textures in `sharedassets0.assets`), so walk the files.
    loaded = [f for f in backend._env.files.values() if hasattr(f, "objects")]
    everything: dict[str, list[Any]] = defaultdict(list)
    for obj in (o for f in loaded for o in f.objects.values()):
        if obj.type.name == "Texture2D":
            everything[obj.peek_name() or obj.read().m_Name].append(obj)
    contents = {
        _resource(reader): _content(reader)
        for group in everything.values() if len(group) > 1 for reader in group
    }
    # Two referenced resources sharing a name are among `everything`'s shared
    # names too, so their contents are already read.
    by_name: dict[str, list[Any]] = defaultdict(list)
    for tex_id, rec in scene.textures.items():
        by_name[rec.name].append(readers[tex_id])

    # The old registry: one entry per name, the last resource of that name to
    # be resolved (`scene.textures` keeps resolution order).
    winner = {rec.name: tex_id for tex_id, rec in scene.textures.items()}
    new_fp = backend.get_texture_fingerprint

    def old_fp(tex_id: tuple[int, int]) -> str | None:
        return new_fp(winner[scene.textures[tex_id].name])

    refused: dict[str, dict[str, str]] = {"new": {}, "old": {}}
    profile, prefix = backend.profile, f"{set_id}-"
    scans = assemble(scene, profile, prefix, fingerprint=new_fp,
                     on_error=lambda scan_id, exc: refused["new"].setdefault(scan_id, str(exc)))
    assemble(scene, profile, prefix, fingerprint=old_fp,
             on_error=lambda scan_id, exc: refused["old"].setdefault(scan_id, str(exc)))

    changes, wet_changes = [], []
    for scan in scans:
        for key, plan in build_material_plans(scan).items():
            for role, file_key in _texture_targets_for_plan(plan):
                for ref in filter(None, texture_sources(plan, role)):
                    old_id = winner[ref.name]
                    if old_id == ref.id or _safe(new_fp, old_id) == _safe(new_fp, ref.id):
                        continue
                    old = scene.textures[old_id]
                    changes.append({
                        "scan": scan.id, "material": key, "file": file_key, "role": ref.role, "name": ref.name,
                        "id": node_id_str(ref.id), "resource": _resource(readers[ref.id]),
                        "oldResource": _resource(readers[old_id]),
                        "size": f"{ref.width}x{ref.height}", "oldSize": f"{old.width}x{old.height}",
                        "fingerprint": _safe(new_fp, ref.id), "oldFingerprint": _safe(new_fp, old_id),
                    })
            dry, wet = plan.texture("specular"), plan.wet_specular
            if dry is None or wet is None:
                continue
            old_same = wet.name == dry.name or _safe(old_fp, wet.id) == _safe(old_fp, dry.id)
            new_same = wet.id == dry.id or _safe(new_fp, wet.id) == _safe(new_fp, dry.id)
            if old_same != new_same:
                wet_changes.append({
                    "scan": scan.id, "material": key, "dry": dry.name, "wet": wet.name,
                    "oldSameSource": old_same, "newSameSource": new_same,
                })

    return {
        "dataset": set_id,
        "dataDir": str(data_dir),
        "profile": profile.id,
        "sourceFiles": [p.name for p in backend.source_files],
        "loadedFiles": [Path(f.name).name for f in loaded],
        "all": _name_table(everything, contents, referenced),
        "referenced": _name_table(by_name, contents, referenced),
        "exportChanges": changes,
        "wetVariantChanges": wet_changes,
        "assemblyChanges": {
            "refusedOnlyNow": {k: v for k, v in refused["new"].items() if k not in refused["old"]},
            "refusedOnlyBefore": {k: v for k, v in refused["old"].items() if k not in refused["new"]},
        },
    }


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    emit({"datasets": [_audit(Path(raw), args.profile) for raw in args.data_dir]}, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
