# vrscan output package — contract v1 (2026-09-22)

This file is the **single source of truth** shared by the converter (`src/vrscan/`),
the viewer (`viewer/`) and the launcher (`src/vrscan/launcher/`). `manifest.py`
(pydantic) must validate exactly this shape; change both together and bump
`schema` when a field's meaning changes.

## Layout

```
out/
  manifest.json                    # written LAST, atomically (tmp + rename)
  .vrscan/                         # cert.pem, cache.json — never served as data
  scans/<scanId>/
    static.<tier>.gltf             # sofa parts, Sofa-Feet, shadow — loaded ONCE per scan
    lod<K>.<tier>.gltf             # body + eyes L/R + eyelashes of LOD level K
    geometry.static.bin            # shared by every tier of static.*.gltf
    geometry.lod<K>.bin            # shared by every tier of lod<K>.*.gltf
    scan.gltf                      # standalone: static + LOD0 of the best AVAILABLE tier (Blender)
    tex/<tier>/<file>              # jpg/png (Ф1), ktx2 (Ф2)
```

A scan directory is built in `scans/<scanId>.tmp/` and renamed into place when
complete; an interrupted run must never look finished.

## Coordinate space (glTF, metres, Y-up, right-handed)

- Every glTF node has an **identity transform**; all Unity node TRS (incl. the
  −90° X rotation, eye/eyelash/shadow/sofa poses) are baked into the vertices,
  relative to the pose root `NN-150K`. The pose root sits at the glTF origin.
- Unity → glTF: `x = −x` on positions and normals; triangle winding reversed;
  `v = 1 − v`. **No TANGENT attribute in Ф1** (the viewer derives tangents);
  `extras.vrscan.tangents = "none"` on each mesh.
- `pivot`, `bounds` (whole scan incl. static parts) and `bodyBounds` (LOD group
  only) are in this same space. `heightM = bodyBounds.max.y − bodyBounds.min.y`.
- Textures are written so that glTF UVs address them directly (no extra flip
  at load time). Normal maps are tangent-space, OpenGL convention (+Y up), Z
  reconstructed by the converter; RGB8 PNG.

## Manifest (v1)

```jsonc
{
  "schema": "vrscan-manifest/1",
  "generator": {"name": "vrscan", "version": "0.1.0"},
  "source": {"app": "VRGirlz-21-Nat-AVR", "unity": "4.6.3f1",
             "files": [{"name": "sharedassets1.assets", "bytes": 0, "sha256": "…"}]},
  "tiers": {                                   // from presets.py; only tiers actually converted
    "2k": {"label": "Quest 2", "baseColor": 2048, "normal": 2048, "orm": 1024, "format": "jpeg", "maxLod": 1},
    "4k": {"label": "Quest 3 / PC VR", "baseColor": 4096, "normal": 4096, "orm": 2048, "format": "jpeg", "maxLod": 0}
  },
  "presets": {"quest2": "2k", "quest3": "4k", "pcvr": "4k"},   // only presets whose tier is in `tiers`
  "sets": {                                    // dataset groups — the viewer's one-button-per-dataset picker
    "vrgirlz-21-nat-avr": {"label": "VRGirlz-21-Nat-AVR", "profile": "vrgirlz-2015", "app": "VRGirlz-21-Nat-AVR"}
  },
  "scans": [{
    "id": "91", "dir": "scans/91",
    "set": "vrgirlz-21-nat-avr",                // key into top-level `sets`; optional (absent on old manifests)
    "source": {"app": "VRGirlz-21-Nat-AVR", "unity": "4.6.3f1",
               "files": [{"name": "sharedassets1.assets", "bytes": 0, "sha256": "…"}]},
    "pivot": [x, y, z], "bounds": {"min": […], "max": […]}, "bodyBounds": {"min": […], "max": […]},
    "heightM": 1.63,
    "tiers": ["2k", "4k"],                     // tiers present for THIS scan
    "static": {"gltf": "static.{tier}.gltf", "bin": "geometry.static.bin", "bytes": 0, "triangles": 0},
    "lods": [                                  // ascending level; level 0 = finest
      {"level": 0, "gltf": "lod0.{tier}.gltf", "bin": "geometry.lod0.bin", "bytes": 0,
       "triangles": 0, "screenHeight": 0.25}   // Unity LODGroup threshold: show this level while
    ],                                         // projected body height / viewport height ≥ screenHeight
    "parts": [                                 // stable ids; node names in glTF == part id
      {"id": "body",      "kind": "body",      "material": "body",     "lod": true},
      {"id": "eyeL",      "kind": "eyes",      "material": "eyes",     "lod": true, "hideable": true, "hidden": true},
      {"id": "eyeR",      "kind": "eyes",      "material": "eyes",     "lod": true, "hideable": true, "hidden": true},
      {"id": "eyelashes", "kind": "eyelashes", "material": "eyelashes","lod": true},
      {"id": "sofa-C1",   "kind": "sofa",      "material": "sofa-C1",  "lod": false, "hideable": true},
      {"id": "sofa-feet", "kind": "feet",      "material": "sofa-feet","lod": false, "hideable": true},
      {"id": "shadow",    "kind": "shadow",    "material": "shadow",   "lod": false, "hideable": true}
    ],
    "materials": {                             // viewer builds ONE THREE material per entry per tier
      "body": {"type": "pbr", "alphaMode": "OPAQUE", "doubleSided": false,
               "textures": {"baseColor": "body_CLR", "normal": "body_NM", "orm": "body_ORM"},
               "baseColorFactor": [0.647, 0.607, 0.571, 1.0],               // Unity material's `_Color`, multiplies baseColor
               "variants": {"wet": {"textures": {"orm": "body_ORM_wet"}}}},   // dry = base, wet = override
      "eyes":      {"type": "pbr", "alphaMode": "BLEND", "blend": "additive", "textures": {"normal": "eyes_NM"},
                    "baseColorFactor": [0, 0, 0, 0.35], "roughnessFactor": 0.08},   // cornea gloss over the photographed eye (see below)
      "eyelashes": {"type": "pbr", "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": true, "textures": {"baseColor": "eyelash_CLR"},
                    "baseColorFactor": [0.21, 0.16, 0.13, 1.0]},                     // `_Color` tints the grey `Eyelash` RGBA
      "shadow":    {"type": "unlit", "alphaMode": "BLEND", "textures": {"baseColor": "shadow_CLR"}},
      "sofa-C1":   {"type": "pbr", "textures": {"baseColor": "sofa-C1_CLR", "normal": "sofa-C1_NM", "orm": "sofa-C1_ORM"}}
    },
    "textures": {                              // per tier: texture key -> file (relative to dir)
      "4k": {"body_CLR": {"file": "tex/4k/body_CLR.jpg", "size": [4096, 4096], "channels": "rgb", "gpuBytes": 89478485},
             "body_NM":  {"file": "tex/4k/body_NM.png",  "size": [4096, 4096], "channels": "rgb", "gpuBytes": 89478485},
             "body_ORM": {"file": "tex/4k/body_ORM.jpg", "size": [2048, 2048], "channels": "rgb", "gpuBytes": 22369621}}
    },
    "budget": {"4k": {"gpuBytesTextures": 0, "gpuBytesLod0": 0}},   // Σ unique textures; geometry of finest LOD
    "provenance": {"sceneRoot": "91-150K", "meshes": ["91_LOD0_MeshPart0", …],
                   "materials": {"body": {"unity": "91", "shader": "IR Deform PreIntegrated Skin Wet Light Override",
                                          "unmapped": ["_MicroMixMap", "_BSSRDF", "_microTex1", "_SSSColor", "_WetBlendTex"]}},
                   "textures": {"body_CLR": "91_CLR", "body_ORM": "91_SPC+91_OCC", "body_ORM_wet": "91WET_SPC+91_OCC"},
                   "normalMaps": {"body_NM": {"source": "91_NM", "fingerprint": "5f0c…", "rule": "ag-reconstruct",
                                              "basis": "audited fingerprint (normal_pins.py)", "status": "verified",
                                              "evidence": "R≈G≈B (mean |R-G| 0.98, |B-G| 0.98 of 255)"}},
                   "approximations": ["roughness = clamp(1 − luma(SPC), 0.25, 0.95)", "wet = WET_SPC roughness", "no SSS", "no micro-detail normals"]}
  }],
  "environments": [{"id": "room", "type": "builtin"}]
}
```

Rules:
- `gpuBytes = w·h·4·1.333` (RGBA8 + mipmaps) for jpg/png; Ф2 KTX2 uses the transcoded format.
- `size` is the written file's `[w, h]`: the source scaled so its longer side is at most the tier's
  size for that role, aspect kept, never upscaled (Nat-21 `Eyelash` 64x512 stays 64x512; a 1024x512
  source under a 512 limit becomes 512x256). `gpuBytes` and the texture cache key use the same `w, h`.
- `provenance.normalMaps[<key>]`: how each normal map was decoded. `status` is `verified` when the
  map's content fingerprint is one the audit established (`src/vrscan/normal_pins.py`), `unverified`
  when the profile's rule was applied to a map nobody audited, `override` for `--normal-swizzle`,
  `skipped` when there was no rule at all (the material then has no `normal` texture). `evidence`
  (R≈G≈B) is a diagnostic only — never what picked the rule.
- glTF materials inside `lod*/static*.gltf` are **valid and self-sufficient** (so
  Blender/gltf-validator work) but the viewer ignores them and applies its own
  `MaterialSet` built from `materials` + `textures[tier]`, one per (scan, tier),
  shared across all LOD files. Node names equal part ids so the viewer can map
  loaded meshes to parts.
- Each `lod<K>.<tier>.gltf` contains only LOD-group parts (`lod: true`). Parts
  absent at a level (e.g. eyelashes without LOD4) are simply missing there.
- A pose with no static parts (four of Nat-18's) still gets its
  `geometry.static.bin` (empty, `static.bytes` 0) and `static.<tier>.gltf`;
  that glTF, like any glTF of a group without geometry, has no `buffers`
  entry — glTF requires a buffer's `byteLength` to be at least 1.
- `screenHeight` compares the projected size of `bodyBounds` to viewport
  height; in XR use the per-eye viewport and take the largest eye. The viewer
  projects the bounding *sphere* of `bodyBounds` at its view-space depth
  (`viewer/src/lod-math.js`), so the estimate doesn't change when the model
  is turned or moved sideways; a sphere is a little more generous than the
  body's height (about 12% on the dev-sample), well inside the hysteresis
  band. Hysteresis is the viewer's business.
- `parts[].hidden` (added 22.09, default `false`): the viewer's *initial*
  visibility for this part is `!hidden`, not always-visible — a part with
  `hidden: true` is always also `hideable: true` (a part the viewer never
  lets you toggle can't start hidden either). Two producers: every `kind:
  "eyes"` part, on every dataset (see "Eyes and eyelashes" below), and a
  `DatasetProfile.hidden_parts` entry naming one specific (scan, kind) —
  currently just Nat-90's eyelashes, a named one-off rather than a general
  heuristic (see below). The viewer's site-wide "Scene" panel (`part-
  toggles.js`) re-derives visibility from a persisted per-category toggle
  layered under this per-part flag, not just this flag alone — see that
  module's doc comment for the exact precedence.
- `scans[].source` (added 22.09, contract-compatible): per-scan source, same
  shape as the top-level `source`. Added once `out/` started merging scans
  converted from more than one dataset (`docs/DATASETS.md`) — the top-level
  `source` describes only the *most recent* `convert` run and can't speak for
  scans that came from an earlier run against a different dataset. Optional
  (`null`/absent on a scan written before this field existed) so an
  already-written `manifest.json` still validates; `cli.py`'s merge step
  keeps such a scan's entry as-is rather than dropping it. The **viewer does
  not read `scans[].source`** — it is provenance for tooling/debugging only,
  same audience as the top-level `source` and `provenance`.
- `sets`/`scans[].set` (added 22.09): group scans by dataset so the viewer
  can show one button per dataset instead of one flat scan list. A set's id
  is `cli.py`'s `_dataset_set_id(data_dir)` — a slug of `_infer_app_name
  (data_dir)` (e.g. `vrgirlz-21-nat-avr`) — deliberately **not**
  `DatasetProfile.id` (`profiles.py`): more than one real-world product can
  share a profile (Lucid Dreams II and Ariel are both Unity 5.x /
  `vrgirlz-2016`), so the profile id alone would collide two datasets into
  one button. `SetEntry.profile` still records the `DatasetProfile.id` used,
  for tooling. `label` is `_infer_app_name(data_dir)` from the run that
  (re)converted that dataset's scans, falling back to `profile.label` if
  that name is empty; `app` records the raw `source.app` the run saw
  (usually the same string as `label`). Scan ids get the same slug as a
  prefix, but only for profiles that opt into prefixing at all
  (`DatasetProfile.prefix_scan_ids`) — Nat's ids stay bare ("79".."91") for
  manifest/URL compatibility; Lucid-style profiles get e.g.
  `vrgirlz-lucid-dreams-ii-21`. `cli.py`'s convert stamps every new scan's
  `set` and upserts/merges the corresponding `sets` entry, leaving other
  datasets' entries and scans untouched. Both fields are optional so a
  manifest written before they existed still validates — the viewer buckets
  scans with no `set` (or a `set` missing from `sets`) under a synthetic
  "Other" group.

### Eyes and eyelashes (established 22.09 on real data)

The iris and sclera are *photographed into the body texture*. The vendor's
`Eyes-Left/Right` spheres sit fractions of a millimetre above them and their
`Eye-T` base texture decodes to all-zero RGBA — they are a cornea gloss layer,
not the eye. The package therefore exports `eyes` as `BLEND`, black
translucent base (`baseColorFactor` alpha 0.35), `roughnessFactor` 0.08, the
`Eye_NM` normal, and `"blend": "additive"`: the viewer renders it with additive
blending so only the environment highlight is added on top of the photographed
eye. Third-party glTF consumers see a plain semi-transparent dark gloss and may
simply hide the `eyeL`/`eyeR` nodes. `Eyelash` is a grey RGBA hair texture;
its colour is the Unity material's `_Color`, exported as `baseColorFactor`.
`baseColorFactor` is linear RGBA and multiplies the base colour texture; the
viewer must honour it for every material (body's `_Color` is not white).

Because the additive gloss is an approximation and not the eye itself
(design review 22.09 point 6), `eyeL`/`eyeR` ship `hideable: true, hidden:
true` — the viewer's "Scene" panel names the re-enable toggle "Corneal
highlight" (not "eyes"), starting OFF by default. Nat-90's `eyelashes`
additionally ships `hideable: true, hidden: true` — that pose's lashes were
found to need hiding by default too, pinned as an explicit, named entry in
`profiles.py`'s `DatasetProfile.hidden_parts` (`{"90": ("eyelashes",)}`)
rather than a general "buried" heuristic, which hasn't been verified yet
and isn't implemented.

## Progress protocol (`vrscan convert --progress jsonl`, stdout, one JSON per line)

```jsonc
{"event": "start",    "scans": ["91"], "tiers": ["2k","4k"], "out": "C:/…/out"}
{"event": "dataset",  "set": "vrgirlz-21-nat-avr", "unity": "4.6.3f1", "profile": "vrgirlz-2015",
                      "support": "supported", "note": "Unity 4.6.3f1, profile vrgirlz-2015"}
{"event": "step",     "scan": "91", "step": "textures", "pct": 12.5, "message": "91: 91_CLR → 4k jpg"}
{"event": "scanDone", "scan": "91", "dir": "scans/91", "seconds": 212.4, "bytes": 0, "pct": 25.0}
{"event": "done",     "seconds": 230.1, "manifest": "manifest.json", "ok": 3, "failed": 1}
{"event": "error",    "scan": "91", "message": "human readable", "detail": "traceback or code"}
```
`pct` is 0–100 for the whole run, by finished poses (a pose's own steps don't
move it). `dataset` comes once the folder is read: its Unity release, the
profile it is read with, and `support` — `supported` (a release that profile
was checked on), `assumed` (only the version prefix matches: converted, and
`note` says to check the result) or `override` (`--profile`).

An `error` with a `scan` is that pose's failure; the run goes on and ends
with `done`, whose `ok`/`failed` count the poses — a partial success is a
`done` with `failed > 0`. An `error` without `scan` ends the run before
`done`, exit code 1: an output folder that can't be written, or a folder
that can't be read as a dataset — not a `*_Data` folder, a Unity release no
profile covers ("not supported"), a file it needs missing (named, e.g. a
`.resS` next to its `.assets`); `detail` only when it was a crash. `vrscan
scan --json` reports the same refusal as `{"error": "…", "scans": []}` and a
readable folder's `dataset` like the event above.

`vrscan serve --progress jsonl` speaks one event of the same kind once it
listens — `{"event": "ready", "scheme": "https", "port": 8444, "addresses":
["127.0.0.1", "192.168.1.5"]}`, the port it actually got (the one asked for,
else the next free one) — or an `error` event and exit code 1.

Every line is ASCII (anything else escaped as `\uXXXX`): a pipe on Windows
is in the ANSI code page. Cancellation = the parent terminates the process;
the converter never leaves a half-written `scans/<id>/` (tmp dir + rename).
Human-readable progress goes to stderr regardless of `--progress`.
