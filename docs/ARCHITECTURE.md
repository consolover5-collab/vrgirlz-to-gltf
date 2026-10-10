# Architecture

Full rationale lives in the approved plan
(`~/.claude/plans/partitioned-exploring-pearl.md`); this is the short
version plus the rejected alternatives, for anyone who lands in this repo
without that context.

## What this is

An unofficial converter that reads a purchased `VRGirlz`-style Unity 4.6
data folder (5 photogrammetry scans of static poses) and turns it into
glTF, servable to a flat-screen or WebXR viewer with a quality switch. No
vendor asset ever ships in this repo or on any server the tool runs.

## Ф0 scope (this checkout)

Reader (`reader/unitypy_backend.py`) → backend-agnostic IR (`ir.py`, via
`reader/backend.py`'s `SceneRec`) → assembler (`assembler.py`) → `vrscan
scan` prints what was found. No geometry or texture export — that's Ф1.

## Key decisions

- **`ReaderBackend` protocol, not a hard UnityPy dependency in
  `assembler.py`.** Only `reader/unitypy_backend.py` imports UnityPy.
  Everything else — assembler, manifest schema, CLI report formatting —
  operates on plain dataclasses (`SceneRec`/`GameObjectRec`/...), so it can
  be fully tested on the VM against `reader/fixture_backend.py` and a
  synthetic scene, with zero vendor bytes anywhere near CI.
- **UnityPy reads exactly `level0` + `sharedassets1.assets`, as path
  strings.** UnityPy turns a path string into a buffered file stream, not
  an eagerly-materialized in-memory buffer, so opening a 1.8 GB file
  doesn't itself blow the RSS budget. `resources.assets` (945 unrelated UI
  textures) is never opened.
- **Texture metadata without pixel decode.** `Texture2D.read()` in UnityPy
  does transiently materialize the raw (still DXT-compressed) `image_data`
  bytes as part of typetree parsing, but the backend never retains the
  parsed object — only `width`/`height`/`format` and the *free* on-disk
  `object_reader.byte_size` end up in `TextureRec`. Actual pixel decode
  (`.image`, DXT decompression into an RGB buffer) only happens for the two
  small (1024²) `_SPC`/`WET_SPC` textures per pose, for `vrscan scan`'s
  brightness histogram — the plan's one explicit "decode is fine here"
  exception.
- **Assembler does a full subtree walk + regex classification, not
  strict parent-nesting assumptions.** Established facts only confirm the
  pose root (`NN-150K`) and its direct LODGroup; deeper nesting (whether
  eyes/eyelashes sit directly under the root or under an intermediate node)
  wasn't spot-checked. Classifying every descendant by name pattern,
  regardless of depth, is robust to either shape and gets validated for
  real on a machine with the vendor data in Ф0 step 5.
- **Parts without their own LOD chain (sofa/feet/shadow) get a single
  `LodMesh` at level 0, `has_lod_chain=False`.** Established facts describe
  a LODGroup covering body/eyes/eyelashes only; sofa pieces, the shared
  `Sofa-Feet`, and the shadow plane are fixed single meshes. This is an
  IR-level judgment call (not explicit in the plan's field list) — flagged
  for the lead to override if the real LODGroup does reference them.
- **`manifest.py` ships the full v1 schema now, populated by nothing yet.**
  It's the contract Ф1's `gltf/writer.py` will produce and the (future)
  viewer will consume — writing it as pydantic models in Ф0 catches shape
  mistakes early via `tests/test_manifest.py`, without needing real
  converted output.
- **Portable peak-RSS via `resource`/`ctypes`, not psutil.** The Windows test machine runs
  Windows, where `resource` doesn't exist; rather than add a dependency not
  in the approved list, `cli.py` uses `GetProcessMemoryInfo` via `ctypes`
  on Windows and `resource.getrusage` elsewhere.

## Rejected alternatives (design review, 22.09.2026)

- **Б. Native Godot/OpenXR viewer + importer.** A standalone Windows/Quest
  app needing no browser or HTTPS, full control over VR interaction.
  Rejected: two separate builds, Quest sideloading instead of a browser tap,
  materials rebuilt from scratch in a different engine — roughly 2-3x the
  work for a benefit (offline-without-a-PC viewing) nobody asked for.
- **В. Bare glTF exporter, no built-in viewer.** Smallest possible project;
  view results in Blender or any third-party glTF viewer. Rejected: it
  drops the actual goal ("give the user a fishing rod, not a fish" — a
  one-click import → VR flow) and pushes viewer setup back onto the user,
  which is exactly the command-line-unfriendly experience the owner asked
  to avoid.
- **Tauri (desktop shell for the viewer).** Considered for Ф3 packaging,
  dropped in favor of a plain Python launcher (pywebview) + system browser
  as the primary VR client: WebXR support inside Tauri's WebView2 wrapper is
  unproven, whereas Chrome/Edge/Quest Browser's WebXR is not. One language
  (Python) end to end, no Rust/sidecar build matrix.
