# Datasets vrscan has been run against

Two vendor asset drops, same "VRGirlz" character rig lineage, different Unity
export era. `src/vrscan/profiles.py` is the one place that knows the
differences below; nothing else in the pipeline hardcodes a dataset name.

| | `vrgirlz-2015` (Nat) | `vrgirlz-2016` (Lucid Dreams II demo) |
|---|---|---|
| Unity version | `4.6.3f1` | `5.3.4p1` |
| Source files | `level0` + `sharedassets1.assets` (fixed pair) | `level0` (33 transforms, messy leftover/menu scene) + `level1` (760 transforms, the real per-pose hierarchies) + `sharedassets0.assets` + `sharedassets1.assets` — resolved by glob (`level*`, `sharedassets*.assets`), not a fixed list |
| `.resS`/`.resource` streaming companions | none observed | present alongside `sharedassets*.assets`; UnityPy resolves them automatically once the primary file is loaded — never listed as a source file themselves |
| Pose root naming | always `NN-150K` | mostly `NN-150K`, but one demo scan is literally named `Nat-42` (reused from the Nat product) — name-pattern matching alone misses it |
| Pose discovery | regex on pose root name is sufficient | must fall back to "nearest LODGroup ancestor with a body LOD chain" (`find_pose_roots`) to catch non-conforming names |
| Body mesh names | `NN_LODk(_MeshPartN)?` | same pattern, unchanged |
| Eyes naming | `Eyes-Left/Right_LODk` | `Eyes-(Clear-)?Left/Right_LODk` — both sub-styles appear depending on the bundled product |
| Eyelashes naming | `Eyelashes_LODk` — one renderer per LOD level | bare `Eyelashes`, no `_LODk` suffix — usually ONE renderer shared across every LOD level (confirmed live on pose `21-150K`: path_id 543 referenced by both L0 and L1) |
| Sofa naming | multi-piece `NN-C1/-L1/-M1/-SR1` + shared `Sofa-Feet` | single mesh `NNsofa`, no piece split, no shared feet |
| Shadow naming | `NNshad` | `NNshad` for some poses, plain `Shad` for others (inconsistent within the same demo scene) |
| Body extra textures | normal + occlusion + specular + wet variant | only `_MainTex` + a *shared* micro-detail texture (`skin02_DMM`) reused by every pose (goes to `provenance.unmapped`, not guessed); no wet material |
| `TEXTURE_ROLE_BY_PROPERTY` changes needed | — | none — every shader property observed (`_MainTex`, `_BumpMap`, `_SpecTex`) was already mapped from Nat's table |
| Eye cornea-gloss material recipe | `Eye-T` decodes to all-zero RGBA | same finding, verified independently on this dataset's `Eye-T` |
| Scan id | unprefixed (`91`) — preserves the existing Nat manifest | prefixed `ld2-` (`ld2-21`, `ld2-nat-42`) so both datasets can share one `out/` without id collisions |

## Recognising a folder (`UnityPyBackend.discover`, `profiles.match_profile`)

A folder is read only as a `*_Data` folder whose `level0` states its Unity
release. A release a profile was checked on (`audited_unity_versions`: the
table's `4.6.3f1` and `5.3.4p1`) is **supported**; another `4.x`/`5.x`
release is read with that profile but only **assumed** — `scan`/`convert`
say so and ask to check the result; any other release is **not supported**
and refused (`--profile` forces a profile). A source file the profile
expects, or a `.resS`/`.resource` companion a texture or mesh streams from,
that isn't in the folder is an error naming the file — not a dataset read
with holes. The messages are the user's (`profiles.DatasetError`), shown by
the launcher as they are.

## Known cross-pose contamination in the source scene

`level0`+`level1`'s GameObject tree has messy/leftover nesting: pose `21`'s
subtree contains a decoy mesh part belonging to a different pose
(`33_LOD1_MeshPart0`), reached through the naive parent/child walk. This is
genuine authoring mess in the vendor scene, not a script bug — verified by
checking `Transform` `path_id` ranges directly (`level0`: 33 transforms in the
34+ range, `level1`: 760 transforms in the 761+ range, zero overlap between
files). `assembler.py`'s `_build_parts` treats LODGroup membership as
authoritative for body/eyes/eyelashes (skip a name-matching renderer that
isn't actually owned by *this* pose's own LODGroup) and pose-number prefix
checks (`_prefix_ok`) as authoritative for sofa/shadow, rather than trusting
tree position.

## Resolver keys (fixed 22.09, three times)

A `path_id` is only unique within one serialized file, and Lucid loads
four. `unitypy_backend.py` therefore keys *everything* — the GameObject /
Transform / renderer / LODGroup scene graph and the mesh/material/texture
resolver caches — by a global id: the pair `(file load-order index,
path_id)` (`reader/backend.py`'s `NodeId`; `GameObjectRec.path_id` and the
manifest's `node_ids` carry it). The pair replaced a packed
`file_index << 40 | path_id` int, which was only collision-free for
`0 <= path_id < 2**40` while Unity pathIDs are signed 64-bit (design review
22.09 point 6). A node id is never handed to JS as a Number; logs and
errors print it via `node_id_str` as `<file>:<pathID>`.

History of the bug this replaced: the first fix (caches only, by the
PPtr's own `(file_id, path_id)`) was not enough: `game_objects[21]` was pose
41's `Shad` node from one file sitting where pose 21's LODGroup expected its
own renderer from another, so lucid-21's shadow part merged materials
`21shad` and `41shad`; and a PPtr's `file_id` is relative to the
*referencing* file's externals, so one shared object got a different key
from every file pointing at it. Mesh *names* are not unique either: Lucid
has seven distinct per-pose meshes all called `Eyelashes` (different local
bounds), and keying by name gave pose `21` another pose's lashes floating
6 cm off the face. Like materials, the first name wins and later
same-named meshes get a `name@path_id` key (visible in
`provenance.meshes`).

Texture2D names are not unique either, and textures are keyed by the same
`(file index, pathID)` pair (`backend.TextureId`); a name is only a label.
Output files are named by material key and role (`tex/<tier>/body_CLR.jpg`),
never by a texture name. The audit of all six sets (22.09,
`scripts/audit-texture-names.py`) found 2–3 names per set carried by more
than one resource: `Glass` and `neuropolitical_0 distance map` with
different content in every set, `black` (Nat-21) with the same. None of
them is in an exported material, so the old name-keyed registry never
changed an exported file.

## Scan ids across datasets

Six datasets are converted into one `out/` (22.09): Nat-21, Lucid Dreams II,
Ariel-09, Nat-18, Chloe-22 and Saph-11 — 47 scans. Every dataset except
those on `vrgirlz-2015` gets its ids prefixed with its set slug
(`vrgirlz-09-ariel-42`). `vrgirlz-2015` keeps bare pose numbers for manifest
compatibility with the first Nat-21 package, and Saph-11 (Unity 4.x, 10
poses, body + shadow only) lands on the same profile — so Nat-21 and
Saph-11 ids are both bare and are unique only because their pose numbers
differ. A pose number shared by two such datasets is refused by `vrscan
convert` before it writes anything (`_check_no_set_collision`), naming the
id and both sets; it can no longer replace the other set's `scans/<id>`.
