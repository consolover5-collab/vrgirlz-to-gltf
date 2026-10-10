"""Intermediate representation (IR) of a converted scene.

This is what `assembler.py` produces from a `SceneRec` (see `reader/backend.py`)
and what everything downstream (`gltf/writer.py`, `manifest.py`, `cli.py`)
consumes. It is deliberately backend-agnostic: nothing here knows about
UnityPy, `.assets` files, or typetrees.

Geometry itself (vertex/index buffers) is NOT carried here — `LodMesh` only
records counts and the source mesh names + renderer GameObject ids (so
`gltf/writer.py` can walk each mesh's own ancestor chain up to the pose root
and bake its *actual* TRS, not assume every MeshPart shares one transform)
that get merged into one primitive: "MeshPart0..2 одного LOD сливаются в
один примитив — разбиение это артефакт 16-битного лимита Unity 4, не
семантика" (assembler.py docstring). The writer fetches the actual vertex
buffers from the `ReaderBackend`/`SceneRec` it was given, keyed by these
mesh names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A GameObject id, `(source file index, Unity pathID)` — see
# `reader/backend.py`'s `NodeId`, which this mirrors (importing it here would
# make the IR depend on the reader package it is deliberately independent of).
NodeId = tuple[int, int]

# A Texture2D id, the same kind of pair — mirrors `reader/backend.py`'s
# `TextureId`, for the same reason.
TextureId = tuple[int, int]

# Part.kind values, see assembler.py PART_PATTERNS for how names map to these.
PartKind = str  # "body" | "eyes" | "eyelashes" | "sofa" | "feet" | "shadow"


@dataclass(frozen=True)
class Transform:
    """A local TRS transform, Unity convention (left-handed, Y-up)."""

    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # quaternion, (x, y, z, w)
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class LodMesh:
    """The geometry a part contributes at one LOD level.

    `level` is an index into `Scan.lods` for parts that actually participate
    in the pose's LODGroup (kind body/eyes/eyelashes). Parts with no LOD
    chain of their own (sofa/feet/shadow) get a single synthetic entry with
    `level=0` — see assembler.py `PARTS_WITHOUT_LOD_CHAIN`.
    """

    level: int
    vertex_count: int
    triangle_count: int
    mesh_names: tuple[str, ...]  # source mesh names, merged into 1 primitive
    node_ids: tuple[NodeId, ...] = ()  # parallel to mesh_names, the renderer GameObject


@dataclass(frozen=True)
class TextureRef:
    role: str  # e.g. "baseColor", "normal", "occlusion", "specular", "specularWet", "unknown"
    id: TextureId  # what the backend knows the texture by; `name` is only its label
    name: str
    width: int
    height: int
    format: str  # Unity TextureFormat name, e.g. "DXT1", "DXT5", "RGB24"
    byte_size: int
    property_name: str = ""  # source shader property, e.g. "_MainTex" — "unknown"-role provenance


@dataclass(frozen=True)
class MaterialDef:
    name: str
    shader: str
    textures: tuple[TextureRef, ...]
    floats: dict[str, float] = field(default_factory=dict)  # shader property -> value, e.g. "_Cutoff"
    colors: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)  # e.g. "_Color"


@dataclass
class Part:
    id: str  # stable within a scan, e.g. "body", "eyeL", "eyeR", "eyelashes", "SR1", "feet", "shadow"
    kind: PartKind
    name: str  # source GameObject name of the representative node
    transform: Transform
    lods: list[LodMesh] = field(default_factory=list)
    material: MaterialDef | None = None
    hideable: bool = False  # sofa parts are hideable per plan manifest schema
    hidden: bool = False  # viewer's initial partVisibility; eyes default true here (assembler.py)
    has_lod_chain: bool = True  # False for sofa/feet/shadow (single fixed mesh)
    wet_material: MaterialDef | None = None  # body only: standalone "NNWet" material, see assembler.py


@dataclass
class LodLevel:
    level: int
    screen_height: float
    triangle_count: int  # sum over all has_lod_chain parts at this level


@dataclass
class Scan:
    id: str  # pose number as string, e.g. "91"
    root_name: str  # e.g. "91-150K"
    root_node_id: NodeId
    pivot: tuple[float, float, float]
    parts: list[Part] = field(default_factory=list)
    lods: list[LodLevel] = field(default_factory=list)
