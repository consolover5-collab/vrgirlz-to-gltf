"""`manifest.json` schema v1 — the contract between the converter, the web
viewer, and the Ф3 launcher/pywebview app. `docs/manifest.md` is the single
source of truth this mirrors field-for-field; change both together and bump
`schema` when a field's meaning changes.

Serialize with `manifest.model_dump(mode="json", by_alias=True, exclude_none=True)`
to get the `"schema"` key (Python can't have a field literally named `schema`
without an alias — `BaseModel.schema()` is a still-present, if deprecated, v1
compat method) and to drop unset optional fields rather than writing nulls.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PartKind = Literal["body", "eyes", "eyelashes", "sofa", "feet", "shadow"]
TextureFormat = Literal["ktx2", "jpeg", "png"]
MaterialType = Literal["pbr", "unlit"]
AlphaMode = Literal["OPAQUE", "MASK", "BLEND"]


class Generator(BaseModel):
    name: str
    version: str


class SourceFile(BaseModel):
    name: str
    bytes: int
    sha256: str


class Source(BaseModel):
    app: str
    unity: str
    files: list[SourceFile]


class Tier(BaseModel):
    """One quality tier — texture resolutions per role, encoding, and how
    many trailing (coarsest) LOD levels it drops. `presets.py` is the single
    source of truth for the concrete 2k/4k/8k values; this is just the shape."""

    label: str
    baseColor: int
    normal: int
    orm: int
    format: TextureFormat
    maxLod: int


class GeometryFile(BaseModel):
    """A `.gltf` + `.bin` pair, shared by every tier (tiers only differ in
    which texture files the material JSON inside the `.gltf` points at)."""

    gltf: str
    bin: str
    bytes: int
    triangles: int


class ScanLod(GeometryFile):
    level: int
    screenHeight: float


class ScanPart(BaseModel):
    id: str
    kind: PartKind
    material: str  # key into ScanEntry.materials
    lod: bool  # True if this part switches geometry across scan.lods (body/eyes/eyelashes)
    hideable: bool = False
    # Viewer's initial partVisibility should be `!hidden`, not always-visible
    # — eyes start hidden by default (the "corneal highlight" toggle turns
    # them back on), and a profile can flag specific scans' otherwise-always-
    # visible parts the same way (see `DatasetProfile.hidden_parts`, e.g.
    # Nat-90's eyelashes). `hideable=False, hidden=True` never happens (a
    # part the viewer never lets you toggle can't start hidden either).
    hidden: bool = False


class MaterialVariant(BaseModel):
    textures: dict[str, str]  # role -> texture key, overriding the base entry's textures


class MaterialEntry(BaseModel):
    type: MaterialType
    alphaMode: AlphaMode = "OPAQUE"
    alphaCutoff: float | None = None
    doubleSided: bool = False
    textures: dict[str, str] = Field(default_factory=dict)  # role -> texture key (into ScanEntry.textures[tier])
    variants: dict[str, MaterialVariant] = Field(default_factory=dict)  # e.g. "wet" -> override
    roughnessFactor: float | None = None
    baseColorFactor: tuple[float, float, float, float] | None = None
    blend: Literal["normal", "additive"] = "normal"  # BLEND only: "additive" = specular-only overlay (eyes cornea)


class ScanTexture(BaseModel):
    file: str
    size: tuple[int, int]
    channels: str  # "rgb" | "rgba"
    gpuBytes: int


class Bounds(BaseModel):
    min: tuple[float, float, float]
    max: tuple[float, float, float]


class TierBudget(BaseModel):
    gpuBytesTextures: int  # sum over unique textures actually used by this scan, this tier
    gpuBytesLod0: int  # geometry byte size of the finest available LOD (or static, if no LOD chain)


class MaterialProvenance(BaseModel):
    unity: str  # source Unity material name
    shader: str
    unmapped: list[str] = Field(default_factory=list)  # shader properties we saw but didn't map to a role


class NormalMapProvenance(BaseModel):
    """How one normal map was decoded and how established that is (the reviewer
    review 22.09 A1): `status` "verified" = its fingerprint is an audited
    one, "unverified" = the profile's rule for a map nobody audited,
    "override" = `--normal-swizzle`, "skipped" = left out of the export."""

    source: str  # source texture name — diagnostic only, not identity
    fingerprint: str | None = None
    rule: str | None = None  # `NormalRule.key()`, None when skipped
    basis: str
    status: Literal["override", "verified", "unverified", "skipped"]
    evidence: str | None = None  # R≈G≈B diagnostic of the source, never the decision


class Provenance(BaseModel):
    """Which source Unity objects a scan's export came from, and which
    approximations were made — for debugging and re-conversion, not consumed
    by the viewer."""

    sceneRoot: str
    meshes: list[str] = Field(default_factory=list)
    materials: dict[str, MaterialProvenance] = Field(default_factory=dict)
    textures: dict[str, str] = Field(default_factory=dict)  # texture key -> source texture name(s)
    normalMaps: dict[str, NormalMapProvenance] = Field(default_factory=dict)  # texture key -> decision
    approximations: list[str] = Field(default_factory=list)


class ScanEntry(BaseModel):
    id: str
    dir: str
    pivot: tuple[float, float, float]
    bounds: Bounds
    bodyBounds: Bounds
    heightM: float
    tiers: list[str]  # tier ids actually converted for this scan
    static: GeometryFile
    lods: list[ScanLod]
    parts: list[ScanPart]
    materials: dict[str, MaterialEntry]
    textures: dict[str, dict[str, ScanTexture]]  # tier id -> texture key -> texture
    budget: dict[str, TierBudget]  # tier id -> budget
    provenance: Provenance
    # Per-scan source, added when `out/` started merging scans from more
    # than one dataset (docs/DATASETS.md) — the top-level `source` above
    # can only describe one. Optional/`None` so a manifest written before
    # this field existed still validates (`cli.py._merge_manifest` keeps
    # such a scan's entry as-is rather than dropping it). The viewer does
    # not read this field — see docs/manifest.md.
    source: Source | None = None
    # Which `sets` entry (below) this scan's poses belong to — the id is a
    # `DatasetProfile.id` (`profiles.py`, e.g. "vrgirlz-2015"). Optional/`None`
    # so a manifest written before this field existed still validates; the
    # viewer groups such scans under a synthetic "Other" bucket.
    set: str | None = None


class EnvironmentEntry(BaseModel):
    id: str
    type: str


class SetEntry(BaseModel):
    """One `sets[id]` entry — the dataset a group of scans came from, keyed
    by a slug of `_infer_app_name(data_dir)` (`cli.py`'s `_dataset_set_id`),
    NOT `DatasetProfile.id` — more than one real-world product can share a
    profile (Lucid Dreams II and Ariel are both Unity 5.x / `vrgirlz-2016`),
    so the profile id alone can't key `sets` without collisions. Powers the
    viewer's one-button-per-dataset picker (docs/manifest.md); optional/
    absent on a manifest written before this field existed."""

    label: str  # human-readable, from `_infer_app_name(data_dir)` (cli.py) or `profile.label` as fallback
    profile: str  # DatasetProfile.id this dataset was converted with — informational, not the dict key
    app: str | None = None  # raw source.app this run saw, if it differed from label


class Manifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: Literal["vrscan-manifest/1"] = Field(alias="schema")
    generator: Generator
    source: Source
    tiers: dict[str, Tier]
    presets: dict[str, str]
    scans: list[ScanEntry]
    environments: list[EnvironmentEntry] = Field(default_factory=list)
    # Dataset groups scans belong to (`ScanEntry.set` keys into this), added
    # when the viewer started needing one button per dataset instead of one
    # flat scan list. Optional/empty so a manifest written before this field
    # existed still validates.
    sets: dict[str, SetEntry] = Field(default_factory=dict)
