"""Per-part-kind material recipes.

Most of what a Unity material means in glTF terms isn't recoverable from the
data (no real roughness/metalness channel, no SSS, "wet" is a runtime
shader-keyword switch, not a second material) — the task's established facts
(`~/.claude/plans/partitioned-exploring-pearl.md`, 22.09 measurement pass)
pin down exactly what each part kind should become, so those are fixed
recipes here rather than something re-derived from shader reflection on
every run. What *is* read from data per scan: which texture is actually
bound to each role (by renderer PPtr, see `assembler.py`), and `_Color` for
`baseColorFactor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vrscan.ir import MaterialDef, Part, Scan, TextureRef
from vrscan.reader.backend import TextureFingerprint

AlphaMode = Literal["OPAQUE", "MASK", "BLEND"]
MaterialType = Literal["pbr", "unlit"]

# Texture key prefix per material key — everywhere the material key itself
# (e.g. "eyelashes" -> texture keys "eyelash_CLR" per docs/manifest.md's
# example; every other kind's prefix equals its material key).
TEXTURE_KEY_PREFIX_OVERRIDE = {"eyelashes": "eyelash"}


@dataclass(frozen=True)
class MaterialRecipe:
    type: MaterialType = "pbr"
    alpha_mode: AlphaMode = "OPAQUE"
    alpha_cutoff: float | None = None
    double_sided: bool = False
    roughness_factor: float | None = None  # fixed override, e.g. eyes = 0.15
    use_normal: bool = False
    use_orm: bool = False  # build R=occlusion, G=roughness(specular), B=0
    use_base_color_factor: bool = False  # read Unity material's `_Color`
    base_color_format: Literal["jpeg", "png_rgba"] = "jpeg"  # eyes/eyelashes/shadow need alpha, PNG
    use_base_color_texture: bool = True  # False = no baseColor export at all (eyes: source `Eye-T` is empty)
    fixed_base_color_factor: tuple[float, float, float, float] | None = None  # constant instead of `_Color`
    blend: Literal["normal", "additive"] = "normal"  # viewer hint for BLEND materials (specular-only overlays)


RECIPES: dict[str, MaterialRecipe] = {
    "body": MaterialRecipe(use_normal=True, use_orm=True, use_base_color_factor=True),
    # Eyes (verified 22.09 on real data): the iris/sclera are photographed into
    # the body texture; the vendor's eye spheres sit ~0.5 mm above them and
    # their `Eye-T` texture decodes to all-zero RGBA. They are a cornea gloss
    # layer, so: no base color texture, black translucent base, low roughness,
    # additive blend in the viewer = only the environment highlight is added.
    "eyes": MaterialRecipe(
        alpha_mode="BLEND", roughness_factor=0.08, use_normal=True, use_base_color_texture=False,
        fixed_base_color_factor=(0.0, 0.0, 0.0, 0.35), blend="additive",
    ),
    # Eyelashes: `Eyelash` texture is grey RGBA (alpha = hair shape), the
    # colour comes from the material's `_Color` (dark brown on Nat).
    "eyelashes": MaterialRecipe(
        alpha_mode="MASK", alpha_cutoff=0.5, double_sided=True, base_color_format="png_rgba",
        use_base_color_factor=True,
    ),
    "shadow": MaterialRecipe(type="unlit", alpha_mode="BLEND", base_color_format="png_rgba"),
    "sofa": MaterialRecipe(use_normal=True, use_orm=True),
    "feet": MaterialRecipe(use_normal=True, use_orm=True),
}


def material_key_for_part(part: Part) -> str:
    """The manifest `materials` dict key a part's material belongs under —
    shared across parts for body/eyes/eyelashes/shadow (one material serves
    every part of that kind in a scan), one per part for sofa pieces/feet
    (different leather per sofa piece)."""

    if part.kind in ("body", "eyes", "eyelashes", "shadow"):
        return "eyes" if part.kind == "eyes" else part.kind
    return part.id  # sofa-C1, sofa-feet, ...


def texture_key_prefix(material_key: str) -> str:
    return TEXTURE_KEY_PREFIX_OVERRIDE.get(material_key, material_key)


@dataclass(frozen=True)
class MaterialPlan:
    """What one manifest `materials[key]` entry needs, resolved from IR —
    `gltf/writer.py` drives both the glTF `Material` it embeds per tier and
    the texture export list (`textures.py`) from this."""

    key: str
    kind: str
    recipe: MaterialRecipe
    source: MaterialDef
    wet_specular: TextureRef | None = None  # body only, from Part.wet_material

    @property
    def prefix(self) -> str:
        return texture_key_prefix(self.key)

    def texture(self, role: str) -> TextureRef | None:
        return next((t for t in self.source.textures if t.role == role), None)

    def unmapped_properties(self) -> list[str]:
        return sorted({t.property_name for t in self.source.textures if t.role == "unknown" and t.property_name})

    def base_color_factor(self) -> tuple[float, float, float, float] | None:
        if self.recipe.fixed_base_color_factor is not None:
            return self.recipe.fixed_base_color_factor
        if not self.recipe.use_base_color_factor:
            return None
        return self.source.colors.get("_Color", (1.0, 1.0, 1.0, 1.0))


def build_material_plans(scan: Scan) -> dict[str, MaterialPlan]:
    """One `MaterialPlan` per distinct material key referenced by `scan.parts`
    (first part of that kind wins — established fact: parts sharing a
    material key always point at the same source material)."""

    plans: dict[str, MaterialPlan] = {}
    for part in scan.parts:
        if part.material is None:
            continue
        key = material_key_for_part(part)
        if key in plans:
            continue
        wet_specular = None
        if part.kind == "body" and part.wet_material is not None:
            wet_specular = next((t for t in part.wet_material.textures if t.role == "specular"), None)
        plans[key] = MaterialPlan(
            key=key, kind=part.kind, recipe=RECIPES[part.kind], source=part.material, wet_specular=wet_specular
        )
    return plans


# Which source texture roles a recipe can actually put into the package —
# used by `material_export_signature` to compare only what is exported (a
# `specularWet` map is a pose-level material of its own, see
# `Part.wet_material`, so it is not part of a renderer's signature).
_SIGNIFICANT_ROLES = ("baseColor", "normal", "occlusion", "specular")


def signature_textures(kind: str, material: MaterialDef) -> dict[str, TextureRef | None]:
    """Role -> the source texture `kind`'s recipe exports for it (None: the
    material has none), over the roles the recipe uses at all."""

    recipe = RECIPES[kind]
    textures: dict[str, TextureRef | None] = {}
    for role in _SIGNIFICANT_ROLES:
        if role == "baseColor" and not recipe.use_base_color_texture:
            continue
        if role == "normal" and not recipe.use_normal:
            continue
        if role in ("occlusion", "specular") and not recipe.use_orm:
            continue
        textures[role] = next((t for t in material.textures if t.role == role), None)
    return textures


def material_export_signature(
    kind: str, material: MaterialDef | None, fingerprint: TextureFingerprint
) -> dict[str, object]:
    """The *effective export signature* of a material: everything about it
    that survives into the glTF package, and nothing that doesn't.

    Two source materials with different names or PPtrs but the same
    signature produce byte-identical output, so they are interchangeable;
    two with different signatures are not, and exporting one part with a
    single one of them would silently ship the wrong look for some LOD
    levels (design review 22.09 point 5). Structured (not hashed) so a
    mismatch can name the field that differs.

    A texture is its content fingerprint (`ReaderBackend.
    get_texture_fingerprint`), not its name/size/format: same pixels under
    another name are the same export, different pixels under the same name
    are not (design review 22.09 B1). The rest of how a texture is used adds
    nothing per material yet — every texture goes through one sampler on
    TEXCOORD_0 with no transform, and the normal-map rule is decided per
    fingerprint (`profiles.NormalRules.decide`). Names stay out of the
    comparison; `signature_textures` gives them for diagnostics."""

    if material is None:
        return {"material": None}
    recipe = RECIPES[kind]
    textures = {
        role: None if tex is None else fingerprint(tex.id)
        for role, tex in signature_textures(kind, material).items()
    }
    plan = MaterialPlan(key=kind, kind=kind, recipe=recipe, source=material)
    return {
        "type": recipe.type,
        "alphaMode": recipe.alpha_mode,
        "alphaCutoff": recipe.alpha_cutoff,
        "doubleSided": recipe.double_sided,
        "roughnessFactor": recipe.roughness_factor,
        "baseColorFormat": recipe.base_color_format,
        "blend": recipe.blend,
        "textures": textures,
        "baseColorFactor": plan.base_color_factor(),
    }
