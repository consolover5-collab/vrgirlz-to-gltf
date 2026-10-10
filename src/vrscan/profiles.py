"""Per-dataset naming/loading rules — the one place that knows *VRGirlz*
scenes come in more than one shape.

Established facts pinned down from two real data folders (22.09):
- `vrgirlz-2015` (Nat, Unity 4.6.3f1): `level0` + `sharedassets1.assets`
  only; pose roots always named `NN-150K`; eyes `Eyes-Left/Right_LODk`;
  eyelashes always `Eyelashes_LODk` (one renderer per level); sofa split
  into pieces `NN-C1/-L1/-M1/-SR1` (+ shared `Sofa-Feet`); shadow `NNshad`.
- `vrgirlz-2016` (Lucid Dreams II demo, Unity 5.3.4p1): geometry spread
  across `level0` (33 objects, a messy leftover/menu scene) + `level1` (760
  objects, the real per-pose hierarchies) + `sharedassets0/1.assets`; pose
  roots are mostly `NN-150K` but one demo scan is named `Nat-42` (literally
  reused from the Nat product) — name-pattern matching alone can't find
  every pose here, see `find_pose_roots` in `assembler.py`. Eyes are
  `Eyes-(Clear-)?Left/Right_LODk` (both sub-styles appear, per source
  product). Eyelashes are usually ONE renderer shared across every LOD level
  (bare `Eyelashes`, no `_LODk` suffix) rather than one renderer per level —
  the same GameObject shows up in more than one `LODGroup` level (found live
  on pose `21-150K`: path_id 543 referenced by both L0 and L1), which is why
  `assembler.py`'s level lookup had to become "levels", plural. Body has no
  per-pose normal/occlusion/specular texture (just `_MainTex` + a *shared*
  micro-detail texture, `skin02_DMM`, reused by every pose — unmapped, not
  guessed at) and no standalone "wet" material. Furniture, where present, is
  a single `NNsofa` mesh (not the C1/L1/M1/SR1 split), no shared feet piece.
  Shadow nodes are named `NNshad` for some poses and plainly `Shad` for
  others (same source scene, inconsistent authoring between the demo's
  bundled products) — `shadow_re` accepts both.

`level0`/`level1` legitimately have disjoint `path_id` ranges in the one
Lucid file pair spot-checked (verified live, 22.09), but a `path_id` is only
guaranteed unique *within* one serialized file — `unitypy_backend.py` keys
every structural dict by `(file index, path_id)` (`reader/backend.py`'s
`NodeId`) so a pair that does collide can't misresolve PPtrs silently.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class NormalRule:
    """How one normal map is actually encoded — not a guess.

    `swizzle` says where X and Y live ("ag" = DXT5nm's X in alpha / Y in
    green, "rg" = X in red / Y in green, "none" = a plain tangent map);
    `flip_x`/`flip_y` cover a source authored in the opposite handedness
    (DirectX-style −Y); `z` says whether the blue channel holds Z or Z has
    to be rebuilt from X and Y (DXT5nm leaves blue to the compressor).
    Whether a rule is *established* for a given map is not a property of
    the rule but of how it was picked — see `NormalDecision`."""

    swizzle: Literal["ag", "rg", "none"]
    flip_x: bool = False
    flip_y: bool = False
    z: Literal["reconstruct", "stored"] = "reconstruct"

    def __post_init__(self) -> None:
        # A plain tangent map has nothing but blue to hold Z; rebuilding it
        # from X and Y would ignore the stored sign and flip z<0 texels
        # (independent review 22.09, P1).
        if self.swizzle == "none" and self.z == "reconstruct":
            raise ValueError("NormalRule(swizzle='none') stores Z in blue — use z='stored', not 'reconstruct'")

    @classmethod
    def for_swizzle(cls, swizzle: Literal["ag", "rg", "none"]) -> NormalRule:
        """The default form of a bare swizzle name: Z read from blue for a
        plain map, rebuilt from X and Y for a repacked one. The single place
        that mapping lives — the CLI flag and the string form of
        `textures.decode_normal_vectors` both go through it, so they cannot
        drift apart into different cache keys."""

        return cls(swizzle=swizzle, z="stored" if swizzle == "none" else "reconstruct")

    def key(self) -> str:
        """Short identity for cache keys — everything that changes the
        output bytes, nothing that doesn't."""

        flips = f"{'X' if self.flip_x else ''}{'Y' if self.flip_y else ''}"
        return f"{self.swizzle}{'-flip' + flips if flips else ''}-{self.z}"

    @classmethod
    def from_key(cls, key: str) -> NormalRule:
        """Inverse of `key()` — how `normal_pins.py` stores the rule of each
        audited map."""

        swizzle, _, rest = key.partition("-")
        flips, z = "", "reconstruct"
        for field_text in filter(None, rest.split("-")):
            if field_text.startswith("flip"):
                flips = field_text[len("flip"):]
            else:
                z = field_text
        return cls(swizzle=swizzle, flip_x="X" in flips, flip_y="Y" in flips, z=z)


NormalStatus = Literal["override", "verified", "unverified", "skipped"]


@dataclass(frozen=True)
class NormalDecision:
    """Which rule one normal map gets, why, and how established that is —
    what `provenance.normalMaps` records per map (design review 22.09 A1:
    "правило, основание выбора и статус проверки"). `rule=None` means the
    map is left out of the export (status "skipped")."""

    rule: NormalRule | None
    basis: str
    status: NormalStatus


@dataclass(frozen=True)
class NormalRules:
    """How a convert run picks each normal map's rule, most authoritative
    first (design review 22.09 A1):

    1. `override` — an explicit `--normal-swizzle`, for every map;
    2. `verified` — the map's content fingerprint is one the audit decoded
       and established (`normal_pins.VERIFIED_NORMAL_MAPS`). Content, not
       the dataset's folder name: renaming a folder changes nothing, and a
       buyer's identical copy of a vendor file is recognised as such;
    3. `profile_rule` — the profile's rule for a map nobody has audited,
       applied with status "unverified" and a warning;
    4. nothing — the map is skipped, with a warning; `--normal-swizzle`
       settles it.

    No channel-statistics autodetect anywhere in this order: it cannot tell
    AG from RG, and was wrong on 36 of the 66 audited maps."""

    override: NormalRule | None = None
    verified: Mapping[str, NormalRule] = field(default_factory=dict)
    profile_rule: NormalRule | None = None
    profile_id: str = ""

    def decide(self, fingerprint: str | None) -> NormalDecision:
        if self.override is not None:
            return NormalDecision(self.override, "--normal-swizzle", "override")
        rule = self.verified.get(fingerprint) if fingerprint is not None else None
        if rule is not None:
            return NormalDecision(rule, "audited fingerprint (normal_pins.py)", "verified")
        if self.profile_rule is not None:
            return NormalDecision(self.profile_rule, f"profile {self.profile_id} rule, map not audited", "unverified")
        return NormalDecision(None, f"no audited fingerprint and no rule in profile {self.profile_id}", "skipped")


@dataclass(frozen=True)
class DatasetProfile:
    id: str
    label: str
    # Unity version string prefixes (`SerializedFile.unity_version`, e.g.
    # "4.6.3f1") that auto-detection maps to this profile.
    unity_version_prefixes: tuple[str, ...]
    # Files to load, relative to the `*_Data` dir, in `UnityPy.load()` order.
    # Glob patterns (`level*`, `sharedassets*.assets`) or exact names —
    # `discover_source_files` resolves them against what's actually on disk.
    source_file_patterns: tuple[str, ...]
    pose_root_re: re.Pattern[str]  # "NN-150K"-style names -> group "pose", used for scan.id when it matches
    body_re: re.Pattern[str]  # group "lod" (required)
    eyes_re: re.Pattern[str]  # groups "side", "lod" (required)
    eyelashes_re: re.Pattern[str]  # group "lod" (optional — a shared renderer has no per-level suffix)
    sofa_re: re.Pattern[str]  # groups "pose" (optional) + "part" (optional, multi-piece sofas only)
    feet_re: re.Pattern[str]  # no groups — always a shared, kind-generic name
    shadow_re: re.Pattern[str]  # group "pose" (optional — some sources drop the pose prefix)
    # Whether scan ids from this profile need disambiguating at all — Nat's
    # ids ("79".."91") stay bare for manifest compatibility. Profiles alone
    # can't supply a *fixed* prefix string any more: more than one real
    # dataset can share a profile (Lucid Dreams II and Ariel are both Unity
    # 5.x / `vrgirlz-2016`), so the actual prefix is per-*dataset* — a slug
    # of `_infer_app_name(data_dir)`, computed by `cli.py` and passed into
    # `scan_id`/`assemble` per run — not per-profile.
    prefix_scan_ids: bool = False
    # Explicit "start hidden, hideable" overrides beyond the universal kind
    # rules (`assembler.py`'s `HIDDEN_BY_DEFAULT_KINDS`, currently just
    # "eyes") — keyed by `local_id(root_name)`, value = kinds to hide for
    # that one scan. Exists for known one-off cases a general heuristic
    # hasn't been built (or verified) for yet — see VRGIRLZ_2015's own
    # `hidden_parts` below for the concrete case this was added for.
    hidden_parts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # The rule for a normal map whose fingerprint the audit has not seen
    # (`NormalRules` step 3) — applied as "unverified", with a warning.
    # None = such maps are skipped instead.
    normal_rule: NormalRule | None = None
    # The exact Unity releases this profile has been converted and checked
    # on. Another release with one of `unity_version_prefixes` is only
    # assumed to read the same way (`match_profile`).
    audited_unity_versions: tuple[str, ...] = ()

    def local_id(self, root_name: str) -> str:
        """The un-prefixed id (e.g. "21", or a slug of the raw name when it
        doesn't match `pose_root_re`, e.g. "Nat-42" -> "nat-42") — what
        sofa/shadow name prefixes are checked against in `assembler.py`."""

        match = self.pose_root_re.match(root_name)
        if match:
            return match.group("pose")
        return re.sub(r"[^a-z0-9]+", "-", root_name.lower()).strip("-")

    def scan_id(self, root_name: str, scan_id_prefix: str = "") -> str:
        """The manifest-facing id — `local_id`, prefixed with
        `scan_id_prefix` (a per-dataset slug, see `prefix_scan_ids` above)
        when this profile opts into prefixing at all. `scan_id_prefix` is
        ignored outright for a profile with `prefix_scan_ids=False`, so
        Nat's ids never move even if a caller passes one."""

        prefix = scan_id_prefix if self.prefix_scan_ids else ""
        return f"{prefix}{self.local_id(root_name)}"


# The profile rule of both profiles. Every one of the 66 maps the audit
# covered is DXT5nm (`normal_pins.py` for what that established), so it is
# the best guess for a map of the same products the audit has not seen —
# still only a guess, hence "unverified".
_DXT5NM_AG = NormalRule(swizzle="ag", z="reconstruct")

VRGIRLZ_2015 = DatasetProfile(
    id="vrgirlz-2015",
    label="VRGirlz 2015 (Nat-style, Unity 4.6)",
    unity_version_prefixes=("4.",),
    source_file_patterns=("level0", "sharedassets1.assets"),
    pose_root_re=re.compile(r"^(?P<pose>\d{2})-150K$"),
    body_re=re.compile(r"^(?P<pose>\d{2})_LOD(?P<lod>\d+)(?:_MeshPart\d+)?$"),
    eyes_re=re.compile(r"^Eyes-(?P<side>Left|Right)_LOD(?P<lod>\d+)$"),
    eyelashes_re=re.compile(r"^Eyelashes_LOD(?P<lod>\d+)$"),
    sofa_re=re.compile(r"^(?P<pose>\d{2})-(?P<part>C1|L1|M1|SR1)$"),
    feet_re=re.compile(r"^Sofa-Feet$"),
    shadow_re=re.compile(r"^(?P<pose>\d{2})shad$"),
    # Pose 90's eyelashes named explicitly in the design review's 22.09
    # evening follow-up ("Уточнение 22.09 вечер", point 6) as needing to
    # start hidden, with manual toggle — same mechanism as eyes, kept a
    # one-off exception rather than a general "buried" heuristic, which the
    # same review explicitly says not to build yet (metric unverified).
    hidden_parts={"90": ("eyelashes",)},
    normal_rule=_DXT5NM_AG,
    audited_unity_versions=("4.6.3f1",),  # Nat-21, Saph-11
)

VRGIRLZ_2016 = DatasetProfile(
    id="vrgirlz-2016",
    label="VRGirlz 2016 (Lucid Dreams II demo-style, Unity 5.x)",
    unity_version_prefixes=("5.",),
    source_file_patterns=("level*", "sharedassets*.assets"),
    pose_root_re=re.compile(r"^(?P<pose>\d{2})-150K$"),
    body_re=re.compile(r"^(?P<pose>\d{2})_LOD(?P<lod>\d+)(?:_MeshPart\d+)?$"),
    eyes_re=re.compile(r"^Eyes-(?:Clear-)?(?P<side>Left|Right)_LOD(?P<lod>\d+)$"),
    eyelashes_re=re.compile(r"^Eyelashes(?:_LOD(?P<lod>\d+))?$"),
    sofa_re=re.compile(r"^(?P<pose>\d{2})sofa$"),
    feet_re=re.compile(r"^Sofa-Feet$"),
    shadow_re=re.compile(r"^(?:(?P<pose>\d{2})shad|Shad)$"),
    prefix_scan_ids=True,
    normal_rule=_DXT5NM_AG,
    audited_unity_versions=("5.3.4p1",),  # Lucid Dreams II, Ariel-09, Nat-18, Chloe-22
)

PROFILES: dict[str, DatasetProfile] = {p.id: p for p in (VRGIRLZ_2015, VRGIRLZ_2016)}

DEFAULT_PROFILE = VRGIRLZ_2015  # preserves existing Nat behaviour when nothing else says otherwise


class DatasetError(ValueError):
    """A folder can't be read as a supported dataset: not a `*_Data`
    folder, a Unity release no profile covers, a file the dataset needs is
    missing. The message is what the user is shown, one line, no
    traceback."""


# How established reading a dataset with its profile is: "supported" — a
# release the profile was checked on; "assumed" — only the version prefix
# matches; "override" — `--profile` chose it.
SupportStatus = Literal["supported", "assumed", "override"]


@dataclass(frozen=True)
class ProfileMatch:
    profile: DatasetProfile
    support: SupportStatus
    note: str  # one line for the user: the release, the profile, and why it's only assumed if it is


def detect_profile(unity_version: str) -> DatasetProfile | None:
    """Unity version string (e.g. "4.6.3f1") -> the profile whose version
    prefix it carries, or None. There is no fallback profile: reading an
    unknown release as some default one would convert it on a guess
    (`match_profile` reports it as not supported; `--profile` is the escape
    hatch)."""

    for profile in PROFILES.values():
        if unity_version.startswith(profile.unity_version_prefixes):
            return profile
    return None


def match_profile(unity_version: str, override: DatasetProfile | None = None) -> ProfileMatch:
    """The profile a dataset of this Unity release is read with, and how
    established that is. Raises `DatasetError` for a release no profile
    covers, unless `override` names one."""

    release = f"Unity {unity_version}" if unity_version else "Unity release unreadable"
    if override is not None:
        return ProfileMatch(override, "override", f"{release}, profile {override.id} (chosen with --profile)")
    profile = detect_profile(unity_version)
    if profile is None:
        known = ", ".join(v for p in PROFILES.values() for v in p.audited_unity_versions)
        raise DatasetError(f"{release}: not supported (supported releases: {known})")
    if unity_version in profile.audited_unity_versions:
        return ProfileMatch(profile, "supported", f"{release}, profile {profile.id}")
    checked = ", ".join(profile.audited_unity_versions)
    return ProfileMatch(
        profile, "assumed",
        f"{release}: recognised tentatively as {profile.id} (checked on {checked}) — check the result",
    )


def discover_source_files(data_dir: Path, profile: DatasetProfile) -> list[Path]:
    """Resolve `profile.source_file_patterns` against what's actually in
    `data_dir`, in pattern order (then name order within a pattern) —
    exact names (Nat: `level0`, `sharedassets1.assets`) pass through
    unchanged; globs (Lucid: `level*`, `sharedassets*.assets`) are filtered
    to plain data files (`levelN`, no `.resS`/`.resource` companion, which
    UnityPy picks up on its own once the primary file is loaded — see
    `docs/DATASETS.md`).

    Every pattern must match: a missing file is a `DatasetError` naming
    it, not a dataset read without it."""

    files: list[Path] = []
    missing: list[str] = []
    for pattern in profile.source_file_patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(
                p for p in data_dir.glob(pattern)
                if p.is_file() and re.fullmatch(r"level\d+|sharedassets\d+\.assets", p.name)
            )
            files.extend(matches)
        else:
            candidate = data_dir / pattern
            matches = [candidate] if candidate.is_file() else []
            files.extend(matches)
        if not matches:
            missing.append(pattern)
    if missing:
        raise DatasetError(
            f"{data_dir.name} is incomplete — missing: {', '.join(missing)} "
            f"(profile {profile.id}; copy the whole *_Data folder)"
        )
    return files
