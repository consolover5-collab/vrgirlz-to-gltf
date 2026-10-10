"""profiles.py — detection, per-profile id derivation, source file discovery."""

from __future__ import annotations

import pytest

from vrscan.profiles import (
    PROFILES,
    VRGIRLZ_2015,
    VRGIRLZ_2016,
    DatasetError,
    detect_profile,
    discover_source_files,
    match_profile,
)


def test_detect_profile_by_unity_version():
    assert detect_profile("4.6.3f1") is VRGIRLZ_2015
    assert detect_profile("5.3.4p1") is VRGIRLZ_2016


def test_unknown_release_is_not_supported_rather_than_read_as_a_default():
    assert detect_profile("2018.4.1f1") is None
    assert detect_profile("") is None
    with pytest.raises(DatasetError, match=r"Unity 2018\.4\.1f1: not supported \(supported releases: 4\.6\.3f1, 5\.3\.4p1\)"):
        match_profile("2018.4.1f1")


def test_a_release_is_supported_only_where_it_was_checked():
    assert match_profile("4.6.3f1").support == "supported"
    assert match_profile("5.3.4p1").support == "supported"
    assumed = match_profile("5.6.0f3")
    assert (assumed.profile, assumed.support) == (VRGIRLZ_2016, "assumed")
    assert "recognised tentatively as vrgirlz-2016 (checked on 5.3.4p1)" in assumed.note
    forced = match_profile("2018.4.1f1", VRGIRLZ_2016)
    assert (forced.profile, forced.support) == (VRGIRLZ_2016, "override")


def test_scan_id_matches_pose_root_pattern():
    assert VRGIRLZ_2015.local_id("91-150K") == "91"
    # Nat ignores any scan_id_prefix outright (prefix_scan_ids=False) — ids
    # stay bare ("91") for manifest compatibility no matter what the caller
    # (cli.py's per-dataset slug) passes in.
    assert VRGIRLZ_2015.scan_id("91-150K", "some-dataset-") == "91"
    assert VRGIRLZ_2016.local_id("21-150K") == "21"
    # Lucid-style profiles (prefix_scan_ids=True) apply whatever prefix
    # they're given — the prefix itself is per-dataset, not baked into the
    # profile (see cli.py's _dataset_set_id).
    assert VRGIRLZ_2016.scan_id("21-150K") == "21"  # no prefix passed -> none applied
    assert VRGIRLZ_2016.scan_id("21-150K", "ld2-") == "ld2-21"


def test_scan_id_falls_back_to_slug_for_nonconforming_root_name():
    assert VRGIRLZ_2016.local_id("Nat-42") == "nat-42"
    assert VRGIRLZ_2016.scan_id("Nat-42", "ld2-") == "ld2-nat-42"


def test_discover_source_files_exact_names(tmp_path):
    (tmp_path / "level0").write_bytes(b"x")
    (tmp_path / "sharedassets1.assets").write_bytes(b"x")
    (tmp_path / "sharedassets0.assets").write_bytes(b"x")  # present but NOT in the 2015 pattern list
    (tmp_path / "resources.assets").write_bytes(b"x")

    files = discover_source_files(tmp_path, VRGIRLZ_2015)
    assert [p.name for p in files] == ["level0", "sharedassets1.assets"]


def test_discover_source_files_glob_picks_up_every_level_and_sharedassets(tmp_path):
    for name in ("level0", "level1", "sharedassets0.assets", "sharedassets1.assets"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sharedassets1.assets.resS").write_bytes(b"x")  # streaming companion — must NOT be listed
    (tmp_path / "sharedassets1.resource").write_bytes(b"x")  # ditto
    (tmp_path / "resources.assets").write_bytes(b"x")  # never opened, per plan — must NOT be listed

    files = discover_source_files(tmp_path, VRGIRLZ_2016)
    assert [p.name for p in files] == ["level0", "level1", "sharedassets0.assets", "sharedassets1.assets"]


def test_discover_source_files_names_what_is_missing(tmp_path):
    (tmp_path / "level0").write_bytes(b"x")
    with pytest.raises(DatasetError, match="missing: sharedassets1.assets"):
        discover_source_files(tmp_path, VRGIRLZ_2015)
    with pytest.raises(DatasetError, match=r"missing: sharedassets\*\.assets"):
        discover_source_files(tmp_path, VRGIRLZ_2016)


def test_profiles_registry_keyed_by_own_id():
    assert PROFILES["vrgirlz-2015"] is VRGIRLZ_2015
    assert PROFILES["vrgirlz-2016"] is VRGIRLZ_2016


def test_vrgirlz_2015_hides_nat_90_eyelashes_only():
    """design review 22.09 point 6 — an explicit, named one-off, not the
    general "50% buried" heuristic (explicitly deferred)."""

    assert VRGIRLZ_2015.hidden_parts == {"90": ("eyelashes",)}
    assert VRGIRLZ_2016.hidden_parts == {}  # no scan-specific overrides for this profile


def test_both_profiles_fall_back_to_dxt5nm_for_unaudited_maps():
    """The profile rule is only the guess for a map the audit has not seen
    (`NormalRules` step 3) — no dataset list gates it any more."""

    for profile in (VRGIRLZ_2015, VRGIRLZ_2016):
        rule = profile.normal_rule
        assert (rule.swizzle, rule.flip_x, rule.flip_y, rule.z) == ("ag", False, False, "reconstruct")
