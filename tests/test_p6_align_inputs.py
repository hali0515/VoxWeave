import math


def _valid_profile(language="en"):
    from voxweave.config import gap_thresholds
    from voxweave.core.layout import default_max_line_length, default_max_lines
    from voxweave.core.segdoc import THRESHOLD_KEYS
    from voxweave.core.smart_split import SplitThresholds

    resolved = SplitThresholds.from_mapping(gap_thresholds(language))
    return {
        "max_line_length": default_max_line_length(language),
        "max_lines": default_max_lines(language),
        **{key: getattr(resolved, key) for key in THRESHOLD_KEYS},
    }


def test_legacy_policy_is_sealed_verbatim_but_v2_validation_is_strict():
    from voxweave.align_inputs import LegacyAlignPolicy, validate_v2_policy

    signed = LegacyAlignPolicy(-0.0, 0.2, 0.5)
    assert math.copysign(1.0, signed.min_cue_sec) == -1.0
    assert validate_v2_policy(signed).kind == "valid"

    nonfinite = validate_v2_policy(LegacyAlignPolicy(float("inf"), 0.2, 0.5))
    assert (nonfinite.kind, nonfinite.detail_code) == (
        "invalid",
        "nonfinite-policy",
    )
    negative = validate_v2_policy(LegacyAlignPolicy(-0.1, 0.2, 0.5))
    assert (negative.kind, negative.detail_code) == ("invalid", "negative-policy")


def test_profile_resolution_uses_closed_sources_and_shared_defaults():
    from voxweave.align_inputs import resolve_align_profile

    absent = resolve_align_profile(None, effective_iso="en")
    assert absent.status.kind == "valid"
    assert absent.status.source == "manifest-absent"

    unsupported = resolve_align_profile(
        {"manifest_version": 99, "engine": "future", "profile": {"bad": True}},
        effective_iso="en",
    )
    assert unsupported.status.kind == "valid"
    assert unsupported.status.source == "unsupported-manifest"
    assert unsupported.profile == absent.profile

    no_profile = resolve_align_profile(
        {"manifest_version": 1, "engine": "legacy-v1", "language": "en"},
        effective_iso="en",
    )
    assert no_profile.status.source == "profile-absent"
    assert no_profile.profile == absent.profile


def test_stored_profile_is_exact_and_invalidity_never_falls_back_silently():
    from voxweave.align_inputs import resolve_align_profile

    profile = _valid_profile()
    valid = resolve_align_profile(
        {
            "manifest_version": 1,
            "engine": "boundary-optimizer-v2",
            "language": "en",
            "profile": profile,
            "descriptive-extra": {"ignored": True},
        },
        effective_iso="en",
    )
    assert valid.status.kind == "valid"
    assert valid.status.source == "stored-profile"
    assert valid.profile is not None
    assert valid.profile.max_line_length == profile["max_line_length"]

    extra = dict(profile, surprise=1)
    invalid = resolve_align_profile(
        {
            "manifest_version": 1,
            "engine": "legacy-v1",
            "language": "en",
            "profile": extra,
        },
        effective_iso="en",
    )
    assert invalid.profile is None
    assert (invalid.status.kind, invalid.status.source, invalid.status.detail_code) == (
        "invalid",
        "stored-profile",
        "profile-shape",
    )

    bool_layout = dict(profile, max_lines=True)
    invalid = resolve_align_profile(
        {
            "manifest_version": 1,
            "engine": "legacy-v1",
            "language": "en",
            "profile": bool_layout,
        },
        effective_iso="en",
    )
    assert invalid.status.detail_code == "profile-domain"


def test_language_override_to_different_iso_ignores_stored_profile():
    from voxweave.align_inputs import resolve_align_profile

    resolved = resolve_align_profile(
        {
            "manifest_version": 1,
            "engine": "legacy-v1",
            "language": "en",
            "profile": None,
        },
        effective_iso="ja",
        stored_iso="en",
    )
    assert resolved.status.kind == "valid"
    assert resolved.status.source == "language-override"
    assert resolved.profile is not None
    assert resolved.profile.language == "ja"


def test_finalizer_evidence_is_strict_sorted_and_closed():
    from voxweave.align_inputs import resolve_finalize_evidence

    valid = resolve_finalize_evidence(
        shot_changes=[3, 1.5, 2.0], sing_spans=[[4, 5], [1.0, 2.0]]
    )
    assert valid.status.kind == "valid"
    assert valid.shots == (1.5, 2.0, 3.0)
    assert valid.sing_spans == ((1.0, 2.0), (4.0, 5.0))

    for shots, sings in [([True], None), (None, [[2.0, 1.0]]), ([float("nan")], [])]:
        invalid = resolve_finalize_evidence(shot_changes=shots, sing_spans=sings)
        assert invalid.status.kind == "invalid"
        assert invalid.status.detail_code == "evidence-domain"
        assert invalid.shots is None and invalid.sing_spans is None
