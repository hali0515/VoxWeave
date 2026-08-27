# tests/test_canonical_text.py
"""The canonical text projection and PD-TEXT hard legality (P5 spec section 3).

Every expected value here is hand-derived in ``p5-w1-api.md`` sections 1.3 and
2.3 and was produced by a read-only probe of the real primitives at base
``99f3605`` -- none is a transcription of what the implementation happens to
return.

Two contracts carry the weight. The projection's primary source is the IMMUTABLE
``word_data`` join, never the delivered text -- a projection that reads what it
previously wrote cannot be a function of the seed. And legality is decided by
DIRECT inspection of the delivered lines and cell widths: ``_fits_budget`` is not
the predicate anywhere in the v2 lane, because it answers "could a rewrap fold
this" rather than "does what ships fit".
"""

import pytest

from voxweave.core.layout import _fits_budget
from voxweave.core.segdoc import DisplayProfile

# ---------------------------------------------------------------- fixtures


def profile(language="en", **over):
    """A resolved profile; every knob explicit so a test never inherits a default."""
    base = dict(
        language=language,
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=0.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11 / 24,
    )
    base.update(over)
    return DisplayProfile(**base)


def ja_profile(**over):
    base = dict(language="ja", max_line_length=18, max_lines=1)
    base.update(over)
    return profile(**base)


def wd(*surfaces):
    """Atom-level word_data, the shape both engines construct."""
    return [
        {"text": text, "start": float(i) / 10, "end": float(i + 1) / 10}
        for i, text in enumerate(surfaces)
    ]


def canon():
    from voxweave.core import canonical_text as module

    return module


# ------------------------------------------------------------ reconstruction


def test_reconstruct_surface_is_the_language_join():
    module = canon()
    assert module.reconstruct_surface(wd("Hello,", "world!!"), "en") == "Hello, world!!"
    assert module.reconstruct_surface(wd("こ", "れ", "は"), "ja") == "これは"


def test_projection_reads_word_data_not_the_delivered_text():
    """The fallback text is IGNORED whenever the reconstruction is usable."""
    module = canon()
    final = module.canonical_text(
        wd("Hello,", "world!!"),
        fallback_text="ALREADY DELIVERED TEXT",
        lang="en",
        profile=profile(),
    )
    assert final.source == "word-data"
    assert final.text == "Hello world"


# ------------------------------------------------------- strip/stutter/wrap


def test_cf9_strip_drops_punctuation_from_the_reading_load():
    """CF-9: raw 13 non-space chars, canonical 10. The gap is FD-1's trigger."""
    module = canon()
    final = module.canonical_text(
        wd("Hello,", "world!!"), fallback_text="", lang="en", profile=profile()
    )
    assert final.text == "Hello world"
    assert final.lines == ("Hello world",)
    assert final.cell_widths == (11,)
    assert final.reading_chars == 10


def test_cf10_bounded_stutter_is_stable_within_four_scans():
    """CF-10: scan1 'I-I I', scan2 'I-I-I', scan3 no change -> 3 charged scans."""
    module = canon()
    final = module.canonical_text(
        wd("I", "I", "I"), fallback_text="", lang="en", profile=profile()
    )
    assert final.text == "I-I-I"
    assert (final.stutter_stable, final.stutter_scans) == (True, 3)
    assert final.reading_chars == 5  # the hyphens count; the load is NOT 3


def test_cf11_injected_five_scan_double_reports_fd9(monkeypatch):
    """CF-11: FD-9's report fires on the fifth needed scan; the 4x text still ships.

    The double is length-NONINCREASING and every scan is charged, so the fixture
    cannot pass by making the loop do less work than the real substitution.
    """
    module = canon()
    remaining = ["eeee", "ddd", "ccc", "bb", "a"]

    def double(text):
        return remaining.pop(0) if remaining else text

    monkeypatch.setattr(module, "_stutter_sub", double)
    text, stable, scans = module.bounded_stutter("fffff")
    assert (text, stable, scans) == ("bb", False, 4)


def test_cf16_reading_chars_is_newline_insensitive():
    """CF-16: the wrap inserts a newline; the CPS load is unchanged by layout."""
    module = canon()
    tokens = [
        "aaaa",
        "bbbb",
        "cccc",
        "dddd",
        "eeee",
        "ffff",
        "gggg",
        "hhhh",
        "iiii",
        "jjjj",
        "kkkk",
    ]
    final = module.canonical_text(
        wd(*tokens), fallback_text="", lang="en", profile=profile()
    )
    assert final.lines == (
        "aaaa bbbb cccc dddd eeee",
        "ffff gggg hhhh iiii jjjj kkkk",
    )
    assert final.cell_widths == (24, 29)
    assert final.reading_chars == 44


# ------------------------------------------------------------ typed fallback


def test_cf14_empty_reconstruction_falls_back_and_is_not_a_truthy_list_test():
    """CF-14: two timed entries with NO surfaces. The list is truthy; the text is not."""
    module = canon()
    final = module.canonical_text(
        [{"start": 0.0, "end": 0.4}, {"start": 0.4, "end": 0.8}],
        fallback_text="salvaged text",
        lang="en",
        profile=profile(),
    )
    assert final.source == "fallback"
    assert final.fallback_reason == "empty-reconstruction"
    assert final.text == "salvaged text"


def test_cf15_footprint_mismatch_falls_back():
    """CF-15: with a footprint supplied, usability IS conservation of it."""
    module = canon()
    final = module.canonical_text(
        wd("abc"),
        fallback_text="abc def",
        lang="en",
        profile=profile(),
        expected_footprint="abc def",
    )
    assert (final.source, final.fallback_reason) == ("fallback", "footprint-mismatch")


def test_granularity_unreconciled_falls_back():
    """No footprint -> the granularity-reconciliation path decides.

    ``3 | . | 75`` joins to ``3 . 75`` for a spaced language. The context-sensitive
    ``[.,](?!\\d)`` rule keeps the dot on the unit side (it is followed by a digit
    there) and drops it on the text side, so ``_surface_ranges`` cannot reconcile
    the two readings and the projection refuses rather than inventing one.
    """
    module = canon()
    final = module.canonical_text(
        wd("3", ".", "75"), fallback_text="3.75", lang="en", profile=profile()
    )
    assert (final.source, final.fallback_reason) == (
        "fallback",
        "granularity-unreconciled",
    )


def test_fallback_reasons_are_a_closed_vocabulary():
    module = canon()
    assert module.FALLBACK_REASONS == (
        "empty-reconstruction",
        "footprint-mismatch",
        "granularity-unreconciled",
    )


# ---------------------------------------------------------- PD-TEXT legality


def test_canonical_legal_is_direct_inspection():
    module = canon()
    legal = module.canonical_text(
        wd(*list("これはみじかいぶんです")),
        fallback_text="",
        lang="ja",
        profile=ja_profile(),
    )
    assert legal.cell_widths == (22,)
    assert module.canonical_legal(legal, ja_profile()) is True


def test_cf13_over_wide_single_line_is_illegal_with_no_indivisible_token():
    """CF-13: 21 kana = 42 cells, budget 36, max_lines 1 -> the wrap has no split budget."""
    module = canon()
    kana = list("これはとてもながいにほんごのぶんしょうです")
    final = module.canonical_text(
        wd(*kana), fallback_text="", lang="ja", profile=ja_profile()
    )
    assert len(final.lines) == 1
    assert final.cell_widths == (42,)
    assert module.canonical_legal(final, ja_profile()) is False
    assert (
        module.over_wide_token(final.lines[0], "ja", module.line_budget(ja_profile()))
        is None
    )


def test_cf12_indivisible_token_is_delivered_and_named():
    """CF-12: wrap cannot break inside a token, so the over-wide line ships."""
    module = canon()
    token = "supercalifragilisticexpialidociousandthensome_extra_tail"
    final = module.canonical_text(
        wd(token), fallback_text="", lang="en", profile=profile()
    )
    assert final.cell_widths == (56,)
    assert module.canonical_legal(final, profile()) is False
    assert module.over_wide_token(final.lines[0], "en", 42) == token


def test_kinsoku_pullback_overflows_a_line_fits_budget_admits():
    """The pinned PD-TEXT counterexample: ``_fits_budget`` says yes, 38 cells ship.

    Eighteen kana fill the 36-cell line exactly and the next line opens on the
    small tsu of ``あっ``. 行頭禁則 cannot leave it there, so ``apply_kinsoku``
    pulls it back onto a line that was already full and the delivered line is
    38 cells wide. ``_fits_budget`` never runs kinsoku -- it scores its own
    greedy repack, which puts the tsu on line two -- so it admits the span.
    This is the whole reason PD-TEXT inspects what ships.
    """
    module = canon()
    text = "あ" * 17 + "あっ" + "い" * 17
    prof = ja_profile(max_lines=2)
    final = module.canonical_text(wd(*text), fallback_text="", lang="ja", profile=prof)
    assert final.lines == ("あ" * 18 + "っ", "い" * 17)
    assert final.cell_widths == (38, 34)
    assert module.canonical_legal(final, prof) is False
    assert _fits_budget(text, prof.max_line_length, prof.max_lines, "ja") is True


def test_line_budget_doubles_the_cjk_preset():
    module = canon()
    assert module.line_budget(profile()) == 42
    assert module.line_budget(ja_profile()) == 36


def test_band_scan_bound_is_a_monotone_true_lower_bound_only():
    """It may only ever prove ILLEGALITY; a legal span must never be broken off."""
    module = canon()
    small = profile(max_line_length=20, max_lines=2)
    assert module.band_scan_lower_bound_exceeded("a" * 81, small) is True
    # 40 cells fit two 20-cell lines exactly: the bound must stay quiet even though
    # the greedy packer's own cell arithmetic would not admit every such span.
    assert (
        module.band_scan_lower_bound_exceeded("a" * 20 + " " + "b" * 19, small) is False
    )


def test_band_scan_bound_discounts_a_removable_line_break_separator():
    """A normalized separator may be discarded at either of two CJK lines."""
    module = canon()
    prof = ja_profile(max_lines=2)
    source = "甲" * 18 + "。" + "乙" * 18

    final = module.canonical_text(
        wd(*source), fallback_text="", lang="ja", profile=prof
    )
    assert final.cell_widths == (36, 36)
    assert module.canonical_legal(final, prof) is True
    assert module.band_scan_lower_bound_exceeded(source, prof) is False

    # Once the removable separator is excluded, one further half-width cell is
    # genuinely above the two-line capacity.
    assert module.band_scan_lower_bound_exceeded(source + "a", prof) is True


# ------------------------------------------------------------- work counter


def test_work_counter_charges_raw_visits_and_caches_projections():
    module = canon()
    work = module.CanonicalWork()
    first = module.canonical_text(
        wd("Hello,", "world!!"),
        fallback_text="",
        lang="en",
        profile=profile(),
        work=work,
    )
    charged = work.canonical_chars
    assert charged > 0
    again = work.cached((0, 2), lambda: first)
    assert again is first
    assert work.canonical_chars == charged  # a cache hit charges nothing


def test_punctuation_flood_charges_raw_visits_not_delivered_chars():
    """``"a" + "!"*4000`` ships one character; the work bound is in RAW chars.

    Charging the projection's *output* would let a band scan walk an arbitrarily
    long span for free, which is exactly the shape the N14 bound exists to
    forbid. The schedule is frozen: reconstruction (4001) + strip (4001) +
    one stutter scan over the stripped text (1) + wrap (1).
    """
    module = canon()
    work = module.CanonicalWork()
    flood = "a" + "!" * 4000
    final = module.canonical_text(
        wd(flood), fallback_text="", lang="en", profile=profile(), work=work
    )
    assert (final.text, final.reading_chars) == ("a", 1)
    assert work.canonical_chars == 4001 + 4001 + 1 + 1


def test_pass_factor_is_the_exact_worst_case():
    module = canon()
    assert (module.STUTTER_MAX_SCANS, module.CANONICAL_PASS_FACTOR) == (4, 6)


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_projection_never_mutates_word_data(lang):
    module = canon()
    source = wd("a", "b")
    before = [dict(entry) for entry in source]
    module.canonical_text(
        source, fallback_text="", lang=lang, profile=profile(language=lang)
    )
    assert source == before
