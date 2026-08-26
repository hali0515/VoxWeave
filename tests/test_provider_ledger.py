# tests/test_provider_ledger.py
"""P3 PIN 2 -- provider identity snapshot + degradation ledger.

Segmentation has seven language-provider surfaces (BudouX, jieba ``cut``, jieba
``posseg``, fugashi/UniDic, pysbd, per-char, whitespace) and every one of them
fails open with no diagnostic today. ``voxweave.core.providers`` adds two
things and changes no behaviour:

* ``provider_snapshot(iso)`` -- the static identity of the providers *this*
  language uses (never loading one it does not: no fugashi for an en document),
* ``note_degraded`` / ``degradation_capture`` -- a ledger of the fallbacks that
  actually fired during one segmentation, plus a once-per-process ``log.warning``
  on the ``voxweave`` logger (never ``warnings.warn`` -- ``ui.install_logging``
  filters swallow those).

Instrumentation is observation only: every wrapped call must return exactly what
it returned before.
"""

import logging
import sys

import pytest

from voxweave.core import breakpoints, kinsoku, providers
from voxweave.core.breakpoints import phrase_atoms
from voxweave.core.kinsoku import ja_pos_end_penalties, zh_pos_boundary_penalties
from voxweave.core.smart_split import _segment_sentences

SLOTS = ("sentences", "atoms", "pos")


@pytest.fixture(autouse=True)
def _reset_warned():
    """The once-per-process warning latch is module state; tests own it."""
    providers._WARNED.clear()
    yield
    providers._WARNED.clear()


@pytest.fixture(autouse=True)
def _reset_posseg_probe():
    """The zh/yue POS probe is lru_cached, so a stubbed import must not leak."""
    providers._posseg_available.cache_clear()
    yield
    providers._posseg_available.cache_clear()


def _reasons(ledger):
    return {(entry["slot"], entry["reason"]): entry["count"] for entry in ledger}


class _Token:
    def __init__(self, word: str, flag: str):
        self.word = word
        self.flag = flag


class _OneTokenPseg:
    """posseg stand-in that tokenizes the whole string as one word, so every
    interior candidate offset disagrees with the atom stream."""

    def cut(self, text, HMM=True):  # noqa: N803 - mirrors jieba's keyword
        yield _Token(text, "n")


class _RaisingPseg:
    def cut(self, text, HMM=True):  # noqa: N803 - mirrors jieba's keyword
        raise RuntimeError("corrupt dict")


# --- snapshot: shape ---------------------------------------------------------


@pytest.mark.parametrize("iso", ["ja", "zh", "yue", "en"])
def test_snapshot_has_exactly_the_three_slots(iso):
    snap = providers.provider_snapshot(iso)
    assert set(snap) == set(SLOTS)
    for slot in SLOTS:
        assert isinstance(snap[slot], dict), slot
        assert "provider" in snap[slot], slot
        assert "version" in snap[slot], slot


@pytest.mark.parametrize("iso", ["ja", "zh", "yue", "en"])
def test_snapshot_is_deterministic(iso):
    assert providers.provider_snapshot(iso) == providers.provider_snapshot(iso)


@pytest.mark.parametrize("iso", ["ja", "zh", "yue", "en"])
def test_snapshot_is_json_scalar_only(iso):
    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str)
                walk(v)
        else:
            assert value is None or isinstance(value, (str, int, float, bool))

    walk(providers.provider_snapshot(iso))


# --- snapshot: per-language identity -----------------------------------------


def test_ja_reports_budoux_atoms_and_fugashi_pos():
    snap = providers.provider_snapshot("ja")
    assert snap["atoms"]["provider"] == "budoux"
    assert snap["pos"]["provider"] == "fugashi-unidic"
    assert snap["pos"]["ja_pos_enabled"] is True
    tagger = kinsoku._load_ja_tagger()
    feature_cls = type(next(iter(tagger("の"))).feature).__name__
    assert feature_cls in snap["pos"]["dict"]


def test_ja_atoms_degrade_to_per_char_without_budoux(monkeypatch):
    monkeypatch.setattr(breakpoints, "_load_parser", lambda lang: None)
    snap = providers.provider_snapshot("ja")
    assert snap["atoms"]["provider"] == "per_char"


def test_ja_pos_reports_the_latched_tagger_state(monkeypatch):
    """VOXWEAVE_JA_POS is latched inside an lru_cache, so the snapshot reports
    what the cached loader actually returned -- not a fresh env read."""
    monkeypatch.setattr(kinsoku, "_load_ja_tagger", lambda: None)
    snap = providers.provider_snapshot("ja")
    assert snap["pos"]["provider"] is None
    assert snap["pos"]["ja_pos_enabled"] is False
    assert snap["pos"]["dict"] is None


def test_zh_prefers_jieba_atoms():
    snap = providers.provider_snapshot("zh")
    assert snap["atoms"]["provider"] == "jieba"
    assert snap["pos"]["provider"] == "jieba-posseg"
    assert snap["pos"]["decode_mode"] == "HMM=False"


def test_zh_atoms_fall_back_to_budoux_then_per_char(monkeypatch):
    """Mirrors the exact branch order of ``breakpoints.phrase_atoms``."""
    monkeypatch.setattr(breakpoints, "_load_jieba", lambda: None)
    assert providers.provider_snapshot("zh")["atoms"]["provider"] == "budoux"

    monkeypatch.setattr(breakpoints, "_load_parser", lambda lang: None)
    assert providers.provider_snapshot("zh")["atoms"]["provider"] == "per_char"


def test_yue_is_a_phantom_provider():
    """yue is declared no-space but has neither jieba nor a BudouX model, while
    still being scored by Mandarin posseg."""
    snap = providers.provider_snapshot("yue")
    assert snap["atoms"]["provider"] == "per_char"
    assert snap["pos"]["provider"] == "jieba-posseg"
    assert snap["pos"]["decode_mode"] == "HMM=False"


def test_zh_pos_slot_probes_the_import_the_scorer_actually_uses(monkeypatch):
    """The POS slot must mirror ``zh_pos_boundary_penalties``' own import.

    The scorer reaches its tagger through ``quiet_import_jieba(posseg=True)``,
    which is a different import from the atoms-side ``_load_jieba()``
    (``posseg=False``). A half-installed jieba answers them differently, and
    reporting the atoms-side one would let the manifest claim ``jieba-posseg``
    ran while its own ``degraded`` list says ``posseg-import-failed``.
    """
    real = breakpoints.quiet_import_jieba

    def only_plain_jieba(*, posseg: bool = False):
        return None if posseg else real()

    monkeypatch.setattr(breakpoints, "quiet_import_jieba", only_plain_jieba)
    breakpoints._load_jieba.cache_clear()
    providers._posseg_available.cache_clear()
    try:
        # the scorer really does fail open and record the degradation ...
        with providers.degradation_capture() as ledger:
            assert zh_pos_boundary_penalties(list("数据中心"), [1, 2], "zh") == {}
        assert _reasons(ledger) == {("pos", "posseg-import-failed"): 1}
        # ... so the snapshot must not claim a provider ran
        for iso in ("zh", "yue"):
            assert providers.provider_snapshot(iso)["pos"]["provider"] is None, iso
            assert providers.provider_snapshot(iso)["pos"]["version"] is None, iso
    finally:
        breakpoints._load_jieba.cache_clear()


def test_zh_pos_probe_is_cached(monkeypatch):
    """One import per process, not one per document."""
    calls = []
    real = breakpoints.quiet_import_jieba

    def counting(*, posseg: bool = False):
        calls.append(posseg)
        return real(posseg=posseg)

    monkeypatch.setattr(breakpoints, "quiet_import_jieba", counting)
    providers._posseg_available.cache_clear()
    providers.provider_snapshot("zh")
    providers.provider_snapshot("zh")
    providers.provider_snapshot("yue")
    assert calls.count(True) == 1


def test_en_and_ja_snapshots_never_probe_posseg(monkeypatch):
    """Laziness survives the probe: only zh/yue documents pay for it."""

    def _no(*args, **kwargs):
        raise AssertionError("probed posseg for a language that never scores with it")

    monkeypatch.setattr(providers, "_posseg_available", _no)
    providers.provider_snapshot("en")
    providers.provider_snapshot("ja")


def test_en_uses_whitespace_atoms_and_has_no_pos_provider():
    snap = providers.provider_snapshot("en")
    assert snap["atoms"]["provider"] == "whitespace"
    assert snap["pos"]["provider"] is None


def test_sentences_provider_reflects_the_pysbd_language_probe():
    assert providers.provider_snapshot("en")["sentences"]["provider"] == "pysbd"
    assert providers.provider_snapshot("ja")["sentences"]["provider"] == "pysbd"
    # pysbd has no yue model: Segmenter(language="yue") raises -> regex fallback.
    assert providers.provider_snapshot("yue")["sentences"]["provider"] == "regex"


def test_versions_come_from_importlib_metadata():
    import importlib.metadata as im

    snap = providers.provider_snapshot("zh")
    assert snap["sentences"]["version"] == im.version("pysbd")
    assert snap["atoms"]["version"] == im.version("jieba")
    assert providers.provider_snapshot("ja")["atoms"]["version"] == im.version("budoux")


# --- snapshot: touches only the providers the language uses ------------------


def _boom(*args, **kwargs):
    raise AssertionError("loaded a provider this language does not use")


def test_en_snapshot_loads_no_cjk_provider(monkeypatch):
    monkeypatch.setattr(kinsoku, "_load_ja_tagger", _boom)
    monkeypatch.setattr(breakpoints, "_load_jieba", _boom)
    monkeypatch.setattr(breakpoints, "_load_parser", _boom)
    providers.provider_snapshot("en")


def test_zh_snapshot_never_loads_fugashi(monkeypatch):
    monkeypatch.setattr(kinsoku, "_load_ja_tagger", _boom)
    providers.provider_snapshot("zh")


def test_ja_snapshot_never_loads_jieba(monkeypatch):
    monkeypatch.setattr(breakpoints, "_load_jieba", _boom)
    providers.provider_snapshot("ja")


# --- ledger mechanics --------------------------------------------------------


def test_note_degraded_without_capture_is_a_noop():
    assert providers.note_degraded("atoms", "jieba-missing:per-char") is None
    with providers.degradation_capture() as ledger:
        pass
    assert ledger == []


def test_capture_aggregates_repeat_events_into_counts():
    with providers.degradation_capture() as ledger:
        providers.note_degraded("pos", "pos-offset-disagreement")
        providers.note_degraded("pos", "pos-offset-disagreement")
        providers.note_degraded("pos", "pos-offset-disagreement")
        providers.note_degraded("atoms", "no-provider:per-char")
    assert _reasons(ledger) == {
        ("pos", "pos-offset-disagreement"): 3,
        ("atoms", "no-provider:per-char"): 1,
    }
    assert all(set(entry) == {"slot", "reason", "count"} for entry in ledger)


def test_capture_restores_the_previous_ledger_on_exit():
    with providers.degradation_capture() as outer:
        providers.note_degraded("atoms", "outer")
        with providers.degradation_capture() as inner:
            providers.note_degraded("atoms", "inner")
        assert _reasons(inner) == {("atoms", "inner"): 1}
        providers.note_degraded("atoms", "outer")
    assert _reasons(outer) == {("atoms", "outer"): 2}
    # capture closed: the ledger is inert again
    providers.note_degraded("atoms", "after")
    assert _reasons(outer) == {("atoms", "outer"): 2}


def test_first_occurrence_warns_once_per_slot_reason_on_the_voxweave_logger(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        with providers.degradation_capture():
            providers.note_degraded("atoms", "jieba-missing:per-char")
            providers.note_degraded("atoms", "jieba-missing:per-char")
            providers.note_degraded("atoms", "budoux-missing:per-char")
    records = [r for r in caplog.records if r.name == "voxweave"]
    assert len(records) == 2
    messages = [r.getMessage() for r in records]
    assert any("jieba-missing:per-char" in m for m in messages)
    assert any("budoux-missing:per-char" in m for m in messages)
    assert all("atoms" in m for m in messages)


def test_warning_fires_even_without_an_active_capture(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        providers.note_degraded("sentences", "pysbd-missing:regex")
    assert [r for r in caplog.records if r.name == "voxweave"]


# --- instrumentation touchpoints --------------------------------------------


def test_yue_per_char_atoms_are_recorded_as_degraded():
    with providers.degradation_capture() as ledger:
        atoms = phrase_atoms("我哋今日去食飯", "yue")
    assert atoms == ["我", "哋", "今", "日", "去", "食", "飯"]
    assert _reasons(ledger) == {("atoms", "no-provider:per-char"): 1}


def test_spaced_language_whitespace_split_is_not_degraded():
    with providers.degradation_capture() as ledger:
        assert phrase_atoms("hello there world", "en") == ["hello", "there", "world"]
    assert ledger == []


def test_healthy_cjk_providers_record_nothing():
    with providers.degradation_capture() as ledger:
        phrase_atoms("数据中心业务", "zh")
        phrase_atoms("これはテストです", "ja")
    assert ledger == []


def test_zh_jieba_absent_falls_back_to_budoux_and_is_recorded(monkeypatch):
    monkeypatch.setattr(breakpoints, "_load_jieba", lambda: None)
    outside = phrase_atoms("数据中心业务", "zh")  # no capture active
    with providers.degradation_capture() as ledger:
        inside = phrase_atoms("数据中心业务", "zh")
    # observation only: the ledger must not change what the call returns
    assert inside == outside
    assert "".join(inside) == "数据中心业务"
    assert _reasons(ledger) == {("atoms", "jieba-missing:budoux-fallback"): 1}


def test_zh_with_no_provider_at_all_is_recorded_as_per_char(monkeypatch):
    monkeypatch.setattr(breakpoints, "_load_jieba", lambda: None)
    monkeypatch.setattr(breakpoints, "_load_parser", lambda lang: None)
    with providers.degradation_capture() as ledger:
        atoms = phrase_atoms("数据中心", "zh")
    assert atoms == ["数", "据", "中", "心"]
    assert _reasons(ledger) == {("atoms", "jieba-missing:per-char"): 1}


def test_ja_budoux_absent_is_recorded_as_per_char(monkeypatch):
    monkeypatch.setattr(breakpoints, "_load_parser", lambda lang: None)
    with providers.degradation_capture() as ledger:
        atoms = phrase_atoms("これはテスト", "ja")
    assert atoms == list("これはテスト")
    assert _reasons(ledger) == {("atoms", "budoux-missing:per-char"): 1}


def test_posseg_import_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(breakpoints, "quiet_import_jieba", lambda **kw: None)
    with providers.degradation_capture() as ledger:
        assert zh_pos_boundary_penalties(list("数据中心"), [1, 2], "zh") == {}
    assert _reasons(ledger) == {("pos", "posseg-import-failed"): 1}


def test_posseg_runtime_exception_is_recorded(monkeypatch):
    monkeypatch.setattr(breakpoints, "quiet_import_jieba", lambda **kw: _RaisingPseg())
    with providers.degradation_capture() as ledger:
        assert zh_pos_boundary_penalties(list("数据中心"), [1, 2], "zh") == {}
    assert _reasons(ledger) == {("pos", "posseg-exception"): 1}


def test_pos_offset_disagreement_aggregates_per_cue_stream(monkeypatch):
    monkeypatch.setattr(breakpoints, "quiet_import_jieba", lambda **kw: _OneTokenPseg())
    with providers.degradation_capture() as ledger:
        assert zh_pos_boundary_penalties(list("数据中心"), [1, 2, 3], "zh") == {}
    assert _reasons(ledger) == {("pos", "pos-offset-disagreement"): 3}


def test_fugashi_unavailable_is_recorded(monkeypatch):
    monkeypatch.setattr(kinsoku, "_load_ja_tagger", lambda: None)
    with providers.degradation_capture() as ledger:
        assert ja_pos_end_penalties("これはテスト") is None
    assert _reasons(ledger) == {("pos", "fugashi-unavailable"): 1}


def test_pysbd_missing_falls_back_to_regex_and_is_recorded(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysbd", None)
    with providers.degradation_capture() as ledger:
        sentences = _segment_sentences("One. Two.", "en")
    assert sentences == ["One.", "Two."]
    assert _reasons(ledger) == {("sentences", "pysbd-missing:regex"): 1}


def test_pysbd_language_unsupported_falls_back_to_regex_and_is_recorded():
    with providers.degradation_capture() as ledger:
        sentences = _segment_sentences("我哋今日。好嘢。", "yue")
    assert sentences == ["我哋今日。", "好嘢。"]
    assert _reasons(ledger) == {("sentences", "pysbd-language-unsupported:regex"): 1}


def test_supported_pysbd_language_records_nothing():
    with providers.degradation_capture() as ledger:
        _segment_sentences("One. Two.", "en")
    assert ledger == []


# --------------------------------- a measurement lane may not take the latch


def test_a_quiet_capture_records_the_event_but_never_claims_the_warning(caplog):
    """Bug pin: a nested measurement could steal production's one log line.

    ``_WARNED`` is process-global, so whichever context reaches a ``(slot,
    reason)`` pair first emits the single warning. A shadow lane re-tokenizes the
    same document inside its own nested capture, so it could win that race and
    leave the shipping run's degradation silent. ``quiet`` suppresses the log for
    the measurement only -- the ledger entry is unaffected.
    """
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        with providers.degradation_capture(quiet=True) as measured:
            providers.note_degraded("atoms", "synthetic:quiet")
        assert measured == [{"slot": "atoms", "reason": "synthetic:quiet", "count": 1}]
        assert not caplog.records

        with providers.degradation_capture() as shipped:
            providers.note_degraded("atoms", "synthetic:quiet")
    assert shipped == [{"slot": "atoms", "reason": "synthetic:quiet", "count": 1}]
    assert [r.getMessage() for r in caplog.records] == [
        "segmentation atoms provider degraded: synthetic:quiet"
    ]


def test_quiet_is_inherited_by_a_capture_nested_inside_a_quiet_one(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        with providers.degradation_capture(quiet=True):
            with providers.degradation_capture() as inner:
                providers.note_degraded("pos", "synthetic:nested")
    assert inner == [{"slot": "pos", "reason": "synthetic:nested", "count": 1}]
    assert not caplog.records
