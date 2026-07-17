from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from voxweave.core import smart_split as smart_split_module
from voxweave.core.layout import _fits_budget
from voxweave.core.smart_split import (
    SplitContext,
    SplitThresholds,
    smart_split_segments,
)
from voxweave.semantic_breaks import SemanticBreakEngine


TH = {
    "clause_ms": 400,
    "vad_skip_ms": 1000,
    "offline_ms": 700,
    "min_cue_s": 0.5,
    "max_cue_s": 7.0,
    "glue_gap_s": 0.3,
    "cps": 9.0,
    "lag_out_s": 0.25,
}


def _segment(text: str, lang: str, *, step: float = 0.25):
    surfaces = text.split() if lang == "en" else [c for c in text if not c.isspace()]
    words = [
        {"word": surface, "start": i * step, "end": i * step + step * 0.75}
        for i, surface in enumerate(surfaces)
    ]
    return {"text": text, "words": words}


class ChoosingEngine:
    def __init__(self, choose):
        self.choose_breaks = choose
        self.calls = []

    def choose(self, tasks, **kwargs):
        self.calls.append((tasks, kwargs))
        return [
            SimpleNamespace(
                source="model", break_indices=tuple(self.choose_breaks(task))
            )
            for task in tasks
        ]


def _engine(value) -> SemanticBreakEngine:
    return cast(SemanticBreakEngine, value)


def _cut_after(task, suffix: str) -> int:
    for index in task.candidate_indices:
        if "".join(task.atoms[:index]).rstrip().endswith(suffix):
            return index
    raise AssertionError(f"no candidate after {suffix!r}: {task.candidate_indices!r}")


def _first_complete_path(task):
    outgoing = {}
    for start, end in task.allowed_edges:
        outgoing.setdefault(start, []).append(end)

    def visit(node, path):
        if node == len(task.atoms):
            return path
        for nxt in outgoing.get(node, ()):
            found = visit(nxt, path + ([nxt] if nxt < len(task.atoms) else []))
            if found is not None:
                return found
        return None

    result = visit(0, [])
    assert result is not None
    return tuple(result)


def test_semantic_mode_bypasses_mechanical_comma_pre_split():
    text = "我们先介绍产品，随后演示完整流程"
    segment = _segment(text, "zh", step=0.3)
    thresholds = {**TH, "glue_gap_s": 0.0}
    baseline = smart_split_segments([segment], "zh", thresholds=thresholds)
    assert len(baseline) == 2

    engine = ChoosingEngine(lambda task: ())
    semantic = smart_split_segments(
        [segment], "zh", thresholds=thresholds, semantic_engine=_engine(engine)
    )

    assert len(semantic) == 1
    assert semantic[0]["text"] == "我们先介绍产品 随后演示完整流程"
    task = engine.calls[0][0][0]
    assert (0, len(task.atoms)) in task.allowed_edges


def test_chinese_model_can_choose_complete_phrases_instead_of_greedy_cut():
    text = "功能现已支持调用MCP连接器使其能够按需构建并执行操作"
    segment = _segment(text, "zh", step=0.16)
    engine = ChoosingEngine(lambda task: (_cut_after(task, "连接器"),))

    cues = smart_split_segments(
        [segment], "zh", thresholds=TH, semantic_engine=_engine(engine)
    )

    assert [cue["text"] for cue in cues] == [
        "功能现已支持调用MCP连接器",
        "使其能够按需构建并执行操作",
    ]
    assert [unit for cue in cues for unit in cue["word_data"]] == segment["words"]
    assert all(_fits_budget(cue["text"], 18, 1, "zh") for cue in cues)
    assert cues[0]["start"] == segment["words"][0]["start"]
    assert cues[-1]["end"] >= segment["words"][-1]["end"]
    for cue in cues:
        unit_start = cue["word_data"][0].get("start")
        unit_end = cue["word_data"][-1].get("end")
        assert unit_start is not None and cue["start"] == unit_start
        assert unit_end is not None and cue["end"] >= unit_end


def test_multilingual_tasks_use_host_legal_paths_for_english_and_japanese():
    cases = [
        (
            "en",
            "OpenAI released a security model for developers around the world",
            18,
            1,
        ),
        ("ja", "今日は新しいモデルを世界中の開発者へ公開しました", 8, 1),
    ]
    for lang, text, max_length, max_lines in cases:
        segment = _segment(text, lang, step=0.5)
        engine = ChoosingEngine(_first_complete_path)
        cues = smart_split_segments(
            [segment],
            lang,
            max_line_length=max_length,
            max_lines=max_lines,
            thresholds={**TH, "cps": 17.0 if lang == "en" else 7.0},
            semantic_engine=_engine(engine),
        )
        task = engine.calls[0][0][0]
        assert task.language == lang
        assert "".join(
            cue["text"].replace(" ", "").replace("\n", "") for cue in cues
        ) == text.replace(" ", "")
        assert all(
            _fits_budget(cue["text"], max_length, max_lines, lang) for cue in cues
        )


def test_long_pause_is_required_and_cannot_be_crossed_by_model():
    text = "we build reliable tools"
    words = [
        {"word": "we", "start": 0.0, "end": 0.2},
        {"word": "build", "start": 0.3, "end": 0.5},
        {"word": "reliable", "start": 2.0, "end": 2.3},
        {"word": "tools", "start": 2.4, "end": 2.7},
    ]
    segment = {"text": text, "words": words}
    baseline = smart_split_segments([segment], "en", thresholds={**TH, "cps": 17.0})
    engine = ChoosingEngine(lambda task: ())

    result = smart_split_segments(
        [segment],
        "en",
        thresholds={**TH, "cps": 17.0},
        semantic_engine=_engine(engine),
    )

    tasks = engine.calls[0][0]
    assert len(tasks) == 2
    assert tasks[0].atoms == ("we", "build")
    assert tasks[1].atoms == ("reliable", "tools")
    assert len(result) == 2
    assert result[0]["end"] < result[1]["start"]
    assert " ".join(cue["text"] for cue in result) == text
    assert [(cue["text"], cue["start"], cue["end"]) for cue in baseline] == [
        (cue["text"], cue["start"], cue["end"]) for cue in result
    ]


def test_duration_and_width_illegal_model_path_returns_exact_baseline():
    text = "one exceptionallylongword two three"
    words = [
        {"word": "one", "start": 0.0, "end": 2.4},
        {"word": "exceptionallylongword", "start": 2.5, "end": 4.9},
        {"word": "two", "start": 5.0, "end": 7.4},
        {"word": "three", "start": 7.5, "end": 9.9},
    ]
    segment = {"text": text, "words": words}
    thresholds = {**TH, "max_cue_s": 3.0, "cps": 17.0}
    baseline = smart_split_segments(
        [segment], "en", max_line_length=22, max_lines=1, thresholds=thresholds
    )
    engine = ChoosingEngine(lambda task: ())

    result = smart_split_segments(
        [segment],
        "en",
        max_line_length=22,
        max_lines=1,
        thresholds=thresholds,
        semantic_engine=_engine(engine),
    )

    task = engine.calls[0][0][0]
    assert (0, len(task.atoms)) not in task.allowed_edges
    assert result == baseline


def test_minimum_duration_cps_and_english_wps_are_soft_scored_not_graph_killers():
    cases = [
        (
            "a compact subtitle phrase",
            [
                (0.0, 0.1),
                (0.15, 0.8),
                (0.9, 1.6),
                (1.7, 2.4),
            ],
            {**TH, "min_cue_s": 0.5, "cps": 100.0},
            1,
        ),
        (
            "abcdefghijklmnop more words later",
            [(0.0, 0.4), (0.5, 1.5), (1.6, 2.5), (2.6, 3.5)],
            {**TH, "min_cue_s": 0.0, "cps": 17.0},
            1,
        ),
        (
            "a b longer final phrase",
            [
                (0.0, 0.15),
                (0.2, 0.35),
                (0.4, 1.4),
                (1.5, 2.5),
                (2.6, 3.5),
            ],
            {**TH, "min_cue_s": 0.0, "cps": 100.0},
            2,
        ),
    ]
    for text, spans, thresholds, illegal_cut in cases:
        words = [
            {"word": word, "start": start, "end": end}
            for word, (start, end) in zip(text.split(), spans)
        ]
        segment = {"text": text, "words": words}
        engine = ChoosingEngine(lambda _task, cut=illegal_cut: (cut,))
        smart_split_segments(
            [segment],
            "en",
            thresholds=thresholds,
            desired_wps=4.0,
            semantic_engine=_engine(engine),
        )
        task = engine.calls[0][0][0]
        quality = {(start, end): score for start, end, score in task.edge_quality}
        assert (0, illegal_cut) in task.allowed_edges
        assert quality[(0, illegal_cut)] < max(quality.values())


def test_materially_worse_soft_timing_path_returns_exact_baseline():
    text = "a compact subtitle phrase"
    spans = [(0.0, 0.1), (0.15, 0.8), (0.9, 1.6), (1.7, 2.4)]
    words = [
        {"word": word, "start": start, "end": end}
        for word, (start, end) in zip(text.split(), spans)
    ]
    segment = {"text": text, "words": words}
    thresholds = {**TH, "min_cue_s": 0.5, "cps": 100.0}
    baseline = smart_split_segments([segment], "en", thresholds=thresholds)
    engine = ChoosingEngine(lambda _task: (1,))

    result = smart_split_segments(
        [segment],
        "en",
        thresholds=thresholds,
        desired_wps=4.0,
        semantic_engine=_engine(engine),
    )

    assert result == baseline


def test_fast_single_segment_is_windowed_into_nonempty_model_tasks():
    text = (
        "大家好欢迎收看今天的AI日报。"
        "功能现已支持调用MCP连接器使其能够按需构建并执行操作。"
        "目前仅面向Pro Max Team和Enterprise用户开放使用。"
        "该模型可自动模拟各类网络攻击用于检测大模型安全漏洞。"
    )
    segment = _segment(text, "zh", step=0.055)
    engine = ChoosingEngine(lambda task: task.fallback_indices)

    cues = smart_split_segments(
        [segment],
        "zh",
        thresholds={**TH, "cps": 9.0},
        semantic_engine=_engine(engine),
    )

    tasks = engine.calls[0][0]
    assert len(tasks) >= 4
    assert all(
        task.atoms and task.allowed_edges and task.edge_quality for task in tasks
    )
    assert any(
        min(score for _start, _end, score in task.edge_quality) < 100 for task in tasks
    )
    assert cues
    assert [unit for cue in cues for unit in cue["word_data"]] == segment["words"]


def test_unpunctuated_long_input_has_bounded_window_and_edge_work(monkeypatch):
    atom_count = 2000
    text = " ".join(f"w{i}" for i in range(atom_count))
    words = [
        {"word": f"w{i}", "start": i * 0.2, "end": i * 0.2 + 0.15}
        for i in range(atom_count)
    ]
    calls = 0
    original = smart_split_module._fits_budget

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(smart_split_module, "_fits_budget", counted)
    thresholds = SplitThresholds(
        min_cue_s=0.5,
        max_cue_s=7.0,
        cps=17.0,
    )
    ctx = SplitContext(
        lang="en",
        max_line_length=42,
        max_lines=2,
        th=thresholds,
        do_new=True,
    )

    plans = smart_split_module._prepare_semantic_plans(
        {"text": text, "words": words}, ctx=ctx, desired_wps=4.0
    )

    assert plans is not None and len(plans) > 1
    assert (
        max(len(plan.atoms) for plan in plans)
        <= smart_split_module.SEMANTIC_WINDOW_MAX_ATOMS
    )
    assert calls < atom_count * 25


def test_engine_error_and_task_local_fallback_are_transactional():
    segments = [
        _segment("we build reliable tools for everyone", "en", step=0.4),
        _segment("teams use them around the world today", "en", step=0.4),
    ]
    thresholds = {**TH, "cps": 17.0}
    baseline = smart_split_segments(segments, "en", thresholds=thresholds)

    class ErrorEngine:
        def choose(self, tasks, **kwargs):
            raise RuntimeError("model unavailable")

    assert (
        smart_split_segments(
            segments,
            "en",
            thresholds=thresholds,
            semantic_engine=_engine(ErrorEngine()),
        )
        == baseline
    )

    class PartialFallbackEngine:
        def choose(self, tasks, **kwargs):
            return [
                SimpleNamespace(
                    source="model", break_indices=_first_complete_path(tasks[0])
                ),
                SimpleNamespace(
                    source="fallback", break_indices=tasks[1].fallback_indices
                ),
            ]

    assert (
        smart_split_segments(
            segments,
            "en",
            thresholds=thresholds,
            semantic_engine=_engine(PartialFallbackEngine()),
        )
        == baseline
    )


def test_semantic_model_override_is_forwarded_without_global_state():
    segment = _segment("we build tools for people everywhere", "en", step=0.4)
    engine = ChoosingEngine(_first_complete_path)
    smart_split_segments(
        [segment],
        "en",
        thresholds={**TH, "cps": 17.0},
        semantic_engine=_engine(engine),
        semantic_model="local/Qwen3.5-fp8",
    )
    assert engine.calls[0][1]["default_model"] == "local/Qwen3.5-fp8"
