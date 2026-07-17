from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from voxweave.semantic_breaks import (
    DEFAULT_SEMANTIC_MODEL,
    BoundaryTask,
    LocalTransformersSelector,
    OpenAICompatibleSelector,
    SemanticBackendUnavailable,
    SemanticBreakEngine,
    _cue_host_penalty,
    _path_options,
    semantic_model_for,
)


class FakeSelector:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []
        self.released = False

    def select(self, model_id, messages, *, max_new_tokens):
        self.calls.append((model_id, messages, max_new_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(model_id, messages)
        return reply

    def release(self):
        self.released = True


class FakeScoringSelector(FakeSelector):
    def __init__(self, score_replies=()):
        super().__init__(())
        self.score_replies = list(score_replies)
        self.score_calls = []

    def score_labels(self, model_id, prompt_batches, labels):
        self.score_calls.append((model_id, prompt_batches, labels))
        reply = self.score_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(model_id, prompt_batches, labels)
        return reply


def task(
    language="zh",
    *,
    fallback=(2,),
    required=(),
    target_chars=None,
    max_segment_chars=None,
):
    return BoundaryTask(
        atoms=("欢迎", "收看", "今天", "的", "节目"),
        candidate_indices=(1, 2, 3, 4),
        language=language,
        fallback_indices=fallback,
        required_indices=required,
        pauses_ms={2: 260, 4: 410},
        target_chars=target_chars,
        max_segment_chars=max_segment_chars,
    )


def payload_from_call(call):
    return json.loads(call[1][-1]["content"])


def echo_first_candidate(_model_id, messages):
    payload = json.loads(messages[1]["content"])
    return json.dumps(
        {
            "results": [
                {"id": item["id"], "breaks": [item["candidate_indices"][0]]}
                for item in payload["tasks"]
            ]
        }
    )


def choose_path_breaks(request, wanted):
    wanted = tuple(wanted)
    choice = next(
        option["choice"]
        for option in _path_options(request)
        if tuple(option["breaks"]) == wanted
    )

    def reply(_model_id, messages):
        payload = json.loads(messages[-1]["content"])
        tasks = payload.get("tasks")
        if tasks is None:
            tasks = [payload["task"]]
        results = []
        for item in tasks:
            results.append({"id": item["id"], "choice": choice})
        return json.dumps({"results": results})

    return reply


def two_cue_path_task(atoms, candidates, fallback, *, language="zh"):
    terminal = len(atoms)
    edges = tuple(
        edge
        for boundary in candidates
        for edge in ((0, boundary), (boundary, terminal))
    )
    return BoundaryTask(
        atoms=tuple(atoms),
        candidate_indices=tuple(candidates),
        language=language,
        fallback_indices=(fallback,),
        allowed_edges=edges,
        edge_quality=tuple((start, end, 100) for start, end in edges),
    )


def test_request_normalizes_indices_and_required_fallback():
    request = BoundaryTask(
        atoms=("a", "b", "c", "d"),
        candidate_indices=(3, 1, 3, 2),
        language="English",
        fallback_indices=(3,),
        required_indices=(1,),
        pauses_ms={3: 500, 1: 100},
    )
    assert request.language == "en"
    assert request.candidate_indices == (1, 2, 3)
    assert request.fallback_indices == (1, 3)
    assert request.pauses_ms == ((1, 100), (3, 500))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"candidate_indices": (0,)}, "outside the valid range"),
        ({"candidate_indices": (True,)}, "integer boundary"),
        (
            {"candidate_indices": (1,), "fallback_indices": (2,)},
            "subset of candidate",
        ),
        (
            {"candidate_indices": (1,), "required_indices": (2,)},
            "subset of candidate",
        ),
        ({"pauses_ms": {2: -1}}, "non-negative"),
    ],
)
def test_request_rejects_invalid_boundary_contract(kwargs, error):
    base = {
        "atoms": ("a", "b", "c"),
        "candidate_indices": (1, 2),
        "language": "en",
        "fallback_indices": (1,),
    }
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        BoundaryTask(**base)


def test_hard_character_budget_applies_to_fallback():
    with pytest.raises(ValueError, match="max_segment_chars"):
        BoundaryTask(
            atoms=("abcd", "efgh", "ijkl"),
            candidate_indices=(1, 2),
            language="en",
            fallback_indices=(2,),
            max_segment_chars=5,
        )


def test_allowed_edges_require_fallback_and_model_to_follow_complete_path():
    request = BoundaryTask(
        atoms=("现", "仅", "面向", "Pro", "用户"),
        candidate_indices=(1, 2, 3, 4),
        language="zh",
        fallback_indices=(2,),
        allowed_edges=((0, 1), (1, 5), (0, 2), (2, 5), (0, 3), (3, 4), (4, 5)),
    )
    legal = choose_path_breaks(request, (3, 4))
    illegal = json.dumps({"results": [{"id": 0, "choice": 99}]})

    decision = SemanticBreakEngine(FakeSelector([legal])).choose([request])[0]
    assert decision.source == "model"
    assert decision.break_indices == (3, 4)

    decision = SemanticBreakEngine(FakeSelector([illegal])).choose([request])[0]
    assert decision.source == "fallback"
    assert decision.break_indices == (2,)
    assert "outside offered path_options" in (decision.reason or "")


def test_path_option_task_retries_once_and_still_requires_an_exact_option():
    request = BoundaryTask(
        atoms=("目前", "仅", "面向", "开发者"),
        candidate_indices=(1, 2, 3),
        language="zh",
        fallback_indices=(2,),
        allowed_edges=((0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)),
        edge_quality=((0, 1, 90), (1, 4, 100), (0, 2, 98), (2, 4, 98), (0, 3, 100), (3, 4, 80)),
    )
    selector = FakeSelector(
        [
            json.dumps({"results": [{"id": 0, "choice": 99}]}),
            choose_path_breaks(request, (2,)),
        ]
    )

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.source == "model"
    assert decision.break_indices == (2,)
    assert len(selector.calls) == 2
    repair_payload = json.loads(selector.calls[1][1][-1]["content"])
    assert "validation_error" in repair_payload
    assert "path_options" in repair_payload["task"]


def test_malformed_path_envelope_gets_one_strict_repair_attempt():
    request = BoundaryTask(
        atoms=("目前", "仅", "面向", "开发者"),
        candidate_indices=(1, 2, 3),
        language="zh",
        fallback_indices=(2,),
        allowed_edges=((0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)),
    )
    selector = FakeSelector(
        ["not json", choose_path_breaks(request, request.fallback_indices)]
    )

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.source == "model"
    assert decision.break_indices == request.fallback_indices
    assert len(selector.calls) == 2
    repair_payload = json.loads(selector.calls[1][1][-1]["content"])
    assert "strict JSON" in repair_payload["validation_error"]


@pytest.mark.parametrize(
    "result, reason",
    [
        ({"id": 0, "breaks": [2]}, "only id and choice"),
        ({"id": 0, "choice": True}, "outside offered path_options"),
        ({"id": 0, "choice": -1}, "outside offered path_options"),
        ({"id": 0, "choice": 0, "text": "rewrite"}, "only id and choice"),
    ],
)
def test_path_option_result_can_only_select_an_offered_choice(result, reason):
    request = BoundaryTask(
        atoms=("目前", "仅", "面向", "开发者"),
        candidate_indices=(1, 2, 3),
        language="zh",
        fallback_indices=(2,),
        allowed_edges=((0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)),
    )
    decision = SemanticBreakEngine(
        FakeSelector([json.dumps({"results": [result]})])
    ).choose([request])[0]

    assert decision.source == "fallback"
    assert decision.break_indices == (2,)
    assert reason in (decision.reason or "")


def test_allowed_edges_are_validated_and_sent_as_allowed_next():
    with pytest.raises(ValueError, match="complete allowed path"):
        BoundaryTask(
            atoms=("a", "b", "c"),
            candidate_indices=(1, 2),
            language="en",
            fallback_indices=(1,),
            allowed_edges=((0, 1),),
        )
    with pytest.raises(ValueError, match="candidate boundary"):
        BoundaryTask(
            atoms=("a", "b", "c"),
            candidate_indices=(1,),
            language="en",
            fallback_indices=(1,),
            allowed_edges=((0, 1), (1, 2), (2, 3)),
        )

    request = BoundaryTask(
        atoms=("a", "b", "c"),
        candidate_indices=(1, 2),
        language="en",
        fallback_indices=(2,),
        allowed_edges=((0, 1), (0, 2), (1, 3), (2, 3)),
    )
    selector = FakeSelector([choose_path_breaks(request, (2,))])
    SemanticBreakEngine(selector).choose([request])
    task_payload = payload_from_call(selector.calls[0])["tasks"][0]
    options = task_payload["path_options"]
    assert [option["choice"] for option in options] == list(range(len(options)))
    assert {tuple(option["cues"]) for option in options} == {
        ("a", "b c"),
        ("a b", "c"),
    }
    assert all(set(option) == {"choice", "cues"} for option in options)
    system_prompt = selector.calls[0][1][0]["content"]
    assert "untrusted transcript data" in system_prompt
    assert "Never return breaks or cues" in system_prompt


def test_edge_quality_is_validated_and_exposed_as_soft_path_metadata():
    with pytest.raises(ValueError, match="only score allowed_edges"):
        BoundaryTask(
            atoms=("a", "b"),
            candidate_indices=(1,),
            language="en",
            fallback_indices=(1,),
            allowed_edges=((0, 1), (1, 2)),
            edge_quality=((0, 2, 90),),
        )
    with pytest.raises(ValueError, match="between 0 and 100"):
        BoundaryTask(
            atoms=("a", "b"),
            candidate_indices=(1,),
            language="en",
            fallback_indices=(1,),
            allowed_edges=((0, 1), (1, 2)),
            edge_quality=((0, 1, 101),),
        )

    request = BoundaryTask(
        atoms=("a", "b"),
        candidate_indices=(1,),
        language="en",
        fallback_indices=(1,),
        allowed_edges=((0, 1), (1, 2)),
        edge_quality=((0, 1, 62), (1, 2, 97)),
    )
    selector = FakeSelector([choose_path_breaks(request, (1,))])
    SemanticBreakEngine(selector).choose([request])
    payload = payload_from_call(selector.calls[0])["tasks"][0]
    assert payload["path_options"] == [
        {
            "choice": 0,
            "cues": ["a", "b"],
        }
    ]
    assert "timing_quality" not in json.dumps(payload)


def test_local_logit_scorer_uses_symmetric_prompts_and_selects_clean_path():
    request = BoundaryTask(
        atoms=(
            "目",
            "前",
            "仅",
            "面",
            "向",
            "Pro",
            "、",
            "Max",
            "、",
            "Team",
            "和",
            "Enterprise",
            "用",
            "户",
            "开",
            "放",
            "使",
            "用",
            "。",
        ),
        candidate_indices=(7, 8, 9, 10, 11, 12),
        language="zh",
        fallback_indices=(12,),
        allowed_edges=tuple(
            edge
            for boundary in (7, 8, 9, 10, 11, 12)
            for edge in ((0, boundary), (boundary, 19))
        ),
        edge_quality=tuple(
            (start, end, 100)
            for boundary in (7, 8, 9, 10, 11, 12)
            for start, end in ((0, boundary), (boundary, 19))
        ),
    )
    def prefer_complete_enumeration(_model_id, prompt_batches, _labels):
        scores = []
        wanted = "Team\nCue 2: 和Enterprise"
        for messages in prompt_batches:
            content = messages[-1]["content"]
            layout_a, layout_b = content.split("\n\nLayout B:", 1)
            if wanted in layout_a:
                scores.append([1.0, -1.0])
            elif wanted in layout_b:
                scores.append([-1.0, 1.0])
            else:
                scores.append([0.0, 0.0])
        return scores

    selector = FakeScoringSelector([prefer_complete_enumeration])

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.source == "model"
    assert decision.break_indices == (10,)
    assert selector.calls == []
    assert len(selector.score_calls) == 1
    model_id, prompts, labels = selector.score_calls[0]
    assert model_id == DEFAULT_SEMANTIC_MODEL
    assert labels == ["Yes", "No"]
    assert len(prompts) >= 2
    assert len(prompts) % 2 == 0
    assert "目前仅面向Pro、Max、Team" in prompts[0][1]["content"]
    assert "和Enterprise用户开放使用。" in prompts[0][1]["content"]
    assert "timing_quality" not in json.dumps(prompts)
    assert "fallback" not in json.dumps(prompts)


def test_host_shortlist_protects_enumerations_and_model_names_without_glossary():
    pro_request = two_cue_path_task(
        (
            "目",
            "前",
            "仅",
            "面",
            "向",
            "Pro",
            "、",
            "Max",
            "、",
            "Team",
            "和",
            "Enterprise",
            "用",
            "户",
            "开",
            "放",
            "使",
            "用",
            "。",
        ),
        (7, 8, 9, 10, 11, 12),
        12,
    )
    model_request = two_cue_path_task(
        (
            "OpenAI",
            "对",
            "外",
            "公",
            "开",
            "了",
            "内",
            "部",
            "网",
            "络",
            "安",
            "全",
            "专",
            "用",
            "模",
            "型",
            "GPT Red",
            "。",
        ),
        (3, 5, 6, 8, 12, 14, 16),
        16,
    )

    pro_options = _path_options(pro_request)
    model_options = _path_options(model_request)

    assert pro_options[1]["breaks"] == [10]
    assert pro_options[1]["cues"] == [
        "目前仅面向Pro、Max、Team",
        "和Enterprise用户开放使用。",
    ]
    assert model_options[1]["breaks"] == [6]
    assert model_options[1]["cues"] == [
        "OpenAI对外公开了",
        "内部网络安全专用模型GPT Red。",
    ]
    assert pro_options[1]["host_penalty"] < pro_options[0]["host_penalty"]
    assert model_options[1]["host_penalty"] < model_options[0]["host_penalty"]


def test_host_shortlist_keeps_aspect_particle_with_following_object():
    text = "官方在今天早些时候也发布了相关预告视频。"
    request = two_cue_path_task(tuple(text), (9, 13), 13)
    options = _path_options(request)

    assert options[1]["breaks"] == [9]
    assert options[1]["cues"] == [
        "官方在今天早些时候",
        "也发布了相关预告视频。",
    ]
    assert options[1]["host_penalty"] + 6 == options[0]["host_penalty"]

    selector = FakeScoringSelector([[[0.0, 0.0], [0.0, 0.0]]])
    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.break_indices == (9,)


def test_host_penalizes_only_long_cues_spanning_substantial_comma_clauses():
    short = BoundaryTask(
        atoms=tuple("我们先介绍产品，随后演示完整流程"),
        candidate_indices=(7,),
        language="zh",
        fallback_indices=(),
        target_chars=18,
    )
    long = BoundaryTask(
        atoms=tuple("在Arena中上线测试，官方在今天早些时候"),
        candidate_indices=(12,),
        language="zh",
        fallback_indices=(),
        target_chars=18,
    )
    long_with_terminal = BoundaryTask(
        atoms=tuple("在Arena中上线测试，官方在今天早些时候。"),
        candidate_indices=(12,),
        language="zh",
        fallback_indices=(),
        target_chars=18,
    )

    assert _cue_host_penalty(short, 0, len(short.atoms)) == 0
    assert _cue_host_penalty(long, 0, len(long.atoms)) == 3
    assert (
        _cue_host_penalty(
            long_with_terminal, 0, len(long_with_terminal.atoms)
        )
        == 3
    )


def test_symmetric_scorer_abstains_from_pure_first_position_bias():
    request = BoundaryTask(
        atoms=("alpha", "beta", "gamma"),
        candidate_indices=(1, 2),
        language="en",
        fallback_indices=(1,),
        allowed_edges=((0, 1), (1, 3), (0, 2), (2, 3)),
        edge_quality=((0, 1, 100), (1, 3, 100), (0, 2, 100), (2, 3, 100)),
    )
    selector = FakeScoringSelector([[[2.0, 0.0], [2.0, 0.0]]])

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.source == "model"
    assert decision.break_indices == request.fallback_indices


@pytest.mark.parametrize(
    ("fallback_worst", "candidate_worst", "expected", "scorer_called"),
    [
        (98, 97, (6,), False),
        (98, 98, (3, 9), True),
        (97, 96, (3, 9), True),
    ],
)
def test_pristine_timing_gate_blocks_only_new_under_floor_tail_layouts(
    fallback_worst, candidate_worst, expected, scorer_called
):
    request = BoundaryTask(
        atoms=(
            "下方",
            "为",
            "命令区",
            "支持",
            "接受",
            "拒绝",
            "语音输入",
            "新建对话",
            "等",
            "常用操作",
        ),
        candidate_indices=(3, 6, 9),
        language="zh",
        fallback_indices=(6,),
        allowed_edges=((0, 6), (6, 10), (0, 3), (3, 9), (9, 10)),
        edge_quality=(
            (0, 6, 100),
            (6, 10, fallback_worst),
            (0, 3, 100),
            (3, 9, 100),
            (9, 10, candidate_worst),
        ),
    )
    selector = FakeScoringSelector([[[2.0, 0.0], [0.0, 2.0]]])

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.break_indices == expected
    assert bool(selector.score_calls) is scorer_called


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([[0.0, 0.0], [0.0, 0.0]], (2,)),
        ([[0.0, 2.0], [2.0, 0.0]], (1,)),
    ],
)
def test_three_point_host_improvement_uses_fused_evidence(scores, expected):
    request = BoundaryTask(
        atoms=("alpha", "bravo", "charlie"),
        candidate_indices=(1, 2),
        language="en",
        fallback_indices=(1,),
        pauses_ms=((2, 300),),
        allowed_edges=((0, 1), (1, 3), (0, 2), (2, 3)),
        edge_quality=(
            (0, 1, 100),
            (1, 3, 100),
            (0, 2, 100),
            (2, 3, 100),
        ),
    )
    selector = FakeScoringSelector([scores])

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.break_indices == expected


def test_logit_scorer_failure_returns_exact_deterministic_fallback():
    request = BoundaryTask(
        atoms=("alpha", "beta", "gamma"),
        candidate_indices=(1, 2),
        language="en",
        fallback_indices=(1,),
        allowed_edges=((0, 1), (1, 3), (0, 2), (2, 3)),
    )
    selector = FakeScoringSelector([RuntimeError("non-finite logits")])

    decision = SemanticBreakEngine(selector).choose([request])[0]

    assert decision.source == "fallback"
    assert decision.break_indices == request.fallback_indices
    assert "non-finite logits" in (decision.reason or "")


def test_batches_zh_ja_en_through_one_multilingual_model_call():
    selector = FakeSelector(
        [
            json.dumps(
                {
                    "results": [
                        {"id": 0, "breaks": [2]},
                        {"id": 1, "breaks": [1, 3]},
                        {"id": 2, "breaks": [3]},
                    ]
                }
            )
        ]
    )
    engine = SemanticBreakEngine(selector)
    decisions = engine.choose([task("zh"), task("ja"), task("en")])

    assert [decision.break_indices for decision in decisions] == [(2,), (1, 3), (3,)]
    assert {decision.source for decision in decisions} == {"model"}
    assert len(selector.calls) == 1
    assert selector.calls[0][0] == DEFAULT_SEMANTIC_MODEL
    payload = payload_from_call(selector.calls[0])
    assert [item["language"] for item in payload["tasks"]] == ["zh", "ja", "en"]
    assert "⟦1⟧" in payload["tasks"][0]["marked_text"]
    assert " ⟦1⟧ " in payload["tasks"][2]["marked_text"]


def test_invalid_one_task_falls_back_without_discarding_valid_sibling():
    selector = FakeSelector(
        [
            json.dumps(
                {
                    "results": [
                        {"id": 0, "breaks": [3]},
                        {"id": 1, "breaks": [99]},
                    ]
                }
            )
        ]
    )
    decisions = SemanticBreakEngine(selector).choose([task("zh"), task("ja")])
    assert decisions[0].source == "model"
    assert decisions[0].break_indices == (3,)
    assert decisions[1].source == "fallback"
    assert decisions[1].break_indices == (2,)
    assert "outside candidate_indices" in (decisions[1].reason or "")


def test_extra_text_field_is_rejected_and_cannot_edit_transcript():
    selector = FakeSelector(
        [
            json.dumps(
                {
                    "results": [
                        {"id": 0, "breaks": [3], "text": "rewritten text"}
                    ]
                }
            )
        ]
    )
    decision = SemanticBreakEngine(selector).choose([task("en")])[0]
    assert decision.source == "fallback"
    assert decision.break_indices == (2,)
    assert "only id and breaks" in (decision.reason or "")


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{\"results\":[]}\n```",
        '{"results":[],"explanation":"because"}',
        "not json",
    ],
)
def test_non_strict_envelope_falls_back_for_whole_batch(raw):
    decisions = SemanticBreakEngine(FakeSelector([raw])).choose([task(), task("ja")])
    assert [decision.source for decision in decisions] == ["fallback", "fallback"]
    assert [decision.break_indices for decision in decisions] == [(2,), (2,)]


def test_missing_task_result_falls_back_only_for_missing_task():
    raw = json.dumps({"results": [{"id": 1, "breaks": [4]}]})
    decisions = SemanticBreakEngine(FakeSelector([raw])).choose([task(), task("ja")])
    assert decisions[0].source == "fallback"
    assert decisions[0].reason == "model response omitted task id"
    assert decisions[1].source == "model"
    assert decisions[1].break_indices == (4,)


def test_required_boundary_and_hard_max_are_validated_on_model_output():
    request = BoundaryTask(
        atoms=("aa", "bb", "cc", "dd"),
        candidate_indices=(1, 2, 3),
        language="en",
        fallback_indices=(1, 2, 3),
        required_indices=(2,),
        max_segment_chars=4,
    )
    raw = json.dumps({"results": [{"id": 0, "breaks": [1]}]})
    decision = SemanticBreakEngine(FakeSelector([raw])).choose([request])[0]
    assert decision.source == "fallback"
    assert decision.break_indices == (1, 2, 3)
    assert "required" in (decision.reason or "")


def test_selector_failure_falls_back_and_skips_later_batches_for_same_model():
    selector = FakeSelector([RuntimeError("server unavailable")])
    engine = SemanticBreakEngine(selector)
    decisions = engine.choose([task(), task("ja")], max_batch_chars=1)
    assert len(selector.calls) == 1
    assert all(decision.source == "fallback" for decision in decisions)
    assert all(decision.break_indices == (2,) for decision in decisions)


def test_small_batch_limit_packs_requests_into_multiple_calls():
    selector = FakeSelector([echo_first_candidate, echo_first_candidate, echo_first_candidate])
    decisions = SemanticBreakEngine(selector).choose(
        [task("zh"), task("ja"), task("en")], max_batch_chars=1
    )
    assert len(selector.calls) == 3
    assert [decision.break_indices for decision in decisions] == [(1,), (1,), (1,)]


def test_per_language_model_routes_group_calls_and_preserve_result_order():
    selector = FakeSelector([echo_first_candidate, echo_first_candidate])
    requests = [task("en"), task("zh"), task("ja")]
    decisions = SemanticBreakEngine(selector).choose(
        requests,
        model_by_language={"zh": "local/chinese", "*": "local/general"},
    )
    assert [call[0] for call in selector.calls] == ["local/general", "local/chinese"]
    assert [decision.break_indices for decision in decisions] == [(1,), (1,), (1,)]
    assert [decision.model_id for decision in decisions] == [
        "local/general",
        "local/chinese",
        "local/general",
    ]


def test_disabled_route_never_calls_selector():
    selector = FakeSelector([])
    decision = SemanticBreakEngine(selector).choose(
        [task()], model_by_language={"zh": "off"}
    )[0]
    assert decision.source == "fallback"
    assert selector.calls == []


def test_semantic_model_for_routing_precedence(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_MODEL", "env/general")
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_MODEL_JA", "env/japanese")
    assert semantic_model_for("ja") == "env/japanese"
    assert semantic_model_for("zh") == "env/general"
    assert semantic_model_for("English", {"en": "map/english"}) == "map/english"
    assert semantic_model_for("zh", {"*": None}) is None


def test_openai_compatible_selector_uses_injected_client_and_deterministic_args():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"results":[]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    selector = OpenAICompatibleSelector(client=client)
    raw = selector.select(
        "local/model",
        [{"role": "user", "content": "payload"}],
        max_new_tokens=123,
    )
    assert raw == '{"results":[]}'
    assert calls == [
        {
            "model": "local/model",
            "messages": [{"role": "user", "content": "payload"}],
            "max_tokens": 123,
            "temperature": 0,
        }
    ]


def test_openai_selector_without_endpoint_is_unavailable_without_importing_client(
    monkeypatch,
):
    monkeypatch.delenv("VOXWEAVE_SEMANTIC_BASE_URL", raising=False)
    selector = OpenAICompatibleSelector()
    with pytest.raises(SemanticBackendUnavailable, match="not configured"):
        selector.select("model", [], max_new_tokens=10)


def test_default_selector_is_isolated_local_unless_server_is_explicit(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_SEMANTIC_BASE_URL", raising=False)
    local_engine = SemanticBreakEngine()
    assert isinstance(local_engine.selector, LocalTransformersSelector)

    monkeypatch.setenv("VOXWEAVE_SEMANTIC_BASE_URL", "http://127.0.0.1:8000/v1")
    server_engine = SemanticBreakEngine()
    assert isinstance(server_engine.selector, OpenAICompatibleSelector)


def test_local_selector_isolates_hf_cache_and_propagates_offline_mode(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", "/tmp/voxweave-test-cache")
    monkeypatch.setenv("VOXWEAVE_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", "/parent/packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/parent/venv")
    monkeypatch.setattr("voxweave.semantic_breaks.shutil.which", lambda _name: "/bin/uv")

    env = LocalTransformersSelector._child_environment()
    assert env["HF_HOME"] == "/tmp/voxweave-test-cache/semantic"
    assert env["HF_HUB_CACHE"] == "/tmp/voxweave-test-cache/semantic"
    assert env["HUGGINGFACE_HUB_CACHE"] == "/tmp/voxweave-test-cache/semantic"
    assert env["UV_CACHE_DIR"] == "/tmp/voxweave-test-cache/semantic/uv"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    command = LocalTransformersSelector._default_command()
    assert "--offline" in command
    assert "--locked" in command


def test_local_selector_respects_cuda_device_with_existing_visibility(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_DEVICE", "cuda:1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha,GPU-beta")
    env = LocalTransformersSelector._child_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-beta"
    assert env["VOXWEAVE_DEVICE"] == "cuda:0"

    monkeypatch.setenv("VOXWEAVE_DEVICE", "cuda:3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = LocalTransformersSelector._child_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["VOXWEAVE_DEVICE"] == "cuda:0"


def test_local_selector_rejects_invalid_or_hidden_cuda_device(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_DEVICE", "cuda:x")
    with pytest.raises(SemanticBackendUnavailable, match="invalid CUDA device"):
        LocalTransformersSelector._child_environment()

    monkeypatch.setenv("VOXWEAVE_DEVICE", "cuda:2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SemanticBackendUnavailable, match="outside CUDA_VISIBLE_DEVICES"):
        LocalTransformersSelector._child_environment()


def test_engine_release_delegates_to_selector():
    selector = FakeSelector([])
    SemanticBreakEngine(selector).release()
    assert selector.released is True


def test_engine_release_is_best_effort_and_never_raises():
    selector = FakeSelector([])

    def fail_release():
        raise RuntimeError("cleanup failed")

    selector.release = fail_release
    SemanticBreakEngine(selector).release()
