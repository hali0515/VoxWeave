from pathlib import Path
from types import SimpleNamespace

import pytest

from voxweave import pipeline, translate
from voxweave.progress import Reporter


class FakeClient:
    """Records received messages and returns chat.completions from a preset queue."""

    def __init__(self, contents):
        self._contents = list(contents)  # each create() call pops one response
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        self.calls.append(messages)
        content = self._contents.pop(0)
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeStreamClient:
    """Returns a chunked iterator when stream=True (simulating SSE); returns a single response otherwise."""

    def __init__(self, pieces):
        self._pieces = list(pieces)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, stream=False, **kw):
        if stream:
            return (
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]
                )
                for p in self._pieces
            )
        full = "".join(self._pieces)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=full))]
        )


class _RecordingReporter(Reporter):
    def __init__(self):
        self.label = None
        self.total = None
        self.advances = 0

    def task(self, label, total):
        self.label, self.total = label, total

    def advance(self, n=1):
        self.advances += n


def test_build_payload_numbers_cues():
    blocks = [
        {"text": "hello", "start": 1.0, "end": 2.0},
        {"text": "world", "start": 2.0, "end": 3.0},
    ]
    payload = translate.build_payload(blocks)
    assert payload == [{"i": 0, "t": "hello"}, {"i": 1, "t": "world"}]


def test_build_payload_merges_multiline_cue():
    blocks = [{"text": "line1\nline2", "start": 1.0, "end": 2.0}]
    payload = translate.build_payload(blocks)
    assert payload == [{"i": 0, "t": "line1 line2"}]


def test_parse_response_standard():
    raw = '{"translations": [{"i": 0, "t": "你好"}, {"i": 1, "t": "世界"}]}'
    assert translate.parse_response(raw) == {0: "你好", 1: "世界"}


def test_parse_response_dict_input():
    obj = {"translations": [{"i": 5, "t": "hi"}]}
    assert translate.parse_response(obj) == {5: "hi"}


def test_parse_response_recovers_from_noise():
    raw = 'Sure!\n{"translations": [{"i": 0, "t": "x"}]}\nDone'
    assert translate.parse_response(raw) == {0: "x"}


def test_parse_response_garbage_returns_empty():
    assert translate.parse_response("not json at all") == {}


def test_parse_response_non_str_non_dict_tolerated():
    # non-str/non-dict input must fall through to {} (not raise TypeError from json.loads)
    assert translate.parse_response(None) == {}
    assert translate.parse_response(123) == {}
    assert translate.parse_response([{"i": 0, "t": "x"}]) == {}


def test_parse_response_ignores_trailing_prose_with_braces():
    raw = '{"translations": [{"i": 0, "t": "x"}]} 备注: 用 {} 表示空'
    assert translate.parse_response(raw) == {0: "x"}


def test_validate_reports_missing_and_empty():
    blocks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    trans = {0: "译A", 1: "   "}
    missing = translate.validate_and_fill(blocks, trans)
    assert missing == [1, 2]


def test_validate_all_present():
    blocks = [{"text": "a"}, {"text": "b"}]
    assert translate.validate_and_fill(blocks, {0: "x", 1: "y"}) == []


def test_render_timestamped():
    blocks = [
        {"text": "hi", "start": 1.0, "end": 2.5},
        {"text": "yo", "start": 3.0, "end": 4.0},
    ]
    out = translate.render_translated_vtt(blocks, {0: "你好", 1: "喲"})
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:01.000 --> 00:00:02.500\n你好" in out
    assert "00:00:03.000 --> 00:00:04.000\n喲" in out


def test_render_falls_back_to_original_when_missing():
    blocks = [{"text": "keepme", "start": 1.0, "end": 2.0}]
    out = translate.render_translated_vtt(blocks, {})
    assert "keepme" in out


def test_render_no_timestamps_plain_blocks():
    blocks = [
        {"text": "a", "start": None, "end": None},
        {"text": "b", "start": None, "end": None},
    ]
    out = translate.render_translated_vtt(blocks, {0: "甲", 1: "乙"})
    assert "-->" not in out
    assert out.startswith("WEBVTT\n\n")
    assert "甲" in out and "乙" in out


def test_format_glossary_dict():
    s = translate.format_glossary({"ミク": "米克", "ユウ": "尤"})
    assert "ミク" in s and "米克" in s


def test_format_glossary_str_passthrough():
    assert translate.format_glossary("随便的术语说明") == "随便的术语说明"


def test_build_messages_has_index_payload_and_target_lang():
    payload = [{"i": 0, "t": "hello"}]
    msgs = translate.build_messages(payload, to="zh")
    assert msgs[0]["role"] == "system"
    assert "zh" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user"
    assert '"i": 0' in msgs[-1]["content"] or '"i":0' in msgs[-1]["content"]


def test_build_messages_injects_context_glossary_and_tail():
    payload = [{"i": 5, "t": "next"}]
    msgs = translate.build_messages(
        payload,
        to="zh",
        context="科幻番剧, 口语化",
        glossary={"A": "甲"},
        tail=[("prev orig", "上一句译")],
    )
    sys = msgs[0]["content"]
    assert "科幻番剧" in sys
    assert "甲" in sys
    joined = sys + msgs[-1]["content"]
    assert "上一句译" in joined


def test_translate_cues_single_batch():
    payload = [{"i": 0, "t": "hello"}, {"i": 1, "t": "bye"}]
    client = FakeClient(['{"translations":[{"i":0,"t":"你好"},{"i":1,"t":"再见"}]}'])
    out = translate.translate_cues(payload, to="zh", model="m", client=client)
    assert out == {0: "你好", 1: "再见"}
    assert len(client.calls) == 1


def test_translate_cues_windows_when_over_threshold():
    payload = [{"i": i, "t": f"c{i}"} for i in range(5)]
    client = FakeClient(
        [
            '{"translations":[{"i":0,"t":"A"},{"i":1,"t":"B"}]}',
            '{"translations":[{"i":2,"t":"C"},{"i":3,"t":"D"}]}',
            '{"translations":[{"i":4,"t":"E"}]}',
        ]
    )
    out = translate.translate_cues(
        payload, to="zh", model="m", client=client, batch=2, context_tail=1
    )
    assert out == {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    assert len(client.calls) == 3
    second_system = client.calls[1][0]["content"]
    assert "B" in second_system


def test_translate_cues_streams_progress_per_cue():
    # reporter provided -> streaming mode; progress bar advances per cue ("i" key) as they arrive (works even for a single batch)
    payload = [{"i": 0, "t": "a"}, {"i": 1, "t": "b"}, {"i": 2, "t": "c"}]
    pieces = [
        '{"translations":[',
        '{"i":0,"t":"甲"},',
        '{"i":1,"t":"乙"},',
        '{"i":2,"t":"丙"}]}',
    ]
    client = FakeStreamClient(pieces)
    rep = _RecordingReporter()
    out = translate.translate_cues(
        payload, to="zh", model="m", client=client, reporter=rep
    )
    assert out == {0: "甲", 1: "乙", 2: "丙"}
    assert rep.total == 3  # denominator = cue count
    assert rep.advances == 3  # three "i" keys streamed -> advanced to 3/3


def test_translate_cues_no_reporter_stays_non_streaming():
    # no reporter -> non-streaming single call (existing behavior unchanged)
    payload = [{"i": 0, "t": "a"}]
    client = FakeClient(['{"translations":[{"i":0,"t":"甲"}]}'])
    out = translate.translate_cues(payload, to="zh", model="m", client=client)
    assert out == {0: "甲"}
    assert len(client.calls) == 1


def _write_vtt(p: Path):
    p.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n\n"
        "00:00:03.000 --> 00:00:04.000\nworld\n",
        encoding="utf-8",
    )


def test_pipeline_translate_writes_sibling(tmp_path, monkeypatch):
    vtt = tmp_path / "ep.vtt"
    _write_vtt(vtt)
    monkeypatch.setattr(
        pipeline.translate_mod,
        "translate_cues",
        lambda payload, **kw: {0: "你好", 1: "世界"},
    )
    out = pipeline.translate(vtt, to="zh")
    assert out == tmp_path / "ep.zh.vtt"
    txt = out.read_text(encoding="utf-8")
    assert "00:00:01.000 --> 00:00:02.000\n你好" in txt
    assert "00:00:03.000 --> 00:00:04.000\n世界" in txt
    assert "hello" in vtt.read_text(encoding="utf-8")


def test_pipeline_translate_rejects_unknown_format(tmp_path):
    txt = tmp_path / "ep.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported subtitle format"):
        pipeline.translate(txt, to="zh")


def test_pipeline_translate_srt_mirrors_format(tmp_path, monkeypatch):
    srt = tmp_path / "ep.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nworld\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline.translate_mod,
        "translate_cues",
        lambda payload, **kw: {0: "你好", 1: "世界"},
    )
    out = pipeline.translate(srt, to="zh")
    assert out == tmp_path / "ep.zh.srt"
    txt = out.read_text(encoding="utf-8")
    assert "1\n00:00:01,000 --> 00:00:02,000\n你好" in txt
    assert "2\n00:00:03,000 --> 00:00:04,000\n世界" in txt


def test_pipeline_translate_ass_mirrors_format(tmp_path, monkeypatch):
    ass = tmp_path / "ep.ass"
    ass.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,hello\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline.translate_mod,
        "translate_cues",
        lambda payload, **kw: {0: "你好"},
    )
    out = pipeline.translate(ass, to="zh")
    assert out == tmp_path / "ep.zh.ass"
    txt = out.read_text(encoding="utf-8")
    assert "[V4+ Styles]" in txt  # rendered as a full ASS script
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,你好" in txt


def test_pipeline_translate_fills_missing_with_retry_then_original(
    tmp_path, monkeypatch
):
    vtt = tmp_path / "ep.vtt"
    _write_vtt(vtt)
    calls = []

    def fake_cues(payload, **kw):
        calls.append([c["i"] for c in payload])
        if len(calls) == 1:
            return {0: "你好"}
        return {}

    monkeypatch.setattr(pipeline.translate_mod, "translate_cues", fake_cues)
    out = pipeline.translate(vtt, to="zh")
    txt = out.read_text(encoding="utf-8")
    assert "你好" in txt
    assert "world" in txt
    assert len(calls) == 2
    assert calls[1] == [1]
    assert (
        txt.count("-->") == 2
    )  # cue count conserved (partial failure must not drop cues)


def test_load_glossary_json(tmp_path):
    p = tmp_path / "g.json"
    p.write_text('{"ミク": "米克"}', encoding="utf-8")
    assert translate.load_glossary(p) == {"ミク": "米克"}


def test_load_glossary_txt_passthrough(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("ミク = 米克\nユウ = 尤", encoding="utf-8")
    assert "米克" in translate.load_glossary(p)
    assert isinstance(translate.load_glossary(p), str)


def test_strip_punct_extended_decorative():
    # quotation marks / brackets / ellipses / dashes -> stripped (replaced with collapsed space)
    assert translate.strip_punct_for_subtitles("喂「等一下」……") == "喂 等一下"
    assert translate.strip_punct_for_subtitles("（旁白）你好—世界") == "旁白 你好 世界"


def test_strip_punct_base_set_and_digits():
    # sentence-final punctuation stripped; decimal/thousands separators within numbers preserved
    assert translate.strip_punct_for_subtitles("你好, 世界。") == "你好 世界"
    assert (
        translate.strip_punct_for_subtitles("共 10,000 元 3.75 秒")
        == "共 10,000 元 3.75 秒"
    )


def test_strip_punct_keeps_name_joiner():
    # · ・ name joiners, ー kana long-vowel mark, - hyphen: none stripped
    assert translate.strip_punct_for_subtitles("米歇尔·奥巴马") == "米歇尔·奥巴马"
    assert translate.strip_punct_for_subtitles("spider-man") == "spider-man"


def test_render_strips_translated_punctuation():
    blocks = [{"text": "x", "start": 1.0, "end": 2.0}]
    out = translate.render_translated_vtt(blocks, {0: "喂「等一下」……"})
    assert "喂 等一下" in out
    assert "「" not in out and "…" not in out


def test_build_messages_mentions_no_decorative_punct():
    msgs = translate.build_messages([{"i": 0, "t": "x"}], to="zh")
    assert "decorative punctuation" in msgs[0]["content"]


def test_build_messages_instructs_use_of_full_context():
    # wording should encourage the model to use full-batch context (disambiguation / consistent proper nouns), not translate each cue in isolation
    sys = translate.build_messages([{"i": 0, "t": "x"}], to="zh")[0]["content"]
    assert "context" in sys
    # must still enforce cue-count contract (no merge / no split / no additions or deletions)
    assert "merge" in sys and "split" in sys
    # must steer toward tone/register-faithful rendering, not literal word-for-word
    assert "tone" in sys


def test_default_batch_threshold_is_large_enough_for_whole_episode():
    # whole-episode single-call translation gives maximum context; real episodes typically 300-500 cues, threshold must be large enough
    assert translate.BATCH_THRESHOLD >= 500


# --------------------------------------------------------------------------- #
# target-language re-layout: cue count/timing are fixed, soft-wrap is the only
# valve when the translation outgrows the target line budget
# --------------------------------------------------------------------------- #


def _cue_lines(vtt: str) -> list[str]:
    return [ln for ln in vtt.splitlines() if ln and "-->" not in ln and ln != "WEBVTT"]


def test_render_translated_wraps_long_zh_to_two_lines():
    blocks = [{"start": 1.0, "end": 4.0, "text": "source"}]
    out = translate.render_translated_vtt(blocks, {0: "字" * 30}, to_iso="zh")
    lines = _cue_lines(out)
    assert len(lines) == 2  # 60 visual width -> balanced two lines
    assert "".join(lines) == "字" * 30  # content preserved


def test_render_translated_short_zh_stays_single_line():
    blocks = [{"start": 1.0, "end": 4.0, "text": "source"}]
    out = translate.render_translated_vtt(blocks, {0: "字" * 10}, to_iso="zh")
    assert len(_cue_lines(out)) == 1


def test_render_translated_no_iso_keeps_legacy_no_wrap():
    blocks = [{"start": 1.0, "end": 4.0, "text": "source"}]
    out = translate.render_translated_vtt(blocks, {0: "字" * 30})
    assert len(_cue_lines(out)) == 1


def test_pipeline_translate_passes_target_iso(tmp_path, monkeypatch):
    vtt = tmp_path / "a.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nhello world\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        translate, "translate_cues", lambda payload, **kw: {0: "字" * 30}
    )
    out = pipeline.translate(vtt, to="zh", reporter=Reporter())
    lines = _cue_lines(out.read_text(encoding="utf-8"))
    assert len(lines) == 2  # wrap applied through the pipeline path
