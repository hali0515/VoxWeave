from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from voxweave import pipeline, speakers, translate
from voxweave.asrfix import render_vtt as render_corrected_vtt
from voxweave.cli import cli
from voxweave.export import (
    export_subtitles,
    render_ass,
    render_srt,
    render_vtt_rows,
)
from voxweave.realign import parse_vtt_blocks, render_vtt
from voxweave.subformats import load_subtitle_blocks, parse_ass_blocks


def _overlap(left, right):
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def test_select_snippets_are_exclusive_voiced_non_singing_and_bounded():
    turns = [
        (0.0, 24.0, "SPEAKER_00"),
        (5.0, 8.0, "SPEAKER_01"),
        (18.0, 20.0, "SPEAKER_01"),
        (30.0, 38.0, "SPEAKER_01"),
    ]
    vad = [(1.0, 22.0), (31.0, 37.0)]
    singing = [(10.0, 13.0), (33.0, 35.0)]

    selected = speakers.select_snippets(turns, vad, singing)

    assert set(selected) == {"SPEAKER_00", "SPEAKER_01"}
    assert 1 <= len(selected["SPEAKER_00"]) <= 3
    for label, clips in selected.items():
        own = [(start, end) for start, end, turn_label in turns if turn_label == label]
        other = [
            (start, end) for start, end, turn_label in turns if turn_label != label
        ]
        for clip in clips:
            assert speakers.MIN_SNIPPET_S <= clip[1] - clip[0] <= speakers.MAX_SNIPPET_S
            assert any(start <= clip[0] and clip[1] <= end for start, end in own)
            assert any(start <= clip[0] and clip[1] <= end for start, end in vad)
            assert not any(_overlap(clip, span) for span in other)
            assert not any(_overlap(clip, span) for span in singing)


def test_select_snippets_intersects_vad_and_subtracts_singing():
    selected = speakers.select_snippets(
        [(0.0, 10.0, "SPEAKER_00")],
        [(3.0, 8.0)],
        [(5.0, 6.0)],
    )
    assert selected == {"SPEAKER_00": [(3.0, 5.0), (6.0, 8.0)]}


def test_select_snippets_keeps_one_six_second_run_as_one_long_clip():
    selected = speakers.select_snippets(
        [(100.0, 106.0, "SPEAKER_00")],
        [(0.0, 700.0)],
        [],
    )

    assert selected == {"SPEAKER_00": [(100.0, 106.0)]}


def test_select_snippets_keeps_close_but_disjoint_clean_utterances():
    selected = speakers.select_snippets(
        [
            (0.0, 2.5, "SPEAKER_00"),
            (3.0, 5.5, "SPEAKER_00"),
            (6.0, 8.5, "SPEAKER_00"),
        ],
        [(0.0, 8.5)],
        [],
    )

    assert selected == {"SPEAKER_00": [(0.0, 2.5), (3.0, 5.5), (6.0, 8.5)]}


def test_select_snippets_fill_can_reuse_one_clean_run_with_a_real_gap():
    selected = speakers.select_snippets(
        [(0.0, 3.5, "SPEAKER_00"), (100.0, 120.0, "SPEAKER_00")],
        [(0.0, 200.0)],
        [],
    )

    assert selected == {"SPEAKER_00": [(0.0, 3.5), (100.0, 106.0), (107.0, 113.0)]}


def test_mapping_reader_ignores_empty_and_unknown_ids_once(tmp_path, caplog):
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {
                    "SPEAKER_00": " Aoi ",
                    "SPEAKER_01": "",
                    "SPEAKER_02": "   ",
                    "OLD_00": "Old",
                    "OLD_01": "Older",
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        names = speakers.load_speaker_mapping(
            mapping, {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
        )

    assert names == {"SPEAKER_00": " Aoi "}
    warnings = [
        record for record in caplog.records if "unknown speaker" in record.message
    ]
    assert len(warnings) == 1
    assert "OLD_00" in warnings[0].message and "OLD_01" in warnings[0].message


@pytest.mark.parametrize(
    "document",
    [
        {"speakers": {}},
        {"version": True, "speakers": {}},
        {"version": 2, "speakers": {}},
        {"version": 1, "speakers": []},
    ],
)
def test_mapping_reader_rejects_invalid_schema(tmp_path, document):
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError):
        speakers.load_speaker_mapping(mapping, set())


def test_vtt_voice_tags_strip_and_render_idempotently():
    source = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "<v Aoi>Hello there</v>\n\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "<v Aoi>-Stay</v>\n"
        "<v Ren>-Go</v>\n"
    )

    blocks = parse_vtt_blocks(source)

    assert blocks[0]["text"] == "Hello there"
    assert blocks[0]["speaker"] == "Aoi"
    assert blocks[1]["text"] == "-Stay\n-Go"
    assert blocks[1]["speakers"] == ["Aoi", "Ren"]
    spans = [(block["start"], block["end"]) for block in blocks]
    assert render_vtt(blocks, spans) == source


@pytest.mark.parametrize(
    "wrapped",
    ["<v>Text</v>", "<v >Text</v>", "<v   >Text</v>", "<v &#10;>Text</v>"],
)
def test_whitespace_only_voice_annotations_strip_as_clean_text(wrapped):
    assert speakers.strip_voice_tags(wrapped) == ("Text", None, None)
    blocks = parse_vtt_blocks(f"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n{wrapped}\n")
    assert blocks == [{"text": "Text", "start": 1.0, "end": 2.0}]


def test_vtt_mixed_named_dash_line_round_trip():
    source = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Aoi>-Named</v>\n-Unnamed\n"
    blocks = parse_vtt_blocks(source)
    assert blocks[0]["text"] == "-Named\n-Unnamed"
    assert blocks[0]["speakers"] == ["Aoi", None]
    assert render_vtt(blocks, [(1.0, 2.0)]) == source


def test_voice_and_lyric_display_flags_round_trip_independently():
    source = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Aoi>♪ la la ♪</v>\n"
    blocks = parse_vtt_blocks(source)
    assert blocks == [
        {"text": "la la", "start": 1.0, "end": 2.0, "speaker": "Aoi", "lyric": True}
    ]
    assert render_vtt(blocks, [(1.0, 2.0)]) == source


def test_named_srt_and_ass_exports_cover_single_dash_and_unnamed(tmp_path):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Aoi>Hello</v>\n\n"
        "00:00:03.000 --> 00:00:04.000\n<v Aoi>-Stay</v>\n<v Ren>-Go</v>\n\n"
        "00:00:05.000 --> 00:00:06.000\nPlain\n",
        encoding="utf-8",
    )

    export_subtitles(vtt, ("srt", "ass"))
    srt = (tmp_path / "episode.srt").read_text(encoding="utf-8")
    ass = (tmp_path / "episode.ass").read_text(encoding="utf-8")

    assert "Aoi: Hello" in srt
    assert "Aoi: -Stay\nRen: -Go" in srt
    assert "\nPlain\n" in srt
    assert "Default,Aoi,0,0,0,,Hello" in ass
    assert "Default,Aoi / Ren,0,0,0,,-Stay\\N-Go" in ass
    assert "Default,,0,0,0,,Plain" in ass


def test_named_exports_sanitize_literal_and_entity_line_breaks(tmp_path):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Ren\nKai>hello</v>\n\n"
        "00:00:03.000 --> 00:00:04.000\n<v Mei&#10;Ling>bye</v>\n\n"
        "00:00:05.000 --> 00:00:06.000\n<v Sane>ok</v>\n",
        encoding="utf-8",
    )

    export_subtitles(vtt, ("ass", "srt"))
    ass = (tmp_path / "episode.ass").read_text(encoding="utf-8")
    srt = (tmp_path / "episode.srt").read_text(encoding="utf-8")

    assert [block["text"] for block in parse_ass_blocks(ass)] == [
        "hello",
        "bye",
        "ok",
    ]
    assert len([line for line in ass.splitlines() if line.startswith("Dialogue:")]) == 3
    assert "Default,Ren Kai,0,0,0,,hello" in ass
    assert "Default,Mei Ling,0,0,0,,bye" in ass
    assert "Ren Kai: hello" in srt
    assert "Mei Ling: bye" in srt


def test_all_named_renderers_share_structural_name_sanitization():
    rows = [(1.0, 2.0, "Hello")]
    blocks = [{"speaker": "Ren,\r\nKai\x1eMei\u2028Lin"}]
    display_safe = "Ren, Kai Mei Lin"
    ass_safe = "Ren， Kai Mei Lin"

    assert f"<v {display_safe}>Hello</v>" in render_vtt_rows(rows, blocks=blocks)
    assert f"{display_safe}: Hello" in render_srt(rows, blocks=blocks)
    ass = render_ass(rows, blocks=blocks)
    assert f"Default,{ass_safe},0,0,0,,Hello" in ass
    assert "\r" not in ass and "\x1e" not in ass and "\u2028" not in ass


def test_name_sanitization_preserves_non_ascii_display_spaces():
    rows = [(1.0, 2.0, "Hello")]
    name = "山田　太郎 / Jean\xa0Luc"
    blocks = [{"speaker": name}]

    assert f"<v {name}>Hello</v>" in render_vtt_rows(rows, blocks=blocks)
    assert f"{name}: Hello" in render_srt(rows, blocks=blocks)
    assert f"Default,{name},0,0,0,,Hello" in render_ass(rows, blocks=blocks)


def test_named_srt_round_trip_recovers_clean_text_and_dash_metadata(tmp_path):
    vtt = tmp_path / "episode.vtt"
    source = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\n<v Aoi>Hello</v>\n\n"
        "00:00:04.000 --> 00:00:06.000\n"
        "<v Aoi>-Stay</v>\n<v Ren>-Go</v>\n"
    )
    vtt.write_text(source, encoding="utf-8")
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {"SPEAKER_00": "Aoi", "SPEAKER_01": "Ren"},
            }
        ),
        encoding="utf-8",
    )

    export_subtitles(vtt, ("srt",))
    srt = tmp_path / "episode.srt"
    blocks = load_subtitle_blocks(srt)
    payload = translate.build_payload(blocks)

    assert blocks[0]["text"] == "Hello" and blocks[0]["speaker"] == "Aoi"
    assert blocks[1]["text"] == "-Stay\n-Go"
    assert blocks[1]["speakers"] == ["Aoi", "Ren"]
    assert payload[1] == {"i": 1, "t": "-Stay -Go", "parts": ["Stay", "Go"]}
    assert "Aoi" not in json.dumps(payload) and "Ren" not in json.dumps(payload)

    vtt.unlink()
    export_subtitles(srt, ("vtt",))
    assert vtt.read_text(encoding="utf-8") == source


def test_translate_keeps_names_out_of_payload_and_restores_vtt_tags():
    blocks = parse_vtt_blocks(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Aoi>Hello</v>\n\n"
        "00:00:03.000 --> 00:00:04.000\n<v Aoi>-Stay</v>\n<v Ren>-Go</v>\n"
    )

    payload = translate.build_payload(blocks)
    rendered = translate.render_translated_vtt(
        blocks, {0: "你好", 1: "留下\n走吧"}, to_iso="zh"
    )

    assert payload[0] == {"i": 0, "t": "Hello"}
    assert payload[1]["parts"] == ["Stay", "Go"]
    assert "Aoi" not in json.dumps(payload)
    assert "<v Aoi>你好</v>" in rendered
    assert "<v Aoi>-留下</v>\n<v Ren>-走吧</v>" in rendered


def test_correct_render_restores_voice_metadata_without_exposing_it_as_text():
    blocks = parse_vtt_blocks(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Aoi>helo</v>\n"
    )
    assert blocks[0]["text"] == "helo"
    assert "<v Aoi>hello</v>" in render_corrected_vtt(blocks, ["hello"])


def test_correct_keeps_all_names_when_a_fix_collapses_dual_speaker_lines(
    tmp_path, monkeypatch
):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "<v Aoi>-Stya here</v>\n<v Ren>-Go now</v>\n",
        encoding="utf-8",
    )

    def fake_correct(payload, **_kwargs):
        assert payload == [
            {
                "i": 0,
                "t": "-Stya here -Go now",
                "parts": ["Stya here", "Go now"],
            }
        ]
        return [
            {
                "i": 0,
                "orig": "-Stya here -Go now",
                "fixed": "-Stay here -Go now",
                "reason": "typo",
            }
        ]

    monkeypatch.setattr(pipeline.asrfix_mod, "correct_cues", fake_correct)
    pipeline.correct(vtt, apply=True)

    rendered = vtt.read_text(encoding="utf-8")
    assert "<v Aoi>-Stay here</v>\n<v Ren>-Go now</v>" in rendered


def test_distinct_line_names_render_unnamed_after_unrecoverable_collapse():
    blocks = parse_vtt_blocks(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "<v Aoi>Hello there my friend</v>\n<v Ren>Hi</v>\n"
    )

    rendered = translate.render_translated_vtt(
        blocks, {0: "你好 我的朋友 嗨"}, to_iso="zh"
    )

    assert "<v " not in rendered
    assert "\n你好 我的朋友 嗨\n" in rendered
    srt = render_srt(
        [(1.0, 4.0, "你好 我的朋友 嗨")],
        blocks=[{"speakers": ["Aoi", "Ren"]}],
    )
    assert "Aoi / Ren" not in srt
    assert "\n你好 我的朋友 嗨\n" in srt
    ass = render_ass(
        [(1.0, 4.0, "你好 我的朋友 嗨")],
        blocks=[{"speakers": ["Aoi", "Ren"]}],
    )
    assert "Default,Aoi / Ren,0,0,0,,你好 我的朋友 嗨" in ass


def test_collapsed_names_do_not_bake_into_srt_round_trip(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "<v Aoi>Hello thre my friend</v>\n<v Ren>Hi</v>\n",
        encoding="utf-8",
    )
    (tmp_path / "episode.speakers.json").write_text(
        '{"version":1,"speakers":{"S0":"Aoi","S1":"Ren"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline.asrfix_mod,
        "correct_cues",
        lambda _payload, **_kwargs: [
            {
                "i": 0,
                "orig": "Hello thre my friend Hi",
                "fixed": "Hello there my friend Hi",
                "reason": "typo",
            }
        ],
    )

    pipeline.correct(vtt, apply=True)
    assert "<v " not in vtt.read_text(encoding="utf-8")
    export_subtitles(vtt, ("srt",))
    srt = tmp_path / "episode.srt"
    assert "Aoi / Ren" not in srt.read_text(encoding="utf-8")
    assert load_subtitle_blocks(srt)[0]["text"] == "Hello there my friend Hi"


def test_correct_audit_records_reflowed_text_written_to_vtt(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "<v Aoi>-Stya here</v>\n<v Ren>-Go now</v>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline.asrfix_mod,
        "correct_cues",
        lambda _payload, **_kwargs: [
            {
                "i": 0,
                "orig": "-Stya here -Go now",
                "fixed": "-Stay here -Go now",
                "reason": "typo",
            }
        ],
    )

    result = pipeline.correct(vtt)
    expected = "-Stay here\n-Go now"
    audit = json.loads(result["audit"].read_text(encoding="utf-8"))

    assert result["applied"][0]["fixed"] == expected
    assert audit["applied"][0]["fixed"] == expected
    assert "<v Aoi>-Stay here</v>\n<v Ren>-Go now</v>" in result["out"].read_text(
        encoding="utf-8"
    )


def test_clip_builder_leaves_audio_stream_selection_to_ffmpeg():
    cmd = speakers.build_clip_command(Path("episode.mkv"), 1.0, 4.0, Path("clip.mp3"))

    assert "-map" not in cmd


def test_clip_builder_and_atomic_extraction_contract(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    output = tmp_path / "clip.mp3"
    seen = {}

    def fake_run(cmd):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "run_clip_command", fake_run)
    speakers.extract_clip(media, 1.25, 4.75, output)

    cmd = seen["cmd"]
    assert cmd[0] == "ffmpeg" and "-nostdin" in cmd
    assert "-map" not in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[-1] != str(output)
    assert output.read_bytes() == b"mp3"


def test_clip_runner_uses_devnull_and_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    speakers.run_clip_command(["ffmpeg", "-nostdin"])

    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["timeout"] == speakers.FFMPEG_TIMEOUT


def test_create_audition_writes_embedded_page_and_empty_mapping(tmp_path, monkeypatch):
    media = tmp_path / "episode.01.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.01.json").write_text(
        json.dumps(
            {
                "speaker_turns": [[0.0, 8.0, "SPEAKER_00"]],
                "vad_speech": [[0.0, 8.0]],
                "sing_spans": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_extract(_media, _start, _end, output):
        Path(output).write_bytes(b"clip-bytes")

    monkeypatch.setattr(speakers, "extract_clip", fake_extract)
    html_path = speakers.create_speaker_audition(media)

    mapping_path = tmp_path / "episode.01.speakers.json"
    page = html_path.read_text(encoding="utf-8")
    assert html_path.name == "episode.01.speakers.html"
    assert "data:audio/mpeg;base64," in page
    assert 'data-speaker="SPEAKER_00"' in page
    assert "JSON.stringify" in page and "https://" not in page
    assert json.loads(mapping_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "speakers": {"SPEAKER_00": ""},
    }


def test_create_audition_refuses_to_overwrite_mapping_before_ffmpeg(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").write_text(
        json.dumps(
            {
                "speaker_turns": [[0.0, 8.0, "SPEAKER_00"]],
                "vad_speech": [[0.0, 8.0]],
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text('{"version":1,"speakers":{"SPEAKER_00":"Aoi"}}')
    called = False

    def fake_extract(*_args):
        nonlocal called
        called = True

    monkeypatch.setattr(speakers, "extract_clip", fake_extract)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        speakers.create_speaker_audition(media)
    assert not called
    assert not (tmp_path / "episode.speakers.html").exists()


def test_create_audition_missing_turns_has_actionable_hint(tmp_path):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        RuntimeError, match=r"run voxweave episode\.mkv --diarize first"
    ):
        speakers.create_speaker_audition(media)


def test_speakers_cli_routes_through_shared_error_wrapper(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    output = tmp_path / "episode.speakers.html"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setattr(speakers, "create_speaker_audition", lambda path: output)

    result = CliRunner().invoke(cli, ["speakers", str(media)])

    assert result.exit_code == 0, result.output
    assert str(output) in result.output


def test_speakers_cli_missing_turns_uses_error_panel(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))

    result = CliRunner().invoke(cli, ["speakers", str(media)])

    assert result.exit_code == 1
    assert "Error" in result.output
    assert "speaker_turns" in result.output
    assert "--diarize first" in result.output


def test_split_named_render_keeps_sibling_json_clean(tmp_path):
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [],
                "word_segments": [
                    {"text": "Hello", "start": 0.0, "end": 0.5},
                    {"text": "there", "start": 0.55, "end": 1.0},
                    {"text": "Go", "start": 1.1, "end": 1.5},
                    {"text": "away", "start": 1.55, "end": 2.0},
                ],
                "vad_speech": [[0.0, 2.0]],
                "speaker_turns": [
                    [0.0, 1.05, "SPEAKER_00"],
                    [1.05, 2.1, "SPEAKER_01"],
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {"SPEAKER_00": "Aoi", "SPEAKER_01": "Ren"},
            }
        ),
        encoding="utf-8",
    )

    vtt_path = pipeline.split(json_path)
    vtt = vtt_path.read_text(encoding="utf-8")
    sibling = json_path.read_text(encoding="utf-8")

    assert "<v Aoi>" in vtt and "<v Ren>" in vtt
    assert "Aoi" not in sibling and "Ren" not in sibling
    assert "speaker_ids" not in sibling
