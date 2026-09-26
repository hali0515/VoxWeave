"""Regression tests for the pack/burn delivery audit (escaping, encodings,
media lookup, stream mapping)."""

from __future__ import annotations

from pathlib import Path

from voxweave import mux

_GBK_SRT = "1\n00:00:00,000 --> 00:00:01,000\n你好，世界\n".encode("gbk")


def test_filter_escape_matches_ffmpeg_documented_two_level_example():
    # ffmpeg-filters "Notes on filtergraph escaping": the level-2 form of
    # `this is a 'string': may contain one, or more, special characters`
    raw = "this is a 'string': may contain one, or more, special characters"
    assert mux._filter_escape(raw) == (
        "this is a \\\\\\'string\\\\\\'\\\\: may contain one\\, or more\\,"
        " special characters"
    )


def test_filter_escape_keeps_posix_backslash_and_escapes_equals():
    assert mux._filter_escape("/tmp/a\\b=c.ass") == "/tmp/a\\\\\\\\b\\\\=c.ass"


def test_resolve_media_peels_sdh_and_asrfix_tags(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    for name in ("ep.sdh.vtt", "ep.asrfix.vtt", "ep.zh.sdh.vtt"):
        assert mux.resolve_media(tmp_path / name, None) == media


def test_resolve_media_untagged_miss_keeps_message(tmp_path):
    import pytest

    (tmp_path / "ep.mkv").write_bytes(b"m")
    with pytest.raises(FileNotFoundError, match="pass --media explicitly"):
        mux.resolve_media(tmp_path / "other.notatag.vtt", None)


def test_utf8_subtitle_passthrough_and_conversion(tmp_path):
    utf8 = tmp_path / "a.srt"
    utf8.write_bytes(
        b"\xef\xbb\xbf" + "1\n00:00:00,000 --> 00:00:01,000\nhi\n".encode()
    )
    temps: list[Path] = []
    assert mux._utf8_subtitle(utf8, temps) == utf8
    assert temps == []

    gbk = tmp_path / "ep.zh.srt"
    gbk.write_bytes(_GBK_SRT)
    copy = mux._utf8_subtitle(gbk, temps)
    assert copy != gbk and copy.name == "ep.zh.srt"  # keeps extension + language tag
    assert copy.read_text(encoding="utf-8").endswith("你好，世界\n")
    assert len(temps) == 1 and copy.parent == temps[0]


def test_pack_feeds_ffmpeg_utf8_copy_of_gbk_subtitle(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    sub = tmp_path / "ep.zh.srt"
    sub.write_bytes(_GBK_SRT)
    seen = {}
    monkeypatch.setattr(
        mux, "probe_streams", lambda _m: [{"codec_type": "video", "index": 0}]
    )

    def fake_run(cmd, *, capture):
        sub_in = Path(cmd[cmd.index("-i", cmd.index("-i") + 1) + 1])
        seen["path"] = sub_in
        seen["text"] = sub_in.read_text(encoding="utf-8")
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"packed")

    monkeypatch.setattr(mux, "_run_ffmpeg", fake_run)
    out = mux.pack([sub], container="mkv", output=tmp_path / "out.mkv")
    assert out.read_bytes() == b"packed"
    assert seen["path"] != sub and "你好，世界" in seen["text"]
    assert "language=chi" in seen["cmd"]  # tag still read from the copy's name
    assert not seen["path"].exists()  # temp copy cleaned up


def test_burn_native_ass_non_utf8_uses_utf8_copy(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    ass = tmp_path / "ep.ass"
    ass.write_bytes(
        (
            "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
            " Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,你好\n"
        ).encode("gbk")
    )
    seen = {}
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: [
            {
                "index": 0,
                "codec_type": "video",
                "width": 64,
                "height": 64,
                "disposition": {"attached_pic": 1},
            },
            {"index": 1, "codec_type": "video", "width": 1280, "height": 720},
        ],
    )
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, force=None: "libx264")

    def fake_run(cmd, *, capture):
        vf = cmd[cmd.index("-vf") + 1]
        path = Path(vf.removeprefix("ass=").split(",format=", 1)[0])
        seen["path"] = path
        seen["text"] = path.read_text(encoding="utf-8")
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"burned")

    monkeypatch.setattr(mux, "_run_ffmpeg", fake_run)
    mux.burn(ass, output=tmp_path / "out.mkv", container="mkv", bitrate_cap=False)
    assert seen["path"] != ass and "你好" in seen["text"]
    assert not seen["path"].exists()
    # the measured (non-cover-art) stream is the one encoded
    cmd = seen["cmd"]
    assert cmd[cmd.index("-map") + 1] == "0:1"


# --- translate ---------------------------------------------------------------


def _fake_chat_client(contents):
    from types import SimpleNamespace

    calls: list = []
    queue = list(contents)

    def create(*, model, messages, **kw):
        calls.append(messages)
        msg = SimpleNamespace(content=queue.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


def test_translate_context_tail_zero_sends_no_previous_window():
    from voxweave import translate

    payload = [{"i": 0, "t": "c0"}, {"i": 1, "t": "c1"}, {"i": 2, "t": "c2"}]
    client, calls = _fake_chat_client(
        [
            '{"translations":[{"i":0,"t":"PREVA"},{"i":1,"t":"PREVB"}]}',
            '{"translations":[{"i":2,"t":"C"}]}',
        ]
    )
    out = translate.translate_cues(
        payload, to="zh", model="m", client=client, batch=2, context_tail=0
    )
    assert out == {0: "PREVA", 1: "PREVB", 2: "C"}
    second = " ".join(m["content"] for m in calls[1])
    assert "PREVA" not in second and "PREVB" not in second


def test_partial_translation_error_reports_one_based_cue_numbers():
    from voxweave import translate

    err = translate.PartialTranslationError([0, 4], 10)
    assert "2 of 10 cues untranslated" in str(err)
    assert "cue numbers 1, 5" in str(err)


def test_make_client_disables_sdk_retries_and_sets_timeout(monkeypatch):
    import openai

    from voxweave import translate

    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    translate._make_client("http://localhost:1/v1", "k")
    assert seen["max_retries"] == 0
    assert seen["timeout"] == translate.LLM_TIMEOUT_S


def test_translate_env_knob_typo_does_not_crash_import():
    import os
    import subprocess
    import sys

    env = dict(
        os.environ, VOXWEAVE_TRANSLATE_BATCH="80O", VOXWEAVE_LLM_TIMEOUT_S="fast"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from voxweave import translate as t;"
            "print(t.BATCH_THRESHOLD, t.LLM_TIMEOUT_S)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["800", "300.0"]
