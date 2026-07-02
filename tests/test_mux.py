# tests/test_mux.py
# pack/burn command construction and helpers: language detection from VTT
# filenames, "VoxWeave <Language>" track titles, sibling media resolution
# (language tag stripped), ffmpeg argv builders (stream mapping, sub codec per
# container, hvc1 tagging, pixel-format/bit-depth policy, mp4 audio fallback),
# and encoder selection. No ffmpeg/ffprobe execution; probed data is injected.
import logging
from pathlib import Path

import pytest

from voxweave import mux
from voxweave import export as export_mod
from voxweave.export import ass_header

VTT_BODY = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n"


def has_seq(cmd, *seq):
    """True when seq appears as a contiguous run inside cmd."""
    n = len(seq)
    return any(tuple(cmd[i : i + n]) == seq for i in range(len(cmd) - n + 1))


class _Proc:
    """Minimal stand-in for subprocess.CompletedProcess (mirrors test_shot_snap.py)."""

    def __init__(self, rc, stdout="", stderr=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


# --- language / title / path helpers ---------------------------------------


def test_detect_subtitle_language():
    assert mux.detect_subtitle_language(Path("ep.zh.vtt")) == "zh"
    assert mux.detect_subtitle_language(Path("ep.en.vtt")) == "en"
    assert mux.detect_subtitle_language(Path("Show S01E01.ja.vtt")) == "ja"
    assert mux.detect_subtitle_language(Path("ep.vtt")) is None
    assert mux.detect_subtitle_language(Path("ep.sdh.vtt")) is None  # not a language
    assert mux.detect_subtitle_language(Path("ep...vtt")) is None  # interior dots
    assert mux.detect_subtitle_language(Path("ep.zh.srt")) == "zh"
    assert mux.detect_subtitle_language(Path("ep.ja.ass")) == "ja"
    assert mux.detect_subtitle_language(Path("ep.ass")) is None


def test_track_title():
    assert mux.track_title("zh") == "VoxWeave Chinese"
    assert mux.track_title("en") == "VoxWeave English"
    assert mux.track_title("ja") == "VoxWeave Japanese"
    assert mux.track_title(None) == "VoxWeave"


def test_resolve_media_sibling_and_language_tag(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"")
    plain = tmp_path / "ep.vtt"
    plain.write_text(VTT_BODY, encoding="utf-8")
    tagged = tmp_path / "ep.zh.vtt"
    tagged.write_text(VTT_BODY, encoding="utf-8")
    assert mux.resolve_media(plain, None) == media
    assert mux.resolve_media(tagged, None) == media  # .zh stripped for lookup
    explicit = tmp_path / "other.mkv"
    explicit.write_bytes(b"")
    assert mux.resolve_media(plain, explicit) == explicit


def test_resolve_media_missing_raises(tmp_path):
    vtt = tmp_path / "lonely.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="--media"):
        mux.resolve_media(vtt, None)


def test_default_output_avoids_overwriting_source(tmp_path):
    media = tmp_path / "ep.mkv"
    assert mux.default_output(media, "mp4", "burn") == tmp_path / "ep.mp4"
    assert mux.default_output(media, "mkv", "pack") == tmp_path / "ep.pack.mkv"


# --- pack command construction ----------------------------------------------


def _streams(*entries):
    return [dict(e, index=i) for i, e in enumerate(entries)]


def test_build_pack_cmd_mkv_keeps_everything():
    streams = _streams(
        {"codec_type": "video", "codec_name": "hevc"},
        {"codec_type": "audio", "codec_name": "flac"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mkv"),
        [Path("ep.zh.vtt")],
        Path("ep.pack.mkv"),
        container="mkv",
        source_streams=streams,
    )
    assert cmd[:2] == ["ffmpeg", "-nostdin"]
    assert ["-map", "0"] == cmd[cmd.index("-map") : cmd.index("-map") + 2]
    assert has_seq(cmd, "-map", "1:0")  # appended VTT
    assert "-c:s:1" in cmd and cmd[cmd.index("-c:s:1") + 1] == "srt"
    assert "-metadata:s:s:1" in cmd
    assert "language=chi" in cmd and "title=VoxWeave Chinese" in cmd
    assert (
        "-disposition:s:1" in cmd
        and cmd[cmd.index("-disposition:s:1") + 1] == "default"
    )
    assert cmd[-1] == "ep.pack.mkv"


def test_build_pack_cmd_mp4_drops_image_subs_tags_hvc1():
    streams = _streams(
        {"codec_type": "video", "codec_name": "hevc"},
        {"codec_type": "audio", "codec_name": "aac"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"codec_type": "subtitle", "codec_name": "subrip"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mp4"),
        [Path("ep.en.vtt")],
        Path("ep.pack.mp4"),
        container="mp4",
        source_streams=streams,
    )
    assert has_seq(cmd, "-map", "0:3")  # text sub kept
    assert "0:2" not in cmd  # PGS dropped for mp4
    assert "-c:s" in cmd and cmd[cmd.index("-c:s") + 1] == "mov_text"
    assert "-tag:v" in cmd and cmd[cmd.index("-tag:v") + 1] == "hvc1"
    # one kept text sub -> new track is s:1
    assert "language=eng" in cmd and "title=VoxWeave English" in cmd
    assert "-disposition:s:1" in cmd


def test_build_pack_cmd_multiple_vtts_index_after_existing():
    streams = _streams(
        {"codec_type": "video", "codec_name": "h264"},
        {"codec_type": "subtitle", "codec_name": "ass"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mkv"),
        [Path("ep.zh.vtt"), Path("ep.ja.vtt")],
        Path("out.mkv"),
        container="mkv",
        source_streams=streams,
    )
    assert "-c:s:1" in cmd and "-c:s:2" in cmd  # after the existing ass track
    assert "title=VoxWeave Chinese" in cmd and "title=VoxWeave Japanese" in cmd
    assert "-disposition:s:1" in cmd  # only the first new track gets default


def test_pack_rejects_plain_text_draft(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\nhello no timestamps\n", encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        mux.pack([vtt])


def test_pack_rejects_unknown_format(tmp_path):
    txt = tmp_path / "ep.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported subtitle format"):
        mux.pack([txt])


def test_burn_rejects_unknown_format(tmp_path):
    txt = tmp_path / "ep.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported subtitle format"):
        mux.burn(txt)


def test_pack_accepts_srt_past_the_gate(tmp_path):
    # .srt passes the format gate and the timed-cue check; it fails later only
    # because there is no sibling media in tmp_path.
    srt = tmp_path / "ep.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no sibling media"):
        mux.pack([srt])


def test_burn_accepts_ass_past_the_gate(tmp_path):
    ass = tmp_path / "ep.ass"
    ass.write_text(
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR,"
        " MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,hi\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="no sibling media"):
        mux.burn(ass)


def test_build_pack_cmd_keeps_ass_codec_in_mkv():
    streams = _streams({"codec_type": "video", "codec_name": "h264"})
    cmd = mux.build_pack_cmd(
        Path("ep.mkv"),
        [Path("ep.zh.ass"), Path("ep.ja.srt")],
        Path("out.mkv"),
        container="mkv",
        source_streams=streams,
    )
    i_ass = cmd.index("-c:s:0")
    i_srt = cmd.index("-c:s:1")
    assert cmd[i_ass + 1] == "ass"  # ASS input kept native in mkv
    assert cmd[i_srt + 1] == "srt"


# --- pack container/codec compatibility --------------------------------------


def test_pack_rejects_incompatible_video_codec_for_webm(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: _streams(
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "opus"},
        ),
    )
    called = []
    monkeypatch.setattr(mux, "_run_ffmpeg", lambda cmd, *, capture: called.append(cmd))
    with pytest.raises(ValueError) as exc:
        mux.pack([vtt], container="webm")
    assert not called, "ffmpeg must not run before the codec/container check"
    msg = str(exc.value).lower()
    assert "h264" in msg and "webm" in msg and "mkv" in msg


def test_pack_rejects_incompatible_audio_codec_for_webm(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: _streams(
            {"codec_type": "video", "codec_name": "vp9"},
            {"codec_type": "audio", "codec_name": "aac"},
        ),
    )
    called = []
    monkeypatch.setattr(mux, "_run_ffmpeg", lambda cmd, *, capture: called.append(cmd))
    with pytest.raises(ValueError) as exc:
        mux.pack([vtt], container="webm")
    assert not called, "ffmpeg must not run before the codec/container check"
    msg = str(exc.value).lower()
    assert "aac" in msg and "webm" in msg and "mkv" in msg


def test_pack_rejects_incompatible_video_codec_for_mp4(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: _streams(
            {"codec_type": "video", "codec_name": "prores"},
            {"codec_type": "audio", "codec_name": "aac"},
        ),
    )
    called = []
    monkeypatch.setattr(mux, "_run_ffmpeg", lambda cmd, *, capture: called.append(cmd))
    with pytest.raises(ValueError) as exc:
        mux.pack([vtt], container="mp4")
    assert not called, "ffmpeg must not run before the codec/container check"
    msg = str(exc.value).lower()
    assert "prores" in msg and "mp4" in msg and "mkv" in msg


# --- burn building blocks ----------------------------------------------------


def test_src_bit_depth_parsing():
    # bits_per_raw_sample is authoritative when present
    assert mux.src_bit_depth({"pix_fmt": "yuv420p", "bits_per_raw_sample": "10"}) == 10
    # otherwise the trailing endianness-suffixed digits of pix_fmt
    assert mux.src_bit_depth({"pix_fmt": "yuv420p10le"}) == 10
    assert mux.src_bit_depth({"pix_fmt": "yuv420p12le"}) == 12
    assert mux.src_bit_depth({"pix_fmt": "p010le"}) == 10
    assert mux.src_bit_depth({"pix_fmt": "yuv420p"}) == 8
    assert mux.src_bit_depth({"pix_fmt": "nv12"}) == 8  # 12 is layout, not depth
    assert mux.src_bit_depth({}) == 8


def test_src_bit_depth_recognizes_suffixless_pix_fmt():
    # some ffprobe builds report pix_fmt without the le/be endianness suffix;
    # the trailing digits still encode the per-component depth.
    assert mux.src_bit_depth({"pix_fmt": "yuv420p10"}) == 10
    assert mux.src_bit_depth({"pix_fmt": "gray16"}) == 16
    # existing suffixed/plain behavior must remain unchanged
    assert mux.src_bit_depth({"pix_fmt": "yuv420p10le"}) == 10
    assert mux.src_bit_depth({"pix_fmt": "yuv420p"}) == 8


def test_burn_pix_fmt_matches_source_depth():
    # depth follows the source, clamped to encoder capability
    assert mux._burn_pix_fmt("hevc_nvenc", 10) == "p010le"
    assert mux._burn_pix_fmt("hevc_nvenc", 8) == "nv12"
    assert mux._burn_pix_fmt("hevc_nvenc", 12) == "p010le"  # NVENC tops out at 10
    assert mux._burn_pix_fmt("av1_nvenc", 10) == "p010le"
    assert mux._burn_pix_fmt("h264_nvenc", 10) == "nv12"  # h264 is always 8-bit
    assert mux._burn_pix_fmt("hevc_videotoolbox", 10) == "p010le"
    assert mux._burn_pix_fmt("libx265", 10) == "yuv420p10le"
    assert mux._burn_pix_fmt("libx265", 12) == "yuv420p12le"  # x265 keeps 12-bit
    assert mux._burn_pix_fmt("libx265", 8) == "yuv420p"
    assert mux._burn_pix_fmt("libsvtav1", 12) == "yuv420p10le"  # svt clamps to 10
    assert mux._burn_pix_fmt("libx264", 10) == "yuv420p"


def test_encoder_args_constant_quality_only():
    nv = mux._encoder_args("hevc_nvenc", 23)
    assert has_seq(nv, "-rc", "vbr") and has_seq(nv, "-cq", "23")
    assert has_seq(nv, "-b:v", "0")  # pure CQ, no bitrate target
    assert has_seq(nv, "-b_ref_mode", "middle") and has_seq(nv, "-temporal-aq", "1")
    assert mux._encoder_args("hevc_videotoolbox", 65) == ["-q:v", "65"]
    assert has_seq(mux._encoder_args("libx265", 23), "-crf", "23")
    assert has_seq(mux._encoder_args("libsvtav1", 30), "-preset", "6")


def test_filter_escape():
    assert mux._filter_escape("/tmp/a.ass") == "/tmp/a.ass"
    assert mux._filter_escape("/tmp/a:b,c.ass") == "/tmp/a\\:b\\,c.ass"
    assert mux._filter_escape("C:\\tmp\\a.ass") == "C\\:/tmp/a.ass"


def test_build_burn_cmd_hevc_mp4():
    cmd = mux.build_burn_cmd(
        Path("ep.mkv"),
        Path("/tmp/s.ass"),
        Path("ep.mp4"),
        encoder="hevc_nvenc",
        quality=23,
        container="mp4",
        src_depth=10,
        audio_codecs=["flac"],
    )
    assert cmd[:2] == ["ffmpeg", "-nostdin"]
    assert has_seq(cmd, "-hwaccel", "cuda")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "ass=/tmp/s.ass,format=p010le"
    assert has_seq(cmd, "-map", "0:v:0") and has_seq(cmd, "-map", "0:a?")
    assert "-sn" not in cmd  # subs dropped by explicit mapping, not -sn
    assert has_seq(cmd, "-tag:v", "hvc1")
    assert has_seq(cmd, "-c:a", "copy")  # mp4 muxer stores flac natively
    assert cmd[-1] == "ep.mp4"


def test_build_burn_cmd_mkv_copies_audio_no_tag():
    cmd = mux.build_burn_cmd(
        Path("ep.mkv"),
        Path("/tmp/s.ass"),
        Path("ep.burn.mkv"),
        encoder="libx264",
        quality=19,
        container="mkv",
        src_depth=8,
        audio_codecs=["flac"],
    )
    assert "-hwaccel" not in cmd  # software path
    assert has_seq(cmd, "-c:a", "copy")
    assert "-tag:v" not in cmd
    assert "format=yuv420p" in cmd[cmd.index("-vf") + 1]


def test_build_burn_cmd_copies_flac_opus_dts_into_mp4():
    # mp4's mov/mp4 muxer stores flac/opus/dts natively; only genuinely
    # incompatible codecs should fall back to an AAC re-encode.
    for codec in ("flac", "opus", "dts"):
        cmd = mux.build_burn_cmd(
            Path("ep.mkv"),
            Path("/tmp/s.ass"),
            Path("ep.mp4"),
            encoder="hevc_nvenc",
            quality=23,
            container="mp4",
            src_depth=8,
            audio_codecs=[codec],
        )
        assert has_seq(cmd, "-c:a", "copy"), f"{codec} should stream-copy into mp4"


# --- encoder-probe caching ----------------------------------------------------


def test_available_encoders_warns_and_is_not_cached_on_ffmpeg_failure(
    monkeypatch, caplog
):
    monkeypatch.setattr(mux, "_ENCODER_CACHE", None)
    monkeypatch.setattr(
        mux.subprocess,
        "run",
        lambda *a, **k: _Proc(1, stderr="ffmpeg: error while loading shared libraries"),
    )
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = mux._available_encoders()
    assert result == frozenset()
    assert any("ffmpeg" in rec.message.lower() for rec in caplog.records)

    # a later, healthy call must not be stuck with the empty cached result
    monkeypatch.setattr(
        mux.subprocess,
        "run",
        lambda *a, **k: _Proc(0, stdout="Encoders:\n V..... libx264   H.264 / AVC\n"),
    )
    result2 = mux._available_encoders()
    assert "libx264" in result2


def test_encoder_works_reprobes_after_negative_but_caches_positive(monkeypatch):
    monkeypatch.setattr(mux, "_ENCODER_WORKS", {})
    calls = []

    def fake_fail(*a, **k):
        calls.append("probe")
        return _Proc(1)

    monkeypatch.setattr(mux.subprocess, "run", fake_fail)
    assert mux._encoder_works("hevc_nvenc") is False
    assert len(calls) == 1

    def fake_ok(*a, **k):
        calls.append("probe")
        return _Proc(0)

    monkeypatch.setattr(mux.subprocess, "run", fake_ok)
    assert mux._encoder_works("hevc_nvenc") is True  # negative result must be re-probed
    assert len(calls) == 2

    assert mux._encoder_works("hevc_nvenc") is True  # positive result stays cached
    assert len(calls) == 2  # no extra subprocess call


def test_pick_encoder_prefers_hardware(monkeypatch):
    monkeypatch.setattr(
        mux, "_available_encoders", lambda: frozenset({"hevc_nvenc", "libx265"})
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: True)
    monkeypatch.setattr(mux.sys, "platform", "linux")
    assert mux.pick_encoder("hevc") == "hevc_nvenc"


def test_pick_encoder_falls_back_to_software(monkeypatch):
    monkeypatch.setattr(
        mux, "_available_encoders", lambda: frozenset({"hevc_nvenc", "libx265"})
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: False)  # no GPU
    monkeypatch.setattr(mux.sys, "platform", "linux")
    assert mux.pick_encoder("hevc") == "libx265"


def test_pick_encoder_videotoolbox_on_darwin(monkeypatch):
    monkeypatch.setattr(
        mux,
        "_available_encoders",
        lambda: frozenset({"hevc_videotoolbox", "libx265"}),
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: True)
    monkeypatch.setattr(mux.sys, "platform", "darwin")
    assert mux.pick_encoder("hevc") == "hevc_videotoolbox"


def test_pick_encoder_force_and_unknown_codec():
    assert mux.pick_encoder("hevc", force="libx265") == "libx265"
    with pytest.raises(ValueError, match="unsupported codec"):
        mux.pick_encoder("vp9")


# --- ASS header scaling (burn renders at the actual frame size) --------------


def test_ass_header_scales_to_frame():
    h = ass_header(width=3840, height=2160)
    assert "PlayResX: 3840" in h and "PlayResY: 2160" in h
    assert ",144," in h  # 72 * 2 font size
    h = ass_header(width=1920, height=1080, font="Noto Sans CJK SC", font_size=58)
    assert "Style: Default,Noto Sans CJK SC,58," in h


# --- atomic output ----------------------------------------------------------


def _fake_ffmpeg_dies_mid_write(cmd, *, capture):
    """Simulate ffmpeg crashing after partially writing its output file
    (last argv element is always the output path)."""
    Path(cmd[-1]).write_bytes(b"half-written garbage")
    raise RuntimeError("ffmpeg failed (exit 1)")


def test_pack_failure_leaves_no_output(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(mux, "probe_streams", lambda _m: [])
    monkeypatch.setattr(mux, "_run_ffmpeg", _fake_ffmpeg_dies_mid_write)
    out = tmp_path / "ep.pack.mkv"
    with pytest.raises(RuntimeError):
        mux.pack([vtt], output=out)
    assert not out.exists()
    assert not list(tmp_path.glob("*.part*"))  # temp cleaned up


def test_pack_failure_preserves_previous_output(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    out = tmp_path / "ep.pack.mkv"
    out.write_bytes(b"good output from yesterday")
    monkeypatch.setattr(mux, "probe_streams", lambda _m: [])
    monkeypatch.setattr(mux, "_run_ffmpeg", _fake_ffmpeg_dies_mid_write)
    with pytest.raises(RuntimeError):
        mux.pack([vtt], output=out)
    assert out.read_bytes() == b"good output from yesterday"


def test_pack_success_lands_at_output(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")

    def fake_ok(cmd, *, capture):
        Path(cmd[-1]).write_bytes(b"muxed")

    monkeypatch.setattr(mux, "probe_streams", lambda _m: [])
    monkeypatch.setattr(mux, "_run_ffmpeg", fake_ok)
    out = mux.pack([vtt])
    assert out == tmp_path / "ep.pack.mkv"
    assert out.read_bytes() == b"muxed"
    assert not list(tmp_path.glob("*.part*"))


def test_burn_failure_leaves_no_output(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: [
            {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}
        ],
    )
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, force=None: "libx264")
    monkeypatch.setattr(mux, "_run_ffmpeg", _fake_ffmpeg_dies_mid_write)
    out = tmp_path / "ep.burn.mp4"
    with pytest.raises(RuntimeError):
        mux.burn(vtt, output=out)
    assert not out.exists()
    assert not list(tmp_path.glob("*.part*"))


# --- output must never be the source media ----------------------------------


def test_pack_rejects_output_equal_to_media(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.zh.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(mux, "probe_streams", lambda _m: [])
    monkeypatch.setattr(mux, "_run_ffmpeg", lambda cmd, *, capture: None)
    with pytest.raises(ValueError, match="source media"):
        mux.pack([vtt], output=media)
    assert media.read_bytes() == b"src"


def test_burn_rejects_output_equal_to_media(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: [
            {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}
        ],
    )
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, force=None: "libx264")
    monkeypatch.setattr(mux, "_run_ffmpeg", lambda cmd, *, capture: None)
    with pytest.raises(ValueError, match="source media"):
        mux.burn(vtt, output=media, container="mkv")
    assert media.read_bytes() == b"src"


# --- burn source-stream selection & diagnostics ------------------------------


def test_burn_warns_when_probed_resolution_is_zero(tmp_path, monkeypatch, caplog):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda _m: [
            {"codec_type": "video", "codec_name": "h264", "width": 0, "height": 0}
        ],
    )
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, force=None: "libx264")
    monkeypatch.setattr(
        mux, "_run_ffmpeg", lambda cmd, *, capture: Path(cmd[-1]).write_bytes(b"x")
    )
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        mux.burn(vtt, output=tmp_path / "out.mkv", container="mkv")
    assert any("resolution" in rec.message.lower() for rec in caplog.records)


def test_burn_skips_attached_pic_cover_art_stream(tmp_path, monkeypatch):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"src")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    streams = [
        {
            "codec_type": "video",
            "codec_name": "mjpeg",
            "width": 300,
            "height": 300,
            "disposition": {"attached_pic": 1},
        },
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "disposition": {"attached_pic": 0},
        },
    ]
    monkeypatch.setattr(mux, "probe_streams", lambda _m: streams)
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, force=None: "libx264")
    monkeypatch.setattr(
        mux, "_run_ffmpeg", lambda cmd, *, capture: Path(cmd[-1]).write_bytes(b"x")
    )
    captured = {}
    real_ass_header = export_mod.ass_header

    def spy_header(**kwargs):
        captured.update(kwargs)
        return real_ass_header(**kwargs)

    monkeypatch.setattr(export_mod, "ass_header", spy_header)
    mux.burn(vtt, output=tmp_path / "out.mkv", container="mkv")
    assert captured.get("width") == 1920  # not the 300x300 attached_pic cover
    assert captured.get("height") == 1080


# --- ffmpeg execution & probing robustness -----------------------------------


def test_run_ffmpeg_error_surfaces_lines_beyond_the_last_8(monkeypatch):
    stderr_lines = ["Error: width not divisible by 2"] + [
        f"info line {i}" for i in range(40)
    ]
    monkeypatch.setattr(
        mux.subprocess,
        "run",
        lambda *a, **k: _Proc(1, stderr="\n".join(stderr_lines)),
    )
    with pytest.raises(RuntimeError) as exc:
        mux._run_ffmpeg(["ffmpeg", "-i", "x"], capture=True)
    assert "divisible" in str(exc.value)


def test_probe_streams_missing_ffprobe_raises_friendly_error(monkeypatch):
    def raise_fnf(*a, **k):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'ffprobe'")

    monkeypatch.setattr(mux.subprocess, "run", raise_fnf)
    with pytest.raises(RuntimeError) as exc:
        mux.probe_streams(Path("ep.mkv"))
    msg = str(exc.value).lower()
    assert "ffmpeg" in msg
    assert "path" in msg or "install" in msg


def test_run_ffmpeg_missing_binary_raises_friendly_error(monkeypatch):
    def raise_fnf(*a, **k):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'ffmpeg'")

    monkeypatch.setattr(mux.subprocess, "run", raise_fnf)
    with pytest.raises(RuntimeError) as exc:
        mux._run_ffmpeg(["ffmpeg", "-i", "x"], capture=True)
    msg = str(exc.value).lower()
    assert "ffmpeg" in msg
    assert "path" in msg or "install" in msg
