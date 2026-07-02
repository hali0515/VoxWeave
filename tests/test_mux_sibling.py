"""RED tests for #26 (mux side): sibling media lookup must be case-insensitive
and must warn (deterministically) when multiple candidate media files exist.

mux.resolve_media delegates to pipeline._find_sibling_media, so these exercise
the same fix from the mux entry point used by pack/burn.
"""

import logging

from voxweave import mux


def test_resolve_media_finds_uppercase_extension_sibling(tmp_path):
    media = tmp_path / "ep.MP4"
    media.write_bytes(b"x")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n", encoding="utf-8")
    assert mux.resolve_media(vtt, None) == media


def test_resolve_media_multiple_candidates_warns_and_is_deterministic(tmp_path, caplog):
    # ep.mkv and ep.mp4 both exist; MEDIA_EXTS lists ".mkv" before ".mp4", so the
    # first-by-order candidate must win, and the ambiguity must be logged.
    mkv = tmp_path / "ep.mkv"
    mkv.write_bytes(b"x")
    mp4 = tmp_path / "ep.mp4"
    mp4.write_bytes(b"x")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n", encoding="utf-8")
    # the warning is emitted from pipeline._find_sibling_media, which logs on
    # the shared "voxweave" logger (not "voxweave.mux").
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        found = mux.resolve_media(vtt, None)
    assert found == mkv
    assert any("multiple" in r.getMessage().lower() for r in caplog.records)
