"""Regressions for audited P6 alignment-evidence defects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_p6_runtime_ao import _stub_public_route, _write_public_align_episode


def test_qwen_align_accepts_skip_after_aligned_cue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A qwen-skip cue after a qwen-call cue is owned by its skip ordinal."""
    from voxweave import pipeline
    from voxweave.align_evidence import verify_align_evidence

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="qwen-crop")
    (tmp_path / "episode.json").write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "你", "start": 0.0, "end": 0.4},
                    {"text": "好", "start": 0.4, "end": 0.8},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Untimed cues: only the first is routable from word_segments, so the
    # inserted second cue becomes a qwen-skip at delivery index 1, skip ordinal 0.
    vtt_path.write_text("WEBVTT\n\n你好\n\n谢谢\n", encoding="utf-8")
    _stub_public_route(monkeypatch, tmp_path, route="qwen-crop", media_path=media_path)

    assert pipeline.align(vtt_path, media_path=media_path, separate=False) == vtt_path

    aligned = vtt_path.read_text(encoding="utf-8")
    assert "你好" in aligned and "谢谢" in aligned
    verified = verify_align_evidence(vtt_path, explicit_media_path=media_path)
    assert verified.integrity is True
    assert verified.detail_code is None


def _claim(kind: str, owner: int, delivery: int, source: int) -> dict[str, object]:
    return {
        "owner_kind": kind,
        "owner_index": owner,
        "delivery_index": delivery,
        "source_index": source,
    }


def _entry(delivery: int, action: str, call_index: int | None) -> dict[str, object]:
    return {
        "delivery_index": delivery,
        "source_index": delivery,
        "action": action,
        "call_index": call_index,
    }


def test_durable_route_projection_expects_skip_ordinal_not_delivery_index() -> None:
    from voxweave.align_evidence import _project_route_mismatch

    entries = [
        _entry(0, "qwen-call", 0),
        _entry(1, "qwen-skip", None),
        _entry(2, "qwen-call", 1),
        _entry(3, "qwen-skip", None),
    ]
    calls = [{}, {}]
    skips = [{}, {}]
    claims = [
        _claim("call", 0, 0, 0),
        _claim("skip", 0, 1, 1),
        _claim("call", 1, 2, 2),
        _claim("skip", 1, 3, 3),
    ]

    assert _project_route_mismatch(claims, entries, calls, skips) is None

    swapped = [*claims[:3], _claim("skip", 0, 3, 3)]
    assert _project_route_mismatch(swapped, entries, calls, skips) == {
        "kind": "owner-crosslink",
        "observation_index": 3,
        "expected_delivery_index": 3,
        "observed_delivery_index": 3,
    }


@pytest.mark.parametrize(
    "raw",
    (b"\xff{}", b"{", b"[]"),
    ids=("invalid-utf8", "syntax", "non-object"),
)
def test_corrupt_sibling_json_hint_names_real_command_and_warns_about_vtt(
    raw: bytes,
) -> None:
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    with pytest.raises(RuntimeError) as caught:
        decode_sibling_json_snapshot("episode.json", raw)

    message = str(caught.value)
    assert "process" not in message
    assert "restore it from a backup" in message
    assert "`voxweave transcribe MEDIA`" in message
    assert "rewrites the VTT" in message


def test_ass_content_in_vtt_hint_points_at_a_working_export() -> None:
    from voxweave.align_snapshot import decode_subtitle_snapshot

    with pytest.raises(RuntimeError) as caught:
        decode_subtitle_snapshot(
            "episode.vtt", b"[Script Info]\n[Events]\nFormat: Start, End, Text\n"
        )

    message = str(caught.value)
    assert message.startswith("episode.vtt: content is ASS/SSA")
    assert "rename it to episode.ass" in message
    assert "`voxweave export episode.ass -f vtt`" in message
