"""3-way segmentation calibration harness.

Compares:
  OLD  -- legacy smart_split_segments (no thresholds, length-only)
  NEW  -- gap-aware smart_split_segments (thresholds=gap_thresholds(iso),
          speech_spans from the JSON's vad_speech when present)
  EN   -- commercial English subtitle stream embedded in the MKV (ASS format),
          speech styles only (signs/credits excluded)

Run:
    uv run python scripts/calib_segmentation.py [VIDEO_DIR]

Notes:
  - JSONs without vad_speech degrade NEW to the offline_ms threshold
    (700ms * 1.4 for ja = 980ms).
  - Mid-phrase-cut rate uses phrase_atoms (jieba zh / BudouX ja); offsets are
    computed over the punctuation-stripped cue stream, never word_segments text.
  - bad line-end % counts internal boundaries whose left cue ends on a
    forward-binding token (line_end_penalty >= 2). Sentence-final function
    words (begin with / lock in) and real >=vad_skip pauses keep an absolute
    floor — read it as a relative gauge between runs.
  - NEW mid-phrase metric is split into len-break and gap-break halves:
      * len-break mid-phrase % = THE quality gate (BudouX gating signal).
        A boundary with no real acoustic silence that lands inside a phrase
        is a genuine defect the BudouX gate should prevent.
      * gap-break mid-phrase % = informational. The speaker paused here so
        the boundary is acoustically correct regardless of BudouX phrase
        alignment; nonzero is expected.
  - OLD keeps the combined mid-phrase % for before/after contrast.
  - EN column skipped per-episode if ffmpeg fails or no subtitle stream.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# percentile helper (stdlib only)
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of sorted data (linear interpolation)."""
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return s[-1]
    return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# ASS subtitle parser
# ---------------------------------------------------------------------------

_SPEECH_STYLES = {
    "Default",
    "DefaultItalics",
    "DefaultTop",
    "DefaultOverlap",
    "DefaultItalicsTop",
    "Flashback",
    "FlashbackItalics",
    "FlashbackTop",
    "FlashbackItalicsTop",
    "Narration",
}

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")


def _ass_time_to_s(t: str) -> float:
    """Convert ASS time H:MM:SS.cc to seconds (centiseconds precision)."""
    h, m, rest = t.split(":")
    s, cs = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def parse_ass(path: str) -> list[dict[str, Any]]:
    """Parse ASS file and return speech-style dialogue cues as list of
    {start, end, text} dicts. Skips Signs / Credits / overlap (end<=start)."""
    cues: list[dict[str, Any]] = []
    format_fields: list[str] = []
    in_events = False

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip().startswith("[Events]"):
                in_events = True
                continue
            if line.strip().startswith("[") and in_events:
                break  # past Events section
            if not in_events:
                continue
            if line.startswith("Format:"):
                # Format: Layer, Start, End, Style, Name, ...
                format_fields = [x.strip() for x in line[len("Format:") :].split(",")]
                continue
            if not line.startswith("Dialogue:"):
                continue
            if not format_fields:
                continue
            # Split only up to len(format_fields) fields; Text may contain commas
            parts = line[len("Dialogue:") :].lstrip().split(",", len(format_fields) - 1)
            if len(parts) < len(format_fields):
                continue
            row = dict(zip(format_fields, parts))
            style = row.get("Style", "")
            if style not in _SPEECH_STYLES:
                continue
            raw_text = row.get("Text", "")
            # Strip ASS override tags {..}
            text = _ASS_TAG_RE.sub("", raw_text)
            # Convert line-break codes to space
            text = text.replace("\\N", " ").replace("\\n", " ").strip()
            if not text:
                continue
            try:
                start = _ass_time_to_s(row["Start"])
                end = _ass_time_to_s(row["End"])
            except (KeyError, ValueError):
                continue
            if end <= start:
                continue
            cues.append({"start": start, "end": end, "text": text})

    return cues


def extract_commercial_en(mkv: Path) -> list[dict[str, Any]] | None:
    """Extract subtitle stream 0 from MKV as ASS and parse it.
    Returns None if ffmpeg fails or no subtitle stream present."""
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(mkv),
                "-map",
                "0:s:0",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        cues = parse_ass(tmp_path)
        return cues if cues else None
    except FileNotFoundError:
        # ffmpeg not available
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


NETFLIX_MIN_S = 5.0 / 6.0  # Netflix minimum display duration


def _tail_token(text: str, iso: str) -> str:
    """Trailing word of a cue for line-end quality scoring.

    Spaced langs: last whitespace token. No-space langs: last phrase atom
    (jieba word / BudouX phrase) of the last whitespace run — line_end_penalty
    needs whole-word semantics for zh.
    """
    from voxweave.core.breakpoints import phrase_atoms  # noqa: PLC0415

    parts = text.replace("\n", " ").split()
    if not parts:
        return ""
    tail = parts[-1]
    if iso in {"zh", "ja", "th", "lo", "my"}:
        atoms = phrase_atoms(tail, iso)
        if atoms:
            tail = atoms[-1]
    return tail


def forbidden_end_rate(cues: list[dict[str, Any]], iso: str) -> float | None:
    """Share of internal cue boundaries whose left cue ends on a forward-binding
    token (line_end_penalty >= 2: en the/of/and..., zh 的/把/被..., ja のをにへ).

    Punctuation is stripped from cue text, so legitimate sentence-final function
    words inflate this slightly — read it as a relative gauge between runs."""
    from voxweave.core.kinsoku import line_end_penalty  # noqa: PLC0415

    if len(cues) < 2:
        return None
    bad = 0
    for c in cues[:-1]:
        tail = _tail_token(c.get("text", ""), iso)
        if tail and line_end_penalty(tail, iso) >= 2:
            bad += 1
    return bad / (len(cues) - 1)


def cue_metrics(cues: list[dict[str, Any]], iso: str) -> dict[str, Any]:
    """Compute per-episode metrics for a cue list."""
    n = len(cues)
    if n == 0:
        return {
            "n": 0,
            "dur_med": 0.0,
            "dur_p90": 0.0,
            "dur_max": 0.0,
            "over7s": 0,
            "under05s": 0,
            "under_min": 0,
            "cps_med": 0.0,
            "cps_p90": 0.0,
            "bad_end": None,
        }
    durs = [c["end"] - c["start"] for c in cues]
    over7 = sum(1 for d in durs if d > 7.0)
    under05 = sum(1 for d in durs if d < 0.5)
    under_min = sum(1 for d in durs if d < NETFLIX_MIN_S)
    # CPS: chars per second; for CJK, len = char count (no spaces)
    cps_vals: list[float] = []
    for c, d in zip(cues, durs):
        if d > 0:
            text = c.get("text", "")
            nc = len(text.replace(" ", "").replace("\n", ""))
            cps_vals.append(nc / d)

    return {
        "n": n,
        "dur_med": percentile(durs, 50),
        "dur_p90": percentile(durs, 90),
        "dur_max": max(durs),
        "over7s": over7,
        "under05s": under05,
        "under_min": under_min,
        "cps_med": percentile(cps_vals, 50) if cps_vals else 0.0,
        "cps_p90": percentile(cps_vals, 90) if cps_vals else 0.0,
        "bad_end": forbidden_end_rate(cues, iso),
    }


def _cue_stream_text(cues: list[dict[str, Any]]) -> str:
    """Concatenated cue text with whitespace removed.

    Phrase-boundary offsets MUST be computed over this exact stream: cue text has
    punctuation stripped to spaces, so offsets derived from the punctuation-bearing
    word_segments text would desync at every former punctuation mark."""
    return "".join(c.get("text", "").replace(" ", "").replace("\n", "") for c in cues)


def mid_phrase_cut_rate(
    cues: list[dict[str, Any]], iso: str, full_text_nospace: str | None = None
) -> float | None:
    """Return combined mid-phrase-cut rate [0,1] for CJK cues using BudouX.

    Used for OLD (legacy) column — single number for before/after contrast.
    Returns None if iso is not CJK or no atoms.
    """
    from voxweave.core.breakpoints import phrase_atoms  # noqa: PLC0415

    no_space_langs = {"zh", "ja", "th", "lo", "my"}
    if iso not in no_space_langs:
        return None

    full_text_nospace = _cue_stream_text(cues)
    if not full_text_nospace:
        return None

    atoms = phrase_atoms(full_text_nospace, iso)
    if not atoms:
        return None

    phrase_start_offsets: set[int] = set()
    c = 0
    for atom in atoms:
        phrase_start_offsets.add(c)
        c += len(atom)

    boundary_offsets: list[int] = []
    c = 0
    for i, cue in enumerate(cues):
        text_ns = cue.get("text", "").replace(" ", "").replace("\n", "")
        c += len(text_ns)
        if i < len(cues) - 1:
            boundary_offsets.append(c)

    if not boundary_offsets:
        return 0.0

    bad = sum(1 for off in boundary_offsets if off not in phrase_start_offsets)
    return bad / len(boundary_offsets)


def mid_phrase_cut_split(
    cues: list[dict[str, Any]],
    iso: str,
    full_text_nospace: str | None,
    offline_s: float,
) -> dict[str, Any] | None:
    """Classify each internal boundary as gap-break or len-break, then measure
    mid-phrase rate separately for each class.

    Classification rule for boundary between cue[i] and cue[i+1]:
      raw_gap = cue[i+1].word_data[0].start - cue[i].word_data[-1].end
      raw_gap >= offline_s  →  gap-break (acoustic silence)
      raw_gap <  offline_s  →  len-break (length/duration forced)

    A boundary is mid-phrase if its cumulative char offset (spaces removed)
    does NOT coincide with a BudouX phrase-start offset.

    Returns dict with keys:
      len_mid_pct    float  len-break mid-phrase %  ← quality gate
      gap_mid_pct    float  gap-break mid-phrase %  ← informational
      len_share_pct  float  len-breaks / all internal boundaries %
      n_len          int    total len-breaks
      n_gap          int    total gap-breaks
      n_len_mid      int    mid-phrase len-breaks
      n_gap_mid      int    mid-phrase gap-breaks

    Returns None if iso not in no_space_langs or no boundaries.
    """
    from voxweave.core.breakpoints import phrase_atoms  # noqa: PLC0415

    no_space_langs = {"zh", "ja", "th", "lo", "my"}
    if iso not in no_space_langs:
        return None

    full_text_nospace = _cue_stream_text(cues)
    if not full_text_nospace or len(cues) < 2:
        return None

    atoms = phrase_atoms(full_text_nospace, iso)
    if not atoms:
        return None

    # Build phrase-start offsets
    phrase_start_offsets: set[int] = set()
    c = 0
    for atom in atoms:
        phrase_start_offsets.add(c)
        c += len(atom)

    # Walk boundaries
    n_len = n_gap = n_len_mid = n_gap_mid = 0
    char_offset = 0

    for i in range(len(cues) - 1):
        cur = cues[i]
        nxt = cues[i + 1]

        # Accumulate char offset at end of cue[i]
        text_ns = cur.get("text", "").replace(" ", "").replace("\n", "")
        char_offset += len(text_ns)

        is_mid = char_offset not in phrase_start_offsets

        # Classify via raw word_data gap
        wd_cur = cur.get("word_data") or []
        wd_nxt = nxt.get("word_data") or []

        if not wd_cur or not wd_nxt:
            # Cannot classify without timing — skip this boundary
            continue

        # Guard: keys may be missing
        end_cur = wd_cur[-1].get("end")
        start_nxt = wd_nxt[0].get("start")
        if end_cur is None or start_nxt is None:
            continue

        raw_gap = start_nxt - end_cur

        if raw_gap >= offline_s:
            n_gap += 1
            if is_mid:
                n_gap_mid += 1
        else:
            n_len += 1
            if is_mid:
                n_len_mid += 1

    n_total = n_len + n_gap
    if n_total == 0:
        return None

    return {
        "len_mid_pct": (n_len_mid / n_len * 100) if n_len > 0 else 0.0,
        "gap_mid_pct": (n_gap_mid / n_gap * 100) if n_gap > 0 else 0.0,
        "len_share_pct": n_len / n_total * 100,
        "n_len": n_len,
        "n_gap": n_gap,
        "n_len_mid": n_len_mid,
        "n_gap_mid": n_gap_mid,
    }


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------


def build_segment(word_segments: list[dict], iso: str) -> dict[str, Any]:
    """Build a single segment dict from word_segments for smart_split_segments."""
    sep = "" if iso in {"zh", "ja", "yue"} else " "
    text = sep.join(u["text"] for u in word_segments)
    words = [
        {"word": u["text"], "start": u["start"], "end": u["end"]} for u in word_segments
    ]
    return {
        "start": word_segments[0]["start"],
        "end": word_segments[-1]["end"],
        "text": text,
        "words": words,
    }


def run_episode(
    mkv: Path,
    json_path: Path,
) -> dict[str, Any]:
    """Process one episode. Returns dict with keys old/new/en metrics + errors."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    word_segments = data.get("word_segments") or []
    iso = data.get("language", "ja")

    if not word_segments:
        return {"error": "empty word_segments"}

    from voxweave import realign  # noqa: PLC0415
    from voxweave.core.smart_split import smart_split_segments  # noqa: PLC0415
    from voxweave.config import gap_thresholds  # noqa: PLC0415

    # Mirror the production path: zh punctuation snapped to jieba word boundaries.
    word_segments = realign.snap_break_punct(word_segments, iso)
    seg = build_segment(word_segments, iso)

    # vad_speech persisted by transcribe (newer JSONs); older JSONs degrade to offline_ms.
    speech_spans = data.get("vad_speech") or None
    if speech_spans:
        speech_spans = [(float(s), float(e)) for s, e in speech_spans]

    # OLD: no thresholds, legacy length-only path
    old_cues = smart_split_segments([seg], iso)

    # NEW: gap-aware
    th = gap_thresholds(iso)
    new_cues = smart_split_segments(
        [seg],
        iso,
        speech_spans=speech_spans,
        thresholds=th,
    )

    # Commercial EN
    en_cues = extract_commercial_en(mkv)

    # Full text no-space for mid-phrase-cut (use word_segments concatenation)
    full_text_nospace = "".join(u["text"] for u in word_segments)

    old_m = cue_metrics(old_cues, iso)
    new_m = cue_metrics(new_cues, iso)
    en_m = cue_metrics(en_cues, "en") if en_cues is not None else None

    # OLD: combined mid-phrase-cut (before/after contrast)
    old_mpc = mid_phrase_cut_rate(old_cues, iso, full_text_nospace)

    # NEW: split by boundary type
    offline_s = th["offline_ms"] / 1000.0
    new_mpc_split = mid_phrase_cut_split(new_cues, iso, full_text_nospace, offline_s)

    return {
        "iso": iso,
        "old": old_m,
        "new": new_m,
        "en": en_m,
        "old_mpc": old_mpc,
        "new_mpc_split": new_mpc_split,
        "en_available": en_cues is not None,
    }


def fmt_mpc(v: float | None) -> str:
    if v is None:
        return "   N/A"
    return f"{v * 100:5.1f}%"


def fmt_f(v: float) -> str:
    return f"{v:5.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:5.1f}%"


def print_episode_table(name: str, r: dict[str, Any]) -> None:
    old = r["old"]
    new = r["new"]
    en = r.get("en")
    split = r.get("new_mpc_split")  # dict or None

    en_str = "(no EN stream)" if en is None else ""

    print(f"\n{'─' * 70}")
    print(f"  Episode: {name}  {en_str}")
    print(f"{'─' * 70}")
    hdr = f"  {'Metric':<28} {'OLD':>10} {'NEW':>10} {'EN (cmcl)':>12}"
    print(hdr)
    print(f"  {'─' * 28} {'─' * 10} {'─' * 10} {'─' * 12}")

    def row(label: str, key: str, fmt_fn=fmt_f) -> None:
        o = fmt_fn(old.get(key, 0))
        n = fmt_fn(new.get(key, 0))
        e = fmt_fn(en.get(key, 0)) if en else "        N/A"
        print(f"  {label:<28} {o:>10} {n:>10} {e:>12}")

    def row_int(label: str, key: str) -> None:
        o = str(old.get(key, 0))
        n = str(new.get(key, 0))
        e = str(en.get(key, 0)) if en else "N/A"
        print(f"  {label:<28} {o:>10} {n:>10} {e:>12}")

    row_int("n (cue count)", "n")
    row("dur median (s)", "dur_med")
    row("dur p90 (s)", "dur_p90")
    row("dur max (s)", "dur_max")
    row_int(">7s cues", "over7s")
    row_int("<0.5s cues", "under05s")
    row_int("<5/6s cues", "under_min")
    row("CPS median", "cps_med")
    row("CPS p90", "cps_p90")
    o_be = fmt_mpc(old.get("bad_end"))
    n_be = fmt_mpc(new.get("bad_end"))
    e_be = fmt_mpc(en.get("bad_end")) if en else "         N/A"
    print(f"  {'bad line-end %':<28} {o_be:>10} {n_be:>10} {e_be:>12}")

    print(f"  {'─' * 28} {'─' * 10} {'─' * 10} {'─' * 12}")
    # OLD: combined mid-phrase-cut %
    print(
        f"  {'mid-phrase-cut % (OLD)':<28} {fmt_mpc(r.get('old_mpc')):>10}"
        f" {'':>10} {'':>12}"
    )
    # NEW: split metrics
    if split is not None:
        print(
            f"  {'  len-break mid-phrase %':<28} {'':>10}"
            f" {fmt_pct(split['len_mid_pct']):>10} {'':>12}"
            f"  <- quality gate"
        )
        print(
            f"  {'  gap-break mid-phrase %':<28} {'':>10}"
            f" {fmt_pct(split['gap_mid_pct']):>10} {'':>12}"
            f"  <- informational"
        )
        print(
            f"  {'  len-break share %':<28} {'':>10}"
            f" {fmt_pct(split['len_share_pct']):>10} {'':>12}"
        )
        print(
            f"  {'  (n_len/n_gap/n_len_mid)':<28} {'':>10}"
            f" {split['n_len']}/{split['n_gap']}/{split['n_len_mid']:>3}{'':>6} {'':>12}"
        )
    else:
        print(f"  {'  mid-phrase split':<28} {'':>10} {'   N/A':>10} {'':>12}")


_MEDIA_EXTS = {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".avi", ".ts"}


def _sibling_json(media: Path) -> Path:
    """Sibling .json path, replacing only the trailing extension (never
    Path.with_suffix on interior-dot names — same contract as pipeline._swap_ext)."""
    return media.with_name(media.name[: -len(media.suffix)] + ".json")


def main(video_dir: str) -> None:
    d = Path(video_dir)
    mkv_files = sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _MEDIA_EXTS
    )

    if not mkv_files:
        print(f"No media files found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    print("\nCalibration harness: gap-aware segmentation")
    print(f"Directory : {video_dir}")
    print(f"Episodes  : {len(mkv_files)} media files found")
    print(
        "Note: speech_spans comes from the JSON's vad_speech when present;"
        " older JSONs degrade to the offline_ms path."
    )

    agg: dict[str, list] = {
        "old_n": [],
        "new_n": [],
        "en_n": [],
        "old_over7": [],
        "new_over7": [],
        "en_over7": [],
        "old_under05": [],
        "new_under05": [],
        "en_under05": [],
        "old_dur_med": [],
        "new_dur_med": [],
        "en_dur_med": [],
        "old_cps_med": [],
        "new_cps_med": [],
        "en_cps_med": [],
        "old_bad_end": [],
        "new_bad_end": [],
        "en_bad_end": [],
        "old_under_min": [],
        "new_under_min": [],
        "en_under_min": [],
        "old_mpc": [],
        # NEW split metrics
        "new_len_mid_pct": [],
        "new_gap_mid_pct": [],
        "new_len_share_pct": [],
    }

    errors: list[str] = []

    for mkv in mkv_files:
        json_path = _sibling_json(mkv)
        if not json_path.exists():
            continue

        ep_name = mkv.stem
        try:
            r = run_episode(mkv, json_path)
        except Exception as exc:
            errors.append(f"{ep_name}: {exc!r}")
            print(f"\n  ERROR processing {ep_name}: {exc!r}", file=sys.stderr)
            continue

        if "error" in r:
            errors.append(f"{ep_name}: {r['error']}")
            continue

        print_episode_table(ep_name, r)

        agg["old_n"].append(r["old"]["n"])
        agg["new_n"].append(r["new"]["n"])
        if r["en"]:
            agg["en_n"].append(r["en"]["n"])
        agg["old_over7"].append(r["old"]["over7s"])
        agg["new_over7"].append(r["new"]["over7s"])
        if r["en"]:
            agg["en_over7"].append(r["en"]["over7s"])
        agg["old_under05"].append(r["old"]["under05s"])
        agg["new_under05"].append(r["new"]["under05s"])
        if r["en"]:
            agg["en_under05"].append(r["en"]["under05s"])
        agg["old_dur_med"].append(r["old"]["dur_med"])
        agg["new_dur_med"].append(r["new"]["dur_med"])
        if r["en"]:
            agg["en_dur_med"].append(r["en"]["dur_med"])
        agg["old_cps_med"].append(r["old"]["cps_med"])
        agg["new_cps_med"].append(r["new"]["cps_med"])
        if r["en"]:
            agg["en_cps_med"].append(r["en"]["cps_med"])
        if r["old"].get("bad_end") is not None:
            agg["old_bad_end"].append(r["old"]["bad_end"])
        if r["new"].get("bad_end") is not None:
            agg["new_bad_end"].append(r["new"]["bad_end"])
        if r["en"] and r["en"].get("bad_end") is not None:
            agg["en_bad_end"].append(r["en"]["bad_end"])
        agg["old_under_min"].append(r["old"]["under_min"])
        agg["new_under_min"].append(r["new"]["under_min"])
        if r["en"]:
            agg["en_under_min"].append(r["en"]["under_min"])
        if r.get("old_mpc") is not None:
            agg["old_mpc"].append(r["old_mpc"])
        split = r.get("new_mpc_split")
        if split is not None:
            agg["new_len_mid_pct"].append(split["len_mid_pct"])
            agg["new_gap_mid_pct"].append(split["gap_mid_pct"])
            agg["new_len_share_pct"].append(split["len_share_pct"])

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 70}")

    eps = len(agg["old_n"])
    print(f"  Episodes processed : {eps}")
    print(f"  EN stream available: {len(agg['en_n'])} / {eps} episodes")

    def agg_sum(k: list[int]) -> int:
        return sum(k)

    def agg_mean(k: list[float]) -> float:
        return sum(k) / len(k) if k else 0.0

    print(f"\n  {'Metric':<34} {'OLD':>10} {'NEW':>10} {'EN (cmcl)':>12}")
    print(f"  {'─' * 34} {'─' * 10} {'─' * 10} {'─' * 12}")

    total_old_n = agg_sum(agg["old_n"])
    total_new_n = agg_sum(agg["new_n"])
    total_en_n = agg_sum(agg["en_n"]) if agg["en_n"] else None
    print(
        f"  {'total cues':<34} {total_old_n:>10} {total_new_n:>10}"
        f" {str(total_en_n) if total_en_n else 'N/A':>12}"
    )

    total_old_o7 = agg_sum(agg["old_over7"])
    total_new_o7 = agg_sum(agg["new_over7"])
    total_en_o7 = agg_sum(agg["en_over7"]) if agg["en_over7"] else None
    print(
        f"  {'total >7s cues':<34} {total_old_o7:>10} {total_new_o7:>10}"
        f" {str(total_en_o7) if total_en_o7 is not None else 'N/A':>12}"
    )

    total_old_u05 = agg_sum(agg["old_under05"])
    total_new_u05 = agg_sum(agg["new_under05"])
    total_en_u05 = agg_sum(agg["en_under05"]) if agg["en_under05"] else None
    print(
        f"  {'total <0.5s cues':<34} {total_old_u05:>10} {total_new_u05:>10}"
        f" {str(total_en_u05) if total_en_u05 is not None else 'N/A':>12}"
    )

    mean_old_dur = agg_mean(agg["old_dur_med"])
    mean_new_dur = agg_mean(agg["new_dur_med"])
    mean_en_dur = agg_mean(agg["en_dur_med"]) if agg["en_dur_med"] else None
    print(
        f"  {'mean(dur median) (s)':<34} {mean_old_dur:>10.2f} {mean_new_dur:>10.2f}"
        f" {f'{mean_en_dur:.2f}' if mean_en_dur is not None else 'N/A':>12}"
    )

    mean_old_cps = agg_mean(agg["old_cps_med"])
    mean_new_cps = agg_mean(agg["new_cps_med"])
    mean_en_cps = agg_mean(agg["en_cps_med"]) if agg["en_cps_med"] else None
    print(
        f"  {'mean(CPS median)':<34} {mean_old_cps:>10.2f} {mean_new_cps:>10.2f}"
        f" {f'{mean_en_cps:.2f}' if mean_en_cps is not None else 'N/A':>12}"
    )

    total_old_um = agg_sum(agg["old_under_min"])
    total_new_um = agg_sum(agg["new_under_min"])
    total_en_um = agg_sum(agg["en_under_min"]) if agg["en_under_min"] else None
    print(
        f"  {'total <5/6s cues':<34} {total_old_um:>10} {total_new_um:>10}"
        f" {str(total_en_um) if total_en_um is not None else 'N/A':>12}"
    )

    mean_old_be = agg_mean(agg["old_bad_end"]) * 100 if agg["old_bad_end"] else None
    mean_new_be = agg_mean(agg["new_bad_end"]) * 100 if agg["new_bad_end"] else None
    mean_en_be = agg_mean(agg["en_bad_end"]) * 100 if agg["en_bad_end"] else None
    print(
        f"  {'mean bad line-end %':<34}"
        f" {f'{mean_old_be:.1f}%' if mean_old_be is not None else 'N/A':>10}"
        f" {f'{mean_new_be:.1f}%' if mean_new_be is not None else 'N/A':>10}"
        f" {f'{mean_en_be:.1f}%' if mean_en_be is not None else 'N/A':>12}"
    )

    print(f"  {'─' * 34} {'─' * 10} {'─' * 10} {'─' * 12}")
    # OLD: combined mid-phrase %
    mean_old_mpc = agg_mean(agg["old_mpc"]) * 100 if agg["old_mpc"] else None
    print(
        f"  {'mean mid-phrase-cut % (OLD)':<34}"
        f" {f'{mean_old_mpc:.1f}%' if mean_old_mpc is not None else 'N/A':>10}"
        f" {'':>10}"
        f" {'N/A':>12}"
    )
    # NEW: split metrics
    mean_new_len_mid = (
        agg_mean(agg["new_len_mid_pct"]) if agg["new_len_mid_pct"] else None
    )
    mean_new_gap_mid = (
        agg_mean(agg["new_gap_mid_pct"]) if agg["new_gap_mid_pct"] else None
    )
    mean_new_len_share = (
        agg_mean(agg["new_len_share_pct"]) if agg["new_len_share_pct"] else None
    )
    print(
        f"  {'  NEW len-break mid-phrase %':<34}"
        f" {'':>10}"
        f" {f'{mean_new_len_mid:.1f}%' if mean_new_len_mid is not None else 'N/A':>10}"
        f" {'':>12}"
        f"  <- quality gate"
    )
    print(
        f"  {'  NEW gap-break mid-phrase %':<34}"
        f" {'':>10}"
        f" {f'{mean_new_gap_mid:.1f}%' if mean_new_gap_mid is not None else 'N/A':>10}"
        f" {'':>12}"
        f"  <- informational"
    )
    print(
        f"  {'  NEW len-break share %':<34}"
        f" {'':>10}"
        f" {f'{mean_new_len_share:.1f}%' if mean_new_len_share is not None else 'N/A':>10}"
        f" {'':>12}"
    )

    # -----------------------------------------------------------------------
    # Acceptance gates
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print("  ACCEPTANCE GATES")
    print(f"{'─' * 70}")

    gate_over7 = total_new_o7 == 0
    print(
        f"  NEW total >7s cues = {total_new_o7}  "
        f"[gate: 0]  {'PASS' if gate_over7 else 'FAIL'}"
    )

    gate_mpc = (mean_new_len_mid is not None) and (mean_new_len_mid < 10.0)
    mpc_str = f"{mean_new_len_mid:.1f}%" if mean_new_len_mid is not None else "N/A"
    print(
        f"  NEW mean len-break mid-phrase = {mpc_str}  "
        f"[gate: <10%]  {'PASS' if gate_mpc else 'FAIL'}"
    )

    if errors:
        print(f"\n  Episodes with errors ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    overall = "PASS" if (gate_over7 and gate_mpc) else "FAIL"
    print(f"\n  OVERALL: {overall}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"usage: {sys.argv[0]} <video_dir>  "
            "(directory of .mkv episodes with sibling voxweave JSONs)",
            file=sys.stderr,
        )
        sys.exit(2)
    main(sys.argv[1])
