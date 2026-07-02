"""Pure logic for re-running forced alignment on edited VTT text.

Parses VTT, routes edited blocks to audio windows via char-level difflib against
old word_segments, back-fills aligner-returned units into each block, and formats
timestamps. Model inference and audio preparation live in :mod:`voxweave.pipeline`.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

log = logging.getLogger("voxweave")

# Korean uses spaces, so excluded.
NO_SPACE_LANGS = {"zh", "ja", "yue"}
# Gaps wider than this between adjacent word_segments are song holes; windows and interpolation must not cross them.
GAP_SEC = 2.0
# Crop window padding. WhisperX uses pad=0; adding pad causes boundary tokens to drift into the
# pad silence (CTC fills available frames to the boundary; measured ja drift 0.1-0.4s). The thin
# 0.1s is kept only so difflib rough spans on edited text don't clip real speech.
PAD_SEC = 0.1
# Forced alignment snaps to tight acoustic boundaries; short interjections (はい 50ms, クレイ 150ms)
# flash by instantly without a floor. Unlike process/smart_split, align must enforce this itself.
MIN_CUE_SEC = 0.8
# Flash-display rescue for very short cues (so/あ/え). Orthogonal to MIN_CUE_SEC: that mechanism
# never overlaps; this one allows side-by-side display with the next cue (at most 1 overlapping neighbor).
TINY_CUE_SEC = 0.2
TINY_CUE_TARGET = 0.5

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


# --------------------------------------------------------------------------- #
# VTT parsing
# --------------------------------------------------------------------------- #
def _parse_ts(token: str) -> float | None:
    """``HH:MM:SS.mmm`` / ``MM:SS.mmm`` → seconds; returns None if no match."""
    m = _TS.fullmatch(token.strip())
    if not m:
        return None
    h = int(m.group(1) or 0)
    mm, ss, ms = int(m.group(2)), int(m.group(3)), int(m.group(4).ljust(3, "0"))
    return h * 3600 + mm * 60 + ss + ms / 1000.0


def parse_vtt_blocks(text: str) -> list[dict]:
    """Parse both plain-text and timestamped VTT → ordered cues ``[{text, start, end}]``.

    Handles both formats: if a ``-->`` timing line is present, start/end are populated
    (re-run scenario); otherwise start/end are None (initial align from plain-text edit).
    Cue id lines, WEBVTT headers, and NOTE/STYLE/REGION blocks are discarded.
    """
    blocks: list[dict] = []
    music_only = 0
    text = text.lstrip("\ufeff")  # a leading BOM must not defeat the header check
    for raw in re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [ln for ln in raw.split("\n")]
        # strip leading/trailing blank lines from block
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue
        head = lines[0].strip()
        if head.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        start = end = None
        body = lines
        # look for timing line within first two lines
        for i, ln in enumerate(lines[:2]):
            if "-->" in ln:
                lhs, _, rhs = ln.partition("-->")
                rhs = rhs.strip().split()[0] if rhs.strip() else ""
                start, end = _parse_ts(lhs), _parse_ts(rhs)
                body = lines[i + 1 :]
                break
        cue = "\n".join(body).strip()
        if not cue:
            continue
        block = {"text": cue, "start": start, "end": end}
        # Keep-lyrics marking: "♪ text ♪" wraps are display decoration, not content.
        # Strip them so alignment/translation see clean text, and flag the block so
        # writers (align rewrite, translated VTT) can restore the wrap.
        if len(cue) > 2 and cue.startswith("♪") and cue.endswith("♪"):
            block["text"] = cue[1:-1].strip()
            block["lyric"] = True
            if not block["text"]:
                music_only += 1
                continue
        blocks.append(block)
    if music_only:
        log.info(
            "dropped %d music-only cue(s) (no text inside the note wrap)", music_only
        )
    return blocks


# --------------------------------------------------------------------------- #
# Character-level alignment (shared by routing and back-fill)
# --------------------------------------------------------------------------- #
def _flatten(texts: list[str]) -> tuple[str, list[int]]:
    """Flatten text segments → (lowercased alnum char stream, per-char owner index).

    Strips punctuation and whitespace: aligner output is clean, but old/new texts may have
    punctuation; pure alnum comparison is more robust across both.
    """
    chars: list[str] = []
    owners: list[int] = []
    for idx, t in enumerate(texts):
        for c in t:
            if c.isalnum():
                chars.append(c.casefold())
                owners.append(idx)
    return "".join(chars), owners


def _seq_map_proportional(a, b):
    """SequenceMatcher equal/replace blocks → (ai, bj) index pairs.

    Replace blocks with differing lengths are mapped proportionally (``j = j1 + (k*nj)//ni``)
    so character substitutions still get an anchor. Insert/delete blocks yield nothing.
    """
    sm = SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("equal", "replace"):
            continue
        ni, nj = i2 - i1, j2 - j1
        if ni == 0 or nj == 0:
            continue
        for k in range(ni):
            yield i1 + k, j1 + (k * nj) // ni


def char_owner_map(item_texts: list[str], ref_texts: list[str]) -> list[set[int]]:
    """Character-level difflib alignment → set of ref indices covered by each item text.

    Used for routing (item=edited blocks, ref=old word_segments) and back-fill
    (item=window blocks, ref=aligner units). Replace blocks still anchor to the ref region.
    """
    ic, io = _flatten(item_texts)
    rc, ro = _flatten(ref_texts)
    res: list[set[int]] = [set() for _ in item_texts]
    if not ic or not rc:
        return res
    for i, j in _seq_map_proportional(ic, rc):
        res[io[i]].add(ro[j])
    return res


def spans_from_sets(
    sets: list[set[int]], units: list[dict]
) -> list[tuple[float, float] | None]:
    """Per-block unit index set → (min start, max end); empty set → None (pure insertion)."""
    out: list[tuple[float, float] | None] = []
    for s in sets:
        if not s:
            out.append(None)
            continue
        out.append((min(units[i]["start"] for i in s), max(units[i]["end"] for i in s)))
    return out


# Sentence/clause delimiters transferred during dual-ASR fusion. Does NOT include ・
# (name separator — e.g. ラスティス・ムーン — including it would split proper names).
_FUSE_PUNCT = set("。、！？，,.!?")


def fuse_punct_into_text(
    text: str,
    units: list[dict],
    punct_units: list[dict],
    strip_existing: bool = True,
) -> str:
    """Insert Qwen sentence punctuation into whisper raw text via char-level content alignment.

    **Why content alignment, not time-based**: whisper and Qwen have independent alignment
    timelines; OOV drift causes the same character to land at different timestamps on each axis.
    Time-based insertion put punctuation after the wrong whisper token (observed: ja ``どの。``
    landed at「造」→ ``番酒造。り``). Char-level difflib maps the Qwen content-char stream onto
    the whisper content-char stream instead, fully decoupled from timestamps.

    **Only equal blocks are used as anchors** (not proportional replace): replace regions are
    where the two ASR outputs disagree. Proportional insertion inside a replace block puts
    punctuation in the middle of whisper words or peppers every character of a misheard lyric
    passage. Punctuation anchored to a replace region falls back to the nearest equal-block
    boundary (≈ sentence boundary) or is dropped.

    ``strip_existing`` (True for spaced languages): strips whisper's own sentence delimiters so
    only Qwen punctuation remains. No-space languages retain whisper's own punctuation (whisper
    large-v3 ja punctuation is reasonable; both coexist after content alignment).

    ``punct_units`` must carry Qwen content chars + punctuation (output of
    :func:`reinject_punct`) so punctuation has content anchors. ``units`` is kept for API
    compatibility only. Pure function.
    """
    # Extract Qwen content-char stream + punctuation events.
    # event (idx, char): insert after the idx-th content char; idx==0 = sentence-initial.
    q_chars: list[str] = []
    events: list[tuple[int, str]] = []
    for u in punct_units:
        for c in u["text"]:
            if c.isalnum():
                q_chars.append(c.casefold())
            elif c in _FUSE_PUNCT:
                events.append((len(q_chars), c))
    if not events:  # no Qwen punctuation: optionally strip whisper's own delimiters
        if strip_existing:
            return "".join(c for c in text if c not in _FUSE_PUNCT)
        return text
    # Whisper content-char stream + each char's position in text.
    w_chars: list[str] = []
    w_pos: list[int] = []
    for i, c in enumerate(text):
        if c.isalnum():
            w_chars.append(c.casefold())
            w_pos.append(i)
    if not w_chars:
        return text
    # Equal blocks only — see docstring for why replace blocks are excluded.
    sm = SequenceMatcher(None, q_chars, w_chars, autojunk=False)
    q2w: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                q2w[i1 + k] = j1 + k
    # Resolve each event to an insertion position in whisper text.
    inserts: dict[int, list[str]] = {}
    for q_idx, punct in events:
        if q_idx <= 0:
            at = 0  # sentence-initial punctuation (rare)
        else:
            wi = q2w.get(q_idx - 1)
            if (
                wi is None
            ):  # anchor in replace region: fall back to nearest equal char (pre-divergence boundary)
                wi = next((q2w[k] for k in range(q_idx - 1, -1, -1) if k in q2w), None)
            if wi is None:
                continue  # no equal char before this punct (e.g. song lyrics) → drop
            at = w_pos[wi] + 1
        inserts.setdefault(at, []).append(punct)
    # Rebuild text, inserting Qwen punctuation at resolved positions.
    out: list[str] = []
    for i, c in enumerate(text):
        for p in inserts.get(i, ()):
            out.append(p)
        if not strip_existing or c not in _FUSE_PUNCT:
            out.append(c)
    for p in inserts.get(len(text), ()):  # trailing punct
        out.append(p)
    return "".join(out)


def explode_units(units: list[dict]) -> list[dict]:
    """Split each unit into single-character pseudo-units, time evenly divided over alnum chars.

    The aligner can group chars that straddle a cue boundary (e.g. ``好柯`` = last char of one
    cue + first of the next). Without exploding, both adjacent cues claim the boundary unit and
    spans overlap. Exploding gives each char an independent owner so the boundary is split cleanly.
    Zero-duration units remain zero after exploding.
    """
    out: list[dict] = []
    for u in units:
        chars = [c for c in u["text"] if c.isalnum()]
        n = len(chars)
        if n == 0:
            continue
        dur = max(0.0, u["end"] - u["start"])
        for k, c in enumerate(chars):
            out.append(
                {
                    "text": c,
                    "start": u["start"] + dur * k / n,
                    "end": u["start"] + dur * (k + 1) / n,
                }
            )
    return out


# Em-dash / ellipsis split points: Qwen 1.7B emits em-dashes that cause "today—a" to become
# a single word_segment, swallowing the pause and preventing any sentence break. Splitting here
# exposes the pause to smart_split's gap-based segmentation. Single hyphen excluded —
# compound words like "well-known" must not be split. The dash is included in the left sub-token.
_BREAK_PUNCT_RE = re.compile(r"(?:—|–|--+|…|\.\.\.)")


def _split_break_spans(token: str) -> list[tuple[int, int]]:
    """Split a whitespace-delimited token at internal em-dashes/ellipses, returning [start, end)
    offsets of each sub-token relative to the token start. The dash is included in the left
    sub-token; splits only when content exists on both sides (excludes leading/trailing dashes
    as in dialogue "—a")."""
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _BREAK_PUNCT_RE.finditer(token):
        a, b = m.start(), m.end()
        if a > last and b < len(token):
            spans.append((last, b))
            last = b
    spans.append((last, len(token)))
    return spans


def reinject_punct(asr_text: str, units: list[dict], iso: str) -> list[dict]:
    """Re-attach punctuation from ASR text back onto aligner units for smart_split consumption.

    The aligner strips punctuation; smart_split needs it for sentence/line breaks. This function
    uses char-level difflib to map each ASR alnum character to a unit timestamp, then rebuilds:

    - No-space languages (zh/ja/yue): one item per non-whitespace character; punctuation becomes
      a zero-width item at the boundary. ``"".join`` of items equals asr_text minus spaces.
    - Spaced languages: one item per whitespace-delimited token with punctuation attached;
      ``" ".join`` of items equals asr_text.

    Falls back to original units if no alnum characters can be aligned.
    """
    if not asr_text or not any(c.isalnum() for c in asr_text):
        return units
    ali = explode_units(units)  # per-alnum-char timestamp stream
    if not ali:
        return units

    asr_pos = [k for k, c in enumerate(asr_text) if c.isalnum()]
    asr_fold = [asr_text[k].casefold() for k in asr_pos]
    ali_fold = [a["text"].casefold() for a in ali]

    # Map each ASR alnum character to a timestamp; substitutions keep their anchor.
    times: list[tuple[float, float] | None] = [None] * len(asr_text)
    for i, j in _seq_map_proportional(asr_fold, ali_fold):
        times[asr_pos[i]] = (ali[j]["start"], ali[j]["end"])

    # Punctuation/spaces/unmatched chars inherit the end of the preceding timed char.
    first_start = next((t[0] for t in times if t is not None), 0.0)
    last = first_start
    filled: list[tuple[float, float]] = []
    for t in times:
        if t is not None:
            filled.append(t)
            last = t[1]
        else:
            filled.append((last, last))

    out: list[dict] = []
    if iso in NO_SPACE_LANGS:
        for k, c in enumerate(asr_text):
            if c.isspace():
                # Spaces between CJK are meaningless, but embedded Latin words need spaces
                # for tokenization (smart_split._tokens bridges ASCII runs across spaces).
                # Append to preceding unit's text rather than creating a new unit: keeps the
                # space in reconstructed text while preserving the one-unit-per-non-whitespace-char contract.
                if out:
                    out[-1]["text"] += c
                continue
            out.append({"text": c, "start": filled[k][0], "end": filled[k][1]})
    else:
        n = len(asr_text)
        i = 0
        while i < n:
            if asr_text[i].isspace():
                i += 1
                continue
            j = i
            while j < n and not asr_text[j].isspace():
                j += 1
            tok = asr_text[i:j]
            # Split at em-dashes/ellipses to expose pauses that would otherwise be swallowed.
            for sa, sb in _split_break_spans(tok):
                ts = filled[i + sa : i + sb]
                out.append(
                    {
                        "text": tok[sa:sb],
                        "start": min(t[0] for t in ts),
                        "end": max(t[1] for t in ts),
                    }
                )
            i = j
    return out or units


# CJK sentence-break delimiters used by snap_break_punct. ASCII .,; excluded — decimal points
# like 14.2 / 1.87 must not be relocated (would split numbers).
_BREAK_PUNCT = set("。！？；，、")


def snap_break_punct(units: list[dict], iso: str, *, max_shift: int = 2) -> list[dict]:
    """zh only: move misplaced sentence-break punctuation to the nearest jieba word boundary.

    Qwen-ASR zh delimiters are frequently off by ≤1 character — e.g. ``同比增长29%。数据中心``
    is transcribed as ``29%数。据中心``, causing smart_split to split mid-word. This function
    snaps each punctuation unit to the nearest word start within ``max_shift`` characters.

    **zh only**: jieba zh segmentation is reliable. ja is **not snapped** — BudouX ja is too
    weak and would introduce new word splits; ja punctuation misplacement is fixed upstream by
    content-aligned fusion (:func:`fuse_punct_into_text`). Returns units unchanged if the
    tokenizer is unavailable. ``units`` must be the per-char sequence from reinject. Pure function.
    """
    if iso != "zh" or len(units) < 3:
        return units
    from voxweave.core import breakpoints

    content: list[dict] = []  # content character units
    puncts: list[
        tuple[int, dict]
    ] = []  # (content position = # content chars before this punct, punct unit)
    for u in units:
        nc = [c for c in u["text"] if not c.isspace()]
        if len(nc) == 1 and nc[0] in _BREAK_PUNCT:
            puncts.append((len(content), u))
        elif len(nc) >= 1:
            content.append(u)
        else:  # pure-whitespace unit (should not occur): bail out for safety
            return units
    if not puncts or len(content) < 2:
        return units
    cstr = "".join(next(c for c in u["text"] if not c.isspace()) for u in content)
    starts = breakpoints.word_starts(cstr, iso)
    if not starts:
        return units

    inserts: dict[
        int, list[dict]
    ] = {}  # target content position → punctuation units (order-preserving)
    for pos, pu in puncts:
        if pos in starts or pos <= 0 or pos >= len(content):
            target = pos  # already at a word boundary / at start or end: no move
        else:
            cand = min(
                starts, key=lambda s: (abs(s - pos), s)
            )  # nearest word start; ties broken toward smaller index
            target = cand if abs(cand - pos) <= max_shift else pos
        inserts.setdefault(target, []).append(pu)

    out: list[dict] = []
    for i, u in enumerate(content):
        for pu in inserts.get(
            i, ()
        ):  # punct before content[i] = break just before this word start
            t = out[-1]["end"] if out else u["start"]
            out.append({"text": pu["text"], "start": t, "end": t})
        out.append(u)
    for pu in inserts.get(len(content), ()):  # trailing punctuation (fallback)
        t = out[-1]["end"] if out else 0.0
        out.append({"text": pu["text"], "start": t, "end": t})
    return out


def route_blocks(
    blocks: list[dict], word_segments: list[dict]
) -> list[tuple[float, float] | None]:
    """Rough audio interval for each edited block (used only for window positioning, not in final output).

    If blocks already carry timestamps (re-run scenario), those are used directly; otherwise
    falls back to character-level matching against the old word_segments.
    """
    if blocks and all(b["start"] is not None and b["end"] is not None for b in blocks):
        return [(b["start"], b["end"]) for b in blocks]
    sets = char_owner_map(
        [b["text"] for b in blocks], [u["text"] for u in word_segments]
    )
    return spans_from_sets(sets, word_segments)


# --------------------------------------------------------------------------- #
# Window cropping
# --------------------------------------------------------------------------- #
def crop_blocks(
    spans: list[tuple[float, float] | None],
    *,
    pad: float = PAD_SEC,
) -> list[tuple[float, float] | None]:
    """Crop each cue to its own tight window ``[start - pad, end + pad]`` (WhisperX equivalent).

    WhisperX confines each segment to its own acoustic range — this is isomorphic. The window is
    **not extended to the next sentence start**: with no extra room, the CTC path cannot drift
    the final word into inter-sentence silence. Window size is independent of neighbor distance,
    so crossing song holes is impossible without needing gap_sec checks.

    Returns the same length as ``spans``; None entries are insertion blocks handled by
    :func:`fill_insert_blocks`.
    """
    crops: list[tuple[float, float] | None] = []
    for sp in spans:
        if sp is None:
            crops.append(None)
            continue
        s, e = sp
        cs = max(0.0, s - pad)
        ce = e + pad
        if ce <= cs:  # degenerate guard (rough span reversed or zero-width)
            ce = cs + 0.1
        crops.append((cs, ce))
    return crops


def join_block_texts(texts: list[str], iso: str) -> str:
    """Join block texts within a window for the aligner: no separator for no-space languages,
    space otherwise; internal wrap newlines are collapsed."""
    sep = "" if iso in NO_SPACE_LANGS else " "
    flat = [sep.join(t.split("\n")).strip() for t in texts]
    return sep.join(t for t in flat if t)


# --------------------------------------------------------------------------- #
# Back-fill and finalization
# --------------------------------------------------------------------------- #
def fill_insert_blocks(
    spans: list[tuple[float, float] | None],
    *,
    gap_sec: float = GAP_SEC,
    default_dur: float = 2.0,
) -> list[tuple[float, float]]:
    """Assign timestamps to pure-insertion blocks (None spans) by neighbor interpolation,
    never crossing a song hole (gap > gap_sec).

    When prev and next neighbors are close enough, divide [prev.end, next.start] evenly
    across the run; otherwise anchor to whichever side has a neighbor and assign default_dur.
    If neither side has a neighbor, fall back to (0, default_dur).
    """
    n = len(spans)
    out: list[tuple[float, float] | None] = list(spans)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        prev = out[i - 1] if i > 0 else None
        nxt = spans[j] if j < n else None
        run = j - i
        if prev and nxt and (nxt[0] - prev[1]) <= gap_sec and nxt[0] > prev[1]:
            step = (nxt[0] - prev[1]) / (run + 1)
            for k in range(run):
                a = prev[1] + step * (k + 1)
                out[i + k] = (a, a + step)
        elif prev:
            a = prev[1]
            for k in range(run):
                out[i + k] = (a + k * default_dur, a + (k + 1) * default_dur)
        elif nxt:
            a = max(0.0, nxt[0] - run * default_dur)
            for k in range(run):
                out[i + k] = (a + k * default_dur, a + (k + 1) * default_dur)
        else:
            for k in range(run):
                out[i + k] = (k * default_dur, (k + 1) * default_dur)
        i = j
    return [s if s is not None else (0.0, default_dur) for s in out]


def snap_zero_duration_units(
    units: list[dict],
    vad: list[tuple[float, float]],
    *,
    eps: float = 0.05,
    max_run: int = 8,
) -> list[dict]:
    """Relocate zero-duration unit runs (Qwen NAR collapse) to the actual speech in the gap.

    Qwen3 NAR aligner tokens placed with low confidence collapse to the edge bin of the
    preceding unit (start==end), while the actual utterance may be 1+ seconds later. reinject
    also promotes punctuation to independent zero-duration units, so ``はい`` becomes a run
    ``。/は/い/。`` of consecutive zeros.

    For each contiguous zero-duration run: find an isolated VAD segment in the gap
    ``(prev_end, next_start)`` closest to the collapse point and spread the run evenly across
    it. If no isolated segment is found (e.g. ``だ`` attached to the preceding word) → leave
    unchanged. Runs longer than ``max_run`` (ASR repetition-collapse walls) → untouched.

    **Only alnum chars are spread**: leading punctuation (sentence-final from the previous cue)
    is left at prev_end — pulling it forward would erase the gap. Trailing punctuation is
    attached to content end. Pure function.
    """
    if not units or not vad:
        return units
    segs = sorted((float(s), float(e)) for s, e in vad if e - s > eps)
    out = [dict(u) for u in units]
    n = len(out)
    i = 0
    while i < n:
        if out[i]["end"] - out[i]["start"] > eps:
            i += 1
            continue
        j = i  # [i, j) = contiguous zero-duration run
        while j < n and out[j]["end"] - out[j]["start"] <= eps:
            j += 1
        run = j - i
        prev_end = out[i - 1]["end"] if i > 0 else 0.0
        next_start = out[j]["start"] if j < n else float("inf")
        alnum_idx = [k for k in range(i, j) if any(c.isalnum() for c in out[k]["text"])]
        if run <= max_run and alnum_idx:
            # isolated speech segments fully within the gap (start >= prev_end, end <= next_start)
            cands = [
                (s, e) for s, e in segs if s >= prev_end - eps and e <= next_start + eps
            ]
            if cands:
                s, e = min(cands, key=lambda se: abs(se[0] - out[i]["start"]))
                s = max(prev_end, s)
                e = min(next_start, e)
                if e - s > eps:
                    m = len(alnum_idx)
                    step = (e - s) / m
                    for r, k in enumerate(alnum_idx):  # spread alnum content evenly
                        out[k]["start"] = s + step * r
                        out[k]["end"] = s + step * (r + 1)
                    core_end = out[alnum_idx[-1]]["end"]
                    for k in range(
                        alnum_idx[-1] + 1, j
                    ):  # trailing punct to content end
                        out[k]["start"] = out[k]["end"] = core_end
                    # leading punct (i .. alnum_idx[0]) stays at prev_end — do not move
        i = j
    return out


def snap_silence_stranded_units(
    units: list[dict],
    vad: list[tuple[float, float]],
    *,
    tol: float = 0.5,
    eps: float = 0.05,
) -> list[dict]:
    """Pull unit runs that drifted into adjacent silence back to the nearest VAD speech edge.

    ``snap_zero_duration_units`` handles Qwen NAR (sparse zero-duration tokens). But ja/en CTC
    aligners give each character a ~20ms frame-point: the whole block is point timestamps, so
    snap sees one long zero-duration run (> max_run) and skips it. Isolated short sentences
    surrounded by silence can land 0.3-0.8s outside the VAD boundary with no correction
    (observed: ``行くよ`` drifted 0.4s past offset; ``のんびり農業`` placed 0.8s before onset).

    Trigger: unit midpoint lands in VAD silence. For a silence-stranded run within ``tol`` of
    the nearest speech edge: pull to that edge (run past offset → pull last char to ≤ offset;
    run before onset → push first char to ≥ onset), clamped by surrounding in-speech units.
    If the nearest edge is > ``tol`` (genuinely dropped audio) → leave unchanged. Pure function.
    """
    if not units or not vad:
        return units
    segs = sorted((float(s), float(e)) for s, e in vad if e - s > 0)
    if not segs:
        return units
    out = [dict(u) for u in units]
    n = len(out)

    def in_speech(u: dict) -> bool:
        # Non-zero-width units use overlap test (carve handles them).
        # Zero-width point timestamps use containment — in-speech only if the point is inside a speech seg.
        s, e = u["start"], u["end"]
        if e > s:
            return any(min(e, ve) > max(s, vs) for vs, ve in segs)
        return any(vs <= s <= ve for vs, ve in segs)

    i = 0
    while i < n:
        if in_speech(out[i]):
            i += 1
            continue
        j = i  # [i, j) = contiguous silence-stranded run
        while j < n and not in_speech(out[j]):
            j += 1
        run_s, run_e = out[i]["start"], out[j - 1]["end"]
        prev_end = out[i - 1]["end"] if i > 0 else 0.0
        next_start = out[j]["start"] if j < n else float("inf")
        left = max(
            (e for _, e in segs if e <= run_s + eps), default=float("-inf")
        )  # nearest speech offset to the left (-inf if none)
        right = min(
            (s for s, _ in segs if s >= run_e - eps), default=float("inf")
        )  # nearest speech onset to the right (+inf if none)
        left_d = run_s - left
        right_d = right - run_e
        alnum_idx = [k for k in range(i, j) if any(c.isalnum() for c in out[k]["text"])]
        if alnum_idx and min(left_d, right_d) <= tol:
            m = len(alnum_idx)
            width = max(run_e - run_s, eps * m)  # preserve span; floor at eps per char
            if left_d <= right_d:  # pull-left: last char back to speech offset
                hi = min(left, next_start)
                lo = max(prev_end, hi - width)
            else:  # push-right: first char to speech onset
                lo = max(prev_end, right)
                hi = min(next_start, lo + width)
            if hi - lo > 0:
                step = (hi - lo) / m
                for r, k in enumerate(alnum_idx):
                    out[k]["start"] = lo + step * r
                    out[k]["end"] = lo + step * (r + 1)
                core_s, core_e = out[alnum_idx[0]]["start"], out[alnum_idx[-1]]["end"]
                for k in range(i, alnum_idx[0]):  # leading punct to content start
                    out[k]["start"] = out[k]["end"] = core_s
                for k in range(alnum_idx[-1] + 1, j):  # trailing punct to content end
                    out[k]["start"] = out[k]["end"] = core_e
        i = j
    return out


def carve_units_over_silence(
    units: list[dict],
    vad: list[tuple[float, float]],
    *,
    min_overhang: float = 0.2,
) -> list[dict]:
    """Trim leading/trailing silence from each unit using the VAD speech map.

    Qwen NAR has no blank token: words inflate to fill the segment including silence (e.g. ``Oh``
    inflated to 2.56s over a 1.8s pause). This clips outer silence, keeping the speech-overlap
    interval. Only trims when the overhang exceeds ``min_overhang`` (debounce). Internal silence
    (speech-silence-speech) is not cut.

    **Continuously voiced** units (long vowels like ``そう〜``, fully covered by VAD) have zero
    overhang and are never trimmed — this is the key safety property: statistical outlier methods
    would misclassify slow speech and drawn-out vowels.

    Units fully in silence → **left unchanged** (snap's domain). Zero-duration units untouched.
    Only shrinks inward, never expands → no overlaps. Pure function.
    """
    if not units or not vad:
        return units
    segs = sorted((float(s), float(e)) for s, e in vad)
    out = [dict(u) for u in units]
    for u in out:
        s, e = u["start"], u["end"]
        if e - s <= 0:  # zero/negative → snap's domain
            continue
        first_ov = last_ov = None  # earliest/latest overlap endpoint with speech segs
        for vs, ve in segs:
            if ve <= s or vs >= e:
                continue
            ov_s, ov_e = max(s, vs), min(e, ve)
            if first_ov is None or ov_s < first_ov:
                first_ov = ov_s
            if last_ov is None or ov_e > last_ov:
                last_ov = ov_e
        if first_ov is None:  # fully in silence → leave unchanged
            continue
        if first_ov - s > min_overhang:
            u["start"] = first_ov
        if e - last_ov > min_overhang:
            u["end"] = last_ov
    return out


def position_units_with_vad(
    units: list[dict], vad: list[tuple[float, float]]
) -> list[dict]:
    """Apply all three VAD positioning steps to aligned units.

    Shared by both transcribe and align paths to prevent behavioral drift. Steps:
    (1) snap_zero_duration_units — Qwen NAR collapsed tokens;
    (2) snap_silence_stranded_units — CTC point-timestamp runs that drifted into silence;
    (3) carve_units_over_silence — trim over-inflated unit edges.
    The three steps are orthogonal and non-interfering. Pure function.
    """
    snapped = snap_silence_stranded_units(snap_zero_duration_units(units, vad), vad)
    return carve_units_over_silence(snapped, vad)


def group_block_spans(
    block_units: list[list[dict]],
) -> tuple[list[tuple[float, float] | None], list[dict]]:
    """Reconstruct per-block spans: ``(first word start, last word end)`` from the cropped window.

    Empty blocks → None (interpolated by :func:`fill_insert_blocks`). Returns ``(spans, flat_units)``.

    **No VAD snap/carve**: tight cropping gives the aligner no room to drift into inter-sentence
    silence, making :func:`position_units_with_vad` unnecessary here (transcribe applies it on
    full-chunk alignments). Zero-duration residues go directly to :func:`clamp_spans` —
    isomorphic to WhisperX's "trust raw + interpolate_nans". Pure function.
    """
    flat = [u for bu in block_units for u in bu]
    spans: list[tuple[float, float] | None] = []
    k = 0
    for bu in block_units:
        if not bu:
            spans.append(None)
            continue
        grp = flat[k : k + len(bu)]
        spans.append((grp[0]["start"], grp[-1]["end"]))
        k += len(bu)
    return spans, flat


def enforce_min_duration(
    spans: list[tuple[float, float]], *, min_dur: float = MIN_CUE_SEC
) -> list[tuple[float, float]]:
    """Ensure every cue displays for at least ``min_dur`` seconds.

    Forced alignment snaps to tight boundaries; short sentences (はい/クレイ) flash by instantly.
    Strategy per cue:
    - Gap after the cue → extend to ``start+min_dur``, capped at next start (clears overlap).
    - No gap but next sentence is long enough → push next sentence start forward to borrow time.
    - Can't push → cap at next sentence start.
    Last cue extended unconditionally. Song holes are unaffected (extension << hole size).
    """
    out = [[a, b] for a, b in spans]
    n = len(out)
    for i in range(n):
        a, b = out[i]
        target = a + min_dur
        if i + 1 >= n:
            out[i][1] = max(b, target)
            continue
        na, nb = out[i + 1]
        if target <= na:
            out[i][1] = min(
                max(b, target), na
            )  # extend to target, capped at next start (clears overlap)
        elif target < nb - min_dur / 2:
            out[i][1] = target  # adjacent: push next sentence start to borrow time
            out[i + 1][0] = target
        else:
            out[i][1] = na  # can't push: cap at next sentence start
    return [(a, max(b, a + 0.001)) for a, b in out]


def rescue_tiny_cues(
    spans: list[tuple[float, float]],
    *,
    trig: float = TINY_CUE_SEC,
    target: float = TINY_CUE_TARGET,
) -> list[tuple[float, float]]:
    """Extend flash-display cues (dur < ``trig``) to ``target`` duration. No-op when ``trig <= 0``.

    Complementary to :func:`enforce_min_duration`: that function never overlaps; this one
    **allows side-by-side display** with the next cue when there's no gap — capped at
    cue-after-next's start so at most 1 overlapping neighbor. Onset never moved.
    """
    if trig <= 0:
        return [(a, b) for a, b in spans]
    out = [[a, b] for a, b in spans]
    n = len(out)
    for i in range(n):
        a, b = out[i]
        if b - a >= trig:
            continue  # not a flash cue, leave it
        want = a + target
        if i + 1 >= n:
            out[i][1] = max(b, want)  # last cue: extend unconditionally
            continue
        nxt = out[i + 1][0]
        if want <= nxt:
            out[i][1] = max(b, want)  # enough gap: no overlap
        else:
            cap = out[i + 2][0] if i + 2 < n else float("inf")
            out[i][1] = max(
                b, min(want, cap)
            )  # side-by-side with next; cap at cue-after-next
    return [(a, b) for a, b in out]


def clamp_spans(
    spans: list[tuple[float, float]], *, min_dur: float = 0.05
) -> list[tuple[float, float]]:
    """Ensure every cue has end > start (at least min_dur)."""
    return [(a, max(b, a + min_dur)) for a, b in spans]


def fmt_ts(seconds: float) -> str:
    """Seconds → VTT timestamp ``HH:MM:SS.mmm``."""
    s = max(0.0, seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def render_cues(rows: list[tuple[float | None, float | None, str]]) -> str:
    """Render WEBVTT from ``(start, end, text)`` rows: a timestamp line is emitted when both
    start and end are present, otherwise the cue is plain text.

    Single source for the WEBVTT skeleton shared by realign / translate / asrfix / pipeline.
    """
    out = ["WEBVTT", ""]
    for start, end, text in rows:
        if start is not None and end is not None:
            out.append(f"{fmt_ts(start)} --> {fmt_ts(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_vtt(blocks: list[dict], spans: list[tuple[float, float]]) -> str:
    """Block text + timestamps → standard timestamped VTT string. Lyric-flagged
    blocks (see :func:`parse_vtt_blocks`) get their music-note wrap restored."""
    return render_cues(
        [
            (a, e, f"♪ {b['text']} ♪" if b.get("lyric") else b["text"])
            for b, (a, e) in zip(blocks, spans)
        ]
    )
