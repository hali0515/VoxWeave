"""Pack (soft-mux) or burn (hard-sub) subtitle files (VTT/SRT/ASS) into video files.

``pack`` remuxes the source media with the subtitles added as a proper subtitle track
(stream copy, instant, reversible). ``burn`` renders the subtitles into the
pixels via a styled ASS + libass filter and re-encodes the video (constant
quality, hardware encoder when available: NVENC on NVIDIA, VideoToolbox on
macOS, libx264/libx265/libsvt-av1 software fallback). Both drop nothing from
the source except, for burn, the now-redundant subtitle tracks.

Command construction is kept separate from probing/execution so the ffmpeg
argv builders stay unit-testable without media files or a GPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from voxweave import lang

logger = logging.getLogger(__name__)

# Containers pack/burn can write, mapped to the text subtitle codec each stores.
SUB_CODEC = {"mkv": "srt", "mp4": "mov_text", "webm": "webvtt"}

# Subtitle codecs that can be transcoded to another text format (image-based
# subs like hdmv_pgs/dvd_subtitle cannot and are dropped when the target
# container will not store them as-is).
_TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}

# Audio codecs safe to stream-copy into mp4; anything else is re-encoded to AAC.
_MP4_SAFE_AUDIO = {"aac", "ac3", "eac3", "mp3", "alac"}

_SOFTWARE_ENCODER = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}

# Constant-quality defaults per encoder. NVENC -cq tracks x264/x265 -crf
# loosely; VideoToolbox -q:v runs 1-100 with higher = better.
_DEFAULT_QUALITY = {
    "h264_nvenc": 19,
    "hevc_nvenc": 23,
    "av1_nvenc": 30,
    "h264_videotoolbox": 65,
    "hevc_videotoolbox": 65,
    "libx264": 19,
    "libx265": 23,
    "libsvtav1": 30,
}


def _run_ffmpeg(cmd: list[str], *, capture: bool) -> None:
    """Run an ffmpeg/ffprobe command; raise RuntimeError with the stderr tail on failure.

    -nostdin + stdin=DEVNULL is a hard requirement: ffmpeg competing for the
    inherited stdin hangs the process (see pipeline.decode_to_wav).
    """
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        text=capture,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:] if capture else []
        detail = ("\n" + "\n".join(tail)) if tail else ""
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}){detail}")


def probe_streams(media: Path) -> list[dict]:
    """ffprobe the media and return its stream dicts (codec_type/codec_name/...)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(media),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media.name}: {proc.stderr.strip()}")
    return json.loads(proc.stdout).get("streams", [])


def detect_subtitle_language(sub: Path) -> str | None:
    """ISO code from a language-tagged subtitle filename ("X.zh.vtt" -> "zh"), else None."""
    from voxweave.subformats import SUBTITLE_EXTS

    p = Path(sub)
    stem = p.name
    if p.suffix.lower() in SUBTITLE_EXTS:
        stem = stem[: -len(p.suffix)]
    if "." not in stem:
        return None
    return lang.to_iso_or(stem.rsplit(".", 1)[1], None)


def track_title(iso: str | None) -> str:
    """Subtitle track title: "VoxWeave Chinese" when the language is known, else "VoxWeave"."""
    return f"VoxWeave {lang.display_name(iso)}" if iso else "VoxWeave"


def resolve_media(vtt: Path, media: Path | None) -> Path:
    """Return the explicit media path, or find the sibling media next to the VTT.

    Translated VTTs carry a language tag ("X.zh.vtt") while the media is named
    "X.<ext>", so the lookup also retries with the language token stripped.
    """
    from voxweave.pipeline import _find_sibling_media, swap_ext

    if media is not None:
        return Path(media)
    vtt = Path(vtt)
    found = _find_sibling_media(vtt)
    if found is None and detect_subtitle_language(vtt) is not None:
        found = _find_sibling_media(swap_ext(vtt, ""))  # drop ".zh" tag and retry
    if found is None:
        raise FileNotFoundError(
            f"no sibling media found for {vtt.name}; pass --media explicitly"
        )
    return found


def _timed_subtitle_check(sub: Path) -> None:
    """Raise early (with the standard hint) when the subtitle file parses to no
    timestamped cues (e.g. a plain-text VTT edit draft)."""
    from voxweave.export import _timed_rows
    from voxweave.subformats import load_subtitle_blocks

    _timed_rows(load_subtitle_blocks(Path(sub)))


def default_output(media: Path, container: str, tag: str) -> Path:
    """Sibling output path "<stem>.<container>"; "<stem>.<tag>.<container>" when
    that would overwrite the source."""
    from voxweave.pipeline import swap_ext

    out = swap_ext(media, f".{container}")
    if out.resolve() == Path(media).resolve():
        out = swap_ext(media, f".{tag}.{container}")
    return out


def _default_container(media: Path) -> str:
    """Keep the source container when pack can write it, else fall back to mkv."""
    ext = Path(media).suffix.lower().lstrip(".")
    return ext if ext in SUB_CODEC else "mkv"


# ---------------------------------------------------------------------------
# pack — soft-mux subtitle tracks


def _packed_sub_codec(sub: Path, container: str) -> str:
    """Codec for an appended subtitle input: mkv stores ASS/SSA natively (keeps
    styling); everything else transcodes to the container's text codec."""
    if container == "mkv" and Path(sub).suffix.lower() in (".ass", ".ssa"):
        return "ass"
    return SUB_CODEC[container]


def build_pack_cmd(
    media: Path,
    vtts: list[Path],
    out: Path,
    *,
    container: str,
    source_streams: list[dict],
) -> list[str]:
    """Build the ffmpeg argv that remuxes ``media`` with each subtitle file
    (VTT/SRT/ASS) appended as a subtitle track (everything stream-copied, new
    subs transcoded to the container's text codec; ASS into mkv is kept as-is).

    mkv targets carry every source stream (including attachments/fonts); mp4 and
    webm targets keep video+audio and only those existing subtitle tracks that
    are text-based (image subs cannot become mov_text/webvtt and are dropped).
    """
    sub_codec = SUB_CODEC[container]
    subs = [s for s in source_streams if s.get("codec_type") == "subtitle"]
    cmd: list[str] = ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(media)]
    for vtt in vtts:
        cmd += ["-i", str(vtt)]

    if container == "mkv":
        cmd += ["-map", "0"]
        kept_subs = len(subs)
    else:
        cmd += ["-map", "0:v", "-map", "0:a?"]
        kept_subs = 0
        for s in subs:
            if s.get("codec_name") in _TEXT_SUB_CODECS:
                cmd += ["-map", f"0:{s['index']}"]
                kept_subs += 1
            else:
                logger.warning(
                    "dropping %s subtitle track (not representable in %s)",
                    s.get("codec_name"),
                    container,
                )
    for i in range(len(vtts)):
        cmd += ["-map", f"{i + 1}:0"]

    cmd += ["-c", "copy"]
    if container == "mkv":
        # existing subs stream-copy; only the appended files are transcoded
        for i, vtt in enumerate(vtts):
            cmd += [f"-c:s:{kept_subs + i}", _packed_sub_codec(vtt, container)]
    else:
        cmd += ["-c:s", sub_codec]  # every kept text sub must be transcoded anyway
        if any(Path(v).suffix.lower() in (".ass", ".ssa") for v in vtts):
            logger.warning(
                "ASS styling is lost when packing into %s (stored as %s)",
                container,
                sub_codec,
            )

    video = next(
        (s for s in source_streams if s.get("codec_type") == "video"),
        None,
    )
    if container == "mp4" and video is not None and video.get("codec_name") == "hevc":
        cmd += ["-tag:v", "hvc1"]  # Apple players require the hvc1 brand

    for i, vtt in enumerate(vtts):
        iso = detect_subtitle_language(vtt)
        spec = f"s:{kept_subs + i}"
        if iso:
            cmd += [f"-metadata:s:{spec}", f"language={lang.to_iso3(iso)}"]
        cmd += [f"-metadata:s:{spec}", f"title={track_title(iso)}"]
    # the first packed track is what the user just produced — make players pick it
    cmd += [f"-disposition:s:{kept_subs}", "default"]
    cmd += [str(out)]
    return cmd


def pack(
    vtts: list[Path],
    *,
    media: Path | None = None,
    container: str | None = None,
    output: Path | None = None,
) -> Path:
    """Soft-mux one or more subtitle files (VTT/SRT/ASS) into the media as
    subtitle tracks; return the output path."""
    if not vtts:
        raise ValueError("at least one subtitle file is required")
    from voxweave.subformats import require_subtitle

    for v in vtts:
        require_subtitle(v)
        _timed_subtitle_check(v)
    src = resolve_media(vtts[0], media)
    cont = container or _default_container(src)
    if cont not in SUB_CODEC:
        raise ValueError(
            f"unsupported container {cont!r} (choose from {', '.join(SUB_CODEC)})"
        )
    out = output or default_output(src, cont, "pack")
    cmd = build_pack_cmd(
        src, vtts, out, container=cont, source_streams=probe_streams(src)
    )
    _run_ffmpeg(cmd, capture=True)
    return out


# ---------------------------------------------------------------------------
# burn — hard-sub re-encode


_ENCODER_CACHE: frozenset[str] | None = None
_ENCODER_WORKS: dict[str, bool] = {}


def _available_encoders() -> frozenset[str]:
    """Encoder names this ffmpeg build was compiled with (cached)."""
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        names = set()
        for line in proc.stdout.splitlines():
            parts = line.split()
            # flag column starts with V for video encoders; skip the "V..... = Video" legend
            if len(parts) >= 2 and parts[0].startswith("V") and parts[1] != "=":
                names.add(parts[1])
        _ENCODER_CACHE = frozenset(names)
    return _ENCODER_CACHE


def _encoder_works(name: str) -> bool:
    """True when a 3-frame test encode succeeds — being compiled in does not mean
    the hardware/driver is present (nvenc on a GPU-less box fails at runtime)."""
    if name not in _ENCODER_WORKS:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=256x256:rate=30",
                "-frames:v",
                "3",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
        _ENCODER_WORKS[name] = proc.returncode == 0
    return _ENCODER_WORKS[name]


def pick_encoder(codec: str, *, force: str | None = None) -> str:
    """Choose the encoder for ``codec`` ("h264"/"hevc"/"av1"): the platform's
    hardware encoder when present and working (VideoToolbox on macOS, NVENC
    elsewhere), else the software fallback."""
    if force:
        return force
    if codec not in _SOFTWARE_ENCODER:
        raise ValueError(
            f"unsupported codec {codec!r} (choose from {', '.join(_SOFTWARE_ENCODER)})"
        )
    hw = f"{codec}_videotoolbox" if sys.platform == "darwin" else f"{codec}_nvenc"
    if hw in _available_encoders() and _encoder_works(hw):
        return hw
    sw = _SOFTWARE_ENCODER[codec]
    if sw not in _available_encoders():
        raise RuntimeError(
            f"no usable {codec} encoder: {hw} unavailable and ffmpeg lacks {sw}"
        )
    return sw


def _encoder_args(encoder: str, quality: int) -> list[str]:
    """Constant-quality rate-control argv for the encoder (never a bitrate target:
    -b:v 0 pure CQ avoids overshoot; the encoder spends bits where the content
    needs them instead of chasing the source's rate)."""
    if encoder.endswith("_nvenc"):
        return [
            "-preset",
            "p6",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-bf",
            "4",
            "-b_ref_mode",
            "middle",
            "-temporal-aq",
            "1",
            "-rc-lookahead",
            "32",
        ]
    if encoder.endswith("_videotoolbox"):
        return ["-q:v", str(quality)]
    if encoder == "libsvtav1":
        return ["-preset", "6", "-crf", str(quality)]
    return ["-preset", "slow", "-crf", str(quality)]  # libx264 / libx265


_BIT_DEPTH_RE = re.compile(r"(\d+)(?:le|be)$")


def src_bit_depth(video_stream: dict) -> int:
    """Per-component bit depth of the source video stream (8 when unknown).

    ``bits_per_raw_sample`` is authoritative when ffprobe reports it; otherwise
    the depth is parsed from the pix_fmt's trailing endianness-suffixed digits
    ("yuv420p10le" -> 10). Plain 8-bit formats ("yuv420p", "nv12") carry no such
    suffix and fall through to 8 -- the 12 in nv12 is layout, not depth.
    """
    raw = str(video_stream.get("bits_per_raw_sample") or "")
    if raw.isdigit():
        return int(raw)
    m = _BIT_DEPTH_RE.search(str(video_stream.get("pix_fmt") or ""))
    return int(m.group(1)) if m else 8


def _burn_pix_fmt(encoder: str, src_depth: int) -> str:
    """Output pixel format: match the source bit depth, clamped to what the
    encoder can produce.

    h264 paths are always 8-bit (h264_nvenc cannot encode 10-bit, and 10-bit
    AVC has poor player support). NVENC/VideoToolbox/SVT-AV1 top out at 10-bit,
    so 12-bit sources clamp to 10 there; libx265 keeps 12-bit. Chroma is always
    4:2:0 for playback compatibility.
    """
    hw = encoder.endswith(("_nvenc", "_videotoolbox"))
    if "264" in encoder:  # h264_nvenc / h264_videotoolbox / libx264
        depth = 8
    elif encoder == "libx265":
        depth = max(d for d in (8, 10, 12) if d <= max(src_depth, 8))
    else:  # hevc/av1 nvenc, hevc_videotoolbox, libsvtav1
        depth = max(d for d in (8, 10) if d <= max(src_depth, 8))
    if depth == 8:
        return "nv12" if hw else "yuv420p"
    if hw:
        return "p010le"  # the only >8-bit 4:2:0 format these encoders take
    return f"yuv420p{depth}le"


def _filter_escape(path: str) -> str:
    """Escape a filename for use inside an ffmpeg filtergraph option value."""
    out = path.replace("\\", "/")
    for ch in ("\\", "'", ":", ",", ";", "[", "]"):
        out = out.replace(ch, "\\" + ch)
    return out


def build_burn_cmd(
    media: Path,
    ass_path: Path,
    out: Path,
    *,
    encoder: str,
    quality: int,
    container: str,
    src_depth: int,
    audio_codecs: list[str],
) -> list[str]:
    """Build the ffmpeg argv that burns the styled ASS into the video.

    Video is re-encoded at constant quality; audio is stream-copied (re-encoded
    to AAC only when the target is mp4 and a source codec cannot live there);
    every source subtitle track is dropped (they are burnt in now).
    """
    pix = _burn_pix_fmt(encoder, src_depth)
    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-y",
    ]
    if encoder.endswith("_nvenc"):
        cmd += ["-hwaccel", "cuda"]  # decode assist; frames return to sysmem for libass
    elif encoder.endswith("_videotoolbox"):
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(media)]
    cmd += ["-vf", f"ass={_filter_escape(str(ass_path))},format={pix}"]
    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-c:v", encoder, *_encoder_args(encoder, quality)]
    if container == "mp4" and encoder.split("_")[0] in ("hevc", "libx265"):
        cmd += ["-tag:v", "hvc1"]
    if container == "mp4" and any(c not in _MP4_SAFE_AUDIO for c in audio_codecs):
        logger.warning("audio re-encoded to AAC (source codec not mp4-compatible)")
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += [str(out)]
    return cmd


def burn(
    vtt: Path,
    *,
    media: Path | None = None,
    codec: str = "hevc",
    encoder: str | None = None,
    quality: int | None = None,
    container: str = "mp4",
    font: str = "Arial",
    font_size: int | None = None,
    output: Path | None = None,
) -> Path:
    """Burn the subtitles into the video pixels and write a clean
    (subtitle-track-free) output; return its path. VTT/SRT inputs are rendered
    to a styled ASS at the actual frame size so proportions match the export
    defaults at any resolution; ASS/SSA inputs go to libass as-is, keeping
    their own styling (--font/--font-size are ignored)."""
    from voxweave.export import _timed_rows, ass_header, render_ass
    from voxweave.subformats import load_subtitle_blocks, require_subtitle

    if container not in ("mp4", "mkv"):
        raise ValueError(f"unsupported container {container!r} (choose mp4 or mkv)")
    require_subtitle(vtt)
    native_ass = Path(vtt).suffix.lower() in (".ass", ".ssa")
    src = resolve_media(vtt, media)
    rows = None if native_ass else _timed_rows(load_subtitle_blocks(Path(vtt)))

    streams = probe_streams(src)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{src.name} has no video stream to burn into")
    width = int(video.get("width") or 1920)
    height = int(video.get("height") or 1080)
    depth = src_bit_depth(video)
    audio_codecs = [
        str(s.get("codec_name") or "")
        for s in streams
        if s.get("codec_type") == "audio"
    ]

    enc = pick_encoder(codec, force=encoder)
    q = quality if quality is not None else _DEFAULT_QUALITY.get(enc, 23)
    out = output or default_output(src, container, "burn")

    tmp_ass: Path | None = None
    if native_ass:
        if font != "Arial" or font_size is not None:
            logger.warning(
                "ASS input keeps its own styling; --font/--font-size ignored"
            )
        ass_path = Path(vtt)
    else:
        header = ass_header(width=width, height=height, font=font, font_size=font_size)
        fd, tmp_name = tempfile.mkstemp(suffix=".ass", prefix="voxweave-burn-")
        os.close(fd)  # mkstemp fds leak one per call unless closed explicitly
        tmp_ass = Path(tmp_name)
        assert rows is not None
        tmp_ass.write_text(render_ass(rows, header=header), encoding="utf-8")
        ass_path = tmp_ass
    try:
        cmd = build_burn_cmd(
            src,
            ass_path,
            out,
            encoder=enc,
            quality=q,
            container=container,
            src_depth=depth,
            audio_codecs=audio_codecs,
        )
        logger.info("burning with %s (quality %s, %s)", enc, q, container)
        _run_ffmpeg(cmd, capture=False)  # let ffmpeg -stats stream to the terminal
    finally:
        if tmp_ass is not None:
            tmp_ass.unlink(missing_ok=True)
    return out
