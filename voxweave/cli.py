from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from voxweave import config, pipeline
from voxweave.ui import (
    RichReporter,
    correct_summary_panel,
    error_panel,
    install_logging,
    summary_panel,
    translate_summary_panel,
)


def _run(fn, *, reporter: bool = True):
    """Run a pipeline call, rendering a unified error panel and exiting 1 on any failure.

    ``fn`` receives a :class:`RichReporter` (or ``None`` when ``reporter=False``). Centralises
    the try/RichReporter/except wrapper shared by every subcommand.
    """
    try:
        if reporter:
            with RichReporter() as rep:
                return fn(rep)
        return fn(None)
    except Exception as exc:  # noqa: BLE001 - top-level catch-all, render unified error panel
        error_panel(exc)
        sys.exit(1)


def llm_options(model_envvar: str, model_help: str):
    """Stack the shared --model/--base-url/--api-key-env options for the LLM subcommands."""

    def decorator(fn):
        fn = click.option(
            "--api-key-env",
            default="OPENAI_API_KEY",
            help="Environment variable to read the API key from (default: OPENAI_API_KEY).",
        )(fn)
        fn = click.option(
            "--base-url",
            default=None,
            envvar="OPENAI_BASE_URL",
            help="OpenAI-compatible endpoint URL.",
        )(fn)
        fn = click.option(
            "--model", default=None, envvar=model_envvar, help=model_help
        )(fn)
        return fn

    return decorator


def _resolve_llm(
    api_key_env: str, model: str | None, base_url: str | None
) -> tuple[str, dict]:
    """Resolve the API key (panel + exit 1 if unset) and build the model/base_url kwargs dict."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        error_panel(
            RuntimeError(
                f"API key not found: set env {api_key_env} (or use --api-key-env to specify another variable)"
            )
        )
        sys.exit(1)
    kwargs: dict = {}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs["base_url"] = base_url
    return api_key, kwargs


class DefaultGroup(click.Group):
    """`voxweave <media>` runs transcription without an explicit subcommand.

    ``default_cmd`` is not in ``self.commands`` (invisible in help, not callable as
    `voxweave transcribe`). When the first token is not a known subcommand or group
    option, the private token is injected at the front of ``args`` during
    ``parse_args`` — this handles ``voxweave --debug a.mkv`` where options precede
    the media arg (injecting at resolve_command time would choke on ``--debug`` first).
    """

    default_cmd: click.Command | None = None
    _GROUP_OPTS = frozenset({"-h", "--help", "-v", "--verbose", "--version"})
    # Not typeable by users; resolve_command returns cmd.name so usage strings still show "transcribe".
    _TOKEN = "\x00voxweave-default"

    def get_command(self, ctx, cmd_name):
        if self.default_cmd is not None and cmd_name == self._TOKEN:
            return self.default_cmd
        return super().get_command(ctx, cmd_name)

    def _needs_default(self, token: str) -> bool:
        return (
            self.default_cmd is not None
            and token != self._TOKEN
            and token not in self.commands
            and token not in self._GROUP_OPTS
        )

    def parse_args(self, ctx, args):
        if args and self._needs_default(args[0]):
            args = [self._TOKEN, *args]
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        # After group-level options are consumed a bare media arg may remain — inject default.
        if args and not args[0].startswith("-") and self._needs_default(args[0]):
            args = [self._TOKEN, *args]
        cmd_name, cmd, rest = super().resolve_command(ctx, args)
        if cmd is not None and cmd is self.default_cmd:
            cmd_name = (
                cmd.name
            )  # show "transcribe" in usage strings, not the private token
        return cmd_name, cmd, rest


@click.group(
    cls=DefaultGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.version_option(package_name="voxweave", message="voxweave %(version)s")
def cli(verbose: bool) -> None:
    """Qwen3 subtitle pipeline orchestrator.

    Run `voxweave <media>` directly to transcribe (no `transcribe` subcommand needed).
    `--debug` implies local mode.
    """
    install_logging(verbose=verbose)
    config.ensure_default_config()  # write default config template on first run


@click.command("transcribe")
@click.argument("media", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--language",
    default=None,
    help="Force language (ISO code or full name); default: auto-detect.",
)
@click.option(
    "--model",
    default=None,
    envvar="VOXWEAVE_ASR_MODEL",
    help=(
        "Local ASR model (default: Qwen3-ASR-0.6B; use qwen3-asr-1.7B or full HF id for higher accuracy; "
        "or faster-whisper: large-v3 / large-v3-turbo / turbo)."
    ),
)
@click.option(
    "--separate/--no-separate",
    default=True,
    help="Separate vocals to remove BGM (default: on; use --no-separate for clean speech to skip GPU separation).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Save intermediate artifacts (fullband/vocals/chunk wavs + ASR raw/alignment) to debug/<stem>/ for"
    " inspection (implies local mode: artifacts are only written during local orchestration).",
)
@click.option(
    "--normalize",
    is_flag=True,
    default=False,
    help="Apply loudnorm to the 16k ASR input; useful for uneven volume or quiet post-separation audio (may boost noise).",
)
@click.option(
    "--skip-songs/--no-skip-songs",
    default=True,
    help="Use PANNs to detect and skip music segments on separated vocals before ASR (default: on;"
    " prevents OP/ED/insert song hallucinations). Use --no-skip-songs to transcribe song lyrics or pure music.",
)
@click.option(
    "--keep-lyrics",
    is_flag=True,
    default=False,
    help="Transcribe detected songs instead of skipping them: sung cues are flagged and"
    " wrapped with music notes (overrides --skip-songs excision; detection still runs;"
    " export to ASS renders them italic).",
)
@click.option(
    "--sdh",
    is_flag=True,
    default=False,
    help="Also write <stem>.sdh.vtt: PANNs-detected non-speech event tags ([explosion],"
    " [phone ringing], ...) merged into the dialogue in speech-free gaps; runs on the"
    " original mix (main VTT/JSON untouched).",
)
@click.option(
    "--diarize",
    is_flag=True,
    default=False,
    help="Run pyannote speaker diarization: two-speaker cues become Netflix dual-speaker"
    " events (-line per speaker), 3+ speaker cues split at speaker boundaries. Needs"
    " 'voxweave[diarize]' + an HF token for the gated checkpoint (VOXWEAVE_HF_TOKEN).",
)
@click.option(
    "--context",
    default=None,
    envvar="VOXWEAVE_ASR_CONTEXT",
    help="ASR bias prompt (free text: names/terms/proper nouns, comma or newline separated);"
    " biases transcription toward these tokens, reducing errors on names and loanwords."
    " Bare term lists are auto-framed as 'Proper nouns: ...' for Qwen (a bare list regresses"
    " accuracy); prose or pre-framed text passes through. Reused for all chunks.",
)
@click.option(
    "--hybrid",
    is_flag=True,
    default=False,
    help="Dual-ASR fusion: whisper for accurate text + Qwen-1.7B for punctuation positions (merged timeline)."
    " Better text than pure Qwen for ja/en; better segmentation than pure whisper (which emits no punctuation)."
    " Runs two ASR passes per chunk (separation only once). Overrides --model."
    " Sub-models: env VOXWEAVE_FUSION_WHISPER / VOXWEAVE_FUSION_QWEN or conf [fusion] whisper/qwen.",
)
@click.option(
    "--timestamps/--no-timestamps",
    default=True,
    help="Include word-level timestamps in VTT (default: on, same precision as align output, ready to use)."
    " Use --no-timestamps for a plain-text editing draft; run align afterwards to re-assign timing.",
)
@click.option(
    "--shot-snap/--no-shot-snap",
    default=True,
    help="Detect video shot changes (one downscaled ffmpeg pass) and snap nearby cue"
    " boundaries onto the cuts, so subtitles change on the cut instead of flashing across"
    " it (default: on; audio-only media skips automatically). Cut times persist to the"
    " sibling JSON for `split` re-runs; window via VOXWEAVE_SHOT_SNAP_MS.",
)
@click.option(
    "--vad-mask",
    is_flag=True,
    default=False,
    help="Suppress CTC emissions outside speech spans during alignment so words cannot"
    " park in music/silence (recommended for sparse-dialogue movies with songs; keep"
    " off when VAD may misjudge sung/whispered speech). Same as"
    " VOXWEAVE_VAD_EMISSION_MASK=1.",
)
def cmd_transcribe(
    media: Path,
    language: str | None,
    model: str | None,
    separate: bool,
    debug: bool,
    normalize: bool,
    skip_songs: bool,
    keep_lyrics: bool,
    sdh: bool,
    diarize: bool,
    context: str | None,
    hybrid: bool,
    timestamps: bool,
    shot_snap: bool,
    vad_mask: bool,
) -> None:
    """Media -> (vocal separation) -> VAD -> local ASR/alignment -> smart_split -> write VTT+JSON."""
    if vad_mask:  # flag form of the env knob; backend reads the env at align time
        os.environ["VOXWEAVE_VAD_EMISSION_MASK"] = "1"
    out = _run(
        lambda rep: pipeline.process(
            media,
            lang_override=language,
            separate=separate,
            reporter=rep,
            debug=debug,
            normalize=normalize,
            skip_songs=skip_songs,
            keep_lyrics=keep_lyrics,
            sdh=sdh,
            diarize=diarize,
            asr_model="fusion" if hybrid else (model or config.conf_asr_model()),
            context=context,
            timestamps=timestamps,
            shot_snap=shot_snap,
        )
    )
    dbg_dir = Path("debug") / media.stem if debug else None
    summary_panel(
        out,
        separated=separate,
        debug_dir=dbg_dir,
        normalized=normalize,
    )
    click.echo(out)  # path -> stdout for script/pipe consumption


cli.default_cmd = (
    cmd_transcribe  # bare `voxweave <media>` routes here; not listed in help
)


@cli.command("split")
@click.argument(
    "json_path",
    metavar="JSON",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--max-line-length", type=int, default=None, help="Maximum characters per line."
)
@click.option("--max-lines", type=int, default=None, help="Maximum lines per cue.")
@click.option(
    "--timestamps/--no-timestamps",
    default=True,
    help="Include timestamps in VTT (default: on; use --no-timestamps for a plain-text editing draft).",
)
def cmd_split(
    json_path: Path,
    max_line_length: int | None,
    max_lines: int | None,
    timestamps: bool,
) -> None:
    """Offline re-layout: re-run smart_split from <stem>.json without running any models."""
    kwargs: dict = {}
    if max_line_length is not None:
        kwargs["max_line_length"] = max_line_length
    if max_lines is not None:
        kwargs["max_lines"] = max_lines
    out = _run(
        lambda _rep: pipeline.split(json_path, timestamps=timestamps, **kwargs),
        reporter=False,
    )
    click.echo(out)


@cli.command("align")
@click.argument("vtt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--media",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Source media path (default: look for same-name file in same directory); required for forced alignment.",
)
@click.option(
    "--language",
    default=None,
    help="Force language (ISO code or full name); default: read from JSON.",
)
@click.option(
    "--separate/--no-separate",
    default=True,
    help="Use separated vocals at 16k for alignment (default: on, prevents BGM interference;"
    " cache hit skips separation; use --no-separate for clean audio sources).",
)
@click.option(
    "--normalize",
    is_flag=True,
    default=False,
    help="Apply loudnorm to the 16k alignment input.",
)
@click.option(
    "--vad-mask",
    is_flag=True,
    default=False,
    help="Suppress CTC emissions outside the JSON's vad_speech spans so words cannot"
    " park in music/silence (recommended for sparse-dialogue movies with songs;"
    " keep off when VAD may misjudge sung/whispered speech). Same as"
    " VOXWEAVE_VAD_EMISSION_MASK=1.",
)
def cmd_align(
    vtt: Path,
    media: Path | None,
    language: str | None,
    separate: bool,
    normalize: bool,
    vad_mask: bool,
) -> None:
    """Re-align after editing: run forced alignment on edited VTT text against the original audio,
    overwrite VTT with timestamps, and update JSON.

    **Loads alignment/separation models locally** (in-process PyTorch, see voxweave.backend); no endpoint calls.
    """
    if vad_mask:  # flag form of the env knob; backend reads the env at align time
        os.environ["VOXWEAVE_VAD_EMISSION_MASK"] = "1"
    out = _run(
        lambda rep: pipeline.align(
            vtt,
            media_path=media,
            separate=separate,
            normalize=normalize,
            lang_override=language,
            reporter=rep,
        )
    )
    summary_panel(out, separated=separate, normalized=normalize)
    click.echo(out)


@cli.command("translate")
@click.argument(
    "vtt",
    metavar="SUBTITLE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--to",
    default="zh",
    help="Target language code (written to <stem>.<to>.<ext>); default: zh.",
)
@click.option(
    "--context", default=None, help="Show/tone context injected into the prompt."
)
@click.option(
    "--glossary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Term/name glossary (.json -> mapping dict; any other format -> passed as raw text prompt).",
)
@llm_options(
    "VOXWEAVE_TRANSLATE_MODEL",
    "Translation model (default: VOXWEAVE_TRANSLATE_MODEL env or gpt-5.3-chat-latest).",
)
def cmd_translate(
    vtt: Path,
    to: str,
    context: str | None,
    glossary: Path | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str,
) -> None:
    """Translate after align: call OpenAI to translate each cue in a subtitle file
    (VTT/SRT/ASS), write <stem>.<to>.<ext> mirroring the input format (original unchanged)."""
    from voxweave.translate import load_glossary

    gloss = load_glossary(glossary) if glossary else None
    api_key, kwargs = _resolve_llm(api_key_env, model, base_url)
    out = _run(
        lambda rep: pipeline.translate(
            vtt,
            to=to,
            context=context,
            glossary=gloss,
            api_key=api_key,
            reporter=rep,
            **kwargs,
        )
    )
    translate_summary_panel(out, to=to)
    click.echo(out)


@cli.command("export")
@click.argument(
    "vtt",
    metavar="SUBTITLE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--to",
    "formats",
    multiple=True,
    type=click.Choice(["srt", "ass", "vtt"]),
    default=("srt",),
    help="Output format(s); repeat for several (e.g. --to srt --to ass). Default: srt.",
)
def cmd_export(vtt: Path, formats: tuple[str, ...]) -> None:
    """Convert between subtitle formats: VTT/SRT/ASS in, SRT/ASS/VTT out.

    The aligned VTT + JSON stay the source of truth for voxweave-produced
    subtitles; foreign SRT/ASS files can be exported to VTT to enter the
    editing workflow.
    """
    from voxweave.export import export_subtitles

    for path in export_subtitles(vtt, formats):
        click.echo(str(path))


@cli.command("pack")
@click.argument(
    "vtts",
    metavar="SUBTITLE...",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--media",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Source media (default: sibling file with the same stem; language tags"
    " like .zh are stripped for the lookup).",
)
@click.option(
    "--to",
    "container",
    type=click.Choice(["mkv", "mp4", "webm"]),
    default=None,
    help="Output container (default: keep the source container when it can store"
    " text subtitles, else mkv).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path (default: <media stem>.<container>, or <stem>.pack.<container>"
    " when that would overwrite the source).",
)
def cmd_pack(
    vtts: tuple[Path, ...],
    media: Path | None,
    container: str | None,
    output: Path | None,
) -> None:
    """Mux subtitle file(s) (VTT/SRT/ASS) into the media as soft subtitle tracks
    (stream copy, no re-encode).

    Each track is titled "VoxWeave <Language>" with the container language tag set
    from the filename (episode.zh.vtt -> chi / "VoxWeave Chinese"); the first
    packed track is flagged default. ASS keeps its styling in mkv targets.
    Existing streams are preserved (mp4/webm targets drop image-based subtitle
    tracks they cannot store).
    """
    from voxweave import mux

    out = _run(
        lambda _rep: mux.pack(
            list(vtts), media=media, container=container, output=output
        ),
        reporter=False,
    )
    click.echo(out)


@cli.command("burn")
@click.argument(
    "vtt",
    metavar="SUBTITLE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--media",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Source media (default: sibling file with the same stem; language tags"
    " like .zh are stripped for the lookup).",
)
@click.option(
    "--codec",
    type=click.Choice(["hevc", "h264", "av1"]),
    default="hevc",
    help="Video codec (default: hevc — 10-bit capable, ~40% smaller than h264 at"
    " equal quality, plays everywhere as hvc1 mp4; pick h264 only for legacy"
    " devices, av1 for maximum compression on recent hardware).",
)
@click.option(
    "--encoder",
    default=None,
    help="Force a specific ffmpeg encoder (default: auto — VideoToolbox on macOS,"
    " NVENC on NVIDIA, libx264/libx265/libsvt-av1 software fallback).",
)
@click.option(
    "--quality",
    type=int,
    default=None,
    help="Constant-quality value (NVENC -cq / x264-x265-svtav1 -crf, lower = better;"
    " VideoToolbox -q:v 1-100, higher = better). Default per encoder: h264 19 /"
    " hevc 23 / av1 30 / VideoToolbox 65. Bitrate is never targeted.",
)
@click.option(
    "--to",
    "container",
    type=click.Choice(["mp4", "mkv"]),
    default="mp4",
    help="Output container (default: mp4 for maximum player compatibility).",
)
@click.option(
    "--font",
    default="Arial",
    help="Subtitle font family (fontconfig resolves fallbacks; e.g."
    " 'Noto Sans CJK SC' for Chinese).",
)
@click.option(
    "--font-size",
    type=int,
    default=None,
    help="Font size in script pixels (default: 72 scaled to the video height).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path (default: <media stem>.<container>, or <stem>.burn.<container>"
    " when that would overwrite the source).",
)
def cmd_burn(
    vtt: Path,
    media: Path | None,
    codec: str,
    encoder: str | None,
    quality: int | None,
    container: str,
    font: str,
    font_size: int | None,
    output: Path | None,
) -> None:
    """Burn subtitles (VTT/SRT/ASS) into the video pixels and drop all subtitle tracks.

    VTT/SRT inputs render to a styled ASS sized to the actual frame; ASS inputs
    keep their own styling. Video re-encodes at constant quality with hardware
    acceleration when available (NVENC / VideoToolbox), preserves the source
    bit depth (10-bit stays 10-bit on hevc/av1), and audio is stream-copied
    (mp4 targets re-encode incompatible codecs to AAC).
    """
    from voxweave import mux

    out = _run(
        lambda _rep: mux.burn(
            vtt,
            media=media,
            codec=codec,
            encoder=encoder,
            quality=quality,
            container=container,
            font=font,
            font_size=font_size,
            output=output,
        ),
        reporter=False,
    )
    click.echo(out)


@cli.command("correct")
@click.argument("vtt", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--glossary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Term/name glossary (.json -> mapping dict; any other format -> raw text prompt); strongly recommended for ambiguous proper nouns.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Overwrite the original VTT in place (no sidecar json) and auto re-align; default: write sidecar <stem>.asrfix.vtt for review.",
)
@click.option(
    "--align/--no-align",
    "do_align",
    default=True,
    help="With --apply, automatically re-run alignment afterwards to refresh timestamps (default: on).",
)
@click.option(
    "--media",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Source media for the auto re-align (default: sibling file with the same stem).",
)
@llm_options(
    "VOXWEAVE_FIX_MODEL",
    "Correction model (default: VOXWEAVE_FIX_MODEL env or gpt-5.3-chat-latest).",
)
def cmd_correct(
    vtt: Path,
    glossary: Path | None,
    apply: bool,
    do_align: bool,
    media: Path | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str,
) -> None:
    """Pre-align LLM correction: fix obvious ASR errors, split words, and garbled proper nouns; produce a reviewable diff.

    By default writes only sidecar ``<stem>.asrfix.vtt`` + audit ``<stem>.asrfix.json`` (original
    VTT untouched). ``--apply`` overwrites the original VTT in place (no audit json) and, since the
    text changed, automatically re-runs alignment to refresh timestamps (use ``--no-align`` to skip).
    Safety gate: only applies revisions where orig matches the original text line-for-line.
    """
    from voxweave.translate import load_glossary

    gloss = load_glossary(glossary) if glossary else None
    api_key, kwargs = _resolve_llm(api_key_env, model, base_url)
    res = _run(
        lambda rep: pipeline.correct(
            vtt,
            glossary=gloss,
            api_key=api_key,
            apply=apply,
            align_after=apply and do_align,
            media_path=media,
            reporter=rep,
            **kwargs,
        )
    )
    correct_summary_panel(res)
    click.echo(res["out"])
