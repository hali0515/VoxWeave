from __future__ import annotations

import os
import sys
from pathlib import Path

import rich_click as click

from voxweave import artifacts, config, pipeline
from voxweave.cli_compat import (
    DefaultGroup,
    DeprecatedAlias,
    renamed_option,
    require_media,
)
from voxweave.cli_speakers import build_speakers_group
from voxweave.progress import Reporter
from voxweave.ui import (
    RichReporter,
    correct_summary_panel,
    error_panel,
    install_logging,
    success_panel,
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


def _report_speaker_service(message: str) -> None:
    """Keep the service URL on stdout and session updates on stderr."""
    click.echo(message, err=not message.startswith("http://127.0.0.1:"))


def _flag(value: bool | None, key: str, builtin: bool) -> bool:
    """Resolve a tri-state boolean flag: explicit CLI value > conf ``[defaults].<key>`` > builtin."""
    return value if value is not None else config.conf_default_flag(key, builtin)


_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off"})


def _flag_source(
    value: bool | None,
    key: str,
    builtin: bool,
    *,
    envvar: str | None = None,
) -> tuple[bool, str]:
    if value is not None:
        return value, f"CLI --{key.replace('_', '-')}"
    if envvar is not None and envvar in os.environ:
        raw = os.environ[envvar].strip().lower()
        if raw in _TRUE_FLAG_VALUES:
            return True, f"environment {envvar}"
        if raw in _FALSE_FLAG_VALUES:
            return False, f"environment {envvar}"
        raise ValueError(f"{envvar} must be one of: 1/0, true/false, yes/no, on/off")
    return config.conf_default_flag_source(key, builtin)


def _apply_vad_mask(vad_mask: bool | None) -> None:
    """Propagate the --vad-mask flag to VOXWEAVE_VAD_EMISSION_MASK (read by the backend
    at align time). Precedence: explicit CLI flag > pre-set env > conf [defaults].vad_mask."""
    env = "VOXWEAVE_VAD_EMISSION_MASK"
    if vad_mask is not None:
        os.environ[env] = "1" if vad_mask else "0"
    elif not os.environ.get(env, "").strip() and config.conf_default_flag(
        "vad_mask", False
    ):
        os.environ[env] = "1"


def llm_options(model_envvar: str, model_help: str):
    """Stack the shared --model/--base-url/--api-key-env options for the LLM subcommands.

    Every value is tri-state: an explicit option wins, then the env var, then
    ``[llm]`` in the config file, then the built-in (see config.resolve_llm_*).
    """

    def decorator(fn):
        fn = click.option(
            "--api-key-env",
            default=None,
            help=(
                "Environment variable holding the API key (default: conf "
                f"[llm].api_key_env or {config.DEFAULT_LLM_API_KEY_ENV}; an empty "
                "value declares a keyless endpoint such as a local vLLM)."
            ),
        )(fn)
        fn = click.option(
            "--base-url",
            default=None,
            envvar="OPENAI_BASE_URL",
            help=(
                "OpenAI-compatible endpoint URL (default: OPENAI_BASE_URL env or "
                "conf [llm].base_url; unset = api.openai.com)."
            ),
        )(fn)
        fn = click.option(
            "--model", default=None, envvar=model_envvar, help=model_help
        )(fn)
        return fn

    return decorator


def _llm_model_help(task: str, task_envvar: str) -> str:
    return (
        f"{task} model (default: {task_envvar} env, conf [llm].model, or "
        f"{config.DEFAULT_LLM_MODEL}; '{config.LLM_MODEL_AUTO}' = the endpoint's "
        "only served model)."
    )


# Sent as the bearer token when the endpoint is declared keyless (api_key_env = "").
# The OpenAI client refuses an empty key; vLLM and friends ignore the value.
_KEYLESS_API_KEY = "EMPTY"


def _resolve_llm(
    api_key_env: str | None,
    model: str | None,
    base_url: str | None,
    *,
    task_envvar: str,
) -> tuple[str, dict]:
    """Resolve the API key (panel + exit 1 if unset) and the model/base_url kwargs.

    Each value follows CLI > env > conf [llm] > built-in. A keyless endpoint
    (``api_key_env`` resolved to "") gets a placeholder key instead of an error.
    """
    key_env = config.resolve_llm_api_key_env(api_key_env)
    if key_env == "":
        api_key = _KEYLESS_API_KEY
    else:
        api_key = os.environ.get(key_env) or ""
        if not api_key:
            error_panel(
                RuntimeError(
                    f"API key not found: set env {key_env} (or use --api-key-env / "
                    'conf [llm].api_key_env to name another variable; "" for a '
                    "keyless endpoint)"
                )
            )
            sys.exit(1)
    kwargs: dict = {"model": config.resolve_llm_model(model, task_envvar=task_envvar)}
    resolved_base_url = config.resolve_llm_base_url(base_url)
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return api_key, kwargs


@click.group(
    cls=DefaultGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.rich_config(
    help_config={
        "style_option": "cyan",
        "style_switch": "cyan",
        "style_command": "cyan",
        "style_options_panel_border": "dim",
        "style_commands_panel_border": "dim",
        "style_errors_panel_border": "red",
        "use_click_short_help": True,
        "command_groups": {
            "*": [
                {"name": "Capture", "commands": ["transcribe"]},
                {
                    "name": "Revise",
                    "commands": ["correct", "align", "render", "speakers"],
                },
                {
                    "name": "Deliver",
                    "commands": ["translate", "export", "pack", "burn"],
                },
            ]
        },
        "option_groups": {
            "*transcribe": [
                {
                    "name": "Recognition",
                    "options": ["--language", "--asr-model", "--hybrid", "--context"],
                },
                {
                    "name": "Audio",
                    "options": [
                        "--separate",
                        "--normalize",
                        "--skip-songs",
                        "--keep-lyrics",
                        "--sdh",
                        "--vad-mask",
                    ],
                },
                {
                    "name": "Speakers",
                    "options": [
                        "--diarize",
                        "--diarize-model",
                        "--voiceprints",
                        "--min-speakers",
                        "--max-speakers",
                    ],
                },
                {"name": "Layout", "options": ["--timestamps", "--shot-snap"]},
                {"name": "Diagnostics", "options": ["--debug", "--help"]},
            ]
        },
    }
)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.version_option(package_name="voxweave", message="voxweave %(version)s")
@click.pass_context
def cli(ctx, verbose: bool) -> None:
    """Turn media into editable subtitles and ready-to-share video.

    Capture: transcribe. Revise: correct -> edit -> align -> render.
    Deliver: translate -> export -> pack or burn.

    Shortcut: voxweave MEDIA runs transcribe. Use COMMAND --help for options.
    """
    install_logging(verbose=verbose)
    if ctx.invoked_subcommand not in {"speakers", "help"}:
        config.ensure_default_config()  # write default config template on first run


@click.command(
    "transcribe", short_help="Create VTT + JSON from audio or video (default)."
)
@click.argument(
    "media",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    callback=require_media,
)
@click.option(
    "--language",
    default=None,
    help="Force language (ISO code or full name); default: auto-detect.",
)
@renamed_option(
    "-m",
    "--asr-model",
    "model",
    legacy="--model",
    default=None,
    envvar="VOXWEAVE_ASR_MODEL",
    help=(
        "Local ASR model (default: Qwen3-ASR-0.6B; use qwen3-asr-1.7B or full HF id for higher accuracy; "
        "or faster-whisper: large-v3 / large-v3-turbo / turbo)."
    ),
)
@click.option(
    "--separate/--no-separate",
    default=None,
    help="Separate vocals to remove BGM (default: on, or conf [defaults].separate;"
    " use --no-separate for clean speech to skip GPU separation).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Save intermediate artifacts (fullband/vocals/chunk wavs + ASR raw/alignment) under the"
    " per-media artifact cache for inspection.",
)
@click.option(
    "--normalize/--no-normalize",
    default=None,
    help="Apply loudnorm to the 16k ASR input; useful for uneven volume or quiet"
    " post-separation audio (may boost noise). Default: off, or conf [defaults].normalize.",
)
@click.option(
    "--skip-songs/--no-skip-songs",
    default=None,
    help="Use PANNs to detect and skip music segments on separated vocals before ASR"
    " (default: on, or conf [defaults].skip_songs; prevents OP/ED/insert song hallucinations)."
    " Use --no-skip-songs to transcribe song lyrics or pure music.",
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
    "--diarize/--no-diarize",
    default=None,
    help="Run pyannote speaker diarization: two-speaker cues become Netflix dual-speaker"
    " events (-line per speaker), 3+ speaker cues split at speaker boundaries. The"
    " default install includes support; the gated checkpoint needs VOXWEAVE_HF_TOKEN,"
    " HF_TOKEN, conf hf_token, or a prior 'hf auth login'."
    " Default: off, or conf [defaults].diarize.",
)
@click.option(
    "--diarize-model",
    default=None,
    metavar="MODEL",
    help=(
        "Diarization pipeline: community-1 (default), 3.1, or any Hugging Face "
        "pipeline ID. Precedence: CLI, VOXWEAVE_DIARIZE_MODEL, conf "
        "[diarize].model."
    ),
)
@click.option(
    "--voiceprints/--no-voiceprints",
    default=None,
    help=(
        "Opt in to a biometric speaker-centroid artifact (cache by default; an existing "
        "legacy sidecar is updated). Requires diarization; "
        "precedence is CLI, VOXWEAVE_VOICEPRINTS, conf [defaults].voiceprints, "
        "then off."
    ),
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="Lower bound on the number of speakers for diarization (only used with --diarize;"
    " pass both --min-speakers and --max-speakers when the count is known to steer pyannote).",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="Upper bound on the number of speakers for diarization (only used with --diarize).",
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
    " Runs two ASR passes per chunk (separation only once). Overrides --asr-model."
    " Sub-models: env VOXWEAVE_FUSION_WHISPER / VOXWEAVE_FUSION_QWEN or conf [fusion] whisper/qwen.",
)
@click.option(
    "--timestamps/--no-timestamps",
    default=None,
    help="Include word-level timestamps in VTT (default: on, or conf [defaults].timestamps;"
    " same precision as align output, ready to use)."
    " Use --no-timestamps for a plain-text editing draft; run align afterwards to re-assign timing.",
)
@click.option(
    "--shot-snap/--no-shot-snap",
    default=None,
    help="Detect video shot changes (one downscaled ffmpeg pass) and snap nearby cue"
    " boundaries onto the cuts, so subtitles change on the cut instead of flashing across"
    " it (default: on, or conf [defaults].shot_snap; audio-only media skips automatically)."
    " Cut times persist to the sibling JSON for `render` re-runs; window via VOXWEAVE_SHOT_SNAP_MS.",
)
@click.option(
    "--vad-mask/--no-vad-mask",
    default=None,
    help="Suppress CTC emissions outside speech spans during alignment so words cannot"
    " park in music/silence (recommended for sparse-dialogue movies with songs; keep"
    " off when VAD may misjudge sung/whispered speech). Default: off, or conf"
    " [defaults].vad_mask; same as VOXWEAVE_VAD_EMISSION_MASK=1.",
)
def cmd_transcribe(
    media: Path,
    language: str | None,
    model: str | None,
    separate: bool | None,
    debug: bool,
    normalize: bool | None,
    skip_songs: bool | None,
    keep_lyrics: bool,
    sdh: bool,
    diarize: bool | None,
    diarize_model: str | None,
    voiceprints: bool | None,
    min_speakers: int | None,
    max_speakers: int | None,
    context: str | None,
    hybrid: bool,
    timestamps: bool | None,
    shot_snap: bool | None,
    vad_mask: bool | None,
) -> None:
    """Transcribe audio or video into sibling VTT subtitles and timing JSON.

    Runs ASR and alignment models locally. Also available as voxweave MEDIA.
    Existing sibling outputs are replaced; preserve reviewed edits before rerunning.
    """
    _apply_vad_mask(vad_mask)
    separate = _flag(separate, "separate", True)
    normalize = _flag(normalize, "normalize", False)
    skip_songs = _flag(skip_songs, "skip_songs", True)
    try:
        diarize, diarize_source = _flag_source(diarize, "diarize", False)
        voiceprints, voiceprints_source = _flag_source(
            voiceprints,
            "voiceprints",
            False,
            envvar="VOXWEAVE_VOICEPRINTS",
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    diarize_model = config.resolve_diarize_model(diarize_model)
    if voiceprints and not diarize:
        raise click.UsageError(
            "voiceprint capture is on from "
            f"{voiceprints_source}, but diarization is off from {diarize_source}; "
            "enable --diarize or disable voiceprints"
        )
    timestamps = _flag(timestamps, "timestamps", True)
    shot_snap = _flag(shot_snap, "shot_snap", True)
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
            diarize_model=diarize_model,
            voiceprints=voiceprints,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            asr_model="fusion" if hybrid else (model or config.conf_asr_model()),
            context=context,
            timestamps=timestamps,
            shot_snap=shot_snap,
        )
    )
    dbg_dir = artifacts.claim_paths(media).debug if debug else None
    summary_panel(
        out,
        separated=separate,
        debug_dir=dbg_dir,
        normalized=normalize,
    )
    click.echo(out)  # path -> stdout for script/pipe consumption


cli.default_cmd = cmd_transcribe
cli.media_extensions = pipeline.MEDIA_EXTS
cli.add_command(cmd_transcribe)


cmd_speakers = build_speakers_group(_run, _report_speaker_service)
cli.add_command(cmd_speakers)


@cli.command(
    "render", short_help="Regenerate subtitle layout from sibling JSON; no model."
)
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
    default=None,
    help="Include timestamps in VTT (default: on, or conf [defaults].timestamps;"
    " use --no-timestamps for a plain-text editing draft).",
)
def cmd_split(
    json_path: Path,
    max_line_length: int | None,
    max_lines: int | None,
    timestamps: bool | None,
) -> None:
    """Regenerate subtitle layout from sibling JSON without running a model.

    Accepts JSON, VTT, or media paths. Replaces the sibling VTT and updates JSON.
    """
    timestamps = _flag(timestamps, "timestamps", True)
    kwargs: dict = {}
    if max_line_length is not None:
        kwargs["max_line_length"] = max_line_length
    if max_lines is not None:
        kwargs["max_lines"] = max_lines
    out = _run(
        lambda rep: pipeline.split(
            json_path,
            timestamps=timestamps,
            reporter=rep,
            **kwargs,
        ),
    )
    success_panel(
        "Render done", [f"VTT  : {out}", f"JSON : {pipeline.swap_ext(out, '.json')}"]
    )
    click.echo(out)


cli.add_command(DeprecatedAlias("split", cmd_split))


@cli.command(
    "align", short_help="Re-align edited VTT against audio; overwrites VTT + JSON."
)
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
    default=None,
    help="Use separated vocals at 16k for alignment (default: on, or conf"
    " [defaults].separate; prevents BGM interference; cache hit skips separation;"
    " use --no-separate for clean audio sources).",
)
@click.option(
    "--normalize/--no-normalize",
    default=None,
    help="Apply loudnorm to the 16k alignment input (default: off, or conf [defaults].normalize).",
)
@click.option(
    "--vad-mask/--no-vad-mask",
    default=None,
    help="Suppress CTC emissions outside the JSON's vad_speech spans so words cannot"
    " park in music/silence (recommended for sparse-dialogue movies with songs;"
    " keep off when VAD may misjudge sung/whispered speech). Default: off, or conf"
    " [defaults].vad_mask; same as VOXWEAVE_VAD_EMISSION_MASK=1.",
)
def cmd_align(
    vtt: Path,
    media: Path | None,
    language: str | None,
    separate: bool | None,
    normalize: bool | None,
    vad_mask: bool | None,
) -> None:
    """Re-align after editing: run forced alignment on edited VTT text against the original audio,
    overwrite VTT with timestamps, and update JSON.

    **Loads alignment/separation models locally** (in-process PyTorch, see voxweave.backend); no endpoint calls.
    """
    _apply_vad_mask(vad_mask)
    separate = _flag(separate, "separate", True)
    normalize = _flag(normalize, "normalize", False)
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


@cli.command("translate", short_help="Translate subtitles into a new language sidecar.")
@click.argument(
    "vtt",
    metavar="SUBTITLE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@renamed_option(
    "-t",
    "--target",
    "to",
    legacy="--to",
    default="zh",
    help="Target language code (written to <stem>.<target>.<ext>); default: zh.",
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
    _llm_model_help("Translation", "VOXWEAVE_TRANSLATE_MODEL"),
)
@click.option(
    "--reasoning-effort",
    default=None,
    metavar="EFFORT",
    help=(
        "Reasoning effort accepted by the served model (e.g. low, medium, xhigh). "
        "Default: VOXWEAVE_TRANSLATE_REASONING_EFFORT env, conf [llm].reasoning_effort, "
        "or the endpoint default. Use 'default' to override a configured effort."
    ),
)
def cmd_translate(
    vtt: Path,
    to: str,
    context: str | None,
    glossary: Path | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str,
    reasoning_effort: str | None,
) -> None:
    """Translate after align: call an OpenAI-compatible endpoint for each subtitle cue
    (VTT/SRT/ASS), write <stem>.<to>.<ext> mirroring the input format (original unchanged)."""
    from voxweave.translate import load_glossary

    gloss = load_glossary(glossary) if glossary else None
    api_key, kwargs = _resolve_llm(
        api_key_env, model, base_url, task_envvar="VOXWEAVE_TRANSLATE_MODEL"
    )
    out = _run(
        lambda rep: pipeline.translate(
            vtt,
            to=to,
            context=context,
            glossary=gloss,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            reporter=rep,
            **kwargs,
        )
    )
    translate_summary_panel(out, to=to)
    click.echo(out)


@cli.command("export", short_help="Convert subtitles between VTT, SRT, and ASS.")
@click.argument(
    "vtt",
    metavar="SUBTITLE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@renamed_option(
    "-f",
    "--format",
    "formats",
    legacy="--to",
    multiple=True,
    type=click.Choice(["srt", "ass", "vtt"]),
    default=("srt",),
    help="Output format(s); repeat for several (e.g. -f srt -f ass). Default: srt.",
)
def cmd_export(vtt: Path, formats: tuple[str, ...]) -> None:
    """Convert between subtitle formats: VTT/SRT/ASS in, SRT/ASS/VTT out.

    The aligned VTT + JSON stay the source of truth for voxweave-produced
    subtitles; foreign SRT/ASS files can be exported to VTT to enter the
    editing workflow.
    """
    from voxweave.export import export_subtitles

    def run_export(rep: Reporter) -> list[Path]:
        rep.plan(("export subtitles",))
        rep.step("export subtitles")
        return export_subtitles(vtt, formats)

    paths = _run(run_export)
    success_panel("Export done", [str(path) for path in paths])
    for path in paths:
        click.echo(str(path))


@cli.command("pack", short_help="Add selectable subtitle tracks; no video re-encode.")
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
@renamed_option(
    "--container",
    "container",
    legacy="--to",
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
        lambda rep: mux.pack(
            list(vtts), media=media, container=container, output=output, reporter=rep
        ),
    )
    success_panel("Pack done", [str(out)])
    click.echo(out)


@cli.command("burn", short_help="Burn subtitles into video pixels; re-encodes video.")
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
@renamed_option(
    "--container",
    "container",
    legacy="--to",
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
        lambda rep: mux.burn(
            vtt,
            media=media,
            codec=codec,
            encoder=encoder,
            quality=quality,
            container=container,
            font=font,
            font_size=font_size,
            output=output,
            reporter=rep,
        ),
    )
    success_panel("Burn done", [str(out)])
    click.echo(out)


@cli.command(
    "correct", short_help="Suggest ASR text corrections in a reviewable sidecar."
)
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
    help="Overwrite the original VTT in place (no correction audit) and auto re-align;"
    " default: write adjacent <stem>.asrfix.vtt plus a cache/legacy-lane audit for review.",
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
    _llm_model_help("Correction", "VOXWEAVE_FIX_MODEL"),
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

    By default writes adjacent sidecar ``<stem>.asrfix.vtt`` plus an audit in the per-media
    artifact cache (or an existing adjacent legacy audit lane); the original VTT is untouched.
    ``--apply`` overwrites the original VTT in place (no audit json) and, since the text changed,
    automatically re-runs alignment to refresh timestamps (use ``--no-align`` to skip).
    Safety gate: only applies revisions where orig matches the original text line-for-line.
    """
    from voxweave.translate import load_glossary

    gloss = load_glossary(glossary) if glossary else None
    api_key, kwargs = _resolve_llm(
        api_key_env, model, base_url, task_envvar="VOXWEAVE_FIX_MODEL"
    )
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


@cli.command("help", hidden=True)
@click.argument("commands", nargs=-1)
@click.pass_context
def cmd_help(ctx, commands: tuple[str, ...]) -> None:
    """Show group or nested command help."""
    parent = ctx.parent
    if parent is None:
        return
    command = parent.command
    for name in commands:
        if not isinstance(command, click.Group):
            raise click.UsageError(f"'{command.name}' has no subcommands.")
        child = command.get_command(parent, name)
        if child is None:
            raise click.UsageError(f"No such command '{name}'.")
        parent = child.context_class(child, info_name=name, parent=parent)
        command = child
    click.echo(command.get_help(parent))
