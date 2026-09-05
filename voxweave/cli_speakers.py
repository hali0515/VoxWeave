"""Speaker command group, compatibility routes, and read-only episode inspection."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table

from voxweave import artifacts, pipeline
from voxweave.cli_compat import DefaultGroup, warn_deprecated


class SpeakersGroup(DefaultGroup):
    """Keep a bare episode path as the audition shortcut."""

    media_extensions = (*pipeline.MEDIA_EXTS, ".vtt", ".json")


def _episode_owner(episode: Path, *, require_media: bool = False) -> Path:
    owner = pipeline._artifact_owner(episode)
    if require_media and owner.suffix.lower() not in pipeline.MEDIA_EXTS:
        raise FileNotFoundError(
            f"source media not found for {episode}; pass the episode's media path"
        )
    return owner


def _voiceprint_state(
    owner: Path, sibling: dict[str, Any]
) -> tuple[str, Path | None, set[str]]:
    from voxweave.voicebase import (
        Phase2DataError,
        load_voiceprints,
        validate_voiceprint_conjunction,
    )

    path = artifacts.legacy_path(owner, ".voiceprints.json")
    if not artifacts.path_present(path):
        cached = artifacts.inspect_paths(owner)
        if cached is None or not artifacts.path_present(cached.voiceprints):
            return "absent", None, set()
        path = cached.voiceprints
    try:
        raw, validated = load_voiceprints(path)
    except (OSError, ValueError, UnicodeError):
        return "invalid", path, set()
    if not {"voiceprint_capture", "voiceprint_media"}.issubset(sibling):
        return "unbound", path, set(validated.speakers)
    try:
        # This checks persisted bindings, never claims to verify today's media bytes.
        validate_voiceprint_conjunction(raw, sibling, sibling["voiceprint_media"])
    except Phase2DataError:
        return "stale", path, set(validated.speakers)
    return "recorded", path, set(validated.speakers)


def _list_episode(episode: Path) -> dict[str, Any]:
    from voxweave.speakers import load_speaker_mapping_bytes
    from voxweave.voicebase import strict_json_object_loads, strict_turn_projection

    owner = _episode_owner(episode)
    sibling_path = pipeline.swap_ext(owner, ".json")
    raw = sibling_path.read_bytes()
    sibling = strict_json_object_loads(
        raw, max_bytes=max(1, len(raw)), source=sibling_path.name
    )
    turns = strict_turn_projection(sibling.get("speaker_turns", []))
    labels = sorted({label for _start, _end, label in turns})
    mapping_path = pipeline.inspect_speakers_mapping_path(owner, reference=sibling_path)
    names = (
        load_speaker_mapping_bytes(
            mapping_path.read_bytes(), labels, source=mapping_path.name
        )
        if artifacts.path_present(mapping_path)
        else {}
    )
    voiceprints, voiceprints_path, recorded = _voiceprint_state(owner, sibling)
    rows = []
    for label in labels:
        spans = [(start, end) for start, end, speaker in turns if speaker == label]
        rows.append(
            {
                "id": label,
                "name": names.get(label),
                "turns": len(spans),
                "duration_seconds": round(sum(end - start for start, end in spans), 3),
                "voiceprint": (
                    voiceprints
                    if label in recorded or voiceprints == "invalid"
                    else "absent"
                ),
            }
        )
    return {
        "episode": str(owner),
        "sibling": str(sibling_path),
        "mapping": str(mapping_path) if artifacts.path_present(mapping_path) else None,
        "voiceprints": {
            "state": voiceprints,
            "path": str(voiceprints_path) if voiceprints_path is not None else None,
            "media_checked": False,
        },
        "speakers": rows,
    }


def _print_list(data: dict[str, Any]) -> None:
    table = Table(title="Episode speakers", box=None, padding=(0, 2))
    for column in ("ID", "Name", "Turns", "Seconds", "Voiceprint"):
        table.add_column(column, no_wrap=column != "Name")
    for speaker in data["speakers"]:
        table.add_row(
            speaker["id"],
            speaker["name"] or "-",
            str(speaker["turns"]),
            f"{speaker['duration_seconds']:.3f}",
            speaker["voiceprint"],
        )
    console = Console(markup=False)
    console.print(table)
    console.print(
        f"Voiceprints: {data['voiceprints']['state']} (saved evidence; media not checked)"
    )


def build_speakers_group(
    run: Callable[..., Any], report: Callable[[str], None]
) -> click.RichGroup:
    """Build the speaker CLI using the application's error wrapper and service output."""

    episode_type = click.Path(exists=False, dir_okay=False, path_type=Path)
    voices_type = click.Path(exists=False, dir_okay=False, path_type=Path)

    @click.group(
        "speakers",
        cls=SpeakersGroup,
        short_help="Review speaker names and manage saved voices.",
    )
    def group() -> None:
        """Review and name speakers, manage saved voices, or inspect an episode.

        A bare EPISODE starts the audition service. Use Ctrl+C after saving.
        """

    @group.command("enroll", short_help="Save reviewed voices to a show store.")
    @click.argument("episode_path", metavar="EPISODE", type=episode_type)
    @click.option("--voices", type=voices_type, help="Voices store to update.")
    @click.option("--show", help="Show name; confirms a discovered voices store.")
    @click.option("--episode", help="Enrollment label (default: media stem).")
    @click.option(
        "--replace", is_flag=True, help="Replace this episode's existing voice samples."
    )
    @click.option("--replace-episode", is_flag=True, hidden=True)
    def enroll_command(
        episode_path: Path,
        voices: Path | None,
        show: str | None,
        episode: str | None,
        replace: bool,
        replace_episode: bool,
    ) -> None:
        """Add reviewed names and voice samples to a voices store."""
        from voxweave.speakers import enroll_speaker_voices

        if replace and replace_episode:
            raise click.UsageError(
                "use either --replace or --replace-episode, not both"
            )
        if replace_episode:
            warn_deprecated(
                "--replace-episode is deprecated; use speakers enroll --replace"
            )
        out = run(
            lambda _rep: enroll_speaker_voices(
                _episode_owner(episode_path, require_media=True),
                voices=voices,
                show=show,
                episode=episode,
                replace_episode=replace or replace_episode,
            ),
            reporter=False,
        )
        click.echo(out)

    @group.command("purge", short_help="Remove voice data; keep reviewed names.")
    @click.argument("episode_path", metavar="EPISODE", type=episode_type)
    def purge_command(episode_path: Path) -> None:
        """Remove cached voice data and suggestions; preserve reviewed names.

        The original media file may be absent.
        """
        from voxweave.speakers import purge_voiceprints

        removed = run(
            lambda _rep: purge_voiceprints(_episode_owner(episode_path)), reporter=False
        )
        for path in removed:
            click.echo(path)

    @group.command("serve", short_help="Review speaker names in a local browser.")
    @click.argument("episode_path", metavar="EPISODE", type=episode_type)
    @click.option(
        "--voices", type=voices_type, help="Explicit voices store for name suggestions."
    )
    @click.option(
        "--show",
        help="Required for suggestions from a discovered store; unnecessary with --voices.",
    )
    @click.option(
        "--manual", is_flag=True, help="Name speakers without voice matching."
    )
    @click.option("--no-match", is_flag=True, hidden=True)
    @click.option(
        "--port",
        type=click.IntRange(0, 65535),
        default=0,
        metavar="PORT",
        show_default=True,
        help="Loopback HTTP port; 0 selects an available port.",
    )
    @click.option(
        "--open/--no-open",
        "open_browser",
        default=True,
        help="Open the audition in a browser.",
    )
    @click.option("--enroll", is_flag=True, hidden=True)
    @click.option("--purge-voiceprints", is_flag=True, hidden=True)
    @click.option("--episode", hidden=True)
    @click.option("--replace-episode", is_flag=True, hidden=True)
    @click.pass_context
    def serve_command(
        ctx: click.Context,
        episode_path: Path,
        voices: Path | None,
        show: str | None,
        manual: bool,
        no_match: bool,
        port: int,
        open_browser: bool,
        enroll: bool,
        purge_voiceprints: bool,
        episode: str | None,
        replace_episode: bool,
    ) -> None:
        """Listen to speakers and save their names in a local browser page."""
        from voxweave.speakers import create_speaker_audition
        from voxweave.speakerserve import serve

        if manual and no_match:
            raise click.UsageError("use either --manual or --no-match, not both")
        if no_match:
            warn_deprecated("--no-match is deprecated; use speakers serve --manual")
        manual = manual or no_match
        if purge_voiceprints:
            if any(
                (
                    voices,
                    show,
                    manual,
                    port,
                    not open_browser,
                    enroll,
                    episode,
                    replace_episode,
                )
            ):
                raise click.UsageError(
                    "--purge-voiceprints is mutually exclusive with generation/enrollment options"
                )
            warn_deprecated(
                "--purge-voiceprints is deprecated; use speakers purge EPISODE"
            )
            ctx.invoke(purge_command, episode_path=episode_path)
            return
        if enroll:
            if manual:
                raise click.UsageError("manual mode cannot be combined with --enroll")
            if port or not open_browser:
                raise click.UsageError(
                    "--port/--no-open cannot be combined with --enroll"
                )
            warn_deprecated("--enroll is deprecated; use speakers enroll EPISODE")
            ctx.invoke(
                enroll_command,
                episode_path=episode_path,
                voices=voices,
                show=show,
                episode=episode,
                replace=False,
                replace_episode=replace_episode,
            )
            return
        if episode is not None or replace_episode:
            raise click.UsageError(
                "--episode/--replace-episode require speakers enroll"
            )

        def prepare(_rep: object) -> Any:
            owner = _episode_owner(episode_path, require_media=True)
            if voices is None and show is None and not manual:
                return create_speaker_audition(owner)
            return create_speaker_audition(
                owner, voices=voices, show=show, no_match=manual
            )

        audition = run(prepare, reporter=False)
        run(
            lambda _rep: serve(
                page=audition.page,
                media_path=audition.media_path,
                mapping_path=audition.mapping_path,
                sibling_path=audition.sibling_json_path,
                speaker_ids=audition.speaker_ids,
                pristine_mapping_generation=audition.pristine_mapping_generation,
                port=port,
                open_browser=open_browser,
                report=report,
            ),
            reporter=False,
        )

    @group.command("list", short_help="Inspect saved speaker IDs, names, and turns.")
    @click.argument("episode_path", metavar="EPISODE", type=episode_type)
    @click.option(
        "--json", "as_json", is_flag=True, help="Print the saved speaker data as JSON."
    )
    def list_command(episode_path: Path, as_json: bool) -> None:
        """List speaker IDs, names, turn counts, and saved voice data without a model."""
        data = run(lambda _rep: _list_episode(episode_path), reporter=False)
        if as_json:
            click.echo(json.dumps(data, ensure_ascii=False))
        else:
            _print_list(data)

    group.default_cmd = serve_command
    return group
