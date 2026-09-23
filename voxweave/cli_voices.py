"""``voxweave voices``: inspect and curate the global voice library."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import rich_click as click
from rich.console import Console
from rich.table import Table

_DIR_HELP = (
    "Voice library directory (default: VOXWEAVE_VOICES_DIR, conf [voices].dir, "
    "or ~/.local/share/voxweave/voices)."
)


def _voices_dir_option(fn: Callable[..., Any]) -> Callable[..., Any]:
    return click.option(
        "--voices-dir",
        type=click.Path(exists=False, file_okay=False, path_type=Path),
        help=_DIR_HELP,
    )(fn)


def _read(voices_dir: Path | None, *, spaces: Sequence[str] | None = None) -> Any:
    from voxweave import voicelibrary

    location = voicelibrary.resolve_voices_dir(voices_dir)
    with voicelibrary.library_lock(location.root, exclusive=False):
        return location, voicelibrary.read_state(location.root, spaces=spaces)


def _counts(row: Mapping[str, object]) -> str:
    counts = cast(Mapping[str, int], row["exemplars"])
    if not counts:
        return "-"
    return ", ".join(f"{space}: {count}" for space, count in counts.items())


def _print_rows(rows: Sequence[Mapping[str, object]], *, title: str) -> None:
    table = Table(title=title, box=None, padding=(0, 2))
    for column in ("ID", "Name", "Scopes", "Voice samples"):
        table.add_column(column, no_wrap=column == "ID")
    for row in rows:
        table.add_row(
            cast(str, row["id"]),
            cast(str, row["display_name"]),
            ", ".join(cast(list[str], row["scopes"])) or "-",
            _counts(row),
        )
    Console(markup=False).print(table)


def _print_identity(detail: Mapping[str, object]) -> None:
    console = Console(markup=False, highlight=False)
    aliases = cast(list[str], detail["aliases"])
    console.print(f"ID       {detail['id']}")
    console.print(f"Name     {detail['display_name']}")
    console.print(f"Aliases  {', '.join(aliases) or '-'}")
    console.print(f"Scopes   {', '.join(cast(list[str], detail['scopes'])) or '-'}")
    console.print(f"Created  {detail['created']}")
    console.print(f"Updated  {detail['updated']}")
    spaces = cast(Mapping[str, Mapping[str, object]], detail["spaces"])
    if not spaces:
        console.print("No voice samples.")
    for name, space in spaces.items():
        exemplars = cast(list[Mapping[str, object]], space["exemplars"])
        console.print(f"{name}: {len(exemplars)} voice sample(s)")
        for item in exemplars:
            media = item["media_path"] or "media path not recorded"
            console.print(
                f"  {item['id']}  {item['scope']} / {item['episode']}  "
                f"{item['added']}  {media}"
            )


def build_voices_group(run: Callable[..., Any]) -> click.RichGroup:
    """Build the voice-library CLI around the application's error wrapper."""

    @click.group(
        "voices",
        cls=click.RichGroup,
        short_help="Inspect, rename, forget, or import saved voices.",
    )
    def group() -> None:
        """Manage the voice library that `speakers enroll` fills.

        The library holds voice biometrics of the people you name, shared by
        every media folder (and by every machine that points at the same
        directory). `forget` removes one person from it completely.
        """

    @group.command("list", short_help="List saved identities.")
    @click.option("--scope", help="Only identities enrolled under this scope.")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON.")
    @_voices_dir_option
    def list_command(scope: str | None, as_json: bool, voices_dir: Path | None) -> None:
        """List identities with their scopes and voice samples per embedding space."""
        from voxweave import voicelibrary

        def collect(_rep: object) -> tuple[Any, list[dict[str, object]]]:
            location, state = _read(voices_dir)
            return location, voicelibrary.list_identities(state, scope=scope)

        location, rows = run(collect, reporter=False)
        if as_json:
            payload = {"library": str(location.root), "identities": rows}
            click.echo(json.dumps(payload, ensure_ascii=False))
            return
        if not rows:
            click.echo(f"No saved voices in {location.root}.")
            return
        _print_rows(rows, title=f"Voice library {location.root}")

    @group.command("show", short_help="Show one identity by id or name.")
    @click.argument("query", metavar="ID|NAME")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON.")
    @_voices_dir_option
    @click.pass_context
    def show_command(
        ctx: click.Context, query: str, as_json: bool, voices_dir: Path | None
    ) -> None:
        """Show an identity's names, scopes and voice samples (never vectors).

        A name shared by several identities lists them and exits non-zero;
        repeat the command with one of the ids.
        """
        from voxweave import voicelibrary

        def collect(_rep: object) -> tuple[Any, list[str]]:
            _location, state = _read(voices_dir)
            return state, voicelibrary.find_identities(state, query)

        state, matches = run(collect, reporter=False)
        if not matches:
            raise click.ClickException(f"no saved identity matches {query!r}")
        if len(matches) > 1:
            click.echo(
                f"{query!r} names {len(matches)} identities; pass one id:", err=True
            )
            _print_rows(
                [voicelibrary.identity_summary(state, item) for item in matches],
                title="Candidates",
            )
            ctx.exit(1)
        detail = voicelibrary.describe_identity(state, matches[0])
        if as_json:
            click.echo(json.dumps(detail, ensure_ascii=False))
        else:
            _print_identity(detail)

    @group.command("rename", short_help="Rename an identity everywhere.")
    @click.argument("identity_id", metavar="ID")
    @click.argument("new_name", metavar="NEW_NAME")
    @_voices_dir_option
    def rename_command(
        identity_id: str, new_name: str, voices_dir: Path | None
    ) -> None:
        """Rename one identity; the name applies in every embedding space."""
        from voxweave import voicelibrary

        def apply(_rep: object) -> None:
            root = voicelibrary.resolve_voices_dir(voices_dir).root
            if not root.is_dir():
                raise voicelibrary.UnknownIdentity(f"no voice library at {root}")
            with voicelibrary.library_lock(root, exclusive=True):
                state = voicelibrary.read_state(root, spaces=[])
                change = voicelibrary.rename_identity(state, identity_id, new_name)
                voicelibrary.commit(state, change)

        run(apply, reporter=False)
        click.echo(f"{identity_id}: {new_name}")

    @group.command("forget", short_help="Delete one person from the library.")
    @click.argument("identity_id", metavar="ID")
    @click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
    @_voices_dir_option
    def forget_command(identity_id: str, yes: bool, voices_dir: Path | None) -> None:
        """Remove an identity and every voice sample of it, in every space.

        The id stays as a tombstone, so it is never suggested, imported or
        enrolled again. The history keeps entries about the id, with no name,
        no scope or episode label and no voice data. Per-folder stores from
        earlier versions are not edited: any that still hold the id are
        listed, to be deleted by hand.
        """
        from voxweave import voicelibrary

        def describe(_rep: object) -> dict[str, object]:
            _location, state = _read(voices_dir)
            return voicelibrary.identity_summary(state, identity_id)

        summary = run(describe, reporter=False)
        samples = sum(cast(Mapping[str, int], summary["exemplars"]).values())
        if not yes:
            click.confirm(
                f"Forget {summary['display_name']} ({identity_id}) and "
                f"{samples} voice sample(s)?",
                abort=True,
            )

        def apply(_rep: object) -> tuple[dict[str, int], list[Path]]:
            root = voicelibrary.resolve_voices_dir(voices_dir).root
            with voicelibrary.library_lock(root, exclusive=True):
                state = voicelibrary.read_state(root)
                candidates = voicelibrary.legacy_store_candidates(state, identity_id)
                change, removed = voicelibrary.forget_identity(state, identity_id)
                voicelibrary.commit(state, change)
                return removed, candidates

        removed, candidates = run(apply, reporter=False)
        click.echo(
            f"forgot {identity_id}: {sum(removed.values())} voice sample(s) removed"
        )
        for path, count in voicelibrary.legacy_stores_holding(candidates, identity_id):
            click.echo(
                f"warning: the per-folder store {path} still holds {count} voice "
                f"sample(s) of {identity_id}. VoxWeave no longer reads them for "
                "this id, but they stay on disk until that file is deleted (it "
                "may hold other people too).",
                err=True,
            )

    @group.command("import", short_help="Merge a per-show voices store.")
    @click.argument(
        "store_path",
        metavar="LEGACY_JSON",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--scope",
        help="Scope for the imported voices (default: for a voxweave.voices.json, "
        "its folder's scope and the store's show; otherwise the store's show). "
        "Importing again with another scope adds it.",
    )
    @_voices_dir_option
    def import_command(
        store_path: Path, scope: str | None, voices_dir: Path | None
    ) -> None:
        """Merge a voxweave.voices.json store into the voice library.

        Ids are kept, so importing the same store again adds nothing. The
        store itself is left unchanged. By default its voices keep the
        scopes under which the store already served suggestions.
        """
        from voxweave import voicelibrary
        from voxweave.voicestore import load_voice_store, shared_store_lock

        def apply(_rep: object) -> tuple[Path, Any]:
            with shared_store_lock(store_path) as handle:
                store, validated = load_voice_store(handle.store_path)
            space_name, _fingerprint = voicelibrary.space_identity(
                cast(Mapping[str, object], store["provenance"])
            )
            scopes = (
                (scope,)
                if scope is not None
                else voicelibrary.default_import_scopes(
                    Path(os.path.abspath(store_path)), validated.show
                )
            )
            location = voicelibrary.resolve_voices_dir(voices_dir)
            root = location.root
            with voicelibrary.library_lock(
                root, exclusive=True, create_parents=location.default
            ):
                state = voicelibrary.read_state(root, spaces=[space_name])
                change, summary = voicelibrary.import_store(
                    state,
                    store,
                    scope=scopes[0],
                    also_scopes=scopes[1:],
                    source_label=str(handle.store_path),
                )
                voicelibrary.commit(state, change)
            return root, summary

        root, summary = run(apply, reporter=False)
        for message in summary.refused:
            click.echo(f"skipped {message}", err=True)
        for forgotten_id in summary.forgotten:
            click.echo(
                f"skipped {forgotten_id}: forgotten in this library "
                "(`voxweave voices forget`)",
                err=True,
            )
        superseded = (
            f", {summary.exemplars_superseded} older than the samples already kept"
            if summary.exemplars_superseded
            else ""
        )
        scopes_added = (
            f"; {summary.scopes_added} scope(s) added to existing identities"
            if summary.scopes_added
            else ""
        )
        scope_list = ("scope " if len(summary.scopes) == 1 else "scopes ") + ", ".join(
            repr(item) for item in summary.scopes
        )
        click.echo(
            f"imported {summary.identities_created} identities and "
            f"{summary.exemplars_added} voice sample(s) into {root} "
            f"({scope_list}, space {summary.space}); "
            f"{summary.exemplars_present} already present{superseded}{scopes_added}"
        )

    @group.command("where", short_help="Print the voice library directory.")
    @_voices_dir_option
    def where_command(voices_dir: Path | None) -> None:
        """Print the library directory, then which setting chose it."""
        from voxweave import voicelibrary

        location = voicelibrary.resolve_voices_dir(voices_dir)
        click.echo(str(location.root))
        click.echo(f"source: {location.source}")

    return group
