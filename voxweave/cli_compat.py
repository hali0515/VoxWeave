"""CLI-only compatibility: explicit commands, path shortcuts, and hidden aliases."""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import rich_click as click
from click.shell_completion import CompletionItem

_WARNED_DEPRECATIONS: set[str] = set()


def warn_deprecated(message: str) -> None:
    """Warn once per deprecated spelling per process, never on stdout."""
    if message not in _WARNED_DEPRECATIONS:
        _WARNED_DEPRECATIONS.add(message)
        click.secho(f"Warning: {message}", fg="yellow", err=True)


class RenamedOption(click.RichOption):
    """Resolve a hidden legacy spelling without silently accepting both spellings."""

    def __init__(self, *args, legacy_name: str, legacy_flag: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.legacy_name = legacy_name
        self.legacy_flag = legacy_flag

    def handle_parse_result(self, ctx, opts, args):
        if self.legacy_name in opts:
            opts = dict(opts)
            if self.name in opts:
                canonical = next(opt for opt in self.opts if opt.startswith("--"))
                raise click.UsageError(
                    f"Use either {canonical} or its legacy alias {self.legacy_flag}, not both.",
                    ctx,
                )
            opts[self.name] = opts[self.legacy_name]
        return super().handle_parse_result(ctx, opts, args)


def renamed_option(*param_decls: str, legacy: str, **kwargs):
    """Define one visible option and a permanently supported, silent hidden alias."""
    legacy_name = "_legacy_" + legacy.lstrip("-").replace("-", "_")

    def decorate(fn):
        fn = click.option(
            legacy,
            legacy_name,
            hidden=True,
            expose_value=False,
            type=kwargs.get("type"),
            multiple=kwargs.get("multiple", False),
        )(fn)
        return click.option(
            *param_decls,
            cls=RenamedOption,
            legacy_name=legacy_name,
            legacy_flag=legacy,
            **kwargs,
        )(fn)

    return decorate


class DeprecatedAlias(click.RichCommand):
    """A hidden command alias sharing the canonical callback and option contract."""

    def __init__(self, name: str, target: click.Command):
        super().__init__(
            name=name,
            callback=target.callback,
            params=target.params,
            help=target.help,
            hidden=True,
            context_settings=target.context_settings,
        )
        self.target_name = target.name

    def invoke(self, ctx):
        warn_deprecated(f"'{self.name}' is deprecated; use '{self.target_name}'.")
        return super().invoke(ctx)


class DefaultGroup(click.RichGroup):
    """Keep native command parsing while allowing a path-shaped default command."""

    default_cmd: click.Command | None = None
    media_extensions: tuple[str, ...] = ()
    _TOKEN = "\x00voxweave-default"

    def get_command(self, ctx, cmd_name):
        if cmd_name == self._TOKEN:
            return self.default_cmd
        return super().get_command(ctx, cmd_name)

    def _path_like(self, token: str) -> bool:
        path = Path(token)
        return (
            any(sep in token for sep in (os.sep, os.altsep) if sep)
            or path.suffix.lower()
            in (*self.media_extensions, ".vtt", ".srt", ".ass", ".ssa", ".json")
            or path.exists()
        )

    def _group_prefix_end(self, ctx, args: list[str]) -> int:
        """Consume only actual group options, including their declared arity."""
        options = {
            spelling: param
            for param in self.get_params(ctx)
            if isinstance(param, click.Option)
            for spelling in (*param.opts, *param.secondary_opts)
        }
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--" or not token.startswith("-"):
                break
            name, equal, _ = token.partition("=")
            option = options.get(name)
            if option is not None:
                index += 1
                if not option.is_flag and not option.count:
                    index += option.nargs - bool(equal)
                continue
            # Native short-option clusters, including an attached option value.
            if token.startswith("--"):
                break
            consumed = 1
            for position, char in enumerate(token[1:], start=1):
                option = options.get("-" + char)
                if option is None:
                    break
                if not option.is_flag and not option.count:
                    consumed += option.nargs - (position + 1 < len(token))
                    break
            if option is None:
                break
            index += consumed
        return index

    def parse_args(self, ctx, args):
        args = list(args)
        if self.default_cmd is not None:
            index = self._group_prefix_end(ctx, args)
            if index < len(args):
                token = args[index]
                if token == "--":
                    # The separator explicitly chooses the file lane, even for
                    # a file named after a command or beginning with a dash.
                    if index + 1 < len(args):
                        args[index:] = [self._TOKEN, *args[index:]]
                elif token not in self.commands and (
                    token.startswith("-") or self._path_like(token)
                ):
                    args.insert(index, self._TOKEN)
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        try:
            name, command, rest = super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if args and "No such command" in exc.message:
                matches = difflib.get_close_matches(
                    args[0],
                    [n for n, cmd in self.commands.items() if not cmd.hidden],
                    n=1,
                )
                hint = (
                    f" Did you mean '{matches[0]}'?"
                    if matches and not getattr(exc, "possibilities", None)
                    else ""
                )
                exc.message += hint + " If this is a media file, pass a path (./name)."
            raise
        if command is self.default_cmd and command is not None:
            name = command.name
        return name, command, rest

    def shell_complete(self, ctx, incomplete):
        results = super().shell_complete(ctx, incomplete)
        if self.default_cmd is not None:
            if incomplete.startswith("-"):
                results.extend(self.default_cmd.shell_complete(ctx, incomplete))
            else:
                results.append(CompletionItem(incomplete, type="file"))
        return results


def require_media(ctx, param, value: Path | None):
    """Reject subtitle/JSON inputs before an ASR invocation can touch artifacts."""
    if value is None or ctx.resilient_parsing:
        return value
    suffix = value.suffix.lower()
    if suffix == ".vtt":
        hint = "Use 'voxweave align FILE.vtt' to align edited text (overwrites VTT and JSON)."
    elif suffix in {".srt", ".ass", ".ssa"}:
        hint = "Use 'voxweave export FILE --format vtt', then explicitly run 'voxweave align FILE.vtt'."
    elif suffix == ".json":
        hint = (
            "Use 'voxweave render FILE.json' to regenerate subtitles without a model."
        )
    else:
        return value
    raise click.BadParameter(f"Transcribe expects audio or video. {hint}", ctx, param)
