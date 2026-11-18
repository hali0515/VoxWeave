"""``voxweave completions``: generate shell completion scripts for bash, zsh, and fish."""

from __future__ import annotations

import rich_click as click
from click.shell_completion import get_completion_class


def build_completions_command() -> click.Command:
    """Build ``voxweave completions``."""

    @click.command(
        "completions",
        short_help="Print shell completion script for bash, zsh, or fish.",
    )
    @click.argument(
        "shell",
        type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    )
    @click.pass_context
    def completions_command(ctx: click.Context, shell: str) -> None:
        """Print the tab-completion script for SHELL to stdout.

        Install examples:

          bash:  eval "$(voxweave completions bash)"
          zsh:   eval "$(voxweave completions zsh)"
          fish:  voxweave completions fish | source
        """
        shell_key = shell.lower()
        comp_cls = get_completion_class(shell_key)
        if comp_cls is None:
            raise click.BadParameter(f"Unsupported shell: {shell}")
        root_cmd = ctx.find_root().command
        comp = comp_cls(root_cmd, {}, "voxweave", "_VOXWEAVE_COMPLETE")
        click.echo(comp.source())

    return completions_command
