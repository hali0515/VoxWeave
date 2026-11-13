"""``voxweave models``: inspect, pre-download, and clean cached model weights."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import rich_click as click
from rich import filesize
from rich.console import Console
from rich.table import Table
from rich.text import Text

from voxweave import config, lang, runtime

ModelKind = Literal["hf_file", "hf_snapshot", "voiceprint"]
CacheRole = Literal["asr", "align", "audio"]


@dataclass(frozen=True)
class ModelEntry:
    """Catalog entry for a managed runtime model."""

    key: str
    role: CacheRole
    purpose: str
    repo: str
    license: str
    kind: ModelKind
    files: tuple[str, ...] = ()
    gated: bool = False


def _hf_repo_cache_dir(base: Path, repo: str) -> Path:
    return base / f"models--{repo.replace('/', '--')}"


def _dir_bytes(path: Path) -> int:
    if path.is_file() and not path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    if not path.is_dir():
        return 0
    total = 0
    for dirpath, _dirs, filenames in os.walk(path):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.is_symlink():
                continue
            try:
                total += fpath.stat().st_size
            except OSError:
                continue
    return total


def catalog() -> list[ModelEntry]:
    """Build the model catalog for the active compute backend."""
    use_mlx = runtime._use_mlx()
    asr_06 = "mlx-community/Qwen3-ASR-0.6B-8bit" if use_mlx else "Qwen/Qwen3-ASR-0.6B"
    asr_17 = "mlx-community/Qwen3-ASR-1.7B-8bit" if use_mlx else "Qwen/Qwen3-ASR-1.7B"
    qwen_align = (
        "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
        if use_mlx
        else "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    return [
        ModelEntry(
            key="separator",
            role="audio",
            purpose="Vocal separation (Mel-Band RoFormer)",
            repo="KimberleyJSN/melbandroformer",
            files=("MelBandRoformer.ckpt",),
            license="MIT",
            kind="hf_file",
        ),
        ModelEntry(
            key="panns",
            role="audio",
            purpose="Song-skip & SDH event tagging (Cnn14)",
            repo="thelou1s/panns-inference",
            files=("Cnn14_mAP=0.431.pth", "class_labels_indices.csv"),
            license="Apache-2.0",
            kind="hf_file",
        ),
        ModelEntry(
            key="qwen3-asr-0.6b",
            role="asr",
            purpose="Default local ASR",
            repo=asr_06,
            license="Apache-2.0",
            kind="hf_snapshot",
        ),
        ModelEntry(
            key="qwen3-asr-1.7b",
            role="asr",
            purpose="High-accuracy / hybrid punctuation ASR",
            repo=asr_17,
            license="Apache-2.0",
            kind="hf_snapshot",
        ),
        ModelEntry(
            key="qwen3-aligner",
            role="align",
            purpose="zh/yue & fallback forced aligner",
            repo=qwen_align,
            license="Apache-2.0",
            kind="hf_snapshot",
        ),
        ModelEntry(
            key="mms-300m",
            role="align",
            purpose="Japanese CTC forced aligner (ONNX)",
            repo="deskpai/ctc_forced_aligner",
            files=("04ac86b67129634da93aea76e0147ef3.onnx",),
            license="CC-BY-NC-4.0",
            kind="hf_file",
        ),
        ModelEntry(
            key="wav2vec2-en",
            role="align",
            purpose="English CTC forced aligner",
            repo="facebook/wav2vec2-large-960h-lv60-self",
            license="Apache-2.0",
            kind="hf_snapshot",
        ),
        ModelEntry(
            key="diarize-community-1",
            role="audio",
            purpose="Default speaker diarization (--diarize)",
            repo=config.COMMUNITY_DIARIZE_MODEL,
            license="CC-BY-4.0",
            kind="hf_snapshot",
            gated=True,
        ),
        ModelEntry(
            key="diarize-3.1",
            role="audio",
            purpose="Legacy speaker diarization (--diarize-model 3.1)",
            repo=config.LEGACY_DIARIZE_MODEL,
            license="MIT",
            kind="hf_snapshot",
            gated=True,
        ),
        ModelEntry(
            key="redimnet2",
            role="audio",
            purpose="Voiceprint embedder (non-Japanese default)",
            repo="PalabraAI/redimnet2 (v1.0.0 release)",
            files=("b6-vb2+vox2+cnc2_v0-lm.pt",),
            license="CC-BY-NC-SA-4.0",
            kind="voiceprint",
        ),
        ModelEntry(
            key="anime-va",
            role="audio",
            purpose="Voiceprint embedder (Japanese default)",
            repo="litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm",
            files=("model.pth",),
            license="MIT (verify corpus)",
            kind="voiceprint",
        ),
    ]


def _entry_cache_target(root: Path, entry: ModelEntry) -> Path:
    role_dir = root / entry.role
    if entry.key == "redimnet2":
        return role_dir / "redimnet2" / entry.files[0]
    return _hf_repo_cache_dir(role_dir, entry.repo)


def _inspect_entry(root: Path, entry: ModelEntry) -> dict[str, Any]:
    target = _entry_cache_target(root, entry)
    size_bytes = _dir_bytes(target)
    cached = size_bytes > 0
    return {
        "key": entry.key,
        "role": entry.role,
        "purpose": entry.purpose,
        "source": entry.repo,
        "license": entry.license,
        "gated": entry.gated,
        "cached": cached,
        "bytes": size_bytes,
        "path": str(target),
    }


def _default_keys(language: str | None) -> list[str]:
    keys = ["separator", "panns", "qwen3-asr-0.6b"]
    if language:
        iso = lang.to_asr_iso(language)
        if iso == "ja":
            keys.append("mms-300m")
        elif iso == "en":
            keys.append("wav2vec2-en")
        else:
            keys.append("qwen3-aligner")
    else:
        keys.extend(["mms-300m", "wav2vec2-en", "qwen3-aligner"])
    return keys


def _pull_entry(root: Path, entry: ModelEntry) -> str:
    role_dir = root / entry.role
    role_dir.mkdir(parents=True, exist_ok=True)
    if entry.kind == "hf_file":
        last = ""
        for fname in entry.files:
            last = runtime._hf_download(entry.repo, fname, cache_dir=str(role_dir))
        return last
    if entry.kind == "hf_snapshot":
        return runtime._hf_snapshot(entry.repo, str(role_dir))
    from voxweave import voiceembed

    spec = voiceembed.REDIMNET2_B6 if entry.key == "redimnet2" else voiceembed.ANIME_VA
    old_audio = config.AUDIO_CACHE
    try:
        config.AUDIO_CACHE = str(role_dir)
        return str(voiceembed.checkpoint_path(spec))
    finally:
        config.AUDIO_CACHE = old_audio


def _print_models_table(rows: Sequence[dict[str, Any]], root: Path) -> None:
    out = Console(highlight=False)
    table = Table(title=f"VoxWeave Models ({root})", box=None, padding=(0, 2))
    table.add_column("Model", no_wrap=True, style="bold")
    table.add_column("Role", no_wrap=True)
    table.add_column("License", no_wrap=True)
    table.add_column("Cached", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Purpose")
    for row in rows:
        lic = f"{row['license']} [gated]" if row["gated"] else str(row["license"])
        status = (
            Text("yes", style="green") if row["cached"] else Text("no", style="dim")
        )
        size_str = filesize.decimal(int(row["bytes"])) if row["cached"] else "-"
        table.add_row(
            str(row["key"]),
            str(row["role"]),
            Text(lic),
            status,
            size_str,
            Text(str(row["purpose"])),
        )
    out.print(table)


def _register_pull_and_clean(group: click.RichGroup, run: Callable[..., Any]) -> None:
    @group.command("pull", short_help="Pre-download model weights into the cache.")
    @click.argument("models", nargs=-1)
    @click.option(
        "--default",
        "pull_default",
        is_flag=True,
        help="Download the default transcribe pipeline models (separator, PANNs, ASR, aligners).",
    )
    @click.option(
        "--language",
        default=None,
        help="Limit --default aligner download to one language (e.g. ja, en, zh).",
    )
    @click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
    def pull_cmd(
        models: tuple[str, ...],
        pull_default: bool,
        language: str | None,
        as_json: bool,
    ) -> None:
        """Download named models (or ``--default``) so offline runs require no network."""
        by_key = {entry.key: entry for entry in catalog()}
        requested = list(models)
        if pull_default:
            try:
                for key in _default_keys(language):
                    if key not in requested:
                        requested.append(key)
            except ValueError as exc:
                raise click.BadParameter(str(exc), param_hint="'--language'") from exc
        if not requested:
            raise click.UsageError(
                "Pass one or more model names (see `voxweave models list`) or --default."
            )
        unknown = [k for k in requested if k not in by_key]
        if unknown:
            raise click.BadParameter(
                f"Unknown model(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(by_key)}",
                param_hint="'MODELS'",
            )

        root = config.cache_root()

        def _do_pull(rep: Any) -> list[dict[str, str]]:
            rep.plan(tuple(f"pull {k}" for k in requested))
            pulled: list[dict[str, str]] = []
            for key in requested:
                rep.step(f"pull {key}")
                local = _pull_entry(root, by_key[key])
                pulled.append({"key": key, "path": str(local)})
            return pulled

        pulled = run(_do_pull)
        if as_json:
            click.echo(
                json.dumps(
                    {"cache_root": str(root), "pulled": pulled},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for item in pulled:
                click.echo(f"{item['key']}: {item['path']}")

    @group.command(
        "clean", short_help="Remove cached model weights to reclaim disk space."
    )
    @click.argument("models", nargs=-1)
    @click.option("--all", "clean_all", is_flag=True, help="Remove all cached models.")
    @click.option(
        "--dry-run", is_flag=True, help="Report what would be removed without deleting."
    )
    @click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
    def clean_cmd(
        models: tuple[str, ...],
        clean_all: bool,
        dry_run: bool,
        yes: bool,
        as_json: bool,
    ) -> None:
        """Delete cached weights for named models or ``--all`` role directories."""
        if bool(models) == clean_all:
            raise click.UsageError("Specify either MODEL names or --all, not both.")
        root = config.cache_root()
        by_key = {entry.key: entry for entry in catalog()}
        targets: list[Path] = []
        if clean_all:
            for sub in ("asr", "align", "audio"):
                candidate = root / sub
                if candidate.exists():
                    targets.append(candidate)
        else:
            unknown = [k for k in models if k not in by_key]
            if unknown:
                raise click.BadParameter(
                    f"Unknown model(s): {', '.join(unknown)}. "
                    f"Choose from: {', '.join(by_key)}",
                    param_hint="'MODELS'",
                )
            for key in models:
                target = _entry_cache_target(root, by_key[key])
                if target.exists():
                    targets.append(target)

        reclaimed = sum(_dir_bytes(t) for t in targets)
        if targets and not dry_run and not yes:
            click.confirm(
                f"Remove {len(targets)} cached target(s) ({filesize.decimal(reclaimed)}) under {root}?",
                abort=True,
            )
        if not dry_run:
            for target in targets:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)

        payload = {
            "dry_run": dry_run,
            "removed": [str(t) for t in targets],
            "bytes": reclaimed,
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            verb = "Would remove" if dry_run else "Removed"
            click.echo(
                f"{verb} {len(targets)} target(s) ({filesize.decimal(reclaimed)})."
            )


def build_models_group(run: Callable[..., Any]) -> click.RichGroup:
    """Build ``voxweave models``."""

    @click.group(
        "models",
        short_help="List, pre-download, or clean local model weights.",
    )
    def group() -> None:
        """Inspect and manage local model weights under ``VOXWEAVE_CACHE_ROOT``."""

    @group.command("path", short_help="Print model cache directories.")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON object.")
    def path_cmd(as_json: bool) -> None:
        """Print the resolved model cache root and role subdirectories."""
        root = config.cache_root()
        info = {
            "root": str(root),
            "asr": str(root / "asr"),
            "align": str(root / "align"),
            "audio": str(root / "audio"),
        }
        if as_json:
            click.echo(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            click.echo(info["root"])

    @group.command(
        "list", short_help="List managed models, licenses, and cache status."
    )
    @click.option("--json", "as_json", is_flag=True, help="Print JSON array.")
    def list_cmd(as_json: bool) -> None:
        """List every managed model, its role, license, and local cache size."""
        root = config.cache_root()
        rows = [_inspect_entry(root, entry) for entry in catalog()]
        if as_json:
            click.echo(
                json.dumps(
                    {"cache_root": str(root), "models": rows},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_models_table(rows, root)

    _register_pull_and_clean(group, run)
    return group
