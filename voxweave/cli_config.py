"""``voxweave config``: inspect effective configuration and manage ``voxweave.conf``."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from voxweave import config, voicelibrary


def _env_or_conf_source(
    env_var: str | None,
    conf_present: bool,
    conf_label: str,
) -> str:
    if env_var and (os.environ.get(env_var) or "").strip():
        return f"env {env_var}"
    if conf_present:
        return f"config {conf_label}"
    return "default"


def _hf_auth_status(raw_conf: dict[str, Any]) -> tuple[str, str]:
    for env_key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (os.environ.get(env_key) or "").strip():
            return "<redacted>", f"env {env_key}"
    conf_val = raw_conf.get("hf_token")
    if isinstance(conf_val, str) and conf_val.strip():
        return "<redacted>", "config hf_token"
    try:
        import huggingface_hub

        if (huggingface_hub.get_token() or "").strip():
            return "<redacted>", "huggingface_hub"
    except Exception:  # noqa: BLE001
        pass
    return "unset", "default"


def _dict_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    sec = raw.get(name)
    return sec if isinstance(sec, dict) else {}


def collect_effective_config() -> dict[str, Any]:
    """Resolve effective settings and their precedence sources without exposing secrets."""
    conf_file = config.config_path()
    raw = config._load()
    voices_loc = voicelibrary.resolve_voices_dir(None)
    hf_val, hf_src = _hf_auth_status(raw)

    asr_val = (
        (os.environ.get("VOXWEAVE_ASR_MODEL") or "").strip()
        or config.conf_asr_model()
        or config.DEFAULT_ASR_MODEL
    )
    asr_src = _env_or_conf_source(
        "VOXWEAVE_ASR_MODEL", config.conf_asr_model() is not None, "asr_model"
    )

    load_val = config.conf_load_strategy()
    load_src = _env_or_conf_source(
        "VOXWEAVE_LOAD_STRATEGY", "load_strategy" in raw, "load_strategy"
    )

    dp_val = config.conf_ctc_max_dp_frames()
    dp_src = _env_or_conf_source(
        "VOXWEAVE_CTC_MAX_DP_FRAMES", "ctc_max_dp_frames" in raw, "ctc_max_dp_frames"
    )

    diarize_sec = _dict_section(raw, "diarize")
    diarize_model = config.resolve_diarize_model(None)
    diarize_model_src = _env_or_conf_source(
        "VOXWEAVE_DIARIZE_MODEL", "model" in diarize_sec, "[diarize].model"
    )
    try:
        clustering = config.resolve_diarize_clustering(None)
    except ValueError:
        clustering = f"invalid ({config.DEFAULT_DIARIZE_CLUSTERING} fallback)"
    clustering_src = _env_or_conf_source(
        config.DIARIZE_CLUSTERING_ENV,
        "clustering" in diarize_sec,
        "[diarize].clustering",
    )

    vp_sec = _dict_section(raw, "voiceprint")
    from voxweave import voiceembed

    try:
        vp_model = voiceembed.resolve_voiceprint_choice(None)
    except ValueError:
        vp_model = "invalid"
    vp_src = _env_or_conf_source(
        voiceembed.ENV_MODEL, "model" in vp_sec, "[voiceprint].model"
    )

    llm_sec = _dict_section(raw, "llm")
    llm_model = config.resolve_llm_model(None, task_envvar="VOXWEAVE_TRANSLATE_MODEL")
    llm_model_src = _env_or_conf_source(
        "VOXWEAVE_TRANSLATE_MODEL", "model" in llm_sec, "[llm].model"
    )
    llm_base = config.resolve_llm_base_url(None) or "https://api.openai.com/v1"
    llm_base_src = _env_or_conf_source(
        "OPENAI_BASE_URL", "base_url" in llm_sec, "[llm].base_url"
    )
    llm_key_env = config.resolve_llm_api_key_env(None)
    llm_key_src = "config [llm].api_key_env" if "api_key_env" in llm_sec else "default"

    flag_defs = (
        ("separate", True, None),
        ("skip_songs", True, None),
        ("normalize", False, None),
        ("diarize", False, None),
        ("voiceprints", False, "VOXWEAVE_VOICEPRINTS"),
        ("timestamps", True, None),
        ("shot_snap", True, None),
        ("vad_mask", False, "VOXWEAVE_VAD_EMISSION_MASK"),
    )
    defaults_out: dict[str, dict[str, Any]] = {}
    for key, builtin, env_key in flag_defs:
        if env_key and (os.environ.get(env_key) or "").strip():
            val_raw = os.environ[env_key].strip().lower()
            val: bool | str = (
                True
                if val_raw in {"1", "true", "yes", "on"}
                else False
                if val_raw in {"0", "false", "no", "off"}
                else f"invalid ({val_raw})"
            )
            src = f"env {env_key}"
        else:
            val, src = config.conf_default_flag_source(key, builtin)
        defaults_out[key] = {"value": val, "source": src}

    entries = [
        {
            "key": "config_file",
            "value": str(conf_file),
            "source": "exists" if conf_file.exists() else "not created",
        },
        {
            "key": "cache_root",
            "value": str(config.cache_root()),
            "source": _env_or_conf_source("VOXWEAVE_CACHE_ROOT", False, ""),
        },
        {
            "key": "voices_dir",
            "value": str(voices_loc.root),
            "source": voices_loc.source,
        },
        {"key": "asr_model", "value": asr_val, "source": asr_src},
        {"key": "load_strategy", "value": load_val, "source": load_src},
        {"key": "ctc_max_dp_frames", "value": dp_val, "source": dp_src},
        {"key": "diarize.model", "value": diarize_model, "source": diarize_model_src},
        {"key": "diarize.clustering", "value": clustering, "source": clustering_src},
        {"key": "voiceprint.model", "value": vp_model, "source": vp_src},
        {"key": "llm.model", "value": llm_model, "source": llm_model_src},
        {"key": "llm.base_url", "value": llm_base, "source": llm_base_src},
        {
            "key": "llm.api_key_env",
            "value": llm_key_env or "(keyless)",
            "source": llm_key_src,
        },
        {"key": "hf_token", "value": hf_val, "source": hf_src},
    ]
    for flag_key, info in defaults_out.items():
        entries.append(
            {
                "key": f"defaults.{flag_key}",
                "value": info["value"],
                "source": info["source"],
            }
        )
    return {
        "config_path": str(conf_file),
        "exists": conf_file.exists(),
        "settings": entries,
    }


def _print_config_table(report: dict[str, Any]) -> None:
    out = Console(highlight=False)
    table = Table(title="VoxWeave Effective Configuration", box=None, padding=(0, 2))
    table.add_column("Setting", no_wrap=True, style="bold")
    table.add_column("Value")
    table.add_column("Source", no_wrap=True, style="dim")
    for row in report["settings"]:
        table.add_row(
            str(row["key"]),
            Text(str(row["value"])),
            Text(str(row["source"])),
        )
    out.print(table)


def build_config_group(run: Callable[..., Any]) -> click.RichGroup:
    """Build ``voxweave config``."""

    @click.group(
        "config",
        short_help="Show effective settings, config file path, or write a template.",
    )
    def group() -> None:
        """Inspect effective settings and manage the user ``voxweave.conf`` file."""

    @group.command("path", short_help="Print the resolved config file path.")
    def path_cmd() -> None:
        """Print the resolved ``voxweave.conf`` path without creating it."""
        click.echo(config.config_path())

    @group.command("show", short_help="Display resolved settings and their sources.")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON object.")
    def show_cmd(as_json: bool) -> None:
        """Display effective configuration values and where each value came from."""
        report = run(lambda _rep: collect_effective_config(), reporter=False)
        if as_json:
            click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_config_table(report)

    @group.command(
        "init", short_help="Write the commented default voxweave.conf template."
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Overwrite an existing config file with the default template.",
    )
    def init_cmd(force: bool) -> None:
        """Create the commented ``voxweave.conf`` template at the resolved path."""

        def _write(_rep: object) -> Path:
            target = config.config_path()
            if target.exists() and not force:
                raise FileExistsError(
                    f"{target} already exists; pass --force to overwrite it"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(config._TEMPLATE, encoding="utf-8")
            return target

        written = run(_write, reporter=False)
        click.echo(written)

    return group
