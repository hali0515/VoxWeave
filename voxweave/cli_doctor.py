"""``voxweave doctor``: read-only environment, backend, and dependency diagnostics."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import rich_click as click
from rich import filesize
from rich.console import Console
from rich.table import Table
from rich.text import Text

from voxweave import config, runtime

CheckStatus = Literal["ok", "warn", "error"]

_HW_ENCODERS = (
    "hevc_nvenc",
    "h264_nvenc",
    "av1_nvenc",
    "hevc_videotoolbox",
    "h264_videotoolbox",
)


@dataclass(frozen=True)
class CheckResult:
    """Result of one diagnostic check."""

    name: str
    status: CheckStatus
    summary: str
    hint: str = ""


def _package_version() -> str:
    try:
        return importlib.metadata.version("voxweave")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_platform() -> CheckResult:
    os_arch = f"{platform.system()} {platform.machine()}"
    py_ver = f"Python {platform.python_version()}"
    pkg_ver = f"voxweave {_package_version()}"
    summary = f"{pkg_ver} on {os_arch} ({py_ver})"
    if sys.platform == "win32":
        return CheckResult(
            "platform",
            "error",
            summary,
            "Native Windows is unsupported (POSIX file locks required); run inside WSL2.",
        )
    return CheckResult("platform", "ok", summary)


def _probe_ffmpeg_encoders(ffmpeg_bin: str) -> list[str]:
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    text = proc.stdout
    return [enc for enc in _HW_ENCODERS if f" {enc} " in text]


def check_ffmpeg() -> CheckResult:
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    missing = [
        name
        for name, path in (("ffmpeg", ffmpeg_bin), ("ffprobe", ffprobe_bin))
        if not path
    ]
    if missing:
        return CheckResult(
            "ffmpeg",
            "error",
            f"missing on PATH: {', '.join(missing)}",
            "Install ffmpeg so both ffmpeg and ffprobe are on PATH.",
        )
    assert ffmpeg_bin is not None
    version_line = "ffmpeg found"
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = (proc.stdout or "").splitlines()
        if first:
            version_line = first[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    hw = _probe_ffmpeg_encoders(ffmpeg_bin)
    accel = f"hw encoders: {', '.join(hw)}" if hw else "software encoders only"
    return CheckResult("ffmpeg", "ok", f"{version_line} ({accel})")


def check_compute() -> CheckResult:
    if importlib.util.find_spec("torch") is None:
        return CheckResult(
            "compute",
            "error",
            "PyTorch is not installed",
            "Install voxweave[cuda] (Linux/NVIDIA) or voxweave[mps] (Apple Silicon).",
        )
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "compute",
            "error",
            f"PyTorch failed to import: {exc}",
            "Reinstall the matching platform extra (voxweave[cuda] or voxweave[mps]).",
        )
    device = runtime.get_device()
    backend_name = "mlx" if runtime._use_mlx() else "torch"
    torch_ver = getattr(torch, "__version__", "unknown")
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(idx)
            vram = filesize.decimal(int(props.total_memory))
            gpu_info = f"{props.name}, {vram}"
        except Exception:  # noqa: BLE001
            gpu_info = "CUDA available"
        return CheckResult(
            "compute",
            "ok",
            f"device={device} ({gpu_info}), backend={backend_name}, torch {torch_ver}",
        )
    if device == "mps":
        return CheckResult(
            "compute",
            "ok",
            f"device=mps (Apple Metal), backend={backend_name}, torch {torch_ver}",
        )
    return CheckResult(
        "compute",
        "warn",
        f"device={device} (CPU fallback), backend={backend_name}, torch {torch_ver}",
        "No GPU accelerator detected; separation and ASR will run much slower on CPU.",
    )


def check_asr_backend() -> CheckResult:
    use_mlx = runtime._use_mlx()
    if use_mlx:
        has_mlx = importlib.util.find_spec("mlx_audio") is not None
        has_whisper = importlib.util.find_spec("mlx_whisper") is not None
        if not has_mlx:
            return CheckResult(
                "asr_backend",
                "error",
                "mlx-audio is missing for the active MLX backend",
                'Install the Apple Silicon extra: uv tool install "voxweave[mps]".',
            )
        whisper_note = "mlx-whisper present" if has_whisper else "mlx-whisper missing"
        return CheckResult("asr_backend", "ok", f"mlx-audio present ({whisper_note})")

    has_qwen = importlib.util.find_spec("qwen_asr") is not None
    has_whisper = importlib.util.find_spec("faster_whisper") is not None
    if not has_qwen:
        return CheckResult(
            "asr_backend",
            "error",
            "qwen-asr is missing for the active PyTorch backend",
            "Install the CUDA extra (see docs/install.md) or run `make install` from a clone.",
        )
    whisper_note = "faster-whisper present" if has_whisper else "faster-whisper missing"
    return CheckResult("asr_backend", "ok", f"qwen-asr present ({whisper_note})")


def check_onnxruntime() -> CheckResult:
    if importlib.util.find_spec("onnxruntime") is None:
        return CheckResult(
            "onnxruntime",
            "error",
            "onnxruntime is not installed",
            "Install voxweave[cuda] or voxweave[mps] so Japanese MMS alignment can run.",
        )
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        ort_ver = getattr(ort, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "onnxruntime",
            "error",
            f"onnxruntime failed to initialize: {exc}",
            "Reinstall onnxruntime-gpu (CUDA) or onnxruntime (macOS).",
        )
    device = runtime.get_device()
    if device.startswith("cuda") and "CUDAExecutionProvider" not in providers:
        return CheckResult(
            "onnxruntime",
            "warn",
            f"onnxruntime {ort_ver} providers={providers} (CUDAExecutionProvider missing)",
            "CPU onnxruntime likely shadowed onnxruntime-gpu; reinstall with --overrides "
            "(see docs/install.md) so Japanese MMS alignment runs on GPU.",
        )
    return CheckResult(
        "onnxruntime", "ok", f"onnxruntime {ort_ver} providers={providers}"
    )


def check_config() -> CheckResult:
    path = config.config_path()
    if not path.exists():
        return CheckResult(
            "config",
            "ok",
            f"{path} (not created yet; built-in defaults active)",
        )
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return CheckResult(
            "config",
            "error",
            f"{path} (invalid TOML: {exc})",
            "Fix the TOML syntax in your config file or remove it to regenerate the template.",
        )
    unknown: list[str] = []
    for key, section in data.items():
        if key not in config._KNOWN_KEYS:
            unknown.append(key)
            continue
        known_inner = config._KNOWN_SECTION_KEYS.get(key)
        if known_inner is not None and isinstance(section, dict):
            for inner in section:
                if inner not in known_inner:
                    unknown.append(f"[{key}].{inner}")
    if unknown:
        return CheckResult(
            "config",
            "warn",
            f"{path} (unknown keys ignored: {', '.join(unknown)})",
            "Check setting names against docs/configuration.md.",
        )
    return CheckResult("config", "ok", f"{path} (valid TOML)")


def _hf_token_source() -> str | None:
    for env_key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return f"env {env_key}"
    conf_tok = config._load().get("hf_token")
    if isinstance(conf_tok, str) and conf_tok.strip():
        return "config hf_token"
    try:
        import huggingface_hub

        if (huggingface_hub.get_token() or "").strip():
            return "huggingface_hub stored token"
    except Exception:  # noqa: BLE001
        pass
    return None


def check_hf_auth() -> CheckResult:
    diarize_default = config.conf_default_flag("diarize", False)
    has_pyannote = importlib.util.find_spec("pyannote") is not None
    if diarize_default and not has_pyannote:
        return CheckResult(
            "hf_auth",
            "error",
            "pyannote.audio is not installed while [defaults].diarize = true",
            "Install the optional [diarize] extra: voxweave[cuda,diarize] or voxweave[mps,diarize].",
        )
    source = _hf_token_source()
    if source is not None:
        note = "" if has_pyannote else ", [diarize] extra not installed"
        return CheckResult("hf_auth", "ok", f"configured ({source}{note})")
    if diarize_default:
        return CheckResult(
            "hf_auth",
            "error",
            "no Hugging Face token found while [defaults].diarize = true",
            "Run `hf auth login` or set VOXWEAVE_HF_TOKEN / HF_TOKEN for gated pyannote models.",
        )
    extra_note = "" if has_pyannote else "; install [diarize] extra to enable"
    return CheckResult(
        "hf_auth",
        "ok",
        f"not configured (only required for --diarize gated models{extra_note})",
    )


def _cache_stats(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        return 0, 0
    files = 0
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if Path(dirpath).name == "snapshots":
            continue
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.is_symlink():
                continue
            try:
                total_bytes += fpath.stat().st_size
                files += 1
            except OSError:
                continue
    return files, total_bytes


def check_cache() -> CheckResult:
    root = config.cache_root()
    if not root.exists():
        return CheckResult("cache", "ok", f"{root} (empty; created on first download)")
    files, total_bytes = _cache_stats(root)
    return CheckResult(
        "cache",
        "ok",
        f"{root} ({files} files, {filesize.decimal(total_bytes)})",
    )


def run_all_checks() -> list[CheckResult]:
    """Run every read-only environment check in deterministic order."""
    return [
        check_platform(),
        check_ffmpeg(),
        check_compute(),
        check_asr_backend(),
        check_onnxruntime(),
        check_config(),
        check_hf_auth(),
        check_cache(),
    ]


_STATUS_STYLE: dict[CheckStatus, str] = {
    "ok": "green",
    "warn": "yellow",
    "error": "red",
}


def _print_report(checks: list[CheckResult]) -> None:
    out = Console(highlight=False)
    table = Table(title="VoxWeave Environment Check", box=None, padding=(0, 2))
    table.add_column("Check", no_wrap=True, style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Details")
    hints: list[tuple[str, CheckStatus, str]] = []
    for item in checks:
        style = _STATUS_STYLE[item.status]
        table.add_row(
            item.name,
            Text(item.status.upper(), style=style),
            Text(item.summary),
        )
        if item.hint:
            hints.append((item.name, item.status, item.hint))
    out.print(table)
    for name, status, hint in hints:
        line = Text(f"  {name}: ", style=_STATUS_STYLE[status])
        line.append(hint)
        out.print(line)


def build_doctor_command() -> click.Command:
    """Build ``voxweave doctor``."""

    @click.command(
        "doctor",
        short_help="Check ffmpeg, GPU backend, ONNX providers, config, and auth.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Print JSON report.")
    @click.pass_context
    def doctor_command(ctx: click.Context, as_json: bool) -> None:
        """Run read-only environment diagnostics without loading models or writing files.

        Exits 0 when every required check passes (warnings stay 0), or 1 when
        any check reports an error.
        """
        checks = run_all_checks()
        ok = all(item.status != "error" for item in checks)
        if as_json:
            payload = {
                "ok": ok,
                "checks": [asdict(item) for item in checks],
            }
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_report(checks)
        if not ok:
            ctx.exit(1)

    return doctor_command
