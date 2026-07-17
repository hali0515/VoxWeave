# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate==1.12.0",
#   "kernels==0.12.0",
#   "torch==2.12.0",
#   "transformers==5.9.0",
# ]
# ///
"""Isolated JSONL worker for optional Qwen3.5/Qwen3.6 boundary inference.

This file is executed with ``uv run --no-project --script``.  Its PEP 723
environment must remain independent of VoxWeave's qwen-asr environment.  Keep
all torch/Transformers imports inside runtime functions so importing this file
for protocol tests has no heavyweight side effects.

Stdout is reserved for JSONL protocol frames.  Third-party logging and ordinary
``print`` calls are redirected to stderr by :func:`main` before any model import.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TextIO, cast


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_PROTOCOL_VERSION = 1
_WORKER_VERSION = "1"


def _offline_enabled() -> bool:
    return os.environ.get("VOXWEAVE_OFFLINE", "").strip().casefold() in _TRUE_ENV_VALUES


def _require_fp8_cuda(torch_module: Any) -> tuple[int, int]:
    """Reject configurations where Transformers silently falls back to BF16."""

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "semantic FP8 inference requires an NVIDIA CUDA GPU; "
            "BF16 fallback is intentionally disabled"
        )
    capability = tuple(torch_module.cuda.get_device_capability(0))
    if len(capability) != 2 or capability < (8, 9):
        actual = ".".join(str(part) for part in capability)
        raise RuntimeError(
            "semantic FP8 inference requires compute capability >= 8.9 "
            f"(found {actual or 'unknown'}); BF16 fallback is intentionally disabled"
        )
    return capability[0], capability[1]


def _configure_fp8_matmul_dispatch(
    capability: tuple[int, int],
    *,
    integration_module: Any | None = None,
) -> str:
    """Use DeepGEMM only on the architecture its downloaded build targets.

    Transformers 5.9 treats every SM90+ device as DeepGEMM-compatible.  The
    current ``kernels-community/deep-gemm`` binary used by this locked runtime
    is built for ``sm_90a`` only; CUDA can load it on Blackwell, but its first
    matmul aborts with ``Unknown recipe`` instead of raising ``ImportError`` and
    reaching Transformers' documented Triton fallback.  Keep Hopper on the
    faster path and force the universal finegrained-FP8 Triton kernel elsewhere.
    Both paths remain W8A8 FP8.
    """

    if capability[0] == 9:
        return "deepgemm"
    if integration_module is None:
        integration_module = importlib.import_module(
            "transformers.integrations.finegrained_fp8"
        )
    loader = getattr(integration_module, "_load_deepgemm_kernel", None)
    if loader is None:
        raise RuntimeError("Transformers fine-grained FP8 dispatch hook is missing")
    clear = getattr(loader, "cache_clear", None)
    if callable(clear):
        clear()

    def triton_only() -> Any:
        raise ImportError(
            "DeepGEMM is disabled on non-Hopper GPUs; using the universal "
            "finegrained-FP8 Triton kernel"
        )

    integration_module._load_deepgemm_kernel = triton_only
    print(
        "semantic FP8 matmul: using Triton on compute capability "
        f"{capability[0]}.{capability[1]}",
        file=sys.stderr,
    )
    return "triton"


def _assert_fp8_model(model: Any, quantization_config: Any, torch_module: Any) -> int:
    """Prove broad FP8 coverage while permitting tiny non-tileable gate weights.

    The 128x128 quantizer intentionally passes through matrices whose dimensions
    cannot form one block.  Qwen3.5's recurrent ``a/b`` gates are 16xhidden and
    require BF16 exponent range (FP16/per-tensor FP8 produced non-finite logits
    in real inference).  They are accepted only when structurally non-tileable,
    below 1% of all text-linear weights, and the rest of the stack is >=80% FP8.
    """

    actual_config = getattr(
        getattr(model, "hf_quantizer", None),
        "quantization_config",
        quantization_config,
    )
    if bool(getattr(quantization_config, "dequantize", False)) or bool(
        getattr(actual_config, "dequantize", False)
    ):
        raise RuntimeError(
            "Transformers attempted to dequantize the semantic model; "
            "broad BF16 fallback is intentionally disabled"
        )
    fp8_dtype = getattr(torch_module, "float8_e4m3fn", None)
    fp16_dtype = getattr(torch_module, "float16", None)
    bf16_dtype = getattr(torch_module, "bfloat16", None)
    if fp8_dtype is None:
        raise RuntimeError("installed Torch build has no float8_e4m3fn support")

    named_modules = getattr(model, "named_modules", None)
    modules = (
        list(cast(Iterable[tuple[str, Any]], named_modules()))
        if callable(named_modules)
        else [("", module) for module in model.modules()]
    )
    block_size = getattr(actual_config, "weight_block_size", (128, 128))
    valid_block = (
        isinstance(block_size, (tuple, list))
        and len(block_size) == 2
        and all(isinstance(value, int) and value > 0 for value in block_size)
    )

    fp8_total = 0
    fp8_layers = 0
    invalid_fp8: list[str] = []
    text_linear_total = 0
    linear_elements = 0
    residual_elements = 0
    torch_linear = getattr(getattr(torch_module, "nn", None), "Linear", None)
    excluded_roots = {
        "visual",
        "vision",
        "projector",
        "multi_modal_projector",
        "lm_head",
        "embed_tokens",
    }

    for name, module in modules:
        module_type = type(module).__name__
        is_fp8 = module_type == "FP8Linear"
        if is_fp8:
            fp8_total += 1
        excluded = bool(set(str(name).casefold().split(".")) & excluded_roots)
        is_plain_linear = module_type == "Linear" or (
            torch_linear is not None and isinstance(module, torch_linear)
        )
        weight = getattr(module, "weight", None)
        if not excluded and (is_fp8 or is_plain_linear):
            text_linear_total += 1
            numel = getattr(weight, "numel", None)
            if callable(numel):
                linear_elements += cast(Callable[[], int], numel)()
        if not is_fp8:
            continue
        verified = (
            weight is not None
            and getattr(weight, "dtype", None) == fp8_dtype
            and callable(getattr(weight, "element_size", None))
            and weight.element_size() == 1
            and getattr(module, "weight_scale_inv", None) is not None
        )
        if verified:
            fp8_layers += 1
            continue

        shape = tuple(getattr(weight, "shape", ()))
        non_tileable = (
            valid_block
            and len(shape) >= 2
            and (
                shape[-2] % int(block_size[0]) != 0
                or shape[-1] % int(block_size[1]) != 0
            )
        )
        residual_dtype = getattr(weight, "dtype", None) in {
            fp16_dtype,
            bf16_dtype,
        }
        if (
            weight is not None
            and non_tileable
            and residual_dtype
            and callable(getattr(weight, "numel", None))
        ):
            residual_elements += int(weight.numel())
            continue
        invalid_fp8.append(
            f"{name or '<unnamed>'}(shape={shape},"
            f"dtype={getattr(weight, 'dtype', None)!s},"
            f"scale={getattr(module, 'weight_scale_inv', None) is not None})"
        )

    if fp8_total < 1 or fp8_layers < 1:
        raise RuntimeError(
            "semantic model load did not produce any verified FP8Linear weights; "
            "refusing an implicit 16/32-bit model"
        )
    if invalid_fp8:
        raise RuntimeError(
            "semantic model contains malformed/unverified FP8Linear weights; "
            f"refusing a partially materialized model ({', '.join(invalid_fp8[:4])})"
        )
    if text_linear_total and fp8_layers / text_linear_total < 0.8:
        raise RuntimeError(
            f"semantic text stack is only {fp8_layers}/{text_linear_total} FP8 "
            "linear layers; refusing broad full-precision coverage"
        )
    if linear_elements and residual_elements / linear_elements > 0.01:
        raise RuntimeError(
            f"semantic model left {residual_elements}/{linear_elements} linear "
            "weight elements outside FP8; refusing broad full-precision coverage"
        )

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    layer_count = getattr(text_config, "num_hidden_layers", 0)
    minimum_layers = (
        2 * int(layer_count)
        if isinstance(layer_count, int) and not isinstance(layer_count, bool)
        else 0
    )
    if minimum_layers > 0 and fp8_layers < minimum_layers:
        raise RuntimeError(
            f"semantic model has only {fp8_layers} verified FP8Linear layers for "
            f"{layer_count} transformer blocks; refusing partial FP8 coverage"
        )
    return fp8_layers


def _load_fp8_model(
    model_id: str,
    *,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[Any, Any, Any, int]:
    """Load a configured Qwen family id through its text-only FP8 mapping."""

    capability = _require_fp8_cuda(torch_module)
    _configure_fp8_matmul_dispatch(capability)
    try:
        config_type = transformers_module.FineGrainedFP8Config
        tokenizer_type = transformers_module.AutoTokenizer
        model_type = transformers_module.AutoModelForCausalLM
    except AttributeError as exc:
        raise RuntimeError(
            "semantic worker needs Transformers >=5.9 with current Qwen and "
            "FineGrainedFP8Config support"
        ) from exc

    quantization_config = config_type(
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
        dequantize=False,
    )
    download_args = {"local_files_only": True} if _offline_enabled() else {}
    tokenizer = tokenizer_type.from_pretrained(model_id, **download_args)
    model = model_type.from_pretrained(
        model_id,
        device_map="cuda",
        quantization_config=quantization_config,
        # Compute-heavy tileable linear weights/activations are W8A8 FP8.
        # Qwen3.5's recurrent state path is numerically unstable in FP16, so
        # non-linear/pass-through tensors keep BF16 (same 2-byte storage); the
        # validator below rejects broad full-precision *linear* fallback.
        dtype=torch_module.bfloat16,
        **download_args,
    )
    fp8_layers = _assert_fp8_model(model, quantization_config, torch_module)
    model.eval()
    return tokenizer, model, quantization_config, fp8_layers


class FP8QwenRuntime:
    """One lazily loaded model resident for sequential JSONL requests."""

    def __init__(self) -> None:
        self.model_id: str | None = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.quantization_config: Any = None
        self.torch: Any = None
        self.torch_version: str | None = None
        self.transformers_version: str | None = None
        self.fp8_layers = 0

    def _release_model(self) -> None:
        self.model_id = None
        self.tokenizer = None
        self.model = None
        self.quantization_config = None
        self.fp8_layers = 0
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _ensure_model(self, model_id: str) -> None:
        if self.model is not None and self.model_id == model_id:
            return
        if self.model is not None:
            self._release_model()
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        tokenizer, model, config, fp8_layers = _load_fp8_model(
            model_id,
            torch_module=torch_module,
            transformers_module=transformers_module,
        )
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.model = model
        self.quantization_config = config
        self.fp8_layers = fp8_layers
        self.torch_version = str(getattr(torch_module, "__version__", "unknown"))
        self.transformers_version = str(
            getattr(transformers_module, "__version__", "unknown")
        )
        self.model_id = model_id

    def load(self, model_id: str) -> dict[str, object]:
        self._ensure_model(model_id)
        return {
            "protocol": _PROTOCOL_VERSION,
            "worker_version": _WORKER_VERSION,
            "model_id": model_id,
            "precision": "fp8",
            "fp8_layers": self.fp8_layers,
            "torch_version": self.torch_version or "unknown",
            "transformers_version": self.transformers_version or "unknown",
            "device": "cuda:0",
        }

    def generate(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        self._ensure_model(model_id)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to("cuda:0", non_blocking=True) for key, value in encoded.items()
        }
        input_length = int(encoded["input_ids"].shape[-1])
        generate_args: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_args["pad_token_id"] = eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(**generate_args)
        generated = output[0, input_length:]
        # This is the model's raw assistant text (with transport/control tokens
        # removed), intentionally not repaired, fenced, stripped, or parsed.
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def score_labels(
        self,
        model_id: str,
        prompt_batches: list[list[dict[str, str]]],
        labels: list[str],
    ) -> list[list[float]]:
        """Read next-token label logits without running a generation loop."""

        self._ensure_model(model_id)
        label_ids: list[int] = []
        for label in labels:
            encoded_label = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(encoded_label) != 1:
                raise ValueError(f"semantic score label must be one token: {label!r}")
            label_ids.append(int(encoded_label[0]))
        scores: list[list[float]] = []
        # Fine-grained FP8 matmul on some non-Hopper GPUs can emit non-finite
        # label logits for a padded multi-prompt batch even though every prompt
        # is stable alone.  Score sequentially inside the resident worker: this
        # still avoids generation and repeated model loading while making the
        # optional stage deterministic across architectures.
        for messages in prompt_batches:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {
                key: value.to("cuda:0", non_blocking=True)
                for key, value in encoded.items()
            }
            with self.torch.inference_mode():
                next_logits = self.model(
                    **encoded,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[0, -1]
            row_scores = [
                float(next_logits[token_id].float().item()) for token_id in label_ids
            ]
            if any(not math.isfinite(score) for score in row_scores):
                raise RuntimeError("semantic label score is not finite")
            scores.append(row_scores)
        return scores

    def release(self) -> None:
        self._release_model()


def _request_id(document: object) -> int | None:
    if not isinstance(document, Mapping):
        return None
    value = document.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_request(document: object) -> tuple[int, str, list[dict[str, str]], int]:
    request_id = _request_id(document)
    if not isinstance(document, dict) or set(document) != {
        "op",
        "id",
        "model_id",
        "messages",
        "max_new_tokens",
    }:
        raise ValueError("invalid semantic worker request schema")
    if document["op"] != "generate":
        raise ValueError("invalid semantic worker operation")
    if request_id is None:
        raise ValueError("semantic worker request id must be an integer")
    model_id = document["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("semantic worker model_id must be non-empty text")
    messages = document["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("semantic worker messages must be a non-empty list")
    clean_messages: list[dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
        ):
            raise ValueError("semantic worker messages have an invalid schema")
        clean_messages.append({"role": message["role"], "content": message["content"]})
    max_new_tokens = document["max_new_tokens"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 4096
    ):
        raise ValueError("semantic worker max_new_tokens must be between 1 and 4096")
    return request_id, model_id, clean_messages, max_new_tokens


def _clean_messages(value: object, *, context: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    clean: list[dict[str, str]] = []
    for message in value:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
        ):
            raise ValueError(f"{context} has an invalid message schema")
        clean.append({"role": message["role"], "content": message["content"]})
    return clean


def _validate_classify_request(
    document: object,
) -> tuple[int, str, list[list[dict[str, str]]], list[str]]:
    request_id = _request_id(document)
    if not isinstance(document, dict) or set(document) != {
        "op",
        "id",
        "model_id",
        "prompt_batches",
        "labels",
    }:
        raise ValueError("invalid semantic worker classify schema")
    if document["op"] != "classify":
        raise ValueError("invalid semantic worker classify operation")
    if request_id is None:
        raise ValueError("semantic worker classify id must be an integer")
    model_id = document["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("semantic worker classify model_id must be non-empty text")
    raw_batches = document["prompt_batches"]
    if not isinstance(raw_batches, list) or not 1 <= len(raw_batches) <= 32:
        raise ValueError("semantic worker classify needs 1..32 prompt batches")
    prompt_batches = [
        _clean_messages(batch, context="semantic worker classify prompt")
        for batch in raw_batches
    ]
    labels = document["labels"]
    if (
        not isinstance(labels, list)
        or not 2 <= len(labels) <= 8
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("semantic worker classify labels must be 2..8 unique strings")
    return request_id, model_id, prompt_batches, labels


def _emit(stream: TextIO, document: Mapping[str, object]) -> None:
    stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _serve(
    source: Iterable[str],
    sink: TextIO,
    *,
    runtime_factory: Callable[[], Any] = FP8QwenRuntime,
) -> int:
    runtime = runtime_factory()
    _emit(
        sink,
        {
            "op": "hello",
            "protocol": _PROTOCOL_VERSION,
            "worker_version": _WORKER_VERSION,
        },
    )
    try:
        for line in source:
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                _emit(
                    sink,
                    {
                        "op": "error",
                        "id": None,
                        "error": f"invalid JSON request: {exc.msg}",
                    },
                )
                continue
            if document == {"op": "shutdown"}:
                return 0
            request_id = _request_id(document)
            if (
                isinstance(document, dict)
                and set(document) == {"op", "id", "model_id"}
                and document.get("op") == "load"
            ):
                try:
                    model_id = document["model_id"]
                    if (
                        request_id is None
                        or not isinstance(model_id, str)
                        or not model_id
                    ):
                        raise ValueError("invalid semantic worker load request")
                    info = runtime.load(model_id)
                    _emit(sink, {"op": "ready", "id": request_id, **info})
                except Exception as exc:  # noqa: BLE001 - report load failure
                    traceback.print_exc(file=sys.stderr)
                    _emit(
                        sink,
                        {
                            "op": "error",
                            "id": request_id,
                            "error": str(exc) or type(exc).__name__,
                        },
                    )
                continue
            if isinstance(document, dict) and document.get("op") == "classify":
                try:
                    request_id, model_id, prompt_batches, labels = (
                        _validate_classify_request(document)
                    )
                    scores = runtime.score_labels(model_id, prompt_batches, labels)
                    _emit(
                        sink, {"op": "label_scores", "id": request_id, "scores": scores}
                    )
                except Exception as exc:  # noqa: BLE001 -- protocol reports runtime errors
                    traceback.print_exc(file=sys.stderr)
                    _emit(
                        sink,
                        {
                            "op": "error",
                            "id": request_id,
                            "error": str(exc) or type(exc).__name__,
                        },
                    )
                continue
            try:
                request_id, model_id, messages, max_new_tokens = _validate_request(
                    document
                )
                text = runtime.generate(model_id, messages, max_new_tokens)
                if not isinstance(text, str):
                    raise TypeError("semantic model returned non-text output")
                _emit(sink, {"op": "result", "id": request_id, "text": text})
            except Exception as exc:  # noqa: BLE001 -- protocol must report runtime errors
                traceback.print_exc(file=sys.stderr)
                _emit(
                    sink,
                    {
                        "op": "error",
                        "id": request_id,
                        "error": str(exc) or type(exc).__name__,
                    },
                )
        return 0
    finally:
        try:
            runtime.release()
        except Exception:  # noqa: BLE001 - process exit cleanup is best-effort
            traceback.print_exc(file=sys.stderr)


def main() -> int:
    # Preserve the original fd 1 for protocol frames, then redirect Python and
    # native-library stdout to stderr so progress bars cannot corrupt JSONL.
    protocol_fd = os.dup(sys.stdout.fileno())
    protocol_out = os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        return _serve(sys.stdin, protocol_out)
    finally:
        protocol_out.close()


if __name__ == "__main__":
    raise SystemExit(main())
