from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from voxweave.semantic_breaks import (
    LocalTransformersSelector,
    SemanticBackendUnavailable,
)
from voxweave.semantic_worker import (
    _assert_fp8_model,
    _configure_fp8_matmul_dispatch,
    _load_fp8_model,
    _require_fp8_cuda,
    _serve,
)


class FakeCuda:
    def __init__(self, capability=(9, 0), available=True):
        self.capability = capability
        self.available = available

    def is_available(self):
        return self.available

    def get_device_capability(self, _index=0):
        return self.capability


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.6-27B-FP8",
    ],
)
def test_fp8_loader_accepts_configured_qwen_family_id_without_dequantize(
    monkeypatch, offline, model_id
):
    if offline:
        monkeypatch.setenv("VOXWEAVE_OFFLINE", "1")
    else:
        monkeypatch.delenv("VOXWEAVE_OFFLINE", raising=False)
    calls = {}

    class Config:
        def __init__(self, **kwargs):
            calls["config"] = kwargs
            self.dequantize = kwargs["dequantize"]

    class TokenizerType:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return "tokenizer"

    class FakeWeight:
        dtype = "fp8"

        @staticmethod
        def element_size():
            return 1

    class FP8Linear:
        weight = FakeWeight()
        weight_scale_inv = object()

    model = SimpleNamespace(
        eval=lambda: calls.setdefault("eval", True), modules=lambda: [FP8Linear()]
    )

    class ModelType:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return model

    torch_module = SimpleNamespace(
        cuda=FakeCuda(), bfloat16="bf16", float8_e4m3fn="fp8"
    )
    transformers_module = SimpleNamespace(
        FineGrainedFP8Config=Config,
        AutoTokenizer=TokenizerType,
        AutoModelForCausalLM=ModelType,
    )

    tokenizer, loaded_model, config, fp8_layers = _load_fp8_model(
        model_id,
        torch_module=torch_module,
        transformers_module=transformers_module,
    )

    assert tokenizer == "tokenizer"
    assert loaded_model is model
    assert config.dequantize is False
    assert fp8_layers == 1
    assert calls["config"] == {
        "activation_scheme": "dynamic",
        "weight_block_size": (128, 128),
        "dequantize": False,
    }
    loaded_model_id, load_kwargs = calls["model"]
    assert loaded_model_id == model_id
    download_args = {"local_files_only": True} if offline else {}
    assert calls["tokenizer"] == (model_id, download_args)
    assert load_kwargs == {
        "device_map": "cuda",
        "quantization_config": config,
        "dtype": "bf16",
        **download_args,
    }
    assert calls["eval"] is True


@pytest.mark.parametrize(
    "cuda, message",
    [
        (FakeCuda(available=False), "requires an NVIDIA CUDA GPU"),
        (FakeCuda(capability=(8, 6)), "compute capability >= 8.9"),
    ],
)
def test_fp8_capability_gate_fails_instead_of_silent_bf16(cuda, message):
    with pytest.raises(RuntimeError, match=message):
        _require_fp8_cuda(SimpleNamespace(cuda=cuda))


def test_blackwell_forces_universal_triton_fp8_instead_of_sm90_deepgemm():
    state = {"cleared": False}

    def loader():
        return object()

    loader.cache_clear = lambda: state.__setitem__("cleared", True)
    integration = SimpleNamespace(_load_deepgemm_kernel=loader)

    assert (
        _configure_fp8_matmul_dispatch((12, 0), integration_module=integration)
        == "triton"
    )
    assert state["cleared"] is True
    with pytest.raises(ImportError, match="universal"):
        integration._load_deepgemm_kernel()


def test_hopper_keeps_deepgemm_fp8_dispatch():
    integration = SimpleNamespace(_load_deepgemm_kernel=lambda: object())
    original = integration._load_deepgemm_kernel
    assert (
        _configure_fp8_matmul_dispatch((9, 0), integration_module=integration)
        == "deepgemm"
    )
    assert integration._load_deepgemm_kernel is original


def test_fp8_proof_rejects_dequantized_or_unverified_model():
    torch_module = SimpleNamespace(float8_e4m3fn="fp8")
    model = SimpleNamespace(modules=lambda: [])
    with pytest.raises(RuntimeError, match="verified FP8Linear"):
        _assert_fp8_model(model, SimpleNamespace(dequantize=False), torch_module)
    with pytest.raises(RuntimeError, match="dequantize"):
        _assert_fp8_model(model, SimpleNamespace(dequantize=True), torch_module)


def test_fp8_proof_rejects_partial_layer_or_text_stack_coverage():
    class Weight:
        def __init__(self, dtype, shape=(128, 128), elements=16_384):
            self.dtype = dtype
            self.shape = shape
            self._elements = elements

        def element_size(self):
            return 1 if self.dtype == "fp8" else 2

        def numel(self):
            return self._elements

    class FP8Linear:
        def __init__(self, dtype="fp8", scale=True, **weight_kwargs):
            self.weight = Weight(dtype, **weight_kwargs)
            self.weight_scale_inv = object() if scale else None

    class Linear:
        weight = Weight("fp16")

    torch_module = SimpleNamespace(float8_e4m3fn="fp8")
    quantization = SimpleNamespace(dequantize=False)

    partially_materialized = SimpleNamespace(
        modules=lambda: [FP8Linear(), FP8Linear(scale=False)]
    )
    with pytest.raises(RuntimeError, match="unverified FP8Linear"):
        _assert_fp8_model(partially_materialized, quantization, torch_module)

    mostly_fp16 = SimpleNamespace(
        named_modules=lambda: [
            ("model.layers.0.q_proj", FP8Linear()),
            *[(f"model.layers.0.fp16_{index}", Linear()) for index in range(9)],
        ]
    )
    with pytest.raises(RuntimeError, match="text stack is only 1/10"):
        _assert_fp8_model(mostly_fp16, quantization, torch_module)

    implausibly_sparse = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=12),
        modules=lambda: [FP8Linear()],
    )
    with pytest.raises(RuntimeError, match="12 transformer blocks"):
        _assert_fp8_model(implausibly_sparse, quantization, torch_module)

    main = [
        (f"model.layers.{index}.q_proj", FP8Linear(elements=1_000_000))
        for index in range(9)
    ]
    residual = FP8Linear(dtype="bf16", shape=(16, 1024))
    mixed_precision = SimpleNamespace(
        named_modules=lambda: [*main, ("model.layers.9.in_proj_a", residual)],
        named_parameters=lambda: [("model.layers.9.in_proj_a.weight", residual.weight)],
    )
    strict_torch = SimpleNamespace(float8_e4m3fn="fp8", float16="fp16", bfloat16="bf16")
    assert _assert_fp8_model(mixed_precision, quantization, strict_torch) == 9


def test_worker_jsonl_returns_model_text_verbatim_and_releases_runtime():
    state = {}

    class Runtime:
        def load(self, model_id):
            state["loaded"] = model_id
            return {
                "protocol": 1,
                "worker_version": "1",
                "model_id": model_id,
                "precision": "fp8",
                "fp8_layers": 12,
                "torch_version": "2.12.0+cu130",
                "transformers_version": "5.9.0",
                "device": "cuda:0",
            }

        def generate(self, model_id, messages, max_new_tokens):
            state["request"] = (model_id, messages, max_new_tokens)
            return '  {"results":[]}\n'

        def release(self):
            state["released"] = True

    load = {"op": "load", "id": 6, "model_id": "Qwen/Qwen3.5-0.8B"}
    request = {
        "op": "generate",
        "id": 7,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "messages": [{"role": "user", "content": "choose"}],
        "max_new_tokens": 96,
    }
    source = io.StringIO(json.dumps(load) + "\n" + json.dumps(request) + "\n")
    sink = io.StringIO()

    assert _serve(source, sink, runtime_factory=Runtime) == 0
    frames = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert frames[0] == {"op": "hello", "protocol": 1, "worker_version": "1"}
    assert frames[1]["op"] == "ready"
    assert frames[1]["precision"] == "fp8"
    assert frames[1]["fp8_layers"] == 12
    assert frames[2] == {
        "op": "result",
        "id": 7,
        "text": '  {"results":[]}\n',
    }
    assert state["loaded"] == "Qwen/Qwen3.5-0.8B"
    assert state["request"] == (
        "Qwen/Qwen3.5-0.8B",
        [{"role": "user", "content": "choose"}],
        96,
    )
    assert state["released"] is True


def test_worker_jsonl_returns_fixed_label_logits_without_generation():
    state = {}

    class Runtime:
        def load(self, model_id):
            return {
                "protocol": 1,
                "worker_version": "1",
                "model_id": model_id,
                "precision": "fp8",
                "fp8_layers": 12,
                "torch_version": "2.12.0+cu130",
                "transformers_version": "5.9.0",
                "device": "cuda:0",
            }

        def score_labels(self, model_id, prompt_batches, labels):
            state["score"] = (model_id, prompt_batches, labels)
            return [[1.25, -0.5], [-0.25, 0.75]]

        def release(self):
            state["released"] = True

    load = {"op": "load", "id": 2, "model_id": "Qwen/Qwen3.5-0.8B"}
    prompts = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    request = {
        "op": "classify",
        "id": 3,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "prompt_batches": prompts,
        "labels": ["Yes", "No"],
    }
    source = io.StringIO(json.dumps(load) + "\n" + json.dumps(request) + "\n")
    sink = io.StringIO()

    assert _serve(source, sink, runtime_factory=Runtime) == 0
    frames = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert frames[2] == {
        "op": "label_scores",
        "id": 3,
        "scores": [[1.25, -0.5], [-0.25, 0.75]],
    }
    assert state["score"] == (
        "Qwen/Qwen3.5-0.8B",
        prompts,
        ["Yes", "No"],
    )
    assert state["released"] is True


def test_local_selector_uses_persistent_jsonl_subprocess_and_returns_raw_text():
    worker_code = r"""
import json, sys
count = 0
print(json.dumps({"op":"hello", "protocol":1, "worker_version":"1"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request == {"op": "shutdown"}:
        break
    if request["op"] == "load":
        print(json.dumps({"op":"ready", "id":request["id"], "protocol":1,
          "worker_version":"1", "model_id":request["model_id"], "precision":"fp8",
          "fp8_layers":10, "torch_version":"2.12.0+cu130",
          "transformers_version":"5.9.0", "device":"cuda:0"}), flush=True)
        continue
    count += 1
    print(json.dumps({"op":"result", "id": request["id"], "text": f"raw-{count}"}), flush=True)
"""
    selector = LocalTransformersSelector(
        command=(sys.executable, "-u", "-c", worker_code), timeout=2
    )
    try:
        first = selector.select(
            "Qwen/Qwen3.5-0.8B",
            [{"role": "user", "content": "one"}],
            max_new_tokens=96,
        )
        second = selector.select(
            "Qwen/Qwen3.5-0.8B",
            [{"role": "user", "content": "two"}],
            max_new_tokens=96,
        )
    finally:
        selector.release()
    assert (first, second) == ("raw-1", "raw-2")


def test_local_selector_validates_label_score_protocol():
    worker_code = r"""
import json, sys
print(json.dumps({"op":"hello", "protocol":1, "worker_version":"1"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request == {"op": "shutdown"}:
        break
    if request["op"] == "load":
        print(json.dumps({"op":"ready", "id":request["id"], "protocol":1,
          "worker_version":"1", "model_id":request["model_id"], "precision":"fp8",
          "fp8_layers":10, "torch_version":"2.12.0+cu130",
          "transformers_version":"5.9.0", "device":"cuda:0"}), flush=True)
        continue
    assert request["op"] == "classify"
    print(json.dumps({"op":"label_scores", "id":request["id"],
      "scores":[[1.0, -1.0], [-0.5, 0.5]]}), flush=True)
"""
    selector = LocalTransformersSelector(
        command=(sys.executable, "-u", "-c", worker_code), timeout=2
    )
    prompts = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    try:
        scores = selector.score_labels("Qwen/Qwen3.5-0.8B", prompts, ["Yes", "No"])
    finally:
        selector.release()

    assert scores == [[1.0, -1.0], [-0.5, 0.5]]


def test_local_selector_maps_timeout_and_crash_to_backend_unavailable():
    timeout_code = (
        'import json,sys,time; print(json.dumps({"op":"hello","protocol":1,'
        '"worker_version":"1"}),flush=True); next(iter(sys.stdin)); time.sleep(10)'
    )
    selector = LocalTransformersSelector(
        command=(sys.executable, "-u", "-c", timeout_code), timeout=0.05
    )
    with pytest.raises(SemanticBackendUnavailable, match="timed out"):
        selector.select("model", [{"role": "user", "content": "x"}], max_new_tokens=10)
    selector.release()


def test_local_selector_large_write_has_a_deadline_after_ready_handshake():
    worker_code = r"""
import json, sys, time
print(json.dumps({"op":"hello", "protocol":1, "worker_version":"1"}), flush=True)
request = json.loads(sys.stdin.readline())
print(json.dumps({"op":"ready", "id":request["id"], "protocol":1,
  "worker_version":"1", "model_id":request["model_id"], "precision":"fp8",
  "fp8_layers":10, "torch_version":"2.12.0+cu130",
  "transformers_version":"5.9.0", "device":"cuda:0"}), flush=True)
time.sleep(10)
"""
    selector = LocalTransformersSelector(
        command=(sys.executable, "-u", "-c", worker_code),
        timeout=2,
        write_timeout=0.05,
    )
    with pytest.raises(SemanticBackendUnavailable, match="write timed out"):
        selector.select(
            "model",
            [{"role": "user", "content": "x" * 2_000_000}],
            max_new_tokens=10,
        )
    selector.release()


def test_local_selector_maps_early_worker_crash_to_backend_unavailable():
    selector = LocalTransformersSelector(
        command=(sys.executable, "-u", "-c", "raise SystemExit(9)"), timeout=2
    )
    with pytest.raises(SemanticBackendUnavailable, match="exited unexpectedly"):
        selector.select("model", [{"role": "user", "content": "x"}], max_new_tokens=10)
    selector.release()
