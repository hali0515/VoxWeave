"""backend pure-logic tests + missing-dependency error paths (no real model loading)."""

import pytest

from voxweave import align_common, align_ctc, align_mms, backend, runtime


@pytest.fixture(autouse=True)
def _force_torch_backend(monkeypatch):
    """Pin the torch backend for this module so ASR/alignment dispatch is deterministic regardless
    of host device (on Apple Silicon _use_mlx() would otherwise route to MLX). The MLX backend has
    its own coverage in test_backend_mlx.py."""
    monkeypatch.setattr(backend, "_use_mlx", lambda: False)


def _block_import(monkeypatch, *names):
    """Force ModuleNotFoundError for the given top-level modules so missing-dependency error paths
    are exercised deterministically whether or not the package is installed in the test env."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if any(name == n or name.startswith(n + ".") for n in names):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)


def test_strip_state_dict_unwraps_and_deprefixes():
    # Lightning wraps with a state_dict layer + model. prefix; both should be stripped
    sd = {"state_dict": {"model.a": 1, "model.b": 2}}
    assert backend._strip_state_dict(sd) == {"a": 1, "b": 2}


def test_strip_state_dict_plain_passthrough():
    assert backend._strip_state_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_transcribe_align_missing_models_raises_friendly(monkeypatch, tmp_path):
    # qwen-asr import blocked -> friendly RuntimeError pointing to voxweave[cuda]/[mps], not bare ModuleNotFoundError
    _block_import(monkeypatch, "qwen_asr")
    backend._asr = None  # ensure not loaded
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend.transcribe_align(wav, language=None, asr_model="qwen3-asr-1.7b")


def test_transcribe_align_forwards_context(monkeypatch, tmp_path):
    # --context bias: when non-empty, forwarded to model.transcribe(context=...); ASR-only so return_time_stamps=False
    calls: dict = {}

    class _Res:
        text, language, time_stamps = "hi", "English", None

    class _Model:
        def transcribe(self, path, **kw):
            calls.update(kw)
            return [_Res()]

    monkeypatch.setattr(backend, "_get_asr", lambda m=None: _Model())
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, lng: [{"text": "hi", "start": 0, "end": 1}]
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    backend.transcribe_align(
        wav, None, asr_model="qwen3-asr-1.7b", context="艾米莉亚, 帕克"
    )
    # bare term lists are auto-framed for the Qwen system slot (see format_qwen_context)
    assert calls.get("context") == "Proper nouns: 艾米莉亚, 帕克."
    assert (
        calls.get("return_time_stamps") is False
    )  # ASR-only, built-in aligner not requested


def test_transcribe_align_omits_context_when_empty(monkeypatch, tmp_path):
    # no context -> kwarg is omitted entirely; preserves legacy behavior (older qwen-asr lacks this param)
    calls: dict = {}

    class _Res:
        text, language, time_stamps = "hi", "English", None

    class _Model:
        def transcribe(self, path, **kw):
            calls.update(kw)
            return [_Res()]

    monkeypatch.setattr(backend, "_get_asr", lambda m=None: _Model())
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, lng: [{"text": "hi", "start": 0, "end": 1}]
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    backend.transcribe_align(wav, None, asr_model="qwen3-asr-1.7b")
    assert "context" not in calls


def test_qwen_align_routes_all_langs_through_align_text(monkeypatch, tmp_path):
    # ASR no longer loads the built-in aligner -> all languages get timestamps from align_text (ja/en CTC, zh falls back internally to Qwen);
    # ASR call must use return_time_stamps=False (no built-in aligner; True would raise ValueError)
    class _Res:
        text, language, time_stamps = "はい", "Japanese", None  # no built-in units

    class _Model:
        def transcribe(self, path, **kw):
            assert kw.get("return_time_stamps") is False
            return [_Res()]

    monkeypatch.setattr(backend, "_get_asr", lambda m=None: _Model())
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    seen: dict = {}

    def _fake_align(wav, text, lang):
        seen.update(text=text, lang=lang)
        return [{"text": "はい", "start": 33.6, "end": 33.9}]

    monkeypatch.setattr(backend, "align_text", _fake_align)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    lang, text, units = backend.transcribe_align(wav, None, asr_model="qwen3-asr-1.7b")
    assert lang == "Japanese" and text == "はい"
    assert units == [{"text": "はい", "start": 33.6, "end": 33.9}]
    assert seen == {"text": "はい", "lang": "Japanese"}


def test_qwen_align_empty_text_skips_align(monkeypatch, tmp_path):
    # empty text -> align_text is not called; return empty directly (consistent with whisper path)
    class _Res:
        text, language, time_stamps = "", "Japanese", None

    class _Model:
        def transcribe(self, path, **kw):
            return [_Res()]

    monkeypatch.setattr(backend, "_get_asr", lambda m=None: _Model())
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    def _boom(*a, **k):
        raise AssertionError("align_text should not be called on empty text")

    monkeypatch.setattr(backend, "align_text", _boom)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    assert backend.transcribe_align(wav, None, asr_model="qwen3-asr-1.7b") == (
        "Japanese",
        "",
        [],
    )


def test_align_text_missing_models_raises_friendly(monkeypatch, tmp_path):
    # qwen-asr import blocked -> align_text also raises a friendly RuntimeError pointing to voxweave[cuda]/[mps]
    _block_import(monkeypatch, "qwen_asr")
    backend._aligner = None
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend.align_text(wav, "你好", "zh")


def test_to_iso3_mapping():
    # ctc-forced-aligner / uroman uses ISO-639-3; zh->chi (library checks ["jpn","chi"] to force per-char mode)
    from voxweave import lang

    assert lang.to_iso3("ja") == "jpn"
    assert lang.to_iso3("zh") == "chi"
    assert lang.to_iso3("en") == "eng"
    assert (
        lang.to_iso3("xyz") == "xyz"
    )  # no mapping (already 3-letter) -> pass through unchanged


def test_uses_mms_ja_yes_en_no(monkeypatch):
    models = {"ja": "mms", "en": "WAV2VEC2_ASR_LARGE_LV60K_960H"}
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: models.get(iso))
    assert backend.uses_mms("ja") is True
    assert backend.uses_mms("en") is False  # wav2vec2, not MMS
    assert backend.uses_mms("zh") is False  # None (goes to Qwen)


def test_mms_providers_cuda_uses_gpu():
    # CUDA build: GPU provider first, CPU fallback.
    assert align_mms._mms_providers("cuda") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert align_mms._mms_providers("cuda:0") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_mms_providers_mps_uses_cpu_not_coreml():
    # macOS/MPS: CPU only. CoreML is deliberately NOT selected -- its Metal context segfaults
    # when it coexists with MLX (per-chunk MLX ASR + MMS align in one process).
    assert align_mms._mms_providers("mps") == ["CPUExecutionProvider"]


def test_mms_providers_cpu():
    assert align_mms._mms_providers("cpu") == ["CPUExecutionProvider"]


def test_align_text_routes_ja_to_mms(monkeypatch, tmp_path):
    # ja + config "mms" -> align_text routes to align_text_mms; align_text_ctc must not be called
    monkeypatch.setattr(
        backend.config, "align_model_for", lambda iso: "mms" if iso == "ja" else None
    )
    seen: dict = {}
    monkeypatch.setattr(
        backend,
        "align_text_mms",
        lambda w, t, iso: (
            seen.update(t=t, iso=iso) or [{"text": "は", "start": 0.0, "end": 0.1}]
        ),
    )

    def _no_ctc(*a):
        raise AssertionError("ja(mms) should not route to align_text_ctc")

    monkeypatch.setattr(backend, "align_text_ctc", _no_ctc)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    units = backend.align_text(wav, "はい", "Japanese")
    assert units == [{"text": "は", "start": 0.0, "end": 0.1}]
    assert seen == {"t": "はい", "iso": "ja"}


def test_align_blocks_full_mms_distributes_by_alnum(monkeypatch, tmp_path):
    # full-audio single pass -> slice flat units back by alnum char count per block (punctuation/spaces not counted)
    import numpy as np

    arr = np.zeros(
        16000, dtype=np.float32
    )  # 1s: far under the DP budget -> single pass
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda p: arr)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    flat = [
        {"text": c, "start": float(i), "end": i + 0.5} for i, c in enumerate("ABCDE")
    ]

    def _emit(wav, text, iso):
        assert wav is arr and iso == "ja"
        return flat

    monkeypatch.setattr(align_mms, "_mms_emit_units", _emit)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    out = backend.align_blocks_full_mms(wav, ["AB", "C。", "D E"], "ja")
    assert [len(b) for b in out] == [2, 1, 2]  # punctuation/spaces not counted as alnum
    assert out[0] == flat[0:2] and out[1] == flat[2:3] and out[2] == flat[3:5]


def test_align_blocks_full_mms_dp_chunks_over_budget(monkeypatch, tmp_path):
    # over the DP budget with bounds -> silence-anchored chunks, each its own full pass,
    # units from later chunks shifted back to absolute time
    import numpy as np

    sr = align_mms.MMS_SR
    arr = np.zeros(40 * sr, dtype=np.float32)  # 40s = 2000 frames
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda p: arr)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    # budget: 1250 frames = 25s; chunk budget 25s * 0.8 = 20s -> must split
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    calls = []

    def _emit(wav, text, iso):
        calls.append((len(wav), text))
        chars = text.replace(" ", "")
        return [
            {"text": c, "start": float(i), "end": i + 0.5} for i, c in enumerate(chars)
        ]

    monkeypatch.setattr(align_mms, "_mms_emit_units", _emit)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    bounds = [(0.0, 8.0), (10.0, 18.0), (22.0, 30.0), (32.0, 39.0)]
    out = backend.align_blocks_full_mms(
        wav, ["AB", "CD", "EF", "GH"], "ja", bounds=bounds
    )
    # split lands in the 18->22 gap (midpoint 20s): two chunks of 20s audio each
    assert calls == [(20 * sr, "AB CD"), (20 * sr, "EF GH")]
    assert [len(b) for b in out] == [2, 2, 2, 2]
    assert out[0][0] == {"text": "A", "start": 0.0, "end": 0.5}
    # second chunk's units are offset by its 20s crop start
    assert out[2][0] == {"text": "E", "start": 20.0, "end": 20.5}
    assert out[3][1] == {"text": "H", "start": 23.0, "end": 23.5}


def test_align_blocks_full_mms_over_budget_without_bounds_raises(monkeypatch, tmp_path):
    import numpy as np

    arr = np.zeros(40 * align_mms.MMS_SR, dtype=np.float32)
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda p: arr)
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(
        align_mms, "_mms_emit_units", lambda *a: pytest.fail("must not emit")
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="DP budget"):
        backend.align_blocks_full_mms(wav, ["AB", "CD"], "ja")


def test_distribute_units_nospace_vs_spaced():
    flat = [{"text": str(i), "start": float(i), "end": i + 1.0} for i in range(5)]
    # no-space language: by alnum char count (punctuation/spaces not counted)
    out = align_common._distribute_units(flat, ["AB", "C。", "D E"], "ja")
    assert [len(b) for b in out] == [2, 1, 2]
    # spaced language: by word count
    out2 = align_common._distribute_units(flat, ["a b", "c", "d e"], "en")
    assert [len(b) for b in out2] == [2, 1, 2]


def test_release_clears_aligner():
    backend._aligner = object()  # pretend it is loaded
    backend.release()
    assert backend._aligner is None


def test_resolve_separator_uses_existing_files(monkeypatch, tmp_path):
    # explicit ckpt + yaml both present -> use directly, no download triggered
    ck = tmp_path / "x.ckpt"
    cf = tmp_path / "x.yaml"
    ck.write_bytes(b"w")
    cf.write_text("model: {}\n")
    monkeypatch.setattr(backend, "SEPARATOR_CKPT", str(ck))
    monkeypatch.setattr(backend, "SEPARATOR_CONFIG", str(cf))

    def _boom(*a, **k):
        raise AssertionError("should not download")

    monkeypatch.setattr(backend, "_hf_download", _boom)
    rck, rcf = backend._resolve_separator_files()
    assert rck == ck and rcf == cf


def test_resolve_separator_downloads_missing_ckpt(monkeypatch, tmp_path):
    # ckpt missing -> download from HF, return cached path; yaml present -> use it as-is
    cf = tmp_path / "x.yaml"
    cf.write_text("model: {}\n")
    cached = tmp_path / "hf" / "MelBandRoformer.ckpt"
    cached.parent.mkdir()
    cached.write_bytes(b"w")
    monkeypatch.setattr(backend, "SEPARATOR_CKPT", "/nonexistent/x.ckpt")
    monkeypatch.setattr(backend, "SEPARATOR_CONFIG", str(cf))
    calls = []

    def _dl(repo, fn, cache_dir=None):
        calls.append((repo, fn, cache_dir))
        return str(cached)

    monkeypatch.setattr(backend, "_hf_download", _dl)
    rck, rcf = backend._resolve_separator_files()
    assert rck == cached and rcf == cf
    # separator weights download into the voxweave audio cache subdir
    assert calls == [
        (
            backend.SEPARATOR_REPO,
            backend.SEPARATOR_REPO_FILE,
            backend.config.AUDIO_CACHE,
        )
    ]


def test_resolve_separator_falls_back_to_bundled_config(monkeypatch, tmp_path):
    # yaml missing -> fall back to the vendor-bundled config (must exist and be parseable)
    ck = tmp_path / "x.ckpt"
    ck.write_bytes(b"w")
    monkeypatch.setattr(backend, "SEPARATOR_CKPT", str(ck))
    monkeypatch.setattr(backend, "SEPARATOR_CONFIG", "/nonexistent/x.yaml")
    rck, rcf = backend._resolve_separator_files()
    assert rck == ck
    assert rcf == backend._BUNDLED_SEPARATOR_CONFIG
    assert rcf.exists()
    cfg = backend._load_yaml(rcf)
    assert cfg["model"]["dim"] == 384 and cfg["model"]["num_bands"] == 60


def test_resolve_separator_download_failure_raises_friendly(monkeypatch):
    # download fails -> friendly RuntimeError (mentions --no-separate / manual weight placement), not bare exception
    monkeypatch.setattr(backend, "SEPARATOR_CKPT", "/nonexistent/x.ckpt")
    monkeypatch.setattr(backend, "SEPARATOR_CONFIG", "/nonexistent/x.yaml")

    def _dl(repo, fn):
        raise OSError("no network")

    monkeypatch.setattr(backend, "_hf_download", _dl)
    with pytest.raises(RuntimeError, match="--no-separate"):
        backend._resolve_separator_files()


def test_release_is_idempotent():
    backend._asr = None
    backend.release()  # must not crash when nothing is loaded
    assert backend._asr is None


def test_demix_reports_progress_per_window():
    # _demix overlap-add window loop is countable -> progress(done, total) called per window, for real progress bars
    torch = pytest.importorskip("torch")

    class _Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self, x
        ):  # x: [1, ch, chunk] -> pass through unchanged (sufficient to test progress counting)
            return x

    cfg = {"audio": {"chunk_size": 16}, "inference": {"num_overlap": 4}}
    mix = torch.zeros(2, 40)  # step=16//4=4, range(0,40,4) -> 10 windows
    calls: list[tuple[int, int]] = []
    backend._demix(
        _Identity(), mix, cfg, progress=lambda d, t: calls.append((d, t)), batch=3
    )
    assert len(calls) == 10
    assert calls[0] == (1, 10) and calls[-1] == (10, 10)
    assert [d for d, _ in calls] == list(range(1, 11))  # monotonically increasing


def test_demix_batched_matches_sequential():
    # window batching is pure bookkeeping: stacking B windows into one forward must
    # reproduce batch=1 overlap-add output exactly (incl. the padded final window)
    torch = pytest.importorskip("torch")

    class _Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):  # [B, ch, chunk] -> per-sample op, batch-size invariant
            return x * 0.5 + 0.1

    cfg = {"audio": {"chunk_size": 16}, "inference": {"num_overlap": 4}}
    mix = torch.randn(2, 43)  # not a multiple of step -> exercises final-window padding
    ref = backend._demix(_Scale(), mix, cfg, batch=1)
    for bs in (2, 3, 64):  # 64 > window count -> single batch
        out = backend._demix(_Scale(), mix, cfg, batch=bs)
        assert torch.equal(out, ref)


def test_resolve_asr_model():
    assert backend.resolve_asr_model(None) == backend.ASR_MODEL
    assert backend.resolve_asr_model("") == backend.ASR_MODEL
    # short name (case-insensitive) -> canonical HF id
    assert backend.resolve_asr_model("qwen3-asr-1.7B") == "Qwen/Qwen3-ASR-1.7B"
    assert backend.resolve_asr_model("1.7b") == "Qwen/Qwen3-ASR-1.7B"
    # full id passes through unchanged
    assert backend.resolve_asr_model("Qwen/Qwen3-ASR-0.6B") == "Qwen/Qwen3-ASR-0.6B"
    # unknown bare name falls back to prepending org
    assert backend.resolve_asr_model("my-asr") == "Qwen/my-asr"


def test_select_engine_routes_whisper_names():
    assert backend._select_engine("large-v3-turbo") == ("whisper", "large-v3-turbo")
    assert backend._select_engine("large-v3") == ("whisper", "large-v3")
    assert backend._select_engine("distil-large-v3") == ("whisper", "distil-large-v3")
    # aliases: turbo / whisper -> canonical large-v3-turbo
    assert backend._select_engine("turbo") == ("whisper", "large-v3-turbo")
    assert backend._select_engine("whisper") == ("whisper", "large-v3-turbo")
    # case-insensitive
    assert backend._select_engine("Large-V3-Turbo") == ("whisper", "large-v3-turbo")


def test_select_engine_default_is_qwen():
    # default = historical Qwen (fusion is opt-in, triggered by CLI --hybrid)
    assert backend._select_engine(None) == ("qwen", backend.ASR_MODEL)
    assert backend._select_engine("") == ("qwen", backend.ASR_MODEL)


def test_select_engine_fusion_aliases():
    # --hybrid sets asr_model to "fusion"; 'fuse' is also recognized
    assert backend._select_engine("fusion") == ("fusion", "")
    assert backend._select_engine("Fuse") == ("fusion", "")
    # config asr_model = "hybrid" is equivalent to CLI --hybrid (both route to fusion)
    assert backend._select_engine("hybrid") == ("fusion", "")
    assert backend._select_engine("Hybrid") == ("fusion", "")


def test_select_engine_routes_qwen_named():
    # explicit Qwen (e.g. for Chinese) still routes to qwen
    assert backend._select_engine("qwen3-asr-1.7B") == ("qwen", "Qwen/Qwen3-ASR-1.7B")
    assert backend._select_engine("Qwen/Qwen3-ASR-0.6B") == (
        "qwen",
        "Qwen/Qwen3-ASR-0.6B",
    )
    # unknown repo containing '/' -> qwen, pass through unchanged
    assert backend._select_engine("openai/whatever") == ("qwen", "openai/whatever")


def test_transcribe_align_routes_fusion(monkeypatch, tmp_path):
    # default (asr_model=None) -> _transcribe_fusion
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    called = {}

    def _fake_fusion(wav_path, language, context):
        called["fusion"] = True
        return "ja", "畑です。", [{"text": "畑", "start": 0.0, "end": 0.2}]

    monkeypatch.setattr(backend, "_transcribe_fusion", _fake_fusion)
    lang, text, units = backend.transcribe_align(wav, None, asr_model="fusion")
    assert called.get("fusion") and lang == "ja" and text == "畑です。"


def test_fusion_merges_whisper_text_with_qwen_punct(monkeypatch, tmp_path):
    # whisper produces accurate words without punctuation, Qwen produces punctuation positions -> fused text = whisper words + Qwen punctuation
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    w_units = [
        {"text": "畑", "start": 1.0, "end": 1.2},
        {"text": "です", "start": 1.2, "end": 1.6},
        {"text": "次", "start": 2.5, "end": 2.7},
    ]
    # Qwen path: text has punctuation, units are aligner output (no punctuation) -> after reinject, punctuation carries timestamps
    monkeypatch.setattr(
        backend,
        "_transcribe_whisper_align",
        lambda *a, **k: ("ja", "畑です次", w_units),
    )
    monkeypatch.setattr(
        backend,
        "_transcribe_qwen_align",
        lambda *a, **k: (
            "Japanese",
            "裸です。次。",
            [
                {"text": "裸", "start": 1.0, "end": 1.2},
                {"text": "で", "start": 1.2, "end": 1.4},
                {"text": "す", "start": 1.4, "end": 1.6},
                {"text": "次", "start": 2.5, "end": 2.7},
            ],
        ),
    )
    lang, text, units = backend._transcribe_fusion(wav, None, None)
    assert "。" in text  # Qwen punctuation is present
    assert "畑" in text and "裸" not in text  # text is from whisper (畑, not 裸)
    assert units == w_units  # units come from whisper


def test_transcribe_chunks_fusion_three_pass_order(monkeypatch, tmp_path):
    # fusion three-pass: all-chunks whisper ASR -> release whisper -> all-chunks Qwen ASR -> release Qwen ->
    # all-chunks align (only CTC resident at this point; whisper/Qwen both released -> fixes Qwen+CTC co-resident OOM on 8GB cards)
    seq: list[str] = []

    def _asr(engine, w, lang, mid, ctx):
        seq.append(f"asr:{engine}:{w.name}")
        return ("ja", f"{engine}-{w.name}", "ja")

    def _align(w, text, alang):
        seq.append(f"align:{text}")
        return [{"text": "x", "start": 0.0, "end": 0.2}]

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(
        backend, "_fuse_chunk", lambda wr, qr, lang: ("ja", "fused", [])
    )
    monkeypatch.setattr(backend, "_release_whisper", lambda: seq.append("REL_W"))
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: seq.append("REL_Q"))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    ticks: list[int] = []
    out = backend.transcribe_chunks(
        wavs, None, asr_model="fusion", on_done=lambda i: ticks.append(i)
    )
    # all whisper ASR -> release -> all Qwen ASR -> release -> all align (each chunk aligned once for whisper text + once for Qwen text)
    assert seq == [
        "asr:whisper:c0.wav",
        "asr:whisper:c1.wav",
        "REL_W",
        "asr:qwen:c0.wav",
        "asr:qwen:c1.wav",
        "REL_Q",
        "align:whisper-c0.wav",
        "align:qwen-c0.wav",
        "align:whisper-c1.wav",
        "align:qwen-c1.wav",
    ]
    assert len(out) == 2 and ticks == [0, 1, 2, 3, 4, 5]  # 3N=6 progress ticks


def test_transcribe_chunks_non_fusion_two_pass_order(monkeypatch, tmp_path):
    # non-fusion (qwen/whisper) also two-pass: all-chunks ASR -> release ASR -> all-chunks align (peak = max not sum;
    # ASR singleton and aligner no longer co-reside -> fixes OOM on 8GB cards)
    seq: list[str] = []

    def _asr(engine, w, lang, mid, ctx):
        seq.append(f"asr:{w.name}")
        return ("Japanese", "はい", "ja")

    def _align(w, text, alang):
        seq.append(f"align:{w.name}")
        return [{"text": "はい", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: seq.append("REL_ASR"))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    ticks: list[int] = []
    out = backend.transcribe_chunks(
        wavs, None, asr_model="qwen3-asr-1.7b", on_done=lambda i: ticks.append(i)
    )
    assert seq == [
        "asr:c0.wav",
        "asr:c1.wav",
        "REL_ASR",
        "align:c0.wav",
        "align:c1.wav",
    ]
    assert len(out) == 2 and ticks == [0, 1, 2, 3]  # 2N progress ticks
    assert out[0] == ("Japanese", "はい", [{"text": "はい", "start": 0.0, "end": 1.0}])


def test_transcribe_chunks_two_pass_releases_whisper_for_whisper_engine(
    monkeypatch, tmp_path
):
    # whisper engine: after the ASR pass, whisper singleton is released (not qwen)
    rel: list[str] = []
    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("ja", "x", "ja")
    )
    monkeypatch.setattr(backend, "align_text", lambda w, t, a: [{"text": "x"}])
    monkeypatch.setattr(backend, "_release_whisper", lambda: rel.append("W"))
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: rel.append("Q"))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wav = tmp_path / "c0.wav"
    wav.write_bytes(b"x")
    backend.transcribe_chunks([wav], None, asr_model="large-v3-turbo")
    assert rel == ["W"]  # whisper engine releases whisper, does not touch qwen


def test_transcribe_chunks_two_pass_skips_align_for_empty_text(monkeypatch, tmp_path):
    # empty ASR text chunk: second pass skips alignment, units stay []
    def _asr(engine, w, lang, mid, ctx):
        return ("ja", "" if w.name == "c1.wav" else "はい", "ja")

    aligned: list[str] = []

    def _align(w, text, alang):
        aligned.append(w.name)
        return [{"text": "はい", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    out = backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")
    assert aligned == ["c0.wav"]  # c1 empty text -> not aligned
    assert out[1] == ("ja", "", [])


# --- fallback paths reclaim VRAM first ------------------------------------- #
def test_full_pass_failure_empties_cache_before_fallback(monkeypatch, tmp_path):
    # an OOM'd full-file pass leaves fragmented VRAM; the per-chunk fallback
    # must start from a clean cache or it cascades into OOM too
    calls: list[str] = []
    monkeypatch.setattr(backend, "_empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        backend.config, "align_model_for", lambda iso: "facebook/wav2vec2-large"
    )

    def _boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(backend, "align_blocks_full_ctc", _boom)
    wav = tmp_path / "full.wav"
    wav.write_bytes(b"x")
    out = backend._full_pass_units(wav, [(0.0, 1.0)], ["hello"], "english")
    assert out is None  # still falls back
    assert calls == ["empty"]


def test_ctc_failure_empties_cache_before_qwen_fallback(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(backend, "_empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: "some/ctc")

    def _boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(backend, "align_text_ctc", _boom)
    monkeypatch.setattr(backend, "_use_mlx", lambda: False)
    seen: list[list[str]] = []

    class _FakeAligner:
        def align(self, wav, text, lang):
            seen.append(list(calls))  # snapshot: cache state when Qwen runs
            return [[]]

    monkeypatch.setattr(backend, "_get_aligner", lambda: _FakeAligner())
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"x")
    out = backend.align_text(wav, "hello", "english")
    assert out == []
    assert seen == [["empty"]]  # cache emptied BEFORE the fallback aligner ran


# --- per-chunk failure containment ---------------------------------------- #
def test_transcribe_chunks_survives_single_asr_failure(monkeypatch, tmp_path):
    # one chunk's ASR blowing up must not kill the whole run: it degrades to
    # empty text (same path as genuine silence) and the rest proceeds
    def _asr(engine, w, lang, mid, ctx):
        if w.name == "c1.wav":
            raise RuntimeError("CUDA error: device-side assert")
        return ("ja", "はい", "ja")

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, a: [{"text": t, "start": 0.0, "end": 1.0}]
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav", tmp_path / "c2.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    ticks: list[int] = []
    out = backend.transcribe_chunks(
        wavs, None, asr_model="qwen3-asr-1.7b", on_done=lambda i: ticks.append(i)
    )
    assert out[1] == (None, "", [])
    assert out[0][1] == "はい" and out[2][1] == "はい"
    assert ticks == [0, 1, 2, 3, 4, 5]  # failed chunk still ticks progress


def test_transcribe_chunks_survives_single_align_failure(monkeypatch, tmp_path):
    # per-chunk alignment failure keeps the transcript text, just without units
    def _align(w, text, alang):
        if w.name == "c1.wav":
            raise RuntimeError("boom")
        return [{"text": text, "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("ja", "はい", "ja")
    )
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    out = backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")
    assert out[0][2] and out[1] == ("ja", "はい", [])


def test_transcribe_chunks_raises_when_all_chunks_fail(monkeypatch, tmp_path):
    # every chunk erroring is a broken run, not a silent empty transcript
    def _asr(engine, w, lang, mid, ctx):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="all 2 chunks"):
        backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")


def test_transcribe_chunks_all_empty_asr_is_not_an_error(monkeypatch, tmp_path):
    # genuine silence (ASR returns empty text, no exception) keeps the old
    # behavior: empty results, pipeline decides what to do
    monkeypatch.setattr(backend, "_asr_only", lambda e, w, lang, m, c: ("ja", "", "ja"))
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wav = tmp_path / "c0.wav"
    wav.write_bytes(b"x")
    out = backend.transcribe_chunks([wav], None, asr_model="qwen3-asr-1.7b")
    assert out == [("ja", "", [])]


def test_transcribe_chunks_fusion_survives_one_engine_failure(monkeypatch, tmp_path):
    # fusion: whisper failing on a chunk leaves the Qwen side to carry it
    def _asr(engine, w, lang, mid, ctx):
        if engine == "whisper" and w.name == "c0.wav":
            raise RuntimeError("whisper died")
        return ("ja", f"{engine}-{w.name}", "ja")

    fused: list[tuple] = []

    def _fuse(wr, qr, lang):
        fused.append((wr, qr))
        return ("ja", "fused", [])

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, a: [{"text": t, "start": 0.0, "end": 0.2}]
    )
    monkeypatch.setattr(backend, "_fuse_chunk", _fuse)
    monkeypatch.setattr(backend, "_release_whisper", lambda: None)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    out = backend.transcribe_chunks(wavs, None, asr_model="fusion")
    assert len(out) == 2
    # chunk 0: whisper side degraded to empty, qwen side intact
    assert fused[0][0][1] == "" and fused[0][1][1] == "qwen-c0.wav"


# --- load strategy: sum (co-resident: same pass structure, no release between passes) ----------- #
def test_transcribe_chunks_sum_strategy_keeps_singletons_resident(
    monkeypatch, tmp_path
):
    # sum: same two-pass structure as peak, but singletons are NOT released between
    # passes (peak VRAM = sum of models). 2N ticks like peak.
    seq: list[str] = []

    def _asr(engine, w, lang, mid, ctx):
        seq.append(f"asr:{w.name}")
        return ("Japanese", "はい", "ja")

    def _align(w, text, alang):
        seq.append(f"align:{w.name}")
        return [{"text": "はい", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(backend, "_asr_only", _asr)
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: seq.append("REL"))
    monkeypatch.setattr(backend, "_release_whisper", lambda: seq.append("RELW"))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    ticks: list[int] = []
    out = backend.transcribe_chunks(
        wavs,
        None,
        asr_model="qwen3-asr-1.7b",
        on_done=lambda i: ticks.append(i),
        strategy="sum",
    )
    assert seq == [
        "asr:c0.wav",
        "asr:c1.wav",
        "align:c0.wav",
        "align:c1.wav",
    ]  # two passes, but no REL between them
    assert len(out) == 2 and ticks == [0, 1, 2, 3]  # 2N like peak
    assert out[0] == ("Japanese", "はい", [{"text": "はい", "start": 0.0, "end": 1.0}])


# --- full-file alignment pass (full_wav + bounds) ----------------------------------------------- #
def test_transcribe_chunks_full_pass_for_ctc_lang(monkeypatch, tmp_path):
    # CTC-configured file language + full_wav/bounds -> ONE align_blocks_full_ctc call
    # over the whole audio; per-chunk align_text never runs; units shifted back to
    # chunk-relative times (transcribe_chunks contract).
    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("English", "hi there", "en")
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setattr(
        backend, "align_text", lambda *a: pytest.fail("per-chunk align must not run")
    )
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: "some-wav2vec2")
    seen = {}

    def _full(wav, texts, iso, model, bounds=None, speech_spans=None):
        seen.update(
            wav=wav,
            texts=texts,
            iso=iso,
            model=model,
            bounds=bounds,
            speech_spans=speech_spans,
        )
        # absolute units, one per chunk
        return [
            [{"text": "hi", "start": 10.5, "end": 11.0}],
            [{"text": "hi", "start": 130.5, "end": 131.0}],
        ]

    monkeypatch.setattr(backend, "align_blocks_full_ctc", _full)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    full = tmp_path / "full.wav"
    full.write_bytes(b"x")
    bounds = [(10.0, 120.0), (130.0, 240.0)]
    spans = [(10.0, 60.0), (130.0, 200.0)]
    out = backend.transcribe_chunks(
        wavs,
        None,
        asr_model="qwen3-asr-1.7b",
        full_wav=full,
        bounds=bounds,
        speech_spans=spans,
    )
    assert seen["wav"] is full and seen["bounds"] == bounds
    assert seen["speech_spans"] == spans  # VAD spans reach the full-pass aligner
    assert seen["texts"] == ["hi there", "hi there"] and seen["iso"] == "en"
    # absolute -> chunk-relative: each chunk's units shifted by -bounds[i].start
    assert out[0][2] == [{"text": "hi", "start": 0.5, "end": 1.0}]
    assert out[1][2] == [{"text": "hi", "start": 0.5, "end": 1.0}]


def test_transcribe_chunks_qwen_lang_stays_per_chunk(monkeypatch, tmp_path):
    # No CTC config for the file language (zh -> Qwen NAR, 180s input cap) -> full-file
    # pass is skipped even when full_wav/bounds are provided; per-chunk align_text runs.
    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("Chinese", "你好", "zh")
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: None)
    aligned: list[str] = []

    def _align(w, text, alang):
        aligned.append(w.name)
        return [{"text": "你好", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(
        backend,
        "align_blocks_full_ctc",
        lambda *a, **k: pytest.fail("full pass must not run for Qwen langs"),
    )
    monkeypatch.setattr(
        backend,
        "align_blocks_full_mms",
        lambda *a, **k: pytest.fail("full pass must not run for Qwen langs"),
    )
    wavs = [tmp_path / "c0.wav"]
    wavs[0].write_bytes(b"x")
    full = tmp_path / "full.wav"
    full.write_bytes(b"x")
    out = backend.transcribe_chunks(
        wavs, None, asr_model="qwen3-asr-1.7b", full_wav=full, bounds=[(0.0, 100.0)]
    )
    assert aligned == ["c0.wav"]
    assert out[0][2] == [{"text": "你好", "start": 0.0, "end": 1.0}]


def test_transcribe_chunks_full_pass_failure_falls_back_per_chunk(
    monkeypatch, tmp_path
):
    # full-file pass raising -> warning + per-chunk fallback (mirrors align_text's CTC->Qwen fallback)
    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("English", "hi", "en")
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: "some-wav2vec2")

    def _boom(*a, **k):
        raise RuntimeError("emission blew up")

    monkeypatch.setattr(backend, "align_blocks_full_ctc", _boom)
    aligned: list[str] = []
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda w, t, a: (
            aligned.append(w.name) or [{"text": "hi", "start": 0.0, "end": 0.5}]
        ),
    )
    wavs = [tmp_path / "c0.wav"]
    wavs[0].write_bytes(b"x")
    full = tmp_path / "full.wav"
    full.write_bytes(b"x")
    out = backend.transcribe_chunks(
        wavs, None, asr_model="qwen3-asr-1.7b", full_wav=full, bounds=[(0.0, 100.0)]
    )
    assert aligned == ["c0.wav"]  # fell back to per-chunk
    assert out[0][2] == [{"text": "hi", "start": 0.0, "end": 0.5}]


def test_weighted_align_lang_votes_by_text_mass():
    # long dialogue dominates a short cold-open insert; empty chunks carry no vote
    asr_out = [
        ("English", "a" * 10, "en"),
        ("Japanese", "あ" * 200, "ja"),
        (None, "", "en"),
    ]
    assert backend._weighted_align_lang(asr_out) == "ja"
    assert backend._weighted_align_lang([(None, "", "en")]) is None


def test_transcribe_chunks_default_strategy_is_peak(monkeypatch, tmp_path):
    # omitting strategy -> peak (two-pass, 2N ticks); default must not change
    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("ja", "x", "ja")
    )
    monkeypatch.setattr(backend, "align_text", lambda w, t, a: [{"text": "x"}])
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for w in wavs:
        w.write_bytes(b"x")
    ticks: list[int] = []
    backend.transcribe_chunks(
        wavs, None, asr_model="qwen3-asr-1.7b", on_done=lambda i: ticks.append(i)
    )
    assert ticks == [0, 1, 2, 3]  # 2N = peak default


def test_chunk_pass_count():
    # pass structure no longer depends on strategy (sum only skips releases between passes)
    assert backend.chunk_pass_count("qwen3-asr-1.7b", "peak") == 2
    assert backend.chunk_pass_count("fusion", "peak") == 3
    assert backend.chunk_pass_count("qwen3-asr-1.7b", "sum") == 2
    assert backend.chunk_pass_count("fusion", "sum") == 3


def test_get_whisper_missing_dep_raises_friendly(monkeypatch):
    # faster-whisper import blocked -> friendly RuntimeError pointing to voxweave[cuda]
    _block_import(monkeypatch, "faster_whisper")
    backend._whisper = None
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend._get_whisper("large-v3-turbo")


def test_release_clears_whisper(monkeypatch):
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    backend._whisper = object()
    backend._whisper_id = "large-v3-turbo"
    backend.release()
    assert backend._whisper is None
    assert backend._whisper_id is None


def test_parse_whisper_device_splits_cuda_index(monkeypatch):
    monkeypatch.setattr(runtime, "_DEVICE", "cuda:0")
    assert backend._parse_whisper_device() == ("cuda", 0)
    monkeypatch.setattr(runtime, "_DEVICE", "cuda:1")
    assert backend._parse_whisper_device() == ("cuda", 1)
    monkeypatch.setattr(runtime, "_DEVICE", "cpu")
    assert backend._parse_whisper_device() == ("cpu", 0)


def _fake_whisper(segs, lang):
    """Fake WhisperModel: .transcribe returns (segment iterator, info); records kwargs to .calls."""

    class _Seg:
        def __init__(self, text):
            self.text = text

    class _Info:
        def __init__(self, language):
            self.language = language

    class _Model:
        def __init__(self):
            self.calls: dict = {}

        def transcribe(self, path, **kw):
            self.calls.update(kw)
            return iter([_Seg(t) for t in segs]), _Info(lang)

    return _Model()


def test_whisper_align_basic_returns_contract(monkeypatch, tmp_path):
    model = _fake_whisper(["Hello", " world"], "en")
    align_calls: dict = {}

    def _fake_align(wav, text, lang):
        align_calls.update(text=text, lang=lang)
        return [{"text": "hello", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(backend, "align_text", _fake_align)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    lang, text, units = backend.transcribe_align(wav, None, asr_model="large-v3-turbo")
    assert lang == "en"
    assert text == "Hello world"
    assert units == [{"text": "hello", "start": 0.0, "end": 1.0}]
    assert align_calls == {"text": "Hello world", "lang": "en"}
    # verify: language=None is passed through to whisper
    assert model.calls.get("language") is None
    assert model.calls.get("word_timestamps") is False


def test_whisper_align_override_language_maps_to_iso(monkeypatch, tmp_path):
    model = _fake_whisper(["こんにちは"], "ja")
    align_calls: dict = {}
    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda w, t, lng: (
            align_calls.update(lang=lng) or [{"text": "x", "start": 0, "end": 1}]
        ),
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    backend.transcribe_align(wav, "japanese", asr_model="large-v3")
    # whisper language= receives iso, alignment receives the override full name
    assert model.calls.get("language") == "ja"
    assert align_calls.get("lang") == "japanese"


def test_whisper_align_unsupported_lang_falls_back_to_en(monkeypatch, tmp_path):
    model = _fake_whisper(
        ["bonjour"], "th"
    )  # Thai is not in the aligner's 11 supported languages
    align_calls: dict = {}
    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda w, t, lng: (
            align_calls.update(lang=lng) or [{"text": "x", "start": 0, "end": 1}]
        ),
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    lang, text, units = backend.transcribe_align(wav, None, asr_model="large-v3")
    assert lang == "th"  # detected language returned as-is
    assert align_calls.get("lang") == "en"  # alignment falls back to en


def test_whisper_align_context_maps_to_initial_prompt(monkeypatch, tmp_path):
    model = _fake_whisper(["hi"], "en")
    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, lng: [{"text": "hi", "start": 0, "end": 1}]
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    backend.transcribe_align(wav, None, asr_model="large-v3", context="艾米莉亚")
    assert model.calls.get("initial_prompt") == "艾米莉亚"


def test_whisper_align_empty_text_skips_alignment(monkeypatch, tmp_path):
    model = _fake_whisper([], "en")  # no segments -> empty text

    def _boom(*a, **k):
        raise AssertionError("align_text should not be called on empty text")

    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(backend, "align_text", _boom)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    lang, text, units = backend.transcribe_align(wav, None, asr_model="large-v3")
    assert (lang, text, units) == ("en", "", [])


def test_whisper_align_cantonese_uses_zh_for_whisper(monkeypatch, tmp_path):
    # whisper has no Cantonese code: --language yue -> whisper receives zh, but alignment still receives yue
    model = _fake_whisper(["你好"], "zh")
    align_calls: dict = {}
    monkeypatch.setattr(backend, "_get_whisper", lambda mid: model)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda w, t, lng: (
            align_calls.update(lang=lng) or [{"text": "x", "start": 0, "end": 1}]
        ),
    )
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    backend.transcribe_align(wav, "yue", asr_model="large-v3")
    assert model.calls.get("language") == "zh"
    assert align_calls.get("lang") == "yue"


def test_transcribe_align_whisper_missing_dep_raises_friendly(monkeypatch, tmp_path):
    # select whisper engine via the public transcribe_align entry point; faster-whisper import blocked -> friendly voxweave[cuda] error
    _block_import(monkeypatch, "faster_whisper")
    backend._whisper = None
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend.transcribe_align(wav, None, asr_model="large-v3-turbo")


# --------------------------------------------------------------------------- #
# wav2vec2 CTC alignment (WhisperX-equivalent path; English default, per-language in voxweave.config)
# --------------------------------------------------------------------------- #
import collections  # noqa: E402

_Span = collections.namedtuple("_Span", "token start end score")

# simulated subset of wav2vec2 960h label set: 0='-'(blank), 1='|'(sep), last entry includes apostrophe
_FAKE_LABELS = list("-|ETAONIHSRDLUMWCFGYPBVKJXQZ'")


def _fake_invocab():
    sep = _FAKE_LABELS.index("|")
    return {c: i for i, c in enumerate(_FAKE_LABELS) if i not in (0, sep)}, sep


_FakeCtcAl = collections.namedtuple("_FakeCtcAl", "invocab")


def test_ctc_build_tokens_wordlevel_star_and_oov():
    al = _FakeCtcAl(invocab=_fake_invocab()[0])
    toks, meta, words = align_ctc._ctc_build_tokens(
        ["Don't well-known cafe, 2"], False, al
    )
    assert words == ["Don't", "well-known", "cafe,", "2"]
    assert max(m for m in meta if m >= 0) == 3  # 4 words -> word_idx 0..3
    assert meta.count(-1) == 5  # <star> at both edges + every word boundary
    assert 0 not in [
        t for t in toks if t is not None
    ]  # hyphen does not mis-hit blank(0)
    # '-' in well-known (word_idx 1) is OOV -> None (wildcard)
    assert any(t is None for t, m in zip(toks, meta) if m == 1)
    # apostrophe is in invocab -> all chars in Don't have non-OOV tokens
    assert all(t is not None for t, m in zip(toks, meta) if m == 0)


def test_ctc_build_tokens_collapses_multispace_and_strips():
    al = _FakeCtcAl(invocab=_fake_invocab()[0])
    toks, meta, words = align_ctc._ctc_build_tokens(["  a   b  "], False, al)
    assert words == ["a", "b"] and max(m for m in meta if m >= 0) == 1
    assert (
        meta.count(-1) == 3
    )  # stars only at edges and the single word boundary (multiple spaces collapsed)


def test_strip_trailing_punct():
    assert align_common._strip_trailing_punct("cafe,") == "cafe"
    assert align_common._strip_trailing_punct("dogs.") == "dogs"
    assert align_common._strip_trailing_punct("well-known") == "well-known"
    assert align_common._strip_trailing_punct("Don't") == "Don't"
    assert (
        align_common._strip_trailing_punct("...") == "..."
    )  # all punctuation -> return original


def test_ctc_words_from_spans():
    spans = [
        _Span(5, 0, 2, 0.9),
        _Span(6, 2, 4, 0.9),  # word0 "ab"
        _Span(1, 4, 5, 0.5),  # sep (meta -1)
        _Span(7, 6, 10, 0.9),  # word1 "cafe,"
    ]
    units = align_ctc._ctc_words_from_spans(
        spans, [0, 0, -1, 1], ["ab", "cafe,"], ratio=0.1
    )
    assert len(units) == 2
    assert units[0] == {"text": "ab", "start": 0.0, "end": 0.4}
    assert units[1] == {
        "text": "cafe",
        "start": 0.6,
        "end": 1.0,
    }  # trailing punctuation stripped; timing covers full word


def test_ctc_emit_full_batched_matches_sequential(monkeypatch):
    # same-length window batching is pure grouping: emissions must equal the
    # batch=1 pass bit-for-bit (per-sample model, identical reduction order)
    torch = pytest.importorskip("torch")

    class _FakeW2V(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):  # [B, n] -> ([B, T, V], lengths), 320x downsample
            b, n = x.shape
            t = n // 320
            f = x[:, : t * 320].reshape(b, t, 320).mean(dim=-1, keepdim=True)
            return torch.cat([f, -f, 2 * f, torch.ones_like(f)], dim=-1), None

    sr = 16000
    torch.manual_seed(0)
    # 100s @ 30s window + 2s context -> spans 32s / 34s / 34s / 12s:
    # exercises singleton groups (unequal first/last) AND a batched interior pair
    wav = torch.randn(100 * sr)
    al = align_ctc.CtcAligner("torchaudio", _FakeW2V(), sr, 0, 1, {}, None)
    monkeypatch.setenv("VOXWEAVE_CTC_BATCH", "1")
    ref = align_ctc._ctc_emit_full(al, wav)
    monkeypatch.setenv("VOXWEAVE_CTC_BATCH", "4")
    out = align_ctc._ctc_emit_full(al, wav)
    assert torch.equal(out, ref)
    assert abs(out.shape[0] - 100 * sr // 320) <= 4  # seamless tiling, no frame loss


def test_align_text_dispatches_ctc_for_configured_lang(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backend.config, "align_model_for", lambda iso: "WAV2VEC2_ASR_LARGE_LV60K_960H"
    )
    sentinel = [{"text": "hi", "start": 0.0, "end": 0.5}]
    seen = {}

    def _fake_ctc(wav, text, iso, model):
        seen["args"] = (text, iso, model)
        return sentinel

    monkeypatch.setattr(backend, "align_text_ctc", _fake_ctc)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    out = backend.align_text(wav, "hi there", "en")
    assert out is sentinel
    assert seen["args"] == ("hi there", "en", "WAV2VEC2_ASR_LARGE_LV60K_960H")


def test_align_text_ctc_failure_falls_back_to_qwen(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backend.config, "align_model_for", lambda iso: "WAV2VEC2_ASR_LARGE_LV60K_960H"
    )

    def _boom(*a):
        raise RuntimeError("ctc broke")

    monkeypatch.setattr(backend, "align_text_ctc", _boom)
    _block_import(monkeypatch, "qwen_asr")
    backend._aligner = None
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    # CTC fails -> falls back to Qwen -> qwen_asr import blocked -> friendly voxweave[cuda]/[mps] error
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend.align_text(wav, "hello there", "en")


def test_align_text_qwen_for_unconfigured_lang(monkeypatch, tmp_path):
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: None)
    _block_import(monkeypatch, "qwen_asr")
    backend._aligner = None
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    # no CTC configured (None) -> skip CTC -> Qwen path -> friendly error
    with pytest.raises(RuntimeError, match=r"voxweave\[cuda\]"):
        backend.align_text(wav, "你好", "zh")


def test_release_clears_ctc(monkeypatch):
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    align_ctc._ctc = object()
    align_ctc._ctc_lang = "en"
    backend.release()
    assert align_ctc._ctc is None and align_ctc._ctc_lang is None


# --------------------------------------------------------------------------- #
# Japanese (no-space) CTC: per-char tokenization + OOV interpolation fallback (Phase B)
# --------------------------------------------------------------------------- #
def test_ctc_build_tokens_nospace_per_char():
    al = _FakeCtcAl(invocab={c: i for i, c in enumerate("私の学校はいええ")})
    toks, meta, words = align_ctc._ctc_build_tokens(["私の学校"], True, al)
    assert words == ["私", "の", "学", "校"]
    assert [m for m in meta if m >= 0] == [0, 1, 2, 3]  # per-char word indices
    assert meta.count(-1) == 5  # <star> at both edges + after every char
    assert all(t is not None for t, m in zip(toks, meta) if m >= 0)


def test_ctc_build_tokens_nospace_skips_punct_and_space():
    al = _FakeCtcAl(invocab={c: i for i, c in enumerate("はいええ")})
    toks, meta, words = align_ctc._ctc_build_tokens(["はい。 ええ"], True, al)
    assert words == [
        "は",
        "い",
        "え",
        "え",
    ]  # 。 and space are skipped, do not become tokens/units
    assert [m for m in meta if m >= 0] == [0, 1, 2, 3]


def test_ctc_build_tokens_nospace_oov_keeps_char():
    al = _FakeCtcAl(invocab={c: i for i, c in enumerate("私の")})  # 学/校 OOV
    toks, meta, words = align_ctc._ctc_build_tokens(["私の学校"], True, al)
    assert words == [
        "私",
        "の",
        "学",
        "校",
    ]  # OOV chars still enter words, no character dropped
    real = [t for t, m in zip(toks, meta) if m >= 0]
    assert real[2] is None and real[3] is None  # 学/校 OOV -> None (wildcard path)


def test_ctc_build_tokens_nospace_no_casing_on_latin():
    # critical invariant: no .lower()/.upper() (xlsr-ja vocab has uppercase A/C/P only; lowercasing would make them OOV)
    al = _FakeCtcAl(invocab={"A": 10, "C": 11, "P": 12, "私": 13})
    toks, meta, words = align_ctc._ctc_build_tokens(["A私"], True, al)
    assert words == ["A", "私"]
    assert [t for t, m in zip(toks, meta) if m >= 0] == [
        10,
        13,
    ]  # A hits directly without case-folding; .lower() would turn it into OOV


def test_interp_missing_monotonic():
    units = [
        {"text": "a", "start": 1.0, "end": 1.5},
        {
            "text": "b",
            "start": 2.0,
            "end": 2.0,
        },  # zero-length -> interpolate from both anchors
        {"text": "c", "start": 3.0, "end": 3.5},
    ]
    out = backend.interp_missing(units)
    assert len(out) == 3
    assert 1.5 <= out[1]["start"] <= 3.0 and out[1]["start"] <= out[1]["end"]


def test_interp_missing_single_anchor():
    units = [
        {"text": "a", "start": 1.0, "end": 1.5},
        {"text": "b", "start": 0.0, "end": 0.0},  # only forward anchor -> ffill
    ]
    out = backend.interp_missing(units)
    assert out[1]["start"] == 1.5


def test_interp_missing_all_invalid_noop():
    units = [{"text": "a", "start": 0.0, "end": 0.0}]
    out = backend.interp_missing(units)
    assert (
        len(out) == 1 and out == units
    )  # no anchors -> return unchanged without crashing


def test_interp_missing_no_unit_lost():
    units = [
        {"text": "a", "start": 0.0, "end": 0.0},
        {"text": "b", "start": 1.0, "end": 1.5},
        {"text": "c", "start": 2.0, "end": 2.0},
    ]
    out = backend.interp_missing(units)
    assert len(out) == len(units)  # never drops a unit


# --------------------------------------------------------------------------- #
# _get_ctc_aligner structural regression (CtcAligner namedtuple; both bundle and HF paths unpacked, no real model)
# --------------------------------------------------------------------------- #
def _fake_torchaudio(bundle_names, bundle=None):
    import types

    pipe = types.ModuleType("torchaudio.pipelines")
    pipe.__all__ = list(bundle_names)
    for name in bundle_names:
        setattr(pipe, name, bundle)
    ta = types.ModuleType("torchaudio")
    ta.pipelines = pipe
    return ta


def test_get_ctc_aligner_bundle_structure(monkeypatch):
    import sys

    class _M:
        def to(self, d):
            return self

        def eval(self):
            return self

    class _Bundle:
        sample_rate = 16000

        @staticmethod
        def get_model():
            return _M()

        @staticmethod
        def get_labels():
            return ("-", "|", "A", "B", "'")  # 0=blank '-', 1=sep '|'

    monkeypatch.setitem(sys.modules, "torchaudio", _fake_torchaudio(["FAKE"], _Bundle))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    align_ctc._ctc = None
    align_ctc._ctc_lang = None
    try:
        al = align_ctc._get_ctc_aligner("en", "FAKE")
        assert al.kind == "torchaudio" and al.blank == 0 and al.sep_id == 1
        assert al.proc is None and al.sr == 16000
        assert (
            al.invocab["A"] == 2 and al.invocab["'"] == 4
        )  # blank(0)/sep(1) excluded from invocab
        assert "-" not in al.invocab and "|" not in al.invocab
    finally:
        align_ctc._ctc = None
        align_ctc._ctc_lang = None


def test_get_ctc_aligner_hf_structure(monkeypatch):
    import sys
    import types

    class _Tok:
        pad_token_id = 0

        def get_vocab(self):
            return {"<pad>": 0, "</s>": 2, "|": 4, "私": 5, "の": 6}

    class _FE:
        sampling_rate = 16000

    seen_cache = []

    class _Proc:
        tokenizer = _Tok()
        feature_extractor = _FE()

        @classmethod
        def from_pretrained(cls, name, local_files_only=True, cache_dir=None):
            seen_cache.append(cache_dir)
            return cls()

    class _Model:
        @classmethod
        def from_pretrained(cls, name, local_files_only=True, cache_dir=None):
            seen_cache.append(cache_dir)
            return cls()

        def to(self, d):
            return self

        def eval(self):
            return self

    tf = types.ModuleType("transformers")
    tf.Wav2Vec2ForCTC = _Model
    tf.Wav2Vec2Processor = _Proc
    # model_name not in bundle __all__ -> takes HF branch
    monkeypatch.setitem(sys.modules, "torchaudio", _fake_torchaudio([]))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    align_ctc._ctc = None
    align_ctc._ctc_lang = None
    try:
        al = align_ctc._get_ctc_aligner("ja", "jonatasgrosman/xlsr-ja")
        assert al.kind == "hf" and al.blank == 0 and al.sep_id == 4
        assert al.proc is not None and al.sr == 16000
        assert (
            al.invocab["私"] == 5 and al.invocab["の"] == 6
        )  # HF path invocab = full vocab
        # processor + model both download into the voxweave align cache subdir
        assert seen_cache == [backend.config.ALIGN_CACHE, backend.config.ALIGN_CACHE]
    finally:
        align_ctc._ctc = None
        align_ctc._ctc_lang = None


# --------------------------------------------------------------------------- #
# Qwen context framing: bare term lists are framed as biasing metadata
# (a bare list regresses WER -- typewhisper-mac#321); prose and pre-framed
# input pass through; whisper initial_prompt never sees this helper.
# --------------------------------------------------------------------------- #
def test_format_qwen_context_frames_bare_list():
    out = backend.format_qwen_context("AVAudioEngine, Roformer\nkinsoku")
    assert out == "Proper nouns: AVAudioEngine, Roformer, kinsoku."


def test_format_qwen_context_passthrough_framed_and_prose():
    framed = "Technical terms: AVAudioEngine, Roformer."
    assert backend.format_qwen_context(framed) == framed
    assert backend.format_qwen_context("Vocabulary: a, b") == "Vocabulary: a, b"
    prose = "A sci-fi story. The hero is called Ryland Grace"
    assert backend.format_qwen_context(prose) == prose


def test_format_qwen_context_blank_is_none():
    assert backend.format_qwen_context(None) is None
    assert backend.format_qwen_context("   ") is None
