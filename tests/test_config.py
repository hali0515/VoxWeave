"""voxweave.config — ~/.config/voxweave.conf (TOML) loading, precedence, and first-run creation. Pure stdlib, no model dependency."""

import logging
import sys
import types
from pathlib import Path

import pytest

from voxweave import config


@pytest.fixture
def conf_at(tmp_path, monkeypatch):
    """Point the config path to a tmp dir; returns that Path (does not exist by default)."""
    p = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(p))
    return p


def test_config_path_env_override(conf_at):
    assert config.config_path() == conf_at


def test_config_path_default(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_CONFIG", raising=False)
    p = config.config_path()
    assert p.name == "voxweave.conf" and p.parent.name == ".config"


def test_load_missing_returns_empty(conf_at):
    assert config._load() == {}


def test_load_malformed_returns_empty(conf_at):
    conf_at.write_text("this is = = not [ valid toml", encoding="utf-8")
    assert (
        config._load() == {}
    )  # malformed file does not crash, treated as empty config


def test_ensure_default_creates_with_builtin(conf_at):
    assert not conf_at.exists()
    config.ensure_default_config()
    txt = conf_at.read_text(encoding="utf-8")
    assert "[align]" in txt and "facebook/wav2vec2-large-960h-lv60-self" in txt


def test_ensure_default_idempotent_no_overwrite(conf_at):
    conf_at.write_text('asr_model = "custom/keep"\n', encoding="utf-8")
    config.ensure_default_config()  # already exists -> no-op
    assert conf_at.read_text(encoding="utf-8") == 'asr_model = "custom/keep"\n'


def test_conf_asr_model_from_file(conf_at):
    conf_at.write_text('asr_model = "Qwen/Qwen3-ASR-1.7B"\n', encoding="utf-8")
    assert config.conf_asr_model() == "Qwen/Qwen3-ASR-1.7B"


def test_conf_asr_model_none_when_absent(conf_at):
    assert config.conf_asr_model() is None


def test_align_model_for_en_builtin(conf_at):
    # no config file -> built-in en -> large wav2vec2 CTC (HF path -> ~/.cache/huggingface/hub)
    assert config.align_model_for("en") == "facebook/wav2vec2-large-960h-lv60-self"


def test_align_model_for_ja_builtin_mms(conf_at):
    # no config file -> built-in ja -> "mms" (MMS-300m + uroman, full-file single pass; = whisperx fork align_ctc)
    assert config.align_model_for("ja") == "mms"


def test_align_model_for_unset_lang_none(conf_at):
    # zh has no built-in (uses Qwen) and is not configured -> None
    assert config.align_model_for("zh") is None


def test_align_model_for_config_override(conf_at):
    conf_at.write_text('[align]\nen = "WAV2VEC2_ASR_BASE_960H"\n', encoding="utf-8")
    assert config.align_model_for("en") == "WAV2VEC2_ASR_BASE_960H"


def test_align_model_for_explicit_empty_disables(conf_at):
    # empty string = explicit fallback to Qwen (user wants to disable CTC for a language)
    conf_at.write_text('[align]\nen = ""\n', encoding="utf-8")
    assert config.align_model_for("en") is None


def test_align_model_for_added_lang(conf_at):
    conf_at.write_text('[align]\nzh = "some/wav2vec2"\n', encoding="utf-8")
    assert config.align_model_for("zh") == "some/wav2vec2"


# --- model load strategy (peak serial / sum concurrent) ------------------- #
def test_load_strategy_default_peak(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_LOAD_STRATEGY", raising=False)
    assert config.conf_load_strategy() == "peak"  # default = serial peak-shaving


def test_load_strategy_from_conf(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_LOAD_STRATEGY", raising=False)
    conf_at.write_text('load_strategy = "sum"\n', encoding="utf-8")
    assert config.conf_load_strategy() == "sum"


def test_load_strategy_env_overrides_conf(conf_at, monkeypatch):
    conf_at.write_text('load_strategy = "peak"\n', encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_LOAD_STRATEGY", "sum")
    assert config.conf_load_strategy() == "sum"  # env > file


def test_load_strategy_invalid_falls_back_peak(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_LOAD_STRATEGY", raising=False)
    conf_at.write_text('load_strategy = "bogus"\n', encoding="utf-8")
    assert (
        config.conf_load_strategy() == "peak"
    )  # invalid value does not crash, falls back to peak


def test_ctc_max_dp_frames_default(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_CTC_MAX_DP_FRAMES", raising=False)
    assert config.conf_ctc_max_dp_frames() == 90000  # ~30min at 50fps


def test_ctc_max_dp_frames_from_conf(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_CTC_MAX_DP_FRAMES", raising=False)
    conf_at.write_text("ctc_max_dp_frames = 150000\n", encoding="utf-8")
    assert config.conf_ctc_max_dp_frames() == 150000


def test_ctc_max_dp_frames_env_overrides_conf(conf_at, monkeypatch):
    conf_at.write_text("ctc_max_dp_frames = 150000\n", encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_CTC_MAX_DP_FRAMES", "200000")
    assert config.conf_ctc_max_dp_frames() == 200000  # env > file


def test_ctc_max_dp_frames_invalid_conf_falls_back(conf_at, monkeypatch):
    monkeypatch.delenv("VOXWEAVE_CTC_MAX_DP_FRAMES", raising=False)
    conf_at.write_text('ctc_max_dp_frames = "lots"\n', encoding="utf-8")
    assert config.conf_ctc_max_dp_frames() == 90000  # non-int does not crash


# --- inference batch sizes ([batch] separate/ctc/mms) --------------------- #
@pytest.fixture
def no_batch_env(monkeypatch):
    for v in config._BATCH_ENV.values():
        monkeypatch.delenv(v, raising=False)


def test_conf_batch_defaults(conf_at, no_batch_env):
    assert config.conf_batch("separate") == 1
    assert config.conf_batch("ctc") == 1
    assert config.conf_batch("mms") == 4


def test_conf_batch_from_conf(conf_at, no_batch_env):
    conf_at.write_text("[batch]\nseparate = 4\nmms = 8\n", encoding="utf-8")
    assert config.conf_batch("separate") == 4
    assert config.conf_batch("mms") == 8
    assert config.conf_batch("ctc") == 1  # unlisted key keeps its default


def test_conf_batch_env_overrides_conf(conf_at, no_batch_env, monkeypatch):
    conf_at.write_text("[batch]\nseparate = 4\n", encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_SEP_BATCH", "8")
    assert config.conf_batch("separate") == 8  # env > file
    # mms keeps its pre-[batch] env name for back-compat
    monkeypatch.setenv("VOXWEAVE_MMS_BATCH", "2")
    assert config.conf_batch("mms") == 2


def test_conf_batch_invalid_falls_back(conf_at, no_batch_env, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CTC_BATCH", "lots")  # non-int env -> next source
    conf_at.write_text('[batch]\nctc = "many"\n', encoding="utf-8")  # non-int file too
    assert config.conf_batch("ctc") == 1


def test_conf_batch_clamps_to_min_1(conf_at, no_batch_env):
    conf_at.write_text("[batch]\nseparate = 0\n", encoding="utf-8")
    assert config.conf_batch("separate") == 1


def test_default_template_has_batch_section(conf_at):
    config.ensure_default_config()
    txt = conf_at.read_text(encoding="utf-8")
    assert "[batch]" in txt and "VOXWEAVE_SEP_BATCH" in txt


# --- hf token precedence (env > conf > huggingface_hub stored token) -------- #
@pytest.fixture
def no_hf_env(monkeypatch):
    """Remove all HF token env vars so file / hub fallbacks are exercised."""
    for v in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(v, raising=False)


def _fake_hub(token_or_exc):
    """Fake ``huggingface_hub`` module whose get_token returns a value or raises."""
    mod = types.ModuleType("huggingface_hub")

    def get_token():
        if isinstance(token_or_exc, Exception):
            raise token_or_exc
        return token_or_exc

    mod.get_token = get_token  # type: ignore[attr-defined]
    return mod


def test_conf_hf_token_from_hub_when_unset(conf_at, no_hf_env, monkeypatch):
    # nothing in env or file -> fall back to the huggingface CLI stored token
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub("hub-tok"))
    assert config.conf_hf_token() == "hub-tok"


def test_conf_hf_token_env_wins_over_hub(conf_at, no_hf_env, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "env-tok")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub("hub-tok"))
    assert config.conf_hf_token() == "env-tok"  # env > hub


def test_conf_hf_token_conf_wins_over_hub(conf_at, no_hf_env, monkeypatch):
    conf_at.write_text('hf_token = "file-tok"\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub("hub-tok"))
    assert config.conf_hf_token() == "file-tok"  # conf > hub


def test_conf_hf_token_hub_raises_returns_none(conf_at, no_hf_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(RuntimeError("boom")))
    assert config.conf_hf_token() is None  # hub failure is swallowed


def test_conf_hf_token_hub_empty_returns_none(conf_at, no_hf_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(""))
    assert config.conf_hf_token() is None  # empty stored token -> None


# --- #24 config silently swallows errors ----------------------------------- #
# (a) unknown/misspelled top-level key -> log.warning naming the key, known
#     keys in the same file still load normally.
def test_unknown_top_level_key_warns(conf_at, caplog):
    conf_at.write_text(
        'asr_model = "Qwen/Qwen3-ASR-1.7B"\nfrobnicate = "typo"\n', encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        conf = config._load()
    assert "unknown config key" in caplog.text.lower()
    assert "frobnicate" in caplog.text
    assert conf.get("asr_model") == "Qwen/Qwen3-ASR-1.7B"  # known key still loads


# (b) type-mismatched value for a known key -> log.warning naming the key,
#     accessor falls back to its built-in default (never raises, never
#     silently misuses the wrongly-typed value).
def test_asr_model_type_mismatch_warns_and_falls_back(conf_at, caplog):
    conf_at.write_text("asr_model = 123\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = config.conf_asr_model()
    assert result is None  # caller falls back to env/built-in default
    assert "asr_model" in caplog.text


def test_ctc_max_dp_frames_type_mismatch_warns(conf_at, monkeypatch, caplog):
    monkeypatch.delenv("VOXWEAVE_CTC_MAX_DP_FRAMES", raising=False)
    conf_at.write_text('ctc_max_dp_frames = "lots"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = config.conf_ctc_max_dp_frames()
    assert result == 90000  # built-in default
    assert "ctc_max_dp_frames" in caplog.text


def test_align_model_for_type_mismatch_falls_back_to_builtin(conf_at, caplog):
    # A stray int under [align] must fall back to the per-language built-in
    # default, not be silently conflated with an explicit "" (= disable, use
    # Qwen). Currently both collapse to None via _nonempty_str -> misuse.
    conf_at.write_text("[align]\nen = 123\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = config.align_model_for("en")
    assert result == "facebook/wav2vec2-large-960h-lv60-self"
    assert "align" in caplog.text or "en" in caplog.text


# (c) syntactically invalid TOML -> log.warning naming the file, defaults used.
def test_load_malformed_logs_warning_with_filename(conf_at, caplog):
    conf_at.write_text("this is = = not [ valid toml", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        conf = config._load()
    assert conf_at.name in caplog.text
    assert conf == {}


# --- #25 migration failure silent ------------------------------------------ #
def test_migration_rename_failure_names_old_path(tmp_path, monkeypatch, caplog):
    """When migrating legacy ~/.config/qsub.conf fails (e.g. cross-device
    rename), the warning must name the OLD config path so the user can
    migrate the file by hand."""
    old = tmp_path / "qsub.conf"
    old.write_text('asr_model = "custom"\n', encoding="utf-8")
    new = tmp_path / ".config" / "voxweave.conf"
    monkeypatch.delenv("VOXWEAVE_CONFIG", raising=False)
    monkeypatch.setattr(config, "_LEGACY_CONFIG", old)
    monkeypatch.setattr(config, "config_path", lambda: new)

    def _raise_rename(self, target):
        raise OSError("cross-device link")

    monkeypatch.setattr(Path, "rename", _raise_rename)

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        config.ensure_default_config()

    assert str(old) in caplog.text
    assert new.exists()  # still falls back to writing a fresh template


# --- [defaults] boolean pipeline flag defaults ---


def test_conf_default_flag_from_file(conf_at):
    conf_at.write_text("[defaults]\nseparate = false\n", encoding="utf-8")
    assert config.conf_default_flag("separate", True) is False


def test_conf_default_flag_true_from_file(conf_at):
    conf_at.write_text("[defaults]\nvad_mask = true\n", encoding="utf-8")
    assert config.conf_default_flag("vad_mask", False) is True


def test_conf_default_flag_missing_uses_builtin(conf_at):
    assert config.conf_default_flag("separate", True) is True
    assert config.conf_default_flag("normalize", False) is False


def test_conf_default_flag_wrong_type_warns_and_falls_back(conf_at, caplog):
    conf_at.write_text('[defaults]\nseparate = "yes"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert config.conf_default_flag("separate", True) is True
    assert "wrong type" in caplog.text


def test_default_template_documents_defaults_section(conf_at):
    config.ensure_default_config()
    txt = conf_at.read_text(encoding="utf-8")
    assert "[defaults]" in txt and "# separate = true" in txt
