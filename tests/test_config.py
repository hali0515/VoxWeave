"""voxweave.config — ~/.config/voxweave.conf (TOML) loading, precedence, and first-run creation. Pure stdlib, no model dependency."""

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
