import pytest

from qwen_mtp.hardware import (
    GPU_BY_ID,
    GPU_PROFILES,
    MODEL_CONFIGS,
    SAMPLING_PRESETS,
    get_gpu,
    get_model,
    vram_required,
)


def test_gpu_profiles_non_empty():
    assert len(GPU_PROFILES) >= 15


def test_gpu_by_id_matches():
    for gpu in GPU_PROFILES:
        assert GPU_BY_ID[gpu.id] is gpu


def test_get_gpu_valid():
    gpu = get_gpu("rtx-3090")
    assert gpu.name == "NVIDIA RTX 3090"
    assert gpu.vram_gb == 24


def test_get_gpu_invalid():
    with pytest.raises(ValueError, match="Unknown GPU"):
        get_gpu("nonexistent-gpu")


def test_get_model_valid():
    model = get_model("Qwen3.6-27B")
    assert model.total_params == "27B"
    assert model.mtp_num_hidden_layers == 1


def test_get_model_35b():
    model = get_model("Qwen3.6-35B-A3B")
    assert model.active_params == "3B"
    assert model.architecture.startswith("MoE")


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("Qwen3.6-99B")


def test_vram_required_bf16():
    vram = vram_required("Qwen3.6-27B", "bf16", include_mtp=True)
    assert vram == 54 + 2.0


def test_vram_required_int4_no_mtp():
    vram = vram_required("Qwen3.6-27B", "int4", include_mtp=False)
    assert vram == 14


def test_vram_required_fp8():
    vram = vram_required("Qwen3.6-35B-A3B", "fp8")
    assert vram == 36 + 2.5


def test_model_configs_have_both():
    assert "Qwen3.6-27B" in MODEL_CONFIGS
    assert "Qwen3.6-35B-A3B" in MODEL_CONFIGS


def test_sampling_presets_exist():
    assert "27B-thinking-general" in SAMPLING_PRESETS
    assert "27B-non-thinking" in SAMPLING_PRESETS
    assert "35B-A3B-thinking-general" in SAMPLING_PRESETS


def test_sampling_presets_differ():
    thinking = SAMPLING_PRESETS["27B-thinking-general"]
    non_thinking = SAMPLING_PRESETS["27B-non-thinking"]
    assert thinking.temperature != non_thinking.temperature
    assert thinking.top_p != non_thinking.top_p


def test_27b_vs_35b_presence_penalty():
    p27 = SAMPLING_PRESETS["27B-thinking-general"]
    p35 = SAMPLING_PRESETS["35B-A3B-thinking-general"]
    assert p27.presence_penalty == 0.0
    assert p35.presence_penalty == 1.5
