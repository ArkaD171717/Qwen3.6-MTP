import pytest

from qwen_mtp.tuner import recommend
from qwen_mtp.types import Objective, Quantization, UseCase


class TestTunerRecommend:
    def test_single_user_latency_enables_mtp(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-4090",
            quantization=Quantization.INT4,
        )
        assert rec.enable is True
        assert rec.num_speculative_tokens >= 2
        assert rec.vllm_command is not None
        assert rec.sglang_command is not None

    def test_multi_user_throughput_disables_mtp(self):
        rec = recommend(
            UseCase.MULTI_USER,
            Objective.MAXIMIZE_THROUGHPUT,
            "rtx-3090",
            quantization=Quantization.INT4,
        )
        assert rec.enable is False
        assert rec.num_speculative_tokens == 0
        assert (
            "throughput loss" in rec.reason.lower() or "overhead" in rec.reason.lower()
        )

    def test_single_user_balanced_enables_mtp(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.BALANCED,
            "rtx-3090",
            quantization=Quantization.INT4,
        )
        assert rec.enable is True

    def test_multi_user_latency_enables_mtp(self):
        rec = recommend(UseCase.MULTI_USER, Objective.MINIMIZE_LATENCY, "h100-sxm")
        assert rec.enable is True

    def test_datacenter_balanced_enables_mtp(self):
        rec = recommend(UseCase.MULTI_USER, Objective.BALANCED, "h100-sxm")
        assert rec.enable is True

    def test_consumer_balanced_multi_disables(self):
        rec = recommend(
            UseCase.MULTI_USER,
            Objective.BALANCED,
            "rtx-3090",
            quantization=Quantization.INT4,
        )
        assert rec.enable is False


class TestTunerTurboquant:
    def test_turboquant_blocks_mtp(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-4090",
            enable_turboquant=True,
        )
        assert rec.enable is False
        assert "BLOCKED" in rec.reason
        assert "40831" in rec.reason
        assert rec.num_speculative_tokens == 0

    def test_turboquant_warning_mentions_bug(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "h100-sxm",
            enable_turboquant=True,
        )
        assert any("degenerate" in w.lower() or "40831" in w for w in rec.warnings)


class TestTunerVram:
    def test_insufficient_vram(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-3080",
            model="Qwen3.6-27B",
            quantization=Quantization.BF16,
        )
        assert rec.enable is False
        assert "insufficient vram" in rec.reason.lower()

    def test_sufficient_vram_int4(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-3090",
            model="Qwen3.6-27B",
            quantization=Quantization.INT4,
        )
        assert rec.enable is True


class TestTunerPrefixCache:
    def test_prefix_cache_warning(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-4090",
            quantization=Quantization.INT4,
            enable_prefix_caching=True,
        )
        assert rec.enable is True
        assert any("L457" in w for w in rec.warnings)

    def test_no_prefix_cache_no_warning(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-4090",
            quantization=Quantization.INT4,
            enable_prefix_caching=False,
        )
        assert not any("L457" in w for w in rec.warnings)


class TestTunerAppleSilicon:
    def test_apple_silicon_warning(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "m4-max",
            model="Qwen3.6-27B",
            quantization=Quantization.INT4,
        )
        assert any("Apple Silicon" in w or "MLX" in w for w in rec.warnings)


class TestTunerSpecTokenCounts:
    def test_single_latency_gets_3(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-4090",
            quantization=Quantization.INT4,
        )
        assert rec.num_speculative_tokens == 3

    def test_multi_latency_gets_2(self):
        rec = recommend(UseCase.MULTI_USER, Objective.MINIMIZE_LATENCY, "h100-sxm")
        assert rec.num_speculative_tokens == 2

    def test_consumer_multi_latency_gets_1(self):
        rec = recommend(
            UseCase.MULTI_USER,
            Objective.MINIMIZE_LATENCY,
            "rtx-3090",
            model="Qwen3.6-27B",
            quantization=Quantization.INT4,
        )
        assert rec.num_speculative_tokens <= 2


class TestTunerOutput:
    def test_to_dict(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.BALANCED,
            "rtx-4090",
            quantization=Quantization.INT4,
        )
        d = rec.to_dict()
        assert "enable" in d
        assert "reason" in d
        assert "vllm_command" in d

    def test_vllm_command_contains_model(self):
        rec = recommend(
            UseCase.SINGLE_USER,
            Objective.BALANCED,
            "rtx-4090",
            quantization=Quantization.INT4,
            model_path="Qwen/Qwen3.6-27B",
        )
        assert "Qwen/Qwen3.6-27B" in rec.vllm_command

    def test_invalid_gpu_raises(self):
        with pytest.raises(ValueError):
            recommend(UseCase.SINGLE_USER, Objective.BALANCED, "fake-gpu")
