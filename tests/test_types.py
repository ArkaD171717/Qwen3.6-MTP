from qwen3_6_mtp.types import (
    Backend,
    BugSeverity,
    BugStatus,
    GpuTier,
    MtpRecommendation,
    Objective,
    Quantization,
    UseCase,
)


def test_backend_values():
    assert Backend.VLLM.value == "vllm"
    assert Backend.SGLANG.value == "sglang"


def test_gpu_tier_values():
    assert GpuTier.CONSUMER.value == "consumer"
    assert GpuTier.DATACENTER.value == "datacenter"


def test_objective_values():
    assert Objective.MINIMIZE_LATENCY.value == "minimize-latency"
    assert Objective.MAXIMIZE_THROUGHPUT.value == "maximize-throughput"
    assert Objective.BALANCED.value == "balanced"


def test_use_case_values():
    assert UseCase.SINGLE_USER.value == "single-user"
    assert UseCase.MULTI_USER.value == "multi-user"


def test_quantization_values():
    assert Quantization.BF16.value == "bf16"
    assert Quantization.FP8.value == "fp8"
    assert Quantization.INT4.value == "int4"


def test_bug_severity_values():
    assert BugSeverity.CRITICAL.value == "critical"
    assert BugSeverity.HIGH.value == "high"


def test_bug_status_values():
    assert BugStatus.OPEN.value == "open"
    assert BugStatus.CLOSED.value == "closed"


def test_recommendation_to_dict():
    rec = MtpRecommendation(
        enable=True,
        reason="test",
        backend=Backend.VLLM,
        num_speculative_tokens=2,
        prefix_caching=False,
        expected_gain="~25%",
        warnings=["warn1"],
        vllm_command="vllm serve ...",
    )
    d = rec.to_dict()
    assert d["enable"] is True
    assert d["backend"] == "vllm"
    assert d["num_speculative_tokens"] == 2
    assert len(d["warnings"]) == 1


def test_recommendation_disabled_to_dict():
    rec = MtpRecommendation(
        enable=False,
        reason="blocked",
        backend=None,
        num_speculative_tokens=0,
        prefix_caching=True,
        expected_gain="N/A",
    )
    d = rec.to_dict()
    assert d["backend"] is None
    assert d["enable"] is False
