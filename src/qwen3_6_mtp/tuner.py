"""MTP configuration advisor.

Decision tree over three inputs (use case, objective, GPU) to produce
an MTP recommendation. Detects and blocks known-broken configurations
(TurboQuant + MTP, prefix cache degradation).
"""

from typing import Optional

from .backends.sglang import sglang_mtp_command
from .backends.vllm import vllm_mtp_command
from .bugs import check_prefix_cache_degradation, check_turboquant_conflict
from .hardware import get_gpu, get_model, vram_required
from .types import (
    Backend,
    GpuProfile,
    MtpRecommendation,
    Objective,
    Quantization,
    UseCase,
)


def recommend(
    use_case: UseCase,
    objective: Objective,
    gpu_id: str,
    model: str = "Qwen3.6-27B",
    quantization: Quantization = Quantization.BF16,
    enable_prefix_caching: bool = True,
    enable_turboquant: bool = False,
    model_path: Optional[str] = None,
) -> MtpRecommendation:
    """Generate an MTP configuration recommendation.

    Returns a recommendation that either enables MTP with specific
    parameters, or disables it with an explanation.
    """
    gpu = get_gpu(gpu_id)
    model_config = get_model(model)
    hf_path = model_path or f"Qwen/{model}"

    tq_bug = check_turboquant_conflict(enable_turboquant, 1)
    if tq_bug:
        return MtpRecommendation(
            enable=False,
            reason=(
                f"BLOCKED: TurboQuant KV + speculative decoding produces "
                f"degenerate token loops ({tq_bug.upstream_issue}). "
                f"Status: {tq_bug.status.value}. "
                f"Use TurboQuant or MTP, not both."
            ),
            backend=None,
            num_speculative_tokens=0,
            prefix_caching=enable_prefix_caching,
            expected_gain="N/A (prevented silent data corruption)",
            warnings=[
                f"TurboQuant + MTP = degenerate loops ({tq_bug.upstream_issue})",
                "TurboQuant + ngram = same bug",
                "Workaround: disable one or the other",
            ],
        )

    required = vram_required(model, quantization.value, include_mtp=True)
    if required > gpu.vram_gb:
        return MtpRecommendation(
            enable=False,
            reason=(
                f"Insufficient VRAM: {model} in {quantization.value} requires "
                f"~{required:.0f}GB (including MTP overhead), but {gpu.name} "
                f"has {gpu.vram_gb}GB."
            ),
            backend=None,
            num_speculative_tokens=0,
            prefix_caching=enable_prefix_caching,
            expected_gain="N/A",
            warnings=[
                f"VRAM required: ~{required:.0f}GB ({quantization.value} + MTP head)",
                f"VRAM available: {gpu.vram_gb}GB ({gpu.name})",
                f"MTP overhead: ~{model_config.mtp_overhead_gb}GB",
            ],
        )

    should_enable = _should_enable(use_case, objective, gpu)
    if not should_enable:
        return _disabled_recommendation(use_case, objective, gpu, enable_prefix_caching)

    num_spec, gain = _pick_spec_tokens(use_case, objective, gpu)
    warnings = []

    if gpu.id in ("m4-ultra", "m4-max"):
        warnings.extend(
            [
                "Apple Silicon: use MLX-LM, not vLLM/SGLang.",
                "Set --ctx-checkpoints 128 for multi-session caching "
                "(default 32 is less reliable).",
            ]
        )

    pc_bug = check_prefix_cache_degradation(enable_prefix_caching, num_spec)
    if pc_bug:
        warnings.append(
            f"L457 BUG: prefix cache hit rate drops ~21% with MTP "
            f"({pc_bug.upstream_issue}). Consider --no-enable-prefix-caching."
        )

    vllm_cfg = vllm_mtp_command(
        model=hf_path,
        num_speculative_tokens=num_spec,
        enable_prefix_caching=enable_prefix_caching,
    )
    sglang_cfg = sglang_mtp_command(
        model=hf_path,
        num_speculative_tokens=num_spec,
        enable_prefix_caching=enable_prefix_caching,
    )

    return MtpRecommendation(
        enable=True,
        reason=(
            f"MTP recommended for {use_case.value} / {objective.value} "
            f"on {gpu.name}. {num_spec} speculative token(s) for {gain}."
        ),
        backend=Backend.VLLM,
        num_speculative_tokens=num_spec,
        prefix_caching=enable_prefix_caching,
        expected_gain=gain,
        warnings=warnings,
        vllm_command=vllm_cfg.command,
        sglang_command=sglang_cfg.command,
    )


def _should_enable(use_case: UseCase, objective: Objective, gpu: GpuProfile) -> bool:
    if use_case == UseCase.SINGLE_USER and objective != Objective.MAXIMIZE_THROUGHPUT:
        return True
    if use_case == UseCase.MULTI_USER and objective == Objective.MINIMIZE_LATENCY:
        return True
    if objective == Objective.BALANCED and gpu.tier.value == "datacenter":
        return True
    return False


def _pick_spec_tokens(use_case: UseCase, objective: Objective, gpu: GpuProfile):
    if objective == Objective.MINIMIZE_LATENCY:
        if use_case == UseCase.SINGLE_USER:
            return 3, "~25-35% latency reduction (projected from 27.5% at k=1 on RTX 3090)"
        return 2, "~20-28% latency reduction (projected)"

    if gpu.tier.value == "consumer" and use_case == UseCase.MULTI_USER:
        return 1, "~10-18% latency reduction (may degrade at batch > 8)"

    return 2, "~15-22% latency reduction, ~5-10% throughput gain at low concurrency"


def _disabled_recommendation(use_case, objective, gpu, prefix_caching):
    if objective == Objective.MAXIMIZE_THROUGHPUT and use_case == UseCase.MULTI_USER:
        reason = (
            "MTP is a net throughput loss under high concurrency. "
            "Speculative tokens consume KV cache capacity, reducing "
            "effective batch size. Maximize batch size instead."
        )
    else:
        reason = (
            f"For {use_case.value} / {objective.value} on {gpu.name}, "
            f"the MTP overhead outweighs benefits."
        )

    return MtpRecommendation(
        enable=False,
        reason=reason,
        backend=None,
        num_speculative_tokens=0,
        prefix_caching=prefix_caching,
        expected_gain="Baseline (no MTP)",
        warnings=[
            "MTP-1 reduces per-token latency but degrades throughput under high concurrency",
            "Without MTP, more concurrent requests fit in the same KV cache",
        ],
    )
