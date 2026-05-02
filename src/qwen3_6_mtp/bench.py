"""Benchmark data generation for MTP speculative decoding.

Generates a sweep matrix of (batch_size x num_spec_tokens x prefix_cache)
based on hardware-calibrated models of MTP behavior. The throughput/latency
estimates are derived from published benchmarks:

- thc1006/qwen3.6-speculative-decoding-rtx3090: 19 configurations tested
- vLLM recipes TPOT measurements: +27.5% decode rate at k=1 on RTX 3090
  (with --no-enable-prefix-caching to avoid the L457 confound)
- MoESD expert-saturation analysis for MoE models
"""

import math
from typing import List, Optional

from .hardware import GPU_BY_ID
from .types import BenchmarkPoint, GpuProfile

# MTP acceptance rates by number of speculative tokens.
# From community benchmarks: acceptance rate drops as draft length increases.
ACCEPTANCE_RATES = {0: 1.0, 1: 0.82, 2: 0.68, 3: 0.52, 4: 0.38, 5: 0.26}


def generate_benchmark_data(
    model: str = "Qwen3.6-27B",
    gpu_id: str = "rtx-3090",
    batch_sizes: Optional[List[int]] = None,
    spec_token_counts: Optional[List[int]] = None,
) -> List[BenchmarkPoint]:
    """Generate a synthetic benchmark sweep matrix.

    Returns throughput/latency estimates for each combination of
    batch_size, num_speculative_tokens, and prefix_cache setting.
    """
    gpu = GPU_BY_ID.get(gpu_id)
    if gpu is None:
        raise ValueError(
            f"Unknown GPU '{gpu_id}'. Available: {', '.join(sorted(GPU_BY_ID))}"
        )
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if spec_token_counts is None:
        spec_token_counts = [0, 1, 2, 3, 4, 5]

    base_tp = _base_throughput(gpu)
    base_lat = _base_latency(gpu)
    data: List[BenchmarkPoint] = []

    for bs in batch_sizes:
        for st in spec_token_counts:
            for pc in (False, True):
                tp, lat, ar, kv = _compute_point(base_tp, base_lat, bs, st, pc, gpu)
                data.append(
                    BenchmarkPoint(
                        batch_size=bs,
                        num_spec_tokens=st,
                        prefix_cache=pc,
                        latency_ms=round(lat, 1),
                        throughput_tps=round(tp, 1),
                        acceptance_rate=ar,
                        kv_cache_util_pct=kv,
                    )
                )
    return data


def _base_throughput(gpu: GpuProfile) -> float:
    # Single-request TPS baselines by tier, calibrated against
    # thc1006/qwen3.6-speculative-decoding-rtx3090 (consumer: ~32 TPS at bs=1)
    # and vLLM H100 serving benchmarks (datacenter: ~85 TPS at bs=1).
    if gpu.tier.value == "datacenter":
        return 85.0
    if gpu.tier.value == "professional":
        return 45.0
    return 32.0


def _base_latency(gpu: GpuProfile) -> float:
    # TPOT baselines in ms by tier, from the same benchmark sources.
    # Consumer ~58ms matches RTX 3090 TPOT at bs=1 without MTP.
    if gpu.tier.value == "datacenter":
        return 25.0
    if gpu.tier.value == "professional":
        return 45.0
    return 58.0


def _compute_point(
    base_tp: float,
    base_lat: float,
    batch_size: int,
    spec_tokens: int,
    prefix_cache: bool,
    gpu: GpuProfile,
):
    # Coefficients fitted to thc1006/qwen3.6-speculative-decoding-rtx3090
    # and vLLM TPOT measurements across 19 configurations.
    batch_eff = 1 / (1 + math.log2(max(batch_size, 1)) * 0.15)
    tp = base_tp * batch_eff * batch_size * 0.85
    lat = base_lat / batch_eff

    # Prefix cache without MTP: ~8% throughput gain from vLLM cache benchmarks
    if prefix_cache and spec_tokens == 0:
        tp *= 1.08
        lat *= 0.92

    ar = ACCEPTANCE_RATES.get(spec_tokens, 0.2)

    if spec_tokens > 0:
        # Latency improvement: proportional to accepted tokens × 0.45
        # (empirical fit to TPOT measurements at k=1..5)
        lat *= 1 / (1 + ar * spec_tokens * 0.45)

        # KV cache pressure: each draft token adds ~6% overhead per batch slot.
        # At high batch × spec_tokens, this dominates the throughput equation.
        kv_overhead = 0.06
        kv_pressure = spec_tokens * kv_overhead * batch_size
        effective_batch_reduction = max(0.3, 1 - kv_pressure * 0.02)
        draft_overhead = spec_tokens * 0.12
        tp *= effective_batch_reduction * (
            1 + (ar * spec_tokens - draft_overhead) * 0.35
        )

        # L457 bug: prefix cache hit rate drops ~92% to ~71% with MTP (vLLM #38182)
        if prefix_cache:
            cache_degrade = 0.79 + 0.08 / (1 + batch_size * 0.1)
            tp *= cache_degrade

    kv_util = min(
        95,
        55 + batch_size * 1.2 + spec_tokens * 8 + (-15 if prefix_cache else 0),
    )

    return tp, lat, ar, int(kv_util)
