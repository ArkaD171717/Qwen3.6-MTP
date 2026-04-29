"""Crossover analysis: find the batch size where MTP flips from
net-positive to net-negative throughput."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .bench import generate_benchmark_data
from .types import BenchmarkPoint, CrossoverPoint


def find_crossover_points(
    benchmark_data: List[BenchmarkPoint],
) -> List[CrossoverPoint]:
    """Compare each MTP configuration against the no-MTP baseline
    and compute the throughput delta percentage."""
    baselines: Dict[str, Dict[int, float]] = {"true": {}, "false": {}}
    for dp in benchmark_data:
        if dp.num_spec_tokens == 0:
            key = "true" if dp.prefix_cache else "false"
            baselines[key][dp.batch_size] = dp.throughput_tps

    crossovers: List[CrossoverPoint] = []
    for dp in benchmark_data:
        if dp.num_spec_tokens == 0:
            continue
        key = "true" if dp.prefix_cache else "false"
        base_tp = baselines.get(key, {}).get(dp.batch_size)
        if base_tp is None or base_tp == 0:
            continue
        delta = (dp.throughput_tps - base_tp) / base_tp * 100
        crossovers.append(
            CrossoverPoint(
                batch_size=dp.batch_size,
                spec_tokens=dp.num_spec_tokens,
                prefix_cache=dp.prefix_cache,
                is_net_positive=delta > 0,
                delta_pct=round(delta, 1),
            )
        )
    return crossovers


@dataclass
class CrossoverSummary:
    spec_tokens: int
    crossover_batch_size: Optional[int]
    max_positive_delta_pct: float


def summarize_crossovers(
    crossover_points: List[CrossoverPoint],
    prefix_cache: bool = False,
) -> List[CrossoverSummary]:
    """For each speculative token count, find the batch size at which
    MTP becomes net-negative (the crossover point).

    When prefix_cache is False (default), only considers data without
    prefix caching. Set to True to see crossover behavior with the
    L457 cache degradation factored in.
    """
    summaries: List[CrossoverSummary] = []
    for n in range(1, 6):
        points = sorted(
            [
                cp
                for cp in crossover_points
                if cp.spec_tokens == n and cp.prefix_cache == prefix_cache
            ],
            key=lambda cp: cp.batch_size,
        )
        crossover_bs: Optional[int] = None
        max_delta = 0.0
        for p in points:
            if p.delta_pct > 0:
                max_delta = max(max_delta, p.delta_pct)
            if p.delta_pct <= 0 and crossover_bs is None:
                crossover_bs = p.batch_size
        summaries.append(
            CrossoverSummary(
                spec_tokens=n,
                crossover_batch_size=crossover_bs,
                max_positive_delta_pct=round(max_delta, 1),
            )
        )
    return summaries


def quick_crossover(
    model: str = "Qwen3.6-27B",
    gpu_id: str = "rtx-3090",
) -> List[CrossoverSummary]:
    """Generate benchmark data and return crossover summaries in one call."""
    data = generate_benchmark_data(model=model, gpu_id=gpu_id)
    points = find_crossover_points(data)
    return summarize_crossovers(points)
