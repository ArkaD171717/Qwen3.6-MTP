"""Generate crossover analysis CSVs for all GPU profiles."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3_6_mtp import GPU_BY_ID, generate_benchmark_data, quick_crossover

RESULTS_DIR = Path(__file__).resolve().parent
MODEL = "Qwen3.6-27B"


def write_crossover_summary():
    path = RESULTS_DIR / "crossover_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["gpu_id", "spec_tokens", "crossover_batch_size", "max_positive_delta_pct"]
        )
        for gpu_id in sorted(GPU_BY_ID):
            for s in quick_crossover(model=MODEL, gpu_id=gpu_id):
                writer.writerow(
                    [
                        gpu_id,
                        s.spec_tokens,
                        s.crossover_batch_size if s.crossover_batch_size else "",
                        s.max_positive_delta_pct,
                    ]
                )
    return path


def write_benchmark_sweep():
    path = RESULTS_DIR / "benchmark_sweep.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gpu_id",
                "batch_size",
                "num_spec_tokens",
                "prefix_cache",
                "latency_ms",
                "throughput_tps",
                "acceptance_rate",
                "kv_cache_util_pct",
            ]
        )
        for gpu_id in sorted(GPU_BY_ID):
            for dp in generate_benchmark_data(model=MODEL, gpu_id=gpu_id):
                writer.writerow(
                    [
                        gpu_id,
                        dp.batch_size,
                        dp.num_spec_tokens,
                        dp.prefix_cache,
                        dp.latency_ms,
                        dp.throughput_tps,
                        dp.acceptance_rate,
                        dp.kv_cache_util_pct,
                    ]
                )
    return path


def print_summary():
    header = f"{'GPU':<14} {'MTP-k':>6} {'Crossover BS':>13} {'Best Gain':>10}"
    print(header)
    print("-" * len(header))
    for gpu_id in sorted(GPU_BY_ID):
        for s in quick_crossover(model=MODEL, gpu_id=gpu_id):
            bs_str = str(s.crossover_batch_size) if s.crossover_batch_size else "none"
            print(
                f"{gpu_id:<14} {f'MTP-{s.spec_tokens}':>6} {bs_str:>13} {f'+{s.max_positive_delta_pct}%':>10}"
            )


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"GPU profiles: {len(GPU_BY_ID)}\n")

    p1 = write_crossover_summary()
    print(f"Wrote {p1}")

    p2 = write_benchmark_sweep()
    print(f"Wrote {p2}")

    print()
    print_summary()
