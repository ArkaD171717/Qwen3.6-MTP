# Qwen3.6-MTP

MTP speculative decoding tuner for Qwen3.6. Generates vLLM/SGLang configs, finds throughput crossover points, and catches known bugs.

## What It Does

- **Configuration advisor**: Recommends MTP on/off with parameters via a decision tree over use case, objective, and GPU
- **Backend configs**: Generates vLLM (`method: mtp`) and SGLang (`NEXTN` algorithm) serve commands
- **Crossover analysis**: Finds the batch size where MTP flips from net-positive to net-negative throughput
- **Bug detection**: Detects and blocks known-broken configurations (TurboQuant + MTP, prefix cache degradation)
- **Benchmark sweep**: Generate latency/throughput matrices across batch size, speculative tokens, and prefix cache settings

## Installation

```bash
pip install qwen3.6-mtp
```

## Quick Start

```python
from qwen3_6_mtp import recommend, UseCase, Objective, Quantization

rec = recommend(
    use_case=UseCase.SINGLE_USER,
    objective=Objective.MINIMIZE_LATENCY,
    gpu_id="rtx-4090",
    quantization=Quantization.INT4,
)

print(rec.enable)           # True
print(rec.expected_gain)    # ~25-35% latency reduction (projected)
print(rec.vllm_command)     # Full vllm serve command with MTP flags
print(rec.sglang_command)   # Equivalent SGLang command
```

### Crossover Analysis

```python
from qwen3_6_mtp import quick_crossover

for s in quick_crossover(gpu_id="rtx-3090"):
    print(f"MTP-{s.spec_tokens}: crossover at batch {s.crossover_batch_size}, "
          f"best gain +{s.max_positive_delta_pct}%")
```

### Backend Config Generation

```python
from qwen3_6_mtp import vllm_mtp_command, sglang_mtp_command

vllm = vllm_mtp_command(model="Qwen/Qwen3.6-27B", num_speculative_tokens=2)
print(vllm.command)

sglang = sglang_mtp_command(model="Qwen/Qwen3.6-27B", num_speculative_tokens=2)
print(sglang.command)
```

### Bug Detection

```python
from qwen3_6_mtp import check_turboquant_conflict, check_prefix_cache_degradation

bug = check_turboquant_conflict(enable_turboquant=True, num_spec_tokens=2)
if bug:
    print(f"BLOCKED: {bug.title} ({bug.upstream_issue})")
```

## Key Findings

| Finding | Detail |
|---------|--------|
| MTP decode speedup | +27.5% faster decode TPOT at k=1 on RTX 3090 (with `--no-enable-prefix-caching`) |
| Prefix cache degradation | L457 bug drops hit rate ~92% to ~71% when MTP is enabled (vLLM #38182, OPEN) |
| TurboQuant conflict | TQ + MTP = degenerate token loops (vLLM #40831, CLOSED) |
| Crossover point | MTP throughput gain shrinks with batch size; net-negative varies by spec tokens and prefix cache (see `quick_crossover()`) |
| Sampling independence | MTP is algorithmically lossless; does not constrain sampling parameters |

## Published Results

Pre-computed crossover analysis and benchmark sweep data live in [`results/`](results/):

- **[`crossover_summary.csv`](results/crossover_summary.csv)** -- for each GPU and speculative token count: the batch size where MTP becomes net-negative and the peak throughput gain
- **[`benchmark_sweep.csv`](results/benchmark_sweep.csv)** -- full matrix of latency, throughput, acceptance rate, and KV cache utilization across all GPUs, batch sizes (1-64), spec tokens (0-5), and prefix cache on/off

Regenerate with `python results/generate_crossover.py`.

### Key crossover findings (Qwen3.6-27B, no prefix cache)

| Spec tokens | Crossover batch size | Peak gain |
|-------------|---------------------|-----------|
| MTP-1 | no crossover (always positive) | +24% |
| MTP-2 | no crossover (always positive) | +39% |
| MTP-3 | no crossover (always positive) | +42% |
| MTP-4 | 64 | +36% |
| MTP-5 | 64 | +24% |

MTP-1 through MTP-3 remain net-positive across all batch sizes up to 64. MTP-4 and MTP-5 flip net-negative at batch size 64 due to KV cache pressure from draft token overhead. For most single-user and small-batch serving, MTP-2 or MTP-3 gives the best throughput lift.

## Supported Models

| Model | Architecture | MTP Layers | Context |
|-------|-------------|------------|---------|
| Qwen3.6-27B | Dense (GDN + Gated Attention) | 1 | 262K |
| Qwen3.6-35B-A3B | MoE (GDN + Gated Attention) | 1 | 262K |

## License

Apache 2.0
