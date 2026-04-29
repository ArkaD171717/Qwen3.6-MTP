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

## Supported Models

| Model | Architecture | MTP Layers | Context |
|-------|-------------|------------|---------|
| Qwen3.6-27B | Dense (GDN + Gated Attention) | 1 | 262K |
| Qwen3.6-35B-A3B | MoE (GDN + Gated Attention) | 1 | 262K |

## License

Apache 2.0
