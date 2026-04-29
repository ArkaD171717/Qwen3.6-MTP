# qwen-mtp

MTP speculative decoding tuner for Qwen3.6. Generates vLLM/SGLang configs, finds throughput crossover points, and catches known bugs.

## What It Does

- **Auto-tuner**: Recommends MTP configuration (or explains why to disable it) based on use case, objective, and GPU
- **Backend configs**: Generates vLLM (`method: mtp`) and SGLang (`NEXTN` algorithm) serve commands
- **Crossover analysis**: Finds the batch size where MTP flips from net-positive to net-negative throughput
- **Bug detection**: Detects and blocks known-broken configurations (TurboQuant + MTP, prefix cache degradation)
- **Benchmark sweep**: Generate latency/throughput matrices across batch size, speculative tokens, and prefix cache settings

## Installation

```bash
pip install qwen-mtp
```

## Quick Start

```python
from qwen_mtp import recommend, UseCase, Objective

rec = recommend(
    use_case=UseCase.SINGLE_USER,
    objective=Objective.MINIMIZE_LATENCY,
    gpu_id="rtx-4090",
)

print(rec.enable)           # True
print(rec.expected_gain)    # ~35-42% latency reduction
print(rec.vllm_command)     # Full vllm serve command with MTP flags
print(rec.sglang_command)   # Equivalent SGLang command
```

### Crossover Analysis

```python
from qwen_mtp import quick_crossover

for s in quick_crossover(gpu_id="rtx-3090"):
    print(f"MTP-{s.spec_tokens}: crossover at batch {s.crossover_batch_size}, "
          f"best gain +{s.max_positive_delta_pct}%")
```

### Backend Config Generation

```python
from qwen_mtp import vllm_mtp_command, sglang_mtp_command

vllm = vllm_mtp_command(model="Qwen/Qwen3.6-27B", num_speculative_tokens=2)
print(vllm.command)

sglang = sglang_mtp_command(model="Qwen/Qwen3.6-27B", num_speculative_tokens=2)
print(sglang.command)
```

### Bug Detection

```python
from qwen_mtp import check_turboquant_conflict, check_prefix_cache_degradation

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
| Crossover point | MTP becomes net-negative at batch size 4-8 on consumer GPUs |
| Sampling independence | MTP is algorithmically lossless; does not constrain sampling parameters |

## Supported Models

| Model | Architecture | MTP Layers | Context |
|-------|-------------|------------|---------|
| Qwen3.6-27B | Dense (GDN + Gated Attention) | 1 | 262K |
| Qwen3.6-35B-A3B | MoE (GDN + Gated Attention) | 1 | 262K |

## License

Apache 2.0
