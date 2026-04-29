"""vLLM MTP speculative decoding configuration."""

import json
from typing import List, Optional

from ..types import Backend, BackendConfig


def vllm_mtp_command(
    model: str = "Qwen/Qwen3.6-27B",
    num_speculative_tokens: int = 2,
    enable_prefix_caching: bool = False,
    tensor_parallel: int = 1,
    max_model_len: int = 262144,
    gpu_memory_utilization: float = 0.9,
    port: int = 8000,
    extra_flags: Optional[List[str]] = None,
) -> BackendConfig:
    """Generate a vLLM serve command with MTP speculative decoding.

    Qwen3.6 models use method "mtp" (not "qwen3_next_mtp", which is
    for Qwen3-Next). The MTP head has mtp_num_hidden_layers=1 and
    shares embeddings with the main model.
    """
    spec_config = json.dumps(
        {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
    )

    flags = [
        f"vllm serve {model}",
        f"  --port {port}",
    ]

    if tensor_parallel > 1:
        flags.append(f"  --tensor-parallel-size {tensor_parallel}")

    flags.append(f"  --max-model-len {max_model_len}")
    flags.append(f"  --gpu-memory-utilization {gpu_memory_utilization}")

    if num_speculative_tokens > 0:
        flags.append(f"  --speculative-config '{spec_config}'")

    if enable_prefix_caching:
        flags.append("  --enable-prefix-caching")
    else:
        flags.append("  --no-enable-prefix-caching")

    flags.append("  --reasoning-parser qwen3")

    if extra_flags:
        flags.extend(f"  {f}" for f in extra_flags)

    caveats = []
    if num_speculative_tokens > 0 and enable_prefix_caching:
        caveats.append(
            "L457 BUG: prefix cache hit rate degrades ~21% when MTP is "
            "enabled (vLLM #38182). Consider --no-enable-prefix-caching."
        )
    if num_speculative_tokens > 0:
        caveats.append(
            f"num_speculative_tokens range: 1-5. Current: {num_speculative_tokens}. "
            "Higher values reduce latency but lower acceptance rate."
        )

    return BackendConfig(
        backend=Backend.VLLM,
        command=" \\\n".join(flags),
        flags=flags,
        description=(
            f"vLLM with mtp method, {num_speculative_tokens} speculative tokens"
            if num_speculative_tokens > 0
            else "vLLM without speculative decoding"
        ),
        caveats=caveats,
    )
