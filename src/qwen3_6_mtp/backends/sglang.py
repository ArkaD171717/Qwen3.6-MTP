"""SGLang MTP speculative decoding configuration."""

from typing import List, Optional

from ..types import Backend, BackendConfig


def sglang_mtp_command(
    model: str = "Qwen/Qwen3.6-27B",
    num_speculative_tokens: int = 2,
    enable_prefix_caching: bool = False,
    tensor_parallel: int = 1,
    context_length: int = 262144,
    mem_fraction_static: float = 0.8,
    port: int = 8000,
    extra_flags: Optional[List[str]] = None,
) -> BackendConfig:
    """Generate an SGLang launch command with NEXTN speculative decoding.

    SGLang uses different algorithm names and parameter semantics than vLLM:
    - Algorithm: NEXTN (vs vLLM's "mtp" method)
    - speculative-num-steps = num_speculative_tokens + 1
    - speculative-eagle-topk = 1 (recommended for Qwen3.6)
    - speculative-num-draft-tokens = num_speculative_tokens + 2
    """
    flags = [
        "python -m sglang.launch_server",
        f"  --model-path {model}",
        f"  --port {port}",
    ]

    if tensor_parallel > 1:
        flags.append(f"  --tp-size {tensor_parallel}")

    flags.append(f"  --mem-fraction-static {mem_fraction_static}")
    flags.append(f"  --context-length {context_length}")

    if num_speculative_tokens > 0:
        num_steps = num_speculative_tokens + 1
        num_draft = num_speculative_tokens + 2
        flags.extend(
            [
                "  --speculative-algo NEXTN",
                f"  --speculative-num-steps {num_steps}",
                "  --speculative-eagle-topk 1",
                f"  --speculative-num-draft-tokens {num_draft}",
            ]
        )

    if enable_prefix_caching:
        flags.append("  --enable-prefix-caching")

    flags.append("  --reasoning-parser qwen3")

    if extra_flags:
        flags.extend(f"  {f}" for f in extra_flags)

    caveats = [
        "SGLang uses NEXTN algorithm (different from vLLM's mtp method).",
        f"speculative-num-steps = {num_speculative_tokens + 1} (num_spec_tokens + 1).",
        "SGLang does not support Apple Silicon natively.",
    ]

    return BackendConfig(
        backend=Backend.SGLANG,
        command=" \\\n".join(flags),
        flags=flags,
        description=(
            f"SGLang with NEXTN algorithm, {num_speculative_tokens + 1} steps"
            if num_speculative_tokens > 0
            else "SGLang without speculative decoding"
        ),
        caveats=caveats,
    )
