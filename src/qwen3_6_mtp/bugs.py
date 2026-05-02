"""Known vLLM bugs affecting MTP speculative decoding on Qwen3.6."""

from typing import List, Optional

from .types import BugReport, BugSeverity, BugStatus

BUG_REPORTS: List[BugReport] = [
    BugReport(
        id="L457",
        title="Prefix cache hit rate drops ~92% to ~71% when MTP is enabled",
        severity=BugSeverity.HIGH,
        status=BugStatus.OPEN,
        upstream_issue="vllm-project/vllm#38182",
        file="vllm/v1/core/single_type_kv_cache_manager.py",
        line=457,
        description=(
            "find_longest_cache_hit() force-drops the last matched block "
            "when MTP/EAGLE is enabled, to recompute hidden states for the "
            "draft head. Combined with large block sizes on high-VRAM GPUs "
            "(H20 at 96GB), the impact is a ~21 percentage-point cache hit "
            "rate drop. Tested on Qwen3.5-35B-A3B with 5500-token shared "
            "prefix, 200-token random input."
        ),
        root_cause=(
            "The full attention manager finds N cached blocks then applies "
            "the EAGLE/MTP drop returning N-1 blocks. This sets max_length = "
            "(N-1) * block_size, making block N-1 unreachable. The drop is "
            "unconditional -- it penalizes the target model's cache even "
            "though only the draft model needs recomputed hidden states."
        ),
        workaround=(
            "Use --no-enable-prefix-caching when MTP is enabled. This "
            "eliminates the cache hit degradation but removes prefix caching. "
            "Alternatively, accept the ~21% cache degradation."
        ),
        affected_versions=["v0.6.x", "v0.7.x", "v0.8.x", "main"],
        reported_date="2026-03-26",
        last_updated="2026-04-25",
    ),
    BugReport(
        id="TQ-40831",
        title="TurboQuant KV + speculative decoding produces degenerate token loops",
        severity=BugSeverity.CRITICAL,
        status=BugStatus.CLOSED,
        upstream_issue="vllm-project/vllm#40831",
        file="vllm/attention/backends/flash_attn.py",
        line=None,
        description=(
            "TurboQuant KV cache quantization combined with MTP or ngram "
            "speculative decoding produces corrupted output: tool-call loops, "
            "first-token repetition at long contexts, streaming stutter. "
            "Short code completions appear unaffected, masking the bug in "
            "initial benchmarks."
        ),
        root_cause=(
            "The MTP draft head runs its own attention over the live KV "
            "cache. When KV entries are stored through TurboQuant (rotate -> "
            "quantize -> pack), the draft head's attention distribution "
            "collapses on rare-token / structured-token contexts."
        ),
        workaround=(
            "Do not enable TurboQuant KV (--kv-cache-dtype turboquant_*) "
            "together with speculative decoding. Use one or the other."
        ),
        affected_versions=["v0.7.4+", "v0.8.x"],
        reported_date="2026-04-21",
        last_updated="2026-04-25",
    ),
    BugReport(
        id="TQ-40880",
        title="MTP + TurboQuant + CUDA graph capture degenerate output on hybrid models",
        severity=BugSeverity.CRITICAL,
        status=BugStatus.CLOSED,
        upstream_issue="vllm-project/vllm#40880",
        file="vllm/worker/worker.py",
        line=None,
        description=(
            "CUDA graph replay captures stale TurboQuant state when "
            "speculative decoding rejects tokens between captures. The "
            "v7.13 ngram fix (PRs #40738, #36138, #40783, #39055) only "
            "covered gdn_attn.py and gdn_linear_attn.py; MTP runs through "
            "eagle.py with the Qwen3_5MTP model class, which was uncovered."
        ),
        root_cause=(
            "CUDA graphs capture the kernel launch sequence including "
            "TurboQuant KV lookup. When speculative decoding rejects tokens, "
            "the next graph replay uses stale block indices. Fixed via "
            "PR #40914."
        ),
        workaround=(
            "Fixed in PR #40914. For older versions: --compilation-config "
            '\'{"cudagraph_mode":"NONE"}\' disables CUDA graphs (~33 TPS '
            "down from ~85 TPS)."
        ),
        affected_versions=["v0.8.x"],
        reported_date="2026-04-25",
        last_updated="2026-04-25",
    ),
    BugReport(
        id="NGRAM-40875",
        title="ngram prompt_lookup_min=2 causes tool-call output corruption",
        severity=BugSeverity.HIGH,
        status=BugStatus.OPEN,
        upstream_issue="vllm-project/vllm#40875",
        file="vllm/v1/spec_decode/ngram_proposer.py",
        line=None,
        description=(
            "ngram speculative decoding with the default "
            "prompt_lookup_min=2 causes tool-call output corruption on "
            "Qwen3-class models. The output generates <tool_call> tags "
            "repeatedly without populating the tool_calls array."
        ),
        root_cause=(
            "Low prompt_lookup_min values cause the ngram proposer to "
            "match very short token sequences that happen to overlap with "
            "structured output tokens (tool calls, JSON delimiters)."
        ),
        workaround="Set prompt_lookup_min=8 in the speculative config.",
        affected_versions=["v0.8.x", "main"],
        reported_date="2026-04-24",
        last_updated="2026-04-25",
    ),
]

_BUG_BY_ID = {bug.id: bug for bug in BUG_REPORTS}


def get_open_bugs() -> List[BugReport]:
    return [b for b in BUG_REPORTS if b.status == BugStatus.OPEN]


def get_critical_bugs() -> List[BugReport]:
    return [b for b in BUG_REPORTS if b.severity == BugSeverity.CRITICAL]


def check_turboquant_conflict(
    enable_turboquant: bool, num_spec_tokens: int
) -> Optional[BugReport]:
    """Return the relevant bug report if TurboQuant + spec decode is attempted."""
    if enable_turboquant and num_spec_tokens > 0:
        return _BUG_BY_ID.get("TQ-40831")
    return None


def check_prefix_cache_degradation(
    enable_prefix_cache: bool, num_spec_tokens: int
) -> Optional[BugReport]:
    """Return the L457 bug report if prefix caching + MTP is enabled."""
    if enable_prefix_cache and num_spec_tokens > 0:
        return _BUG_BY_ID.get("L457")
    return None
