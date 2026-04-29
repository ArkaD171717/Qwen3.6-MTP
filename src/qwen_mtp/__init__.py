"""Hardware-aware MTP speculative decoding auto-tuner for Qwen3.6:
vLLM/SGLang backend normalization, crossover analysis, and bug detection."""

from .backends import sglang_mtp_command, vllm_mtp_command
from .bench import generate_benchmark_data
from .bugs import (
    BUG_REPORTS,
    check_prefix_cache_degradation,
    check_turboquant_conflict,
    get_critical_bugs,
    get_open_bugs,
)
from .crossover import (
    CrossoverSummary,
    find_crossover_points,
    quick_crossover,
    summarize_crossovers,
)
from .hardware import (
    GPU_BY_ID,
    GPU_PROFILES,
    MODEL_CONFIGS,
    SAMPLING_PRESETS,
    get_gpu,
    get_model,
    vram_required,
)
from .tuner import recommend
from .types import (
    Backend,
    BackendConfig,
    BenchmarkPoint,
    BugReport,
    BugSeverity,
    BugStatus,
    CrossoverPoint,
    GpuProfile,
    GpuTier,
    ModelConfig,
    MtpRecommendation,
    Objective,
    Quantization,
    SamplingPreset,
    UseCase,
)

__version__ = "0.1.0"

__all__ = [
    "recommend",
    "vllm_mtp_command",
    "sglang_mtp_command",
    "generate_benchmark_data",
    "find_crossover_points",
    "summarize_crossovers",
    "quick_crossover",
    "CrossoverSummary",
    "check_turboquant_conflict",
    "check_prefix_cache_degradation",
    "get_open_bugs",
    "get_critical_bugs",
    "BUG_REPORTS",
    "GPU_PROFILES",
    "GPU_BY_ID",
    "MODEL_CONFIGS",
    "SAMPLING_PRESETS",
    "get_gpu",
    "get_model",
    "vram_required",
    "Backend",
    "BackendConfig",
    "BenchmarkPoint",
    "BugReport",
    "BugSeverity",
    "BugStatus",
    "CrossoverPoint",
    "GpuProfile",
    "GpuTier",
    "ModelConfig",
    "MtpRecommendation",
    "Objective",
    "Quantization",
    "SamplingPreset",
    "UseCase",
]
