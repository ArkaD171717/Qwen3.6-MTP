import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class GpuTier(str, enum.Enum):
    CONSUMER = "consumer"
    PROFESSIONAL = "professional"
    DATACENTER = "datacenter"


class Backend(str, enum.Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class Objective(str, enum.Enum):
    MINIMIZE_LATENCY = "minimize-latency"
    MAXIMIZE_THROUGHPUT = "maximize-throughput"
    BALANCED = "balanced"


class UseCase(str, enum.Enum):
    SINGLE_USER = "single-user"
    MULTI_USER = "multi-user"


class Quantization(str, enum.Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    INT4 = "int4"


class BugSeverity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class BugStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class GpuProfile:
    id: str
    name: str
    vram_gb: int
    memory_bandwidth_gbs: int
    compute_tflops: str
    tier: GpuTier
    tensor_cores: bool
    supports_bf16: bool


@dataclass(frozen=True)
class ModelConfig:
    name: str
    total_params: str
    active_params: str
    architecture: str
    mtp_num_hidden_layers: int
    context_length: int
    vram_bf16_gb: int
    vram_fp8_gb: int
    vram_int4_gb: int
    mtp_overhead_gb: float


@dataclass(frozen=True)
class SamplingPreset:
    """Recommended sampling parameters from the Qwen3.6 model cards."""

    temperature: float
    top_p: float
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0


@dataclass(frozen=True)
class BenchmarkPoint:
    batch_size: int
    num_spec_tokens: int
    prefix_cache: bool
    latency_ms: float
    throughput_tps: float
    acceptance_rate: float
    kv_cache_util_pct: int


@dataclass(frozen=True)
class CrossoverPoint:
    batch_size: int
    spec_tokens: int
    is_net_positive: bool
    delta_pct: float


@dataclass(frozen=True)
class BugReport:
    id: str
    title: str
    severity: BugSeverity
    status: BugStatus
    upstream_issue: str
    file: str
    line: Optional[int]
    description: str
    root_cause: str
    workaround: str
    affected_versions: List[str]
    reported_date: str
    last_updated: str


@dataclass(frozen=True)
class BackendConfig:
    backend: Backend
    command: str
    flags: List[str]
    description: str
    caveats: List[str]


@dataclass
class MtpRecommendation:
    enable: bool
    reason: str
    backend: Optional[Backend]
    num_speculative_tokens: int
    prefix_caching: bool
    expected_gain: str
    warnings: List[str] = field(default_factory=list)
    vllm_command: Optional[str] = None
    sglang_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable": self.enable,
            "reason": self.reason,
            "backend": self.backend.value if self.backend else None,
            "num_speculative_tokens": self.num_speculative_tokens,
            "prefix_caching": self.prefix_caching,
            "expected_gain": self.expected_gain,
            "warnings": self.warnings,
            "vllm_command": self.vllm_command,
            "sglang_command": self.sglang_command,
        }
