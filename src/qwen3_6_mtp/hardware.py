"""GPU hardware profiles and Qwen3.6 model configurations."""

from .types import (
    GpuProfile,
    GpuTier,
    ModelConfig,
)

GPU_PROFILES = [
    GpuProfile(
        "rtx-4090", "NVIDIA RTX 4090", 24, 1008, "82.6", GpuTier.CONSUMER, True, True
    ),
    GpuProfile(
        "rtx-4080", "NVIDIA RTX 4080", 16, 717, "48.7", GpuTier.CONSUMER, True, True
    ),
    GpuProfile(
        "rtx-3090", "NVIDIA RTX 3090", 24, 936, "35.6", GpuTier.CONSUMER, True, False
    ),
    GpuProfile(
        "rtx-3080", "NVIDIA RTX 3080", 10, 760, "29.8", GpuTier.CONSUMER, True, False
    ),
    GpuProfile(
        "rtx-4070ti",
        "NVIDIA RTX 4070 Ti",
        12,
        504,
        "40.1",
        GpuTier.CONSUMER,
        True,
        True,
    ),
    GpuProfile(
        "rtx-4060ti",
        "NVIDIA RTX 4060 Ti",
        16,
        288,
        "22.1",
        GpuTier.CONSUMER,
        True,
        True,
    ),
    GpuProfile(
        "a6000", "NVIDIA A6000", 48, 768, "38.9", GpuTier.PROFESSIONAL, True, True
    ),
    GpuProfile(
        "a5000", "NVIDIA A5000", 24, 768, "27.8", GpuTier.PROFESSIONAL, True, True
    ),
    GpuProfile(
        "h100-sxm", "NVIDIA H100 SXM", 80, 3350, "989", GpuTier.DATACENTER, True, True
    ),
    GpuProfile(
        "h100-pcie", "NVIDIA H100 PCIe", 80, 2039, "756", GpuTier.DATACENTER, True, True
    ),
    GpuProfile("h20", "NVIDIA H20", 96, 4000, "148", GpuTier.DATACENTER, True, True),
    GpuProfile(
        "a100-sxm", "NVIDIA A100 SXM", 80, 2039, "312", GpuTier.DATACENTER, True, True
    ),
    GpuProfile(
        "a100-pcie", "NVIDIA A100 PCIe", 80, 1555, "312", GpuTier.DATACENTER, True, True
    ),
    GpuProfile("l40s", "NVIDIA L40S", 48, 864, "91.6", GpuTier.DATACENTER, True, True),
    GpuProfile("t4", "NVIDIA T4", 16, 320, "65.1", GpuTier.DATACENTER, True, False),
    GpuProfile(
        "m4-ultra", "Apple M4 Ultra", 192, 819, "~55", GpuTier.CONSUMER, False, False
    ),
    GpuProfile(
        "m4-max", "Apple M4 Max", 128, 546, "~40", GpuTier.CONSUMER, False, False
    ),
]

GPU_BY_ID = {gpu.id: gpu for gpu in GPU_PROFILES}

MODEL_CONFIGS = {
    "Qwen3.6-35B-A3B": ModelConfig(
        name="Qwen3.6-35B-A3B",
        total_params="35B",
        active_params="3B",
        architecture="MoE (Gated Delta Networks + Gated Attention)",
        mtp_num_hidden_layers=1,
        context_length=262144,
        vram_bf16_gb=72,
        vram_fp8_gb=36,
        vram_int4_gb=18,
        mtp_overhead_gb=2.5,
    ),
    "Qwen3.6-27B": ModelConfig(
        name="Qwen3.6-27B",
        total_params="27B",
        active_params="27B",
        architecture="Dense (Gated Delta Networks + Gated Attention)",
        mtp_num_hidden_layers=1,
        context_length=262144,
        vram_bf16_gb=54,
        vram_fp8_gb=27,
        vram_int4_gb=14,
        mtp_overhead_gb=2.0,
    ),
}


def get_gpu(gpu_id: str) -> GpuProfile:
    if gpu_id not in GPU_BY_ID:
        raise ValueError(
            f"Unknown GPU '{gpu_id}'. Available: {', '.join(sorted(GPU_BY_ID))}"
        )
    return GPU_BY_ID[gpu_id]


def get_model(model_name: str) -> ModelConfig:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(sorted(MODEL_CONFIGS))}"
        )
    return MODEL_CONFIGS[model_name]


def vram_required(
    model_name: str, quantization: str, include_mtp: bool = True
) -> float:
    model = get_model(model_name)
    base = {
        "bf16": model.vram_bf16_gb,
        "fp8": model.vram_fp8_gb,
        "int4": model.vram_int4_gb,
    }
    if quantization not in base:
        raise ValueError(
            f"Unknown quantization '{quantization}'. Available: {', '.join(sorted(base))}"
        )
    vram = base[quantization]
    if include_mtp:
        vram += model.mtp_overhead_gb
    return vram
