from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MODEL_FAMILY_TO_TYPE = {
    "qwen3-vl": "vlm",
    "smolvlm2": "vlm",
    "pi0_fast": "vla",
    "pi05": "vla",
}

MODEL_FAMILY_TO_BACKENDS = {
    "qwen3-vl": {"vllm"},
    "smolvlm2": {"vllm"},
    "pi0_fast": {"lerobot"},
    "pi05": {"lerobot"},
}


@dataclass
class ModelConfig:
    model_id: str
    family: str
    model_path: str | None = None
    dtype: str = "float16"
    max_new_tokens: int = 128
    max_pixels: int | None = None
    trust_remote_code: bool = True


@dataclass
class QuantConfig:
    method: str = "none"
    bits: int | None = None
    group_size: int | None = None
    calibration: str | None = None
    keep_fp16_modules: list[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    backend: str = "vllm"
    batch_size: int = 1
    concurrency: int = 1
    device: str = "cuda"
    visual_keep_ratio: float = 1.0
    kv_cache_dtype: str = "fp16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    disable_flashinfer_sampler: bool = True
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    speculative_config: dict[str, Any] | None = None
    enable_prefix_kv_cache: bool = True


@dataclass
class EvalConfig:
    dataset: str = "self_test"
    sample_count: int = 10
    sample_strategy: str = "first"
    output_dir: str = "outputs"
    episodes: list[int] = field(default_factory=lambda: [0])
    mode: str = "reset"


@dataclass
class ProfileConfig:
    warmup: int = 3
    repeats: int = 10
    output_dir: str = "profiling"
    measure_ttft: bool = False


@dataclass
class ExperimentConfig:
    name: str
    model: ModelConfig
    quant: QuantConfig = field(default_factory=QuantConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    root = _require_mapping(data, "root")
    model = ModelConfig(**_require_mapping(root.get("model", {}), "model"))
    quant = QuantConfig(**_require_mapping(root.get("quant", {}), "quant"))
    runtime = RuntimeConfig(**_require_mapping(root.get("runtime", {}), "runtime"))
    eval_cfg = EvalConfig(**_require_mapping(root.get("eval", {}), "eval"))
    profile = ProfileConfig(**_require_mapping(root.get("profile", {}), "profile"))
    cfg = ExperimentConfig(
        name=str(root.get("name", path.stem)),
        model=model,
        quant=quant,
        runtime=runtime,
        eval=eval_cfg,
        profile=profile,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if not cfg.model.model_id:
        raise ValueError("model.model_id is required")
    if cfg.runtime.batch_size < 1:
        raise ValueError("runtime.batch_size must be >= 1")
    if cfg.runtime.concurrency < 1:
        raise ValueError("runtime.concurrency must be >= 1")
    if cfg.eval.sample_count < 0:
        raise ValueError("eval.sample_count must be >= 0")
    episodes = cfg.eval.episodes
    if isinstance(episodes, int):
        episodes = [episodes]
        cfg.eval.episodes = episodes
    if not episodes:
        raise ValueError("eval.episodes must not be empty")
    if any(e < 0 for e in episodes):
        raise ValueError("eval.episodes must all be >= 0")
    if cfg.model.max_new_tokens < 0:
        raise ValueError("model.max_new_tokens must be >= 0")
    if cfg.model.max_pixels is not None and cfg.model.max_pixels < 1:
        raise ValueError("model.max_pixels must be >= 1")
    if not 0 < cfg.runtime.visual_keep_ratio <= 1:
        raise ValueError("runtime.visual_keep_ratio must be in (0, 1]")
    if cfg.quant.bits is not None and cfg.quant.bits not in {4, 8, 16}:
        raise ValueError("quant.bits must be one of 4, 8, 16")
    if cfg.eval.sample_strategy not in {"first", "stratified"}:
        raise ValueError("eval.sample_strategy must be one of: first, stratified")
    if cfg.eval.mode not in {"reset", "queue"}:
        raise ValueError("eval.mode must be one of: reset, queue")
    if cfg.runtime.backend not in {"vllm", "pytorch", "lerobot", "transformers"}:
        raise ValueError("runtime.backend must be one of: vllm, pytorch, lerobot, transformers")
    family = cfg.model.family.lower()
    supported_backends = MODEL_FAMILY_TO_BACKENDS.get(family)
    if supported_backends and cfg.runtime.backend not in supported_backends:
        known = ", ".join(sorted(supported_backends))
        raise ValueError(f"{cfg.model.family} configs must use runtime.backend in: {known}")
    if cfg.runtime.backend == "vllm" and cfg.runtime.batch_size != 1:
        raise ValueError("vLLM OCRBench benchmark currently expects runtime.batch_size=1")
    if cfg.runtime.backend == "vllm" and cfg.quant.method not in {
        "none",
        "awq",
        "gptq",
        "bitsandbytes",
    }:
        raise ValueError(
            "vLLM benchmark supports quant.method in: none, awq, gptq, bitsandbytes"
        )
    if cfg.runtime.max_model_len < 1:
        raise ValueError("runtime.max_model_len must be >= 1")
    if not 0 < cfg.runtime.gpu_memory_utilization <= 1:
        raise ValueError("runtime.gpu_memory_utilization must be in (0, 1]")
    if cfg.runtime.speculative_config is not None:
        if not isinstance(cfg.runtime.speculative_config, dict):
            raise ValueError("runtime.speculative_config must be a mapping")
        num_speculative_tokens = cfg.runtime.speculative_config.get("num_speculative_tokens")
        if num_speculative_tokens is not None and num_speculative_tokens < 1:
            raise ValueError("runtime.speculative_config.num_speculative_tokens must be >= 1")


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "model": cfg.model.__dict__,
        "quant": cfg.quant.__dict__,
        "runtime": cfg.runtime.__dict__,
        "eval": cfg.eval.__dict__,
        "profile": cfg.profile.__dict__,
    }


def model_type_from_config(cfg: ExperimentConfig) -> str:
    family = cfg.model.family.lower()
    if family not in MODEL_FAMILY_TO_TYPE:
        known = ", ".join(sorted(MODEL_FAMILY_TO_TYPE))
        raise ValueError(f"Unknown model family '{cfg.model.family}'. Known families: {known}")
    return MODEL_FAMILY_TO_TYPE[family]


def model_load_path(cfg: ExperimentConfig) -> str:
    return cfg.model.model_path or cfg.model.model_id
