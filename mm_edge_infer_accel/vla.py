from __future__ import annotations

import math
from typing import Iterable

from .config import ExperimentConfig, config_to_dict


def benchmark_plan(cfg: ExperimentConfig) -> dict:
    if cfg.model.family == "pi05" and cfg.runtime.backend == "lerobot":
        kind = "pi05_lerobot_action_inference"
    else:
        kind = "action_prediction_benchmark"
    return {
        "name": cfg.name,
        "model_type": "vla",
        "kind": kind,
        "status": "planned",
        "config": config_to_dict(cfg),
        "metrics": [
            "action_latency",
            "generated_tokens",
            "tokens_per_second",
            "request_latency_ms_mean",
            "vllm_first_token_latency_ms_mean",
            "vllm_prefill_latency_ms_mean",
            "vllm_decode_latency_ms_mean",
            "vllm_inference_latency_ms_mean",
            "action_mae",
            "action_cosine",
            "gpu_memory_after_load_mb",
        ],
    }


def run_benchmark(cfg: ExperimentConfig, output: str | None = None) -> dict:
    if cfg.model.family == "pi05" and cfg.runtime.backend == "lerobot":
        from . import vla_lerobot

        return vla_lerobot.run_benchmark(cfg, output=output)
    raise NotImplementedError(
        f"VLA benchmark is not implemented for family={cfg.model.family}, "
        f"backend={cfg.runtime.backend}"
    )


def mean_absolute_error(a: Iterable[float], b: Iterable[float]) -> float:
    left = list(a)
    right = list(b)
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(abs(x - y) for x, y in zip(left, right)) / len(left) if left else 0.0


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    left = list(a)
    right = list(b)
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(left, right))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(y * y for y in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def action_report(reference: Iterable[float], candidate: Iterable[float]) -> dict:
    ref = list(reference)
    pred = list(candidate)
    return {
        "dimensions": len(ref),
        "mae": mean_absolute_error(ref, pred),
        "cosine_similarity": cosine_similarity(ref, pred),
    }
