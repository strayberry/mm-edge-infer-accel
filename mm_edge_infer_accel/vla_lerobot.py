from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig, config_to_dict, model_load_path
from .env import collect_environment
from .pi05_runtime import load_policy_only, run_libero_action_inference
from .profiling import gpu_memory_snapshot_mb, nvtx_range


def _import_pi05_policy():
    try:
        from lerobot.policies.pi05 import PI05Policy
    except ImportError as exc:
        raise RuntimeError(
            "Pi0.5 load-only check requires LeRobot with Pi0.5 support. Use the isolated "
            "data-disk env at /root/autodl-tmp/envs/pi05 or install LeRobot main "
            "with pi extras in a Python 3.12 environment."
        ) from exc
    return PI05Policy


def _aggregate_metrics(
    load_seconds: float,
    memory_before_load: dict,
    memory_after_load: dict,
    sample_count: int,
) -> dict:
    model_load_memory_delta = (
        round(
            memory_after_load["gpu_used_memory_mb"] - memory_before_load["gpu_used_memory_mb"],
            2,
        )
        if memory_after_load["gpu_used_memory_mb"] is not None
        and memory_before_load["gpu_used_memory_mb"] is not None
        else None
    )
    return {
        "load_seconds": round(load_seconds, 4),
        "sample_count": sample_count,
        "success_count": 0,
        "failed_count": 0,
        "failure_rate": 0.0,
        "generated_tokens": 0,
        "tokens_per_second": None,
        "action_metric_sample_count": 0,
        "gpu_total_memory_mb": memory_after_load["gpu_total_memory_mb"],
        "gpu_memory_before_load_mb": memory_before_load["gpu_used_memory_mb"],
        "gpu_memory_after_load_mb": memory_after_load["gpu_used_memory_mb"],
        "model_load_memory_delta_mb": model_load_memory_delta,
    }


def run_benchmark(cfg: ExperimentConfig, output: Optional[str] = None) -> dict:
    if cfg.runtime.backend != "lerobot":
        raise ValueError("Pi0.5 benchmark requires runtime.backend=lerobot")
    if cfg.eval.dataset != "pi05_load_only":
        return run_libero_action_inference(
            model_id=model_load_path(cfg),
            dataset_id=cfg.eval.dataset,
            episodes=cfg.eval.episodes,
            sample_count=cfg.eval.sample_count,
            mode=cfg.eval.mode,
            warmup=cfg.profile.warmup,
            output=output or str(Path(cfg.eval.output_dir) / f"{cfg.name}.json"),
        )

    load_path = model_load_path(cfg)
    memory_before_load = gpu_memory_snapshot_mb()
    started = time.perf_counter()
    with nvtx_range("vla_pi05_lerobot_load_model"):
        _import_pi05_policy()
        policy, device = load_policy_only(load_path, cfg.runtime.device)
    load_seconds = time.perf_counter() - started
    memory_after_load = gpu_memory_snapshot_mb()

    result = {
        "name": cfg.name,
        "model_type": "vla",
        "backend": "lerobot",
        "measurement": "pi05_lerobot_load_only",
        "dataset": cfg.eval.dataset,
        "model_id": cfg.model.model_id,
        "model_path": cfg.model.model_path,
        "load_path": load_path,
        "dtype": cfg.model.dtype,
        "device": device,
        "policy_class": type(policy).__name__,
        "quant": cfg.quant.__dict__,
        "config": config_to_dict(cfg),
        "system_info": collect_environment(),
        "metrics": _aggregate_metrics(load_seconds, memory_before_load, memory_after_load, 0),
        "samples": [],
        "notes": [
            "This check verifies Pi0.5 policy loading only.",
            "Action inference is available through scripts/run_pi05_action_inference.py.",
        ],
    }

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
