from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .profiling import nvtx_range


def set_pi05_cache_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pi05_inference_optimizations() -> dict:
    import torch

    tf32_enabled = _env_flag("MM_EDGE_PI05_TF32", True)
    num_inference_steps = os.environ.get("MM_EDGE_PI05_NUM_INFERENCE_STEPS")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
    if tf32_enabled:
        torch.set_float32_matmul_precision("high")

    return {
        "tf32": tf32_enabled,
        "compile_model": _env_flag("MM_EDGE_PI05_COMPILE", False),
        "compile_mode": os.environ.get("MM_EDGE_PI05_COMPILE_MODE", "reduce-overhead"),
        "num_inference_steps": int(num_inference_steps) if num_inference_steps else None,
        "patch_sample_actions": True,
    }


def cuda_snapshot() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }


def action_vector_metrics(reference, candidate) -> dict:
    import torch

    ref = reference.detach().float().cpu().flatten()
    pred = candidate.detach().float().cpu().flatten()
    if ref.numel() != pred.numel():
        return {
            "error": f"shape mismatch: reference={list(ref.shape)} candidate={list(pred.shape)}"
        }
    mae = torch.mean(torch.abs(ref - pred)).item()
    denom = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(pred)
    cosine = (torch.dot(ref, pred) / denom).item() if denom.item() else 0.0
    return {"mae": round(mae, 6), "cosine_similarity": round(cosine, 6)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def time_summary(values: list[float]) -> dict:
    if not values:
        return {}
    mean_value = fmean(values)
    return {
        "mean": round(mean_value, 6),
        "p50": round(median(values), 6),
        "p90": round(percentile(values, 0.90), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "hz_mean_inverse": round(1.0 / mean_value, 4) if mean_value else None,
    }


def make_synthetic_batch():
    import torch

    image = torch.zeros(3, 256, 256, dtype=torch.float32)
    image[0, 64:160, 64:160] = 1.0
    image2 = torch.zeros(3, 256, 256, dtype=torch.float32)
    image2[2, 96:192, 96:192] = 1.0
    return {
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.state": torch.zeros(8, dtype=torch.float32),
        "task": "pick up the object and place it in the target zone",
    }


def load_policy_and_processors(model_id: str, enable_prefix_kv_cache: bool = True):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    import lerobot.policies.pi05.processor_pi05  # noqa: F401 - registers Pi0.5 processor step
    import lerobot.processor as lerobot_processor

    PolicyProcessorPipeline = lerobot_processor.PolicyProcessorPipeline
    batch_to_transition = lerobot_processor.batch_to_transition
    transition_to_batch = lerobot_processor.transition_to_batch
    policy_action_to_transition = getattr(
        lerobot_processor, "policy_action_to_transition", batch_to_transition
    )
    transition_to_policy_action = getattr(
        lerobot_processor, "transition_to_policy_action", transition_to_batch
    )

    optimizations = pi05_inference_optimizations()
    with nvtx_range("pi05_load_config_processors"):
        config = PreTrainedConfig.from_pretrained(model_id, local_files_only=True)
        config.compile_model = optimizations["compile_model"]
        config.compile_mode = optimizations["compile_mode"]
        if optimizations["num_inference_steps"] is not None:
            config.num_inference_steps = optimizations["num_inference_steps"]
        config.gradient_checkpointing = False
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    started = time.perf_counter()
    with nvtx_range("pi05_load_policy"):
        policy = PI05Policy.from_pretrained(model_id, config=config, local_files_only=True).eval()
    if enable_prefix_kv_cache:
        from .pi05_optimizations import apply_pi05_optimizations

        patch_result = apply_pi05_optimizations(policy, enabled=True)
        optimizations["patch_result"] = patch_result.as_dict()
    config.mm_edge_inference_optimizations = optimizations
    return policy, preprocessor, postprocessor, config, time.perf_counter() - started


def load_policy_only(model_id: str, device_name: str):
    import torch
    from lerobot.policies.pi05 import PI05Policy

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    policy = PI05Policy.from_pretrained(model_id)
    return policy.to(device).eval(), str(device)


def _raw_batch_from_libero_item(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.image": raw["observation.images.image"],
        "observation.images.image2": raw["observation.images.image2"],
        "observation.state": raw["observation.state"],
        "task": raw["task"],
    }


def run_synthetic_action(model_id: str, output: str, enable_prefix_kv_cache: bool = True) -> dict:
    set_pi05_cache_env()
    log(f"loading policy {model_id}")
    policy, preprocessor, postprocessor, config, load_seconds = load_policy_and_processors(
        model_id, enable_prefix_kv_cache=enable_prefix_kv_cache
    )
    log(f"policy loaded in {load_seconds:.2f}s; {cuda_snapshot()}")

    preprocess_started = time.perf_counter()
    batch = preprocessor(make_synthetic_batch())
    preprocess_seconds = time.perf_counter() - preprocess_started
    batch_shapes = {
        key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")
    }

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    action_started = time.perf_counter()
    action = policy.select_action(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    action_seconds = time.perf_counter() - action_started

    postprocess_started = time.perf_counter()
    final_action = postprocessor(action)
    postprocess_seconds = time.perf_counter() - postprocess_started

    result = {
        "model_id": model_id,
        "source": "synthetic",
        "policy_class": type(policy).__name__,
        "load_seconds": round(load_seconds, 4),
        "preprocess_seconds": round(preprocess_seconds, 4),
        "action_seconds": round(action_seconds, 4),
        "postprocess_seconds": round(postprocess_seconds, 4),
        "raw_action_shape": list(action.shape),
        "final_action_shape": list(final_action.shape),
        "raw_action": action.detach().cpu().tolist(),
        "final_action": final_action.detach().cpu().tolist(),
        "batch_shapes": batch_shapes,
        "input_mode": "checkpoint_preprocessor",
        "cuda": cuda_snapshot(),
    }
    write_json(output, result)
    return result


def _warmup_policy(policy, preprocessor, dataset, mode: str, warmup: int, device: str) -> None:
    if warmup <= 0:
        return
    warmup_count = min(warmup, len(dataset))
    log(f"running warmup: {warmup_count} frames, mode={mode}")
    with nvtx_range("pi05_warmup"):
        import torch

        for idx in range(warmup_count):
            with nvtx_range("pi05_warmup_preprocess"):
                batch = preprocessor(_raw_batch_from_libero_item(dataset[idx]))
            if mode == "reset":
                with nvtx_range("pi05_warmup_reset"):
                    policy.reset()
            with nvtx_range("pi05_warmup_select_action"):
                with torch.inference_mode():
                    _ = policy.select_action(batch)
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    policy.reset()


def run_libero_action_inference(
    model_id: str,
    dataset_id: str,
    episodes: list[int],
    sample_count: int,
    mode: str,
    warmup: int,
    output: str,
    enable_prefix_kv_cache: bool = True,
) -> dict:
    set_pi05_cache_env()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    log(f"loading dataset {dataset_id} episodes={episodes}")
    dataset_started = time.perf_counter()
    with nvtx_range("pi05_load_dataset"):
        dataset = LeRobotDataset(repo_id=dataset_id, episodes=episodes)
    dataset_seconds = time.perf_counter() - dataset_started
    log(f"dataset ready: len={len(dataset)}")

    log(f"loading policy {model_id}")
    policy, preprocessor, postprocessor, config, load_seconds = load_policy_and_processors(
        model_id, enable_prefix_kv_cache=enable_prefix_kv_cache,
    )
    log(f"policy loaded in {load_seconds:.2f}s; {cuda_snapshot()}")
    _warmup_policy(policy, preprocessor, dataset, mode, warmup, config.device)

    samples = []
    total_count = min(sample_count * len(episodes), len(dataset))
    total_started = time.perf_counter()
    for idx in range(total_count):
        with nvtx_range("pi05_dataset_getitem"):
            raw = dataset[idx]
        preprocess_started = time.perf_counter()
        with nvtx_range("pi05_preprocess"):
            batch = preprocessor(_raw_batch_from_libero_item(raw))
        preprocess_seconds = time.perf_counter() - preprocess_started

        queue_len_before = len(getattr(policy, "_action_queue", []))
        if mode == "reset":
            with nvtx_range("pi05_policy_reset"):
                policy.reset()
            queue_len_before = 0
        if config.device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        action_started = time.perf_counter()
        with nvtx_range("pi05_select_action"):
            with torch.inference_mode():
                raw_action = policy.select_action(batch)
        if config.device == "cuda":
            torch.cuda.synchronize()
        action_seconds = time.perf_counter() - action_started
        queue_len_after = len(getattr(policy, "_action_queue", []))

        postprocess_started = time.perf_counter()
        with nvtx_range("pi05_postprocess"):
            final_action = postprocessor(raw_action)
        postprocess_seconds = time.perf_counter() - postprocess_started
        with nvtx_range("pi05_action_metrics"):
            metrics = action_vector_metrics(raw["action"], final_action.squeeze(0))

        item = {
            "dataset_index": int(raw["index"].item())
            if hasattr(raw["index"], "item")
            else int(raw["index"]),
            "episode_index": int(raw["episode_index"].item()),
            "frame_index": int(raw["frame_index"].item()),
            "task": raw["task"],
            "preprocess_seconds": round(preprocess_seconds, 4),
            "action_seconds": round(action_seconds, 4),
            "postprocess_seconds": round(postprocess_seconds, 4),
            "end_to_end_seconds": round(
                preprocess_seconds + action_seconds + postprocess_seconds, 4
            ),
            "queue_len_before": queue_len_before,
            "queue_len_after": queue_len_after,
            "reference_action": raw["action"].detach().cpu().tolist(),
            "raw_action": raw_action.detach().cpu().tolist(),
            "final_action": final_action.detach().cpu().tolist(),
            "raw_action_shape": list(raw_action.shape),
            "final_action_shape": list(final_action.shape),
            "action_metrics": metrics,
            "cuda": cuda_snapshot(),
        }
        samples.append(item)
        log(
            f"sample [{idx}]: frame={item['frame_index']} episode={item['episode_index']} "
            f"action={item['action_seconds']:.4f}s "
            f"queue={queue_len_before}->{queue_len_after} mae={metrics.get('mae')}"
        )

    total_seconds = time.perf_counter() - total_started
    total_count = len(samples)
    action_times = [s["action_seconds"] for s in samples]
    e2e_times = [s["end_to_end_seconds"] for s in samples]
    mae_values = [s["action_metrics"]["mae"] for s in samples if "mae" in s["action_metrics"]]
    chunk_predict_count = sum(1 for s in samples if s["queue_len_before"] == 0)
    result = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "episodes": episodes,
        "sample_count": total_count,
        "source": "libero",
        "mode": mode,
        "warmup": warmup,
        "dataset_seconds": round(dataset_seconds, 4),
        "load_seconds": round(load_seconds, 4),
        "measured_loop_seconds": round(total_seconds, 4),
        "action_seconds_mean": round(sum(action_times) / len(action_times), 4)
        if action_times
        else None,
        "action_time_summary": time_summary(action_times),
        "end_to_end_time_summary": time_summary(e2e_times),
        "loop_hz": round(total_count / total_seconds, 4) if total_seconds else None,
        "chunk_predict_count": chunk_predict_count,
        "chunk_predict_hz": round(chunk_predict_count / total_seconds, 4)
        if total_seconds
        else None,
        "action_mae_mean": round(sum(mae_values) / len(mae_values), 6) if mae_values else None,
        "policy_class": type(policy).__name__,
        "inference_optimizations": getattr(config, "mm_edge_inference_optimizations", {}),
        "features": dataset.features,
        "samples": samples,
    }
    write_json(output, result)
    return result


def write_json(output: str, payload: dict) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {path}")
