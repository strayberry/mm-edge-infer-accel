from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

import torch

from mm_edge_infer_accel.pi05_runtime import (
    _raw_batch_from_libero_item,
    action_vector_metrics,
    cuda_snapshot,
    load_policy_and_processors,
    set_pi05_cache_env,
    time_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Pi0.5 num_inference_steps with one policy load.")
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--dataset-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--steps", default="3,5,8,10")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=("reset", "queue"), default="reset")
    parser.add_argument("--output", default="outputs/pi05_inference_steps_sweep_ep0_20.json")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def set_num_inference_steps(policy, steps: int) -> None:
    policy.config.num_inference_steps = steps
    if hasattr(policy, "model") and hasattr(policy.model, "config"):
        policy.model.config.num_inference_steps = steps


def run_warmup(policy, preprocessor, dataset, warmup: int, mode: str, device: str) -> None:
    for idx in range(min(warmup, len(dataset))):
        batch = preprocessor(_raw_batch_from_libero_item(dataset[idx]))
        if mode == "reset":
            policy.reset()
        with torch.inference_mode():
            _ = policy.select_action(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    policy.reset()


def run_one(policy, preprocessor, postprocessor, dataset, sample_count: int, mode: str, device: str) -> dict:
    samples = []
    actual_sample_count = min(sample_count, len(dataset))
    started = time.perf_counter()
    for idx in range(actual_sample_count):
        raw = dataset[idx]
        batch = preprocessor(_raw_batch_from_libero_item(raw))

        queue_len_before = len(getattr(policy, "_action_queue", []))
        if mode == "reset":
            policy.reset()
            queue_len_before = 0

        if device == "cuda":
            torch.cuda.synchronize()
        action_started = time.perf_counter()
        with torch.inference_mode():
            raw_action = policy.select_action(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        action_seconds = time.perf_counter() - action_started

        final_action = postprocessor(raw_action)
        metrics = action_vector_metrics(raw["action"], final_action.squeeze(0))
        samples.append(
            {
                "frame_index": int(raw["frame_index"].item()),
                "action_seconds": round(action_seconds, 6),
                "queue_len_before": queue_len_before,
                "queue_len_after": len(getattr(policy, "_action_queue", [])),
                "action_metrics": metrics,
            }
        )

    elapsed = time.perf_counter() - started
    action_times = [sample["action_seconds"] for sample in samples]
    mae_values = [
        sample["action_metrics"]["mae"] for sample in samples if "mae" in sample["action_metrics"]
    ]
    return {
        "sample_count": actual_sample_count,
        "measured_loop_seconds": round(elapsed, 4),
        "loop_hz": round(actual_sample_count / elapsed, 4) if elapsed else None,
        "action_time_summary": time_summary(action_times),
        "action_mae_mean": round(fmean(mae_values), 6) if mae_values else None,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    set_pi05_cache_env()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = parse_ints(args.episodes)
    steps_values = parse_ints(args.steps)

    print(f"[{time.strftime('%H:%M:%S')}] loading policy {args.model_id}", flush=True)
    policy, preprocessor, postprocessor, config, load_seconds = load_policy_and_processors(args.model_id)
    print(f"[{time.strftime('%H:%M:%S')}] policy loaded in {load_seconds:.2f}s; {cuda_snapshot()}")

    datasets = {}
    for episode in episodes:
        print(f"[{time.strftime('%H:%M:%S')}] loading dataset {args.dataset_id} episode={episode}")
        datasets[episode] = LeRobotDataset(repo_id=args.dataset_id, episodes=[episode])

    runs = []
    for steps in steps_values:
        set_num_inference_steps(policy, steps)
        for episode in episodes:
            dataset = datasets[episode]
            run_warmup(policy, preprocessor, dataset, args.warmup, args.mode, config.device)
            print(
                f"[{time.strftime('%H:%M:%S')}] running steps={steps} "
                f"episode={episode} samples={args.sample_count} mode={args.mode}",
                flush=True,
            )
            result = run_one(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                sample_count=args.sample_count,
                mode=args.mode,
                device=config.device,
            )
            result.update({"num_inference_steps": steps, "episode": episode})
            runs.append(result)
            summary = result["action_time_summary"]
            print(
                f"[{time.strftime('%H:%M:%S')}] steps={steps} episode={episode}: "
                f"action_mean={summary.get('mean')}s loop_hz={result['loop_hz']} "
                f"mae={result['action_mae_mean']}",
                flush=True,
            )

    output = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "episodes": episodes,
        "steps": steps_values,
        "mode": args.mode,
        "warmup": args.warmup,
        "sample_count": args.sample_count,
        "load_seconds": round(load_seconds, 4),
        "inference_optimizations": getattr(config, "mm_edge_inference_optimizations", {}),
        "runs": runs,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
