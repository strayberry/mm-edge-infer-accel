from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mm_edge_infer_accel.pi05_runtime import (
    _raw_batch_from_libero_item,
    action_vector_metrics,
    cuda_snapshot,
    load_policy_and_processors,
    set_pi05_cache_env,
    time_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Pi0.5 LIBERO inference with torch.profiler.")
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--dataset-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=("reset", "queue"), default="reset")
    parser.add_argument("--sort-by", default="cuda_time_total")
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace-output", default="profiling/pi05_torch_profile_reset3.json")
    parser.add_argument("--table-output", default="profiling/pi05_torch_profile_reset3_table.txt")
    parser.add_argument("--summary-output", default="outputs/pi05_torch_profile_reset3_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_pi05_cache_env()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[{time.strftime('%H:%M:%S')}] loading dataset {args.dataset_id} episode={args.episode}")
    dataset = LeRobotDataset(repo_id=args.dataset_id, episodes=[args.episode])
    print(f"[{time.strftime('%H:%M:%S')}] loading policy {args.model_id}")
    policy, preprocessor, postprocessor, config, load_seconds = load_policy_and_processors(
        args.model_id
    )
    print(f"[{time.strftime('%H:%M:%S')}] policy loaded in {load_seconds:.2f}s; {cuda_snapshot()}")

    for idx in range(min(args.warmup, len(dataset))):
        batch = preprocessor(_raw_batch_from_libero_item(dataset[idx]))
        if args.mode == "reset":
            policy.reset()
        with torch.inference_mode():
            _ = policy.select_action(batch)
    if config.device == "cuda":
        torch.cuda.synchronize()
    policy.reset()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    samples = []
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for idx in range(min(args.sample_count, len(dataset))):
            with record_function("pi05_dataset_getitem"):
                raw = dataset[idx]
            with record_function("pi05_preprocess"):
                batch = preprocessor(_raw_batch_from_libero_item(raw))

            queue_len_before = len(getattr(policy, "_action_queue", []))
            if args.mode == "reset":
                with record_function("pi05_policy_reset"):
                    policy.reset()
                queue_len_before = 0

            if config.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with record_function("pi05_select_action"):
                with torch.inference_mode():
                    raw_action = policy.select_action(batch)
            if config.device == "cuda":
                torch.cuda.synchronize()
            action_seconds = time.perf_counter() - started

            with record_function("pi05_postprocess"):
                final_action = postprocessor(raw_action)
            with record_function("pi05_action_metrics"):
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
            print(
                f"[{time.strftime('%H:%M:%S')}] sample {idx}: "
                f"action={action_seconds:.4f}s queue={samples[-1]['queue_len_before']}"
                f"->{samples[-1]['queue_len_after']} mae={metrics.get('mae')}"
            )

    trace_path = Path(args.trace_output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit)
    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n")
    print(table)

    action_times = [sample["action_seconds"] for sample in samples]
    summary = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "episode": args.episode,
        "mode": args.mode,
        "warmup": args.warmup,
        "sample_count": len(samples),
        "load_seconds": round(load_seconds, 4),
        "action_time_summary": time_summary(action_times),
        "samples": samples,
        "trace_output": str(trace_path),
        "table_output": str(table_path),
        "inference_optimizations": getattr(config, "mm_edge_inference_optimizations", {}),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
