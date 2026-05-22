from __future__ import annotations

import argparse
import json

from mm_edge_infer_accel.pi05_runtime import run_libero_action_inference, run_synthetic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pi0.5 synthetic or LIBERO action inference.")
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--source", choices=("synthetic", "libero"), default="libero")
    parser.add_argument("--dataset-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("reset", "queue"),
        default="reset",
        help=(
            "reset: clear the action queue before every frame and force a new chunk prediction; "
            "queue: keep Pi0.5's action queue and measure control-loop action output Hz."
        ),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output:
        output = args.output
    elif args.source == "synthetic":
        output = "outputs/pi05_synthetic_action.json"
    else:
        output = "outputs/pi05_libero_action_inference.json"

    if args.source == "synthetic":
        result = run_synthetic_action(args.model_id, output)
    else:
        result = run_libero_action_inference(
            model_id=args.model_id,
            dataset_id=args.dataset_id,
            episodes=[args.episode],
            sample_count=args.sample_count,
            mode=args.mode,
            warmup=args.warmup,
            output=output,
        )

    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
