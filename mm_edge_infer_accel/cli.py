from __future__ import annotations

import argparse
import json
from typing import Any

from .common import profile_command, quantization_plan
from .config import config_to_dict, load_config, model_type_from_config, validate_config
from .env import collect_environment
from . import vla, vlm


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _apply_benchmark_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.concurrency is not None:
        cfg.runtime.concurrency = args.concurrency
    if args.sample_count is not None:
        cfg.eval.sample_count = args.sample_count
    if args.sample_strategy is not None:
        cfg.eval.sample_strategy = args.sample_strategy
    if args.max_new_tokens is not None:
        cfg.model.max_new_tokens = args.max_new_tokens
    if args.max_pixels is not None:
        cfg.model.max_pixels = args.max_pixels
    if args.mode is not None:
        cfg.eval.mode = args.mode
    if args.episode is not None:
        cfg.eval.episodes = args.episode
    validate_config(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mm-edge-infer-accel")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env-check")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", required=True)
    bench.add_argument("--dry-run", action="store_true")
    bench.add_argument("--run", action="store_true")
    bench.add_argument("--output")
    bench.add_argument("--concurrency", type=int)
    bench.add_argument("--sample-count", type=int)
    bench.add_argument("--sample-strategy", choices=["first", "stratified"])
    bench.add_argument("--max-new-tokens", type=int)
    bench.add_argument("--max-pixels", type=int)
    bench.add_argument("--mode", choices=["reset", "queue"],
                       help="VLA action inference mode")
    bench.add_argument("--episode", type=int, action="append",
                       help="VLA episode(s) to run (can be specified multiple times)")


    quant = sub.add_parser("quantize")
    quant.add_argument("--config", required=True)
    quant.add_argument("--dry-run", action="store_true")

    prof = sub.add_parser("profile")
    prof.add_argument("--config", required=True)
    prof.add_argument("--tool", choices=["nsys", "ncu"], required=True)
    prof.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "env-check":
        _print(collect_environment())
        return 0

    cfg = load_config(args.config)
    if args.command == "benchmark":
        _apply_benchmark_overrides(cfg, args)
    cfg_dict = config_to_dict(cfg)
    model_type = model_type_from_config(cfg)
    if args.command == "benchmark":
        if args.run:
            if model_type == "vla":
                _print(vla.run_benchmark(cfg, output=args.output))
            else:
                _print(vlm.run_benchmark(cfg, output=args.output))
        else:
            _print(vla.benchmark_plan(cfg) if model_type == "vla" else vlm.benchmark_plan(cfg))
    elif args.command == "quantize":
        _print(quantization_plan(cfg_dict))
    elif args.command == "profile":
        _print(profile_command(cfg.name, args.config, args.tool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
