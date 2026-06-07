from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain what edge AI inference optimization means in one paragraph.",
    "List three practical ways to reduce LLM inference latency.",
    "What is TensorRT-LLM used for?",
    "Summarize the tradeoff between quantization and accuracy.",
    "Write a short description of CUDA Graphs.",
    "What does KV cache store during autoregressive decoding?",
    "Give a concise definition of throughput in model serving.",
    "Why can smaller models be useful on edge devices?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-test helper: benchmark text-only Qwen3-0.6B with vLLM"
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-0.6B",
        help="Local HuggingFace Qwen3-0.6B model directory",
    )
    parser.add_argument(
        "--output",
        default="outputs/qwen3_0_6b_vllm_bf16_baseline.json",
        help="Output benchmark JSON path",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--disable-enforce-eager",
        action="store_true",
        help="Allow vLLM CUDA Graph/compile paths instead of the current eager baseline",
    )
    return parser.parse_args()


def _positive_delta(end: float, start: float) -> float | None:
    if end <= 0 or start <= 0:
        return None
    return max(0.0, end - start)


def _extract_metrics(metrics) -> dict[str, float | int | bool | None]:
    if metrics is None:
        return {}
    queued_ts = float(getattr(metrics, "queued_ts", 0.0) or 0.0)
    scheduled_ts = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    first_token_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_token_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    return {
        "queue_ms": _to_ms(_positive_delta(scheduled_ts, queued_ts)),
        "prefill_ms": _to_ms(_positive_delta(first_token_ts, scheduled_ts)),
        "decode_ms": _to_ms(_positive_delta(last_token_ts, first_token_ts)),
        "inference_ms": _to_ms(_positive_delta(last_token_ts, scheduled_ts)),
        "first_token_latency_ms": _to_ms(first_token_latency or None),
        "num_generation_tokens": getattr(metrics, "num_generation_tokens", None),
        "is_corrupted": getattr(metrics, "is_corrupted", None),
    }


def _to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 1000, 3)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[int(ratio * (len(values) - 1))]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from vllm import LLM, SamplingParams

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("loading vLLM model")
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.disable_enforce_eager,
        disable_log_stats=False,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    for idx in range(args.warmup):
        llm.generate([DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()

    samples = []
    latencies = []
    generated_tokens = []
    first_token_latencies = []
    for index, prompt in enumerate(DEFAULT_PROMPTS):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        request_output = outputs[0]
        generation = request_output.outputs[0]
        token_ids = getattr(generation, "token_ids", None)
        token_count = len(token_ids) if token_ids is not None else None
        metrics = _extract_metrics(getattr(request_output, "metrics", None))
        if metrics.get("first_token_latency_ms") is not None:
            first_token_latencies.append(metrics["first_token_latency_ms"])
        latencies.append(elapsed)
        if token_count is not None:
            generated_tokens.append(token_count)
        samples.append(
            {
                "index": index,
                "prompt": prompt,
                "prediction": generation.text,
                "latency_ms": round(elapsed * 1000, 3),
                "generated_tokens": token_count,
                "tokens_per_second": round(token_count / elapsed, 3) if token_count else None,
                **metrics,
            }
        )
        print(index, round(elapsed * 1000, 3), "ms", "new", token_count)

    lat_ms = [value * 1000 for value in latencies]
    result = {
        "backend": "vllm",
        "model": args.model_path,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "sample_count": len(DEFAULT_PROMPTS),
        "load_seconds": round(load_seconds, 4),
        "latency_mean_ms": round(statistics.mean(lat_ms), 3),
        "latency_p50_ms": round(statistics.median(lat_ms), 3),
        "latency_p95_ms": round(percentile(lat_ms, 0.95), 3),
        "tokens_per_second_mean": round(sum(generated_tokens) / sum(latencies), 3),
        "first_token_latency_mean_ms": round(statistics.mean(first_token_latencies), 3)
        if first_token_latencies
        else None,
        "enforce_eager": not args.disable_enforce_eager,
        "samples": samples,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
