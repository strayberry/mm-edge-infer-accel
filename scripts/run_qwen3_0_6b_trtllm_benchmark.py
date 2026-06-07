from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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


def _prepend_existing_library_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing + ([current] if current else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _ensure_trtllm_library_path() -> None:
    if os.environ.get("MM_EDGE_TRTLLM_LD_READY") == "1":
        return
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_dir / "site-packages"
    library_paths = [
        site_packages / "nvidia" / "cu13" / "lib",
        site_packages / "tensorrt_libs",
    ]
    _prepend_existing_library_paths(library_paths)
    os.environ["MM_EDGE_TRTLLM_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-test helper: benchmark a text-only Qwen3-0.6B TensorRT-LLM engine"
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-0.6B",
        help="Local HuggingFace Qwen3-0.6B model directory for tokenizer loading",
    )
    parser.add_argument(
        "--engine-dir",
        default="outputs/tensorrt/qwen3_0_6b_trtllm_bf16_engine",
        help="TensorRT-LLM engine directory",
    )
    parser.add_argument(
        "--output",
        default="outputs/qwen3_0_6b_trtllm_bf16_engine_baseline.json",
        help="Output benchmark JSON path",
    )
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--quant-algo", default="none", choices=["none", "w8a16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def generate_one(
    runner,
    tokenizer,
    prompt: str,
    sampling_config,
    max_input_len: int,
) -> tuple[list[int], list[int], float]:
    import torch

    ids = tokenizer.encode(
        prompt, add_special_tokens=True, truncation=True, max_length=max_input_len
    )
    input_ids = [torch.tensor(ids, dtype=torch.int32)]
    start = time.perf_counter()
    result = runner.generate(input_ids, sampling_config=sampling_config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    seq = result["output_ids"][0][0].detach().cpu().tolist()
    return ids, seq, elapsed


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[int(ratio * (len(values) - 1))]


def main() -> int:
    args = parse_args()
    _ensure_trtllm_library_path()

    import torch
    from transformers import AutoTokenizer
    from tensorrt_llm.runtime import ModelRunner
    from tensorrt_llm.runtime.generation import SamplingConfig

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("loading tokenizer and TensorRT-LLM engine")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    runner = ModelRunner.from_dir(args.engine_dir, max_output_len=args.max_new_tokens)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    sampling_config = SamplingConfig(
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=1,
        return_dict=True,
    )

    for idx in range(args.warmup):
        generate_one(
            runner,
            tokenizer,
            DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)],
            sampling_config,
            args.max_input_len,
        )

    samples = []
    latencies = []
    new_tokens = []
    for index, prompt in enumerate(DEFAULT_PROMPTS):
        ids, seq, elapsed = generate_one(
            runner, tokenizer, prompt, sampling_config, args.max_input_len
        )
        generated_tokens = max(0, len(seq) - len(ids))
        latencies.append(elapsed)
        new_tokens.append(generated_tokens)
        samples.append(
            {
                "index": index,
                "prompt": prompt,
                "latency_ms": round(elapsed * 1000, 3),
                "input_tokens": len(ids),
                "new_tokens": generated_tokens,
                "text": tokenizer.decode(seq, skip_special_tokens=True),
            }
        )
        print(index, round(elapsed * 1000, 3), "ms", "input", len(ids), "new", generated_tokens)

    lat_ms = [value * 1000 for value in latencies]
    result = {
        "backend": "tensorrt_llm_engine",
        "model": args.model_path,
        "engine": args.engine_dir,
        "dtype": "bfloat16",
        "quant_algo": args.quant_algo,
        "max_model_len": args.max_input_len,
        "max_new_tokens": args.max_new_tokens,
        "sample_count": len(DEFAULT_PROMPTS),
        "load_seconds": round(load_seconds, 4),
        "latency_mean_ms": round(statistics.mean(lat_ms), 3),
        "latency_p50_ms": round(statistics.median(lat_ms), 3),
        "latency_p95_ms": round(percentile(lat_ms, 0.95), 3),
        "tokens_per_second_mean": round(sum(new_tokens) / sum(latencies), 3),
        "samples": samples,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
