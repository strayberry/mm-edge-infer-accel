from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import ExperimentConfig, config_to_dict, model_load_path
from .datasets import load_ocrbench
from .env import collect_environment
from .metrics import contains_answer, latency_metrics
from .profiling import gpu_memory_snapshot_mb, nvtx_range, synchronize, timed_stage
from .runners.vllm_runner import VLLMRunner


def benchmark_plan(cfg: ExperimentConfig) -> dict:
    return {
        "name": cfg.name,
        "model_type": "vlm",
        "kind": "vllm_ocrbench_benchmark",
        "status": "planned",
        "config": config_to_dict(cfg),
        "metrics": [
            "accuracy",
            "correct",
            "success_count",
            "failed_count",
            "failure_rate",
            "input_tokens",
            "generated_tokens",
            "tokens_per_second",
            "request_latency_ms_mean",
            "generate_latency_ms_mean",
            "preprocess_latency_ms_mean",
            "decode_latency_ms_mean",
            "vllm_first_token_latency_ms_mean",
            "vllm_prefill_latency_ms_mean",
            "vllm_decode_latency_ms_mean",
            "vllm_inference_latency_ms_mean",
            "vllm_queue_latency_ms_mean",
            "load_seconds",
            "warmup_seconds",
            "gpu_memory_before_load_mb",
            "gpu_memory_after_load_mb",
            "model_load_memory_delta_mb",
            "concurrency",
            "workload_wall_seconds",
            "requests_per_second",
            "serving_tokens_per_second",
        ],
    }


def _warmup_model(runner: VLLMRunner, samples, warmup_count: int) -> float:
    if warmup_count <= 0:
        return 0.0
    started = time.perf_counter()
    warmup_samples = samples.select(range(min(warmup_count, len(samples))))
    with nvtx_range("vlm_vllm_warmup"):
        for sample in tqdm(warmup_samples, desc="Warmup"):
            image = sample["image"].convert("RGB")
            prompt = runner.build_prompt(image, sample["question"])
            runner.generate(prompt, image, max_tokens=1)
    synchronize()
    return time.perf_counter() - started


def _run_one_sample(runner: VLLMRunner, sample: dict, max_new_tokens: int) -> dict:
    image = sample["image"].convert("RGB")
    question = sample["question"]
    answers = list(sample["answer"])
    timings: dict = {}

    with timed_stage("preprocess", timings):
        prompt = runner.build_prompt(image, question)
        input_tokens = runner.token_count(prompt)

    with timed_stage("generate", timings):
        generated = runner.generate(prompt, image, max_tokens=max_new_tokens)

    with timed_stage("decode", timings):
        decoded = generated.text
        generated_tokens = generated.token_count

    return {
        "dataset": sample.get("dataset"),
        "question_type": sample.get("question_type"),
        "question": question,
        "answers": answers,
        "prediction": decoded,
        "correct": contains_answer(decoded, answers),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        **generated.vllm_metrics,
        "request_seconds": sum(timings.values()),
        "sample_tokens_per_second": generated_tokens / timings["generate_seconds"]
        if timings.get("generate_seconds")
        else None,
        **timings,
    }


def _prepare_sample(runner: VLLMRunner, sample: dict) -> dict:
    image = sample["image"].convert("RGB")
    question = sample["question"]
    timings: dict = {}
    with timed_stage("preprocess", timings):
        prompt = runner.build_prompt(image, question)
        input_tokens = runner.token_count(prompt)
    return {
        "dataset": sample.get("dataset"),
        "question_type": sample.get("question_type"),
        "question": question,
        "answers": list(sample["answer"]),
        "image": image,
        "prompt": prompt,
        "input_tokens": input_tokens,
        **timings,
    }


def _finish_sample(prepared: dict, generated, generate_seconds: float) -> dict:
    timings = {
        "preprocess_seconds": prepared["preprocess_seconds"],
        "generate_seconds": generate_seconds,
    }
    with timed_stage("decode", timings):
        decoded = generated.text
        generated_tokens = generated.token_count
    return {
        "dataset": prepared["dataset"],
        "question_type": prepared["question_type"],
        "question": prepared["question"],
        "answers": prepared["answers"],
        "prediction": decoded,
        "correct": contains_answer(decoded, prepared["answers"]),
        "input_tokens": prepared["input_tokens"],
        "generated_tokens": generated_tokens,
        **generated.vllm_metrics,
        "request_seconds": sum(timings.values()),
        "sample_tokens_per_second": generated_tokens / generate_seconds
        if generate_seconds
        else None,
        **timings,
    }


def _run_sample_batch(
    runner: VLLMRunner,
    samples: list[dict],
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    prepared = [_prepare_sample(runner, sample) for sample in samples]
    timings: dict = {}
    with timed_stage("generate", timings):
        generated = runner.generate_batch(
            [(item["prompt"], item["image"]) for item in prepared],
            max_new_tokens,
        )
    generate_seconds = timings["generate_seconds"]
    return (
        [
            _finish_sample(item, generated_item, generate_seconds)
            for item, generated_item in zip(prepared, generated)
        ],
        generate_seconds,
    )


def _aggregate_metrics(
    per_sample: list[dict],
    load_seconds: float,
    warmup_seconds: float,
    memory_before_load: dict,
    memory_after_load: dict,
    concurrency: int = 1,
    workload_wall_seconds: float | None = None,
    serving_generate_seconds: float | None = None,
) -> dict:
    sample_count = len(per_sample)
    total_generated_tokens = sum(item["generated_tokens"] for item in per_sample)
    total_input_tokens = sum(item["input_tokens"] for item in per_sample)
    total_preprocess_seconds = sum(item["preprocess_seconds"] for item in per_sample)
    total_generate_seconds = sum(item["generate_seconds"] for item in per_sample)
    total_decode_seconds = sum(item["decode_seconds"] for item in per_sample)
    correct = sum(1 for item in per_sample if item["correct"])
    model_load_memory_delta = (
        round(
            memory_after_load["gpu_used_memory_mb"] - memory_before_load["gpu_used_memory_mb"],
            2,
        )
        if memory_after_load["gpu_used_memory_mb"] is not None
        and memory_before_load["gpu_used_memory_mb"] is not None
        else None
    )

    workload_wall_seconds = workload_wall_seconds or (
        total_preprocess_seconds + total_generate_seconds + total_decode_seconds
    )
    serving_generate_seconds = serving_generate_seconds or total_generate_seconds

    return {
        "load_seconds": round(load_seconds, 4),
        "warmup_seconds": round(warmup_seconds, 4),
        "concurrency": concurrency,
        "sample_count": sample_count,
        "success_count": sample_count,
        "failed_count": 0,
        "failure_rate": 0.0,
        "accuracy": round(correct / sample_count, 4) if sample_count else None,
        "correct": correct,
        "input_tokens": total_input_tokens,
        "input_tokens_mean": round(total_input_tokens / sample_count, 4)
        if sample_count
        else None,
        "input_tokens_max": max((item["input_tokens"] for item in per_sample), default=None),
        "generated_tokens": total_generated_tokens,
        "generated_tokens_mean": round(total_generated_tokens / sample_count, 4)
        if sample_count
        else None,
        "generated_tokens_max": max(
            (item["generated_tokens"] for item in per_sample), default=None
        ),
        "preprocess_seconds": round(total_preprocess_seconds, 4),
        "generate_seconds": round(total_generate_seconds, 4),
        "decode_seconds": round(total_decode_seconds, 4),
        "total_workload_seconds": round(
            total_preprocess_seconds + total_generate_seconds + total_decode_seconds,
            4,
        ),
        "workload_wall_seconds": round(workload_wall_seconds, 4),
        "requests_per_second": round(sample_count / workload_wall_seconds, 4)
        if workload_wall_seconds
        else None,
        "tokens_per_second": round(total_generated_tokens / total_generate_seconds, 4)
        if total_generate_seconds
        else None,
        "serving_tokens_per_second": round(total_generated_tokens / serving_generate_seconds, 4)
        if serving_generate_seconds
        else None,
        **latency_metrics(per_sample),
        "gpu_total_memory_mb": memory_after_load["gpu_total_memory_mb"],
        "gpu_memory_before_load_mb": memory_before_load["gpu_used_memory_mb"],
        "gpu_memory_after_load_mb": memory_after_load["gpu_used_memory_mb"],
        "model_load_memory_delta_mb": model_load_memory_delta,
    }


def run_benchmark(cfg: ExperimentConfig, output: Optional[str] = None) -> dict:
    if cfg.runtime.backend != "vllm":
        raise ValueError("VLM benchmark requires runtime.backend=vllm")
    if cfg.eval.dataset != "ocrbench":
        raise ValueError(f"VLM benchmark expects eval.dataset=ocrbench, got {cfg.eval.dataset}")

    samples = load_ocrbench(cfg.eval.sample_count, cfg.eval.sample_strategy)
    load_path = model_load_path(cfg)
    memory_before_load = gpu_memory_snapshot_mb()
    started = time.perf_counter()
    with nvtx_range("vlm_vllm_load_model"):
        runner = VLLMRunner(cfg)
    load_seconds = time.perf_counter() - started
    memory_after_load = gpu_memory_snapshot_mb()

    warmup_seconds = _warmup_model(runner, samples, cfg.profile.warmup)
    per_sample = []
    serving_generate_seconds = 0.0
    workload_started = time.perf_counter()
    with nvtx_range("vlm_vllm_ocrbench_loop"):
        if cfg.runtime.concurrency == 1:
            for sample in tqdm(samples, desc=f"Benchmark {cfg.name}"):
                item = _run_one_sample(runner, sample, cfg.model.max_new_tokens)
                per_sample.append(item)
                serving_generate_seconds += item["generate_seconds"]
        else:
            sample_list = list(samples)
            for start in tqdm(
                range(0, len(sample_list), cfg.runtime.concurrency),
                desc=f"Benchmark {cfg.name} c={cfg.runtime.concurrency}",
            ):
                batch = sample_list[start : start + cfg.runtime.concurrency]
                batch_results, batch_generate_seconds = _run_sample_batch(
                    runner,
                    batch,
                    cfg.model.max_new_tokens,
                )
                per_sample.extend(batch_results)
                serving_generate_seconds += batch_generate_seconds
    workload_wall_seconds = time.perf_counter() - workload_started

    result = {
        "name": cfg.name,
        "model_type": "vlm",
        "backend": "vllm",
        "measurement": "vllm_ocrbench_nvtx",
        "dataset": cfg.eval.dataset,
        "dataset_id": "echo840/OCRBench",
        "model_id": cfg.model.model_id,
        "model_path": cfg.model.model_path,
        "load_path": load_path,
        "dtype": cfg.model.dtype,
        "model_max_pixels": cfg.model.max_pixels,
        "quant": cfg.quant.__dict__,
        "vllm_config": {
            "max_model_len": cfg.runtime.max_model_len,
            "kv_cache_dtype": cfg.runtime.kv_cache_dtype,
            "gpu_memory_utilization": cfg.runtime.gpu_memory_utilization,
            "enforce_eager": cfg.runtime.enforce_eager,
            "disable_flashinfer_sampler": cfg.runtime.disable_flashinfer_sampler,
            "concurrency": cfg.runtime.concurrency,
            "mm_processor_kwargs": {
                **(
                    {"max_pixels": cfg.model.max_pixels}
                    if cfg.model.max_pixels is not None
                    else {}
                ),
                **cfg.runtime.mm_processor_kwargs,
            },
            "speculative_config": cfg.runtime.speculative_config,
        },
        "sampling_params": {
            "max_tokens": cfg.model.max_new_tokens,
            "temperature": 0.0,
        },
        "system_info": collect_environment(),
        "metrics": _aggregate_metrics(
            per_sample,
            load_seconds,
            warmup_seconds,
            memory_before_load,
            memory_after_load,
            concurrency=cfg.runtime.concurrency,
            workload_wall_seconds=workload_wall_seconds,
            serving_generate_seconds=serving_generate_seconds,
        ),
        "samples": per_sample,
    }

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
