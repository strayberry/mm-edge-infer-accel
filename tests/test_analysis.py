from mm_edge_infer_accel.vla import (
    cosine_similarity,
    mean_absolute_error,
)
from mm_edge_infer_accel.metrics import latency_metrics
from mm_edge_infer_accel.vlm import _aggregate_metrics


def test_small_metrics():
    assert mean_absolute_error([1, 2, 3], [1, 1, 5]) == 1.0
    assert round(cosine_similarity([1, 0], [1, 0]), 6) == 1.0


def test_latency_metrics_include_vllm_internal_timings():
    per_sample = [
        {
            "request_seconds": 1.0,
            "generate_seconds": 0.8,
            "preprocess_seconds": 0.1,
            "decode_seconds": 0.1,
            "sample_tokens_per_second": 10.0,
            "vllm_first_token_latency_seconds": 0.2,
            "vllm_queue_seconds": 0.01,
            "vllm_prefill_seconds": 0.15,
            "vllm_decode_seconds": 0.6,
            "vllm_inference_seconds": 0.75,
            "vllm_num_generation_tokens": 8,
            "vllm_is_corrupted": False,
        },
        {
            "request_seconds": 2.0,
            "generate_seconds": 1.6,
            "preprocess_seconds": 0.2,
            "decode_seconds": 0.2,
            "sample_tokens_per_second": 5.0,
            "vllm_first_token_latency_seconds": 0.3,
            "vllm_queue_seconds": 0.02,
            "vllm_prefill_seconds": 0.25,
            "vllm_decode_seconds": 1.2,
            "vllm_inference_seconds": 1.45,
            "vllm_num_generation_tokens": 8,
            "vllm_is_corrupted": True,
        },
    ]

    metrics = latency_metrics(per_sample)

    assert metrics["request_latency_ms_mean"] == 1500.0
    assert metrics["vllm_first_token_latency_ms_mean"] == 250.0
    assert metrics["vllm_prefill_latency_ms_mean"] == 200.0
    assert metrics["vllm_decode_latency_ms_mean"] == 900.0
    assert metrics["vllm_inference_latency_ms_mean"] == 1100.0
    assert metrics["vllm_queue_latency_ms_mean"] == 15.0
    assert metrics["vllm_generation_tokens"] == 16
    assert metrics["vllm_generation_tokens_mean"] == 8.0
    assert metrics["vllm_corrupted_count"] == 1


def test_vlm_aggregate_metrics_include_serving_throughput():
    per_sample = [
        {
            "correct": True,
            "input_tokens": 10,
            "generated_tokens": 5,
            "preprocess_seconds": 0.1,
            "generate_seconds": 1.0,
            "decode_seconds": 0.1,
            "request_seconds": 1.2,
            "sample_tokens_per_second": 5.0,
        },
        {
            "correct": False,
            "input_tokens": 12,
            "generated_tokens": 7,
            "preprocess_seconds": 0.1,
            "generate_seconds": 1.0,
            "decode_seconds": 0.1,
            "request_seconds": 1.2,
            "sample_tokens_per_second": 7.0,
        },
    ]

    metrics = _aggregate_metrics(
        per_sample,
        load_seconds=1.0,
        warmup_seconds=0.0,
        memory_before_load={"gpu_used_memory_mb": 100.0},
        memory_after_load={"gpu_total_memory_mb": 1000.0, "gpu_used_memory_mb": 300.0},
        concurrency=2,
        workload_wall_seconds=1.5,
        serving_generate_seconds=1.0,
    )

    assert metrics["concurrency"] == 2
    assert metrics["requests_per_second"] == 1.3333
    assert metrics["serving_tokens_per_second"] == 12.0
    assert metrics["tokens_per_second"] == 6.0
