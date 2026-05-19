from __future__ import annotations

import statistics
from typing import Optional


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip() if ch.isalnum())


def contains_answer(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    return any(normalize_text(answer) in pred for answer in answers)


def mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], value: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * value)
    return ordered[index]


def round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None


def seconds_to_ms(value: Optional[float]) -> Optional[float]:
    return round(value * 1000, 3) if value is not None else None


def _present_numbers(per_sample: list[dict], key: str) -> list[float]:
    return [
        item[key]
        for item in per_sample
        if item.get(key) is not None and isinstance(item.get(key), (int, float))
    ]


def _latency_summary(per_sample: list[dict], key: str, prefix: str) -> dict:
    values = _present_numbers(per_sample, key)
    return {
        f"{prefix}_ms_mean": seconds_to_ms(mean(values)),
        f"{prefix}_ms_p50": seconds_to_ms(percentile(values, 0.50)),
        f"{prefix}_ms_p90": seconds_to_ms(percentile(values, 0.90)),
        f"{prefix}_ms_p95": seconds_to_ms(percentile(values, 0.95)),
        f"{prefix}_ms_max": seconds_to_ms(max(values) if values else None),
    }


def vllm_latency_metrics(per_sample: list[dict]) -> dict:
    generation_tokens = _present_numbers(per_sample, "vllm_num_generation_tokens")
    corrupted = [item.get("vllm_is_corrupted") for item in per_sample]
    corrupted_count = sum(1 for value in corrupted if value is True)

    return {
        **_latency_summary(
            per_sample,
            "vllm_first_token_latency_seconds",
            "vllm_first_token_latency",
        ),
        **_latency_summary(per_sample, "vllm_queue_seconds", "vllm_queue_latency"),
        **_latency_summary(per_sample, "vllm_prefill_seconds", "vllm_prefill_latency"),
        **_latency_summary(per_sample, "vllm_decode_seconds", "vllm_decode_latency"),
        **_latency_summary(per_sample, "vllm_inference_seconds", "vllm_inference_latency"),
        "vllm_generation_tokens": int(sum(generation_tokens)) if generation_tokens else None,
        "vllm_generation_tokens_mean": round_optional(mean(generation_tokens), 4),
        "vllm_corrupted_count": corrupted_count if corrupted else None,
    }


def latency_metrics(per_sample: list[dict]) -> dict:
    request_seconds = [item["request_seconds"] for item in per_sample]
    generate_seconds = [item["generate_seconds"] for item in per_sample]
    preprocess_seconds = [item["preprocess_seconds"] for item in per_sample]
    decode_seconds = [item["decode_seconds"] for item in per_sample]
    sample_tps = [
        item["sample_tokens_per_second"]
        for item in per_sample
        if item.get("sample_tokens_per_second") is not None
    ]

    return {
        "request_latency_ms_mean": seconds_to_ms(mean(request_seconds)),
        "request_latency_ms_p50": seconds_to_ms(percentile(request_seconds, 0.50)),
        "request_latency_ms_p90": seconds_to_ms(percentile(request_seconds, 0.90)),
        "request_latency_ms_p95": seconds_to_ms(percentile(request_seconds, 0.95)),
        "request_latency_ms_max": seconds_to_ms(max(request_seconds) if request_seconds else None),
        "generate_latency_ms_mean": seconds_to_ms(mean(generate_seconds)),
        "generate_latency_ms_p50": seconds_to_ms(percentile(generate_seconds, 0.50)),
        "generate_latency_ms_p90": seconds_to_ms(percentile(generate_seconds, 0.90)),
        "generate_latency_ms_p95": seconds_to_ms(percentile(generate_seconds, 0.95)),
        "generate_latency_ms_max": seconds_to_ms(
            max(generate_seconds) if generate_seconds else None
        ),
        "preprocess_latency_ms_mean": seconds_to_ms(mean(preprocess_seconds)),
        "decode_latency_ms_mean": seconds_to_ms(mean(decode_seconds)),
        "sample_tokens_per_second_mean": round_optional(mean(sample_tps), 4),
        "sample_tokens_per_second_p50": round_optional(percentile(sample_tps, 0.50), 4),
        "sample_tokens_per_second_p90": round_optional(percentile(sample_tps, 0.90), 4),
        **vllm_latency_metrics(per_sample),
    }
