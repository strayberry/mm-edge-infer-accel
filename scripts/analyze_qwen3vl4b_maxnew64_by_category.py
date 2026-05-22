from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_BASELINE = "outputs/qwen3vl_4b_bf16_vllm_ocrbench_1000.json"
DEFAULT_CANDIDATE = "outputs/qwen3vl_4b_bf16_maxnew64_ocrbench_1000.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3-VL-4B BF16 max_new_tokens=64 by OCRBench category."
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-max-tokens", type=int, default=128)
    parser.add_argument("--candidate-max-tokens", type=int, default=64)
    return parser.parse_args()


def load_result(path: str) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def mean(values: list[float]) -> float | None:
    return round(fmean(values), 4) if values else None


def rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def ms(seconds: float | None) -> float | None:
    return round(seconds * 1000, 3) if seconds is not None else None


def group_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get("question_type", "unknown"))].append(sample)
    return dict(groups)


def summarize_group(
    question_type: str,
    baseline_samples: list[dict[str, Any]],
    candidate_samples: list[dict[str, Any]],
    baseline_max_tokens: int,
    candidate_max_tokens: int,
) -> dict[str, Any]:
    if len(baseline_samples) != len(candidate_samples):
        raise ValueError(
            f"Mismatched sample count for {question_type}: "
            f"{len(baseline_samples)} vs {len(candidate_samples)}"
        )
    n = len(candidate_samples)
    baseline_correct = sum(1 for item in baseline_samples if item["correct"])
    candidate_correct = sum(1 for item in candidate_samples if item["correct"])
    regressions = sum(
        1
        for base, candidate in zip(baseline_samples, candidate_samples)
        if base["correct"] and not candidate["correct"]
    )
    improvements = sum(
        1
        for base, candidate in zip(baseline_samples, candidate_samples)
        if not base["correct"] and candidate["correct"]
    )

    baseline_request = mean([item["request_seconds"] * 1000 for item in baseline_samples])
    candidate_request = mean([item["request_seconds"] * 1000 for item in candidate_samples])
    baseline_generated = mean([item["generated_tokens"] for item in baseline_samples])
    candidate_generated = mean([item["generated_tokens"] for item in candidate_samples])
    candidate_decode = mean(
        [
            item["vllm_decode_seconds"] * 1000
            for item in candidate_samples
            if item.get("vllm_decode_seconds") is not None
        ]
    )
    candidate_prefill = mean(
        [
            item["vllm_prefill_seconds"] * 1000
            for item in candidate_samples
            if item.get("vllm_prefill_seconds") is not None
        ]
    )

    return {
        "question_type": question_type,
        "n": n,
        "acc_128": rate(baseline_correct, n),
        "acc_64": rate(candidate_correct, n),
        "acc_delta": round(candidate_correct / n - baseline_correct / n, 4) if n else None,
        "regressions": regressions,
        "improvements": improvements,
        "req_128_ms": baseline_request,
        "req_64_ms": candidate_request,
        "latency_delta_ms": round(candidate_request - baseline_request, 4)
        if baseline_request is not None and candidate_request is not None
        else None,
        "latency_reduction_pct": round((baseline_request - candidate_request) / baseline_request, 4)
        if baseline_request
        else None,
        "gen_128": baseline_generated,
        "gen_64": candidate_generated,
        "trunc_128": rate(
            sum(1 for item in baseline_samples if item["generated_tokens"] == baseline_max_tokens),
            n,
        ),
        "trunc_64": rate(
            sum(
                1
                for item in candidate_samples
                if item["generated_tokens"] == candidate_max_tokens
            ),
            n,
        ),
        "vllm_prefill_64_ms": candidate_prefill,
        "vllm_decode_64_ms": candidate_decode,
    }


def format_value(value: Any, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, bool]]) -> str:
    lines = [
        "| " + " | ".join(header for header, _, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key), pct=pct) for _, key, pct in columns)
            + " |"
        )
    return "\n".join(lines)


def build_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_path: str,
    candidate_path: str,
) -> str:
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    by_drop = sorted(rows, key=lambda row: (row["acc_delta"], -row["n"]))
    by_latency = sorted(rows, key=lambda row: (row["latency_delta_ms"], -row["n"]))
    by_trunc = sorted(rows, key=lambda row: (-row["trunc_64"], row["acc_delta"]))

    accuracy_columns = [
        ("question_type", "question_type", False),
        ("n", "n", False),
        ("acc128", "acc_128", False),
        ("acc64", "acc_64", False),
        ("delta", "acc_delta", False),
        ("reg", "regressions", False),
        ("imp", "improvements", False),
        ("trunc64", "trunc_64", True),
        ("gen64", "gen_64", False),
        ("decode64_ms", "vllm_decode_64_ms", False),
    ]
    latency_columns = [
        ("question_type", "question_type", False),
        ("n", "n", False),
        ("req128_ms", "req_128_ms", False),
        ("req64_ms", "req_64_ms", False),
        ("delta_ms", "latency_delta_ms", False),
        ("reduction", "latency_reduction_pct", True),
        ("gen128", "gen_128", False),
        ("gen64", "gen_64", False),
        ("trunc64", "trunc_64", True),
    ]

    return "\n\n".join(
        [
            "# Qwen3-VL-4B BF16 max_new_tokens=64 Full1000 Category Analysis",
            "## Inputs\n\n"
            f"- Baseline: `{baseline_path}`\n"
            f"- Candidate: `{candidate_path}`",
            "## Overall\n\n"
            "| Metric | max_new_tokens=128 | max_new_tokens=64 |\n"
            "| --- | ---: | ---: |\n"
            f"| Accuracy | {baseline_metrics['accuracy']} | {candidate_metrics['accuracy']} |\n"
            f"| Correct | {baseline_metrics['correct']} / 1000 | "
            f"{candidate_metrics['correct']} / 1000 |\n"
            f"| Request mean | {baseline_metrics['request_latency_ms_mean']} ms | "
            f"{candidate_metrics['request_latency_ms_mean']} ms |\n"
            f"| Request p90 | {baseline_metrics['request_latency_ms_p90']} ms | "
            f"{candidate_metrics['request_latency_ms_p90']} ms |\n"
            f"| Generated tokens mean | {baseline_metrics['generated_tokens_mean']} | "
            f"{candidate_metrics['generated_tokens_mean']} |\n"
            f"| Generated tokens max | {baseline_metrics['generated_tokens_max']} | "
            f"{candidate_metrics['generated_tokens_max']} |",
            "## Accuracy Drop By Category\n\n"
            + markdown_table(by_drop, accuracy_columns),
            "## Latency Change By Category\n\n"
            + markdown_table(by_latency, latency_columns),
            "## Highest Truncation Rate At 64 Tokens\n\n"
            + markdown_table(by_trunc, accuracy_columns),
            "## Interpretation\n\n"
            "`max_new_tokens=64` reduces latency mainly by reducing generated tokens and vLLM "
            "decode time. The categories with high `trunc64` and negative accuracy delta are "
            "the strongest candidates for a larger per-category token budget. Categories with "
            "low truncation and stable accuracy can use a smaller budget in a dynamic "
            "`max_new_tokens` policy.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)
    baseline_groups = group_samples(baseline["samples"])
    candidate_groups = group_samples(candidate["samples"])
    if set(baseline_groups) != set(candidate_groups):
        raise ValueError("Baseline and candidate question_type sets differ.")

    rows = [
        summarize_group(
            question_type,
            baseline_groups[question_type],
            candidate_groups[question_type],
            args.baseline_max_tokens,
            args.candidate_max_tokens,
        )
        for question_type in sorted(candidate_groups)
    ]
    report = build_report(baseline, candidate, rows, args.baseline, args.candidate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
