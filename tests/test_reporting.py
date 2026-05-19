import pytest

from scripts.reporting.analyze_qwen3vl4b_maxnew64_by_category import (
    build_report,
    group_samples,
    markdown_table,
    summarize_group,
)


def _sample(question_type, correct, request_seconds, generated_tokens, decode=0.01):
    return {
        "question_type": question_type,
        "correct": correct,
        "request_seconds": request_seconds,
        "generated_tokens": generated_tokens,
        "vllm_decode_seconds": decode,
        "vllm_prefill_seconds": 0.02,
    }


def test_summarize_group_counts_accuracy_latency_and_truncation():
    baseline = [
        _sample("doc", True, 1.0, 128),
        _sample("doc", False, 2.0, 20),
    ]
    candidate = [
        _sample("doc", False, 0.5, 64),
        _sample("doc", True, 1.0, 10),
    ]

    row = summarize_group("doc", baseline, candidate, 128, 64)

    assert row["acc_128"] == 0.5
    assert row["acc_64"] == 0.5
    assert row["regressions"] == 1
    assert row["improvements"] == 1
    assert row["req_128_ms"] == 1500.0
    assert row["req_64_ms"] == 750.0
    assert row["trunc_128"] == 0.5
    assert row["trunc_64"] == 0.5


def test_summarize_group_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="Mismatched sample count"):
        summarize_group("doc", [_sample("doc", True, 1.0, 1)], [], 128, 64)


def test_group_samples_and_markdown_table():
    groups = group_samples([_sample("doc", True, 1.0, 1), _sample("scene", False, 2.0, 2)])

    assert sorted(groups) == ["doc", "scene"]
    table = markdown_table(
        [{"question_type": "doc", "trunc_64": 0.5}],
        [("type", "question_type", False), ("trunc", "trunc_64", True)],
    )
    assert "| doc | 50.0% |" in table


def test_build_report_includes_overall_metrics():
    baseline = {
        "metrics": {
            "accuracy": 0.8,
            "correct": 8,
            "request_latency_ms_mean": 100.0,
            "request_latency_ms_p90": 200.0,
            "generated_tokens_mean": 10.0,
            "generated_tokens_max": 20,
        }
    }
    candidate = {
        "metrics": {
            "accuracy": 0.7,
            "correct": 7,
            "request_latency_ms_mean": 80.0,
            "request_latency_ms_p90": 150.0,
            "generated_tokens_mean": 8.0,
            "generated_tokens_max": 16,
        }
    }
    rows = [
        {
            "question_type": "doc",
            "n": 2,
            "acc_128": 1.0,
            "acc_64": 0.5,
            "acc_delta": -0.5,
            "regressions": 1,
            "improvements": 0,
            "req_128_ms": 100.0,
            "req_64_ms": 80.0,
            "latency_delta_ms": -20.0,
            "latency_reduction_pct": 0.2,
            "gen_128": 10.0,
            "gen_64": 8.0,
            "trunc_64": 0.5,
            "vllm_decode_64_ms": 20.0,
        }
    ]

    report = build_report(baseline, candidate, rows, "base.json", "candidate.json")

    assert "# Qwen3-VL-4B BF16" in report
    assert "| Accuracy | 0.8 | 0.7 |" in report
    assert "base.json" in report
