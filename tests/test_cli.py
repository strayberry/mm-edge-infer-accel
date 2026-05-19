import json

from mm_edge_infer_accel import cli


def _read_stdout_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_benchmark_prints_vlm_plan(capsys):
    assert cli.main(["benchmark", "--config", "configs/vlm/qwen3vl_4b_bf16.yaml"]) == 0

    payload = _read_stdout_json(capsys)

    assert payload["kind"] == "vllm_ocrbench_benchmark"
    assert payload["model_type"] == "vlm"
    assert payload["status"] == "planned"


def test_cli_benchmark_accepts_concurrency_override(capsys):
    assert (
        cli.main(
            [
                "benchmark",
                "--config",
                "configs/vlm/qwen3vl_4b_awq_local.yaml",
                "--concurrency",
                "4",
            ]
        )
        == 0
    )

    payload = _read_stdout_json(capsys)

    assert payload["config"]["runtime"]["concurrency"] == 4


def test_cli_benchmark_accepts_dataset_and_model_overrides(capsys):
    assert (
        cli.main(
            [
                "benchmark",
                "--config",
                "configs/vlm/qwen3vl_4b_bf16.yaml",
                "--sample-count",
                "1000",
                "--sample-strategy",
                "first",
                "--max-new-tokens",
                "64",
                "--max-pixels",
                "501760",
            ]
        )
        == 0
    )

    payload = _read_stdout_json(capsys)

    assert payload["config"]["eval"]["sample_count"] == 1000
    assert payload["config"]["eval"]["sample_strategy"] == "first"
    assert payload["config"]["model"]["max_new_tokens"] == 64
    assert payload["config"]["model"]["max_pixels"] == 501760


def test_cli_quantize_prints_config_plan(capsys):
    assert cli.main(["quantize", "--config", "configs/vlm/qwen3vl_4b_gptq_local.yaml"]) == 0

    payload = _read_stdout_json(capsys)

    assert payload["kind"] == "quantization"
    assert payload["status"] == "planned"
    assert payload["config"]["quant"]["method"] == "gptq"


def test_cli_profile_prints_nsys_command(capsys):
    config_path = "configs/vlm/qwen3vl_4b_bf16.yaml"

    assert cli.main(["profile", "--tool", "nsys", "--config", config_path]) == 0

    payload = _read_stdout_json(capsys)

    assert payload["kind"] == "profile"
    assert payload["tool"] == "nsys"
    assert config_path in payload["command"]
    assert "nsys profile" in payload["command"]


def test_cli_env_check_uses_collector(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_environment", lambda: {"python": "test"})

    assert cli.main(["env-check"]) == 0

    assert _read_stdout_json(capsys) == {"python": "test"}
