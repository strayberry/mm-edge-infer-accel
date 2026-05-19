import pytest

from mm_edge_infer_accel.config import load_config, model_type_from_config, validate_config


def test_unknown_model_family_is_rejected():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    cfg.model.family = "unknown"

    with pytest.raises(ValueError, match="Unknown model family"):
        model_type_from_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size"),
        ("concurrency", 0, "concurrency"),
        ("visual_keep_ratio", 0.0, "visual_keep_ratio"),
        ("max_model_len", 0, "max_model_len"),
        ("gpu_memory_utilization", 1.5, "gpu_memory_utilization"),
    ],
)
def test_runtime_numeric_validation(field, value, message):
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    setattr(cfg.runtime, field, value)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_quant_bits_validation():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    cfg.quant.bits = 3

    with pytest.raises(ValueError, match="quant.bits"):
        validate_config(cfg)


def test_sample_strategy_validation():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    cfg.eval.sample_strategy = "random"

    with pytest.raises(ValueError, match="sample_strategy"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("eval", "sample_count", -1, "sample_count"),
        ("model", "max_new_tokens", -1, "max_new_tokens"),
        ("model", "max_pixels", 0, "max_pixels"),
    ],
)
def test_benchmark_override_field_validation(section, field, value, message):
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    setattr(getattr(cfg, section), field, value)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_speculative_config_validation():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    cfg.runtime.speculative_config = {"num_speculative_tokens": 0}

    with pytest.raises(ValueError, match="num_speculative_tokens"):
        validate_config(cfg)
