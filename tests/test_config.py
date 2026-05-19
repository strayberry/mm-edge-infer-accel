from mm_edge_infer_accel import vlm
from mm_edge_infer_accel.config import (
    load_config,
    model_load_path,
    model_type_from_config,
    validate_config,
)


def test_load_qwen_config():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    assert cfg.name == "qwen3vl_4b_bf16"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert cfg.model.model_path == "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
    assert model_load_path(cfg) == "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
    assert cfg.runtime.backend == "vllm"
    assert cfg.runtime.visual_keep_ratio == 1.0
    assert cfg.runtime.max_model_len == 1024
    assert cfg.runtime.disable_flashinfer_sampler is True
    assert cfg.runtime.mm_processor_kwargs == {"truncation": False}
    assert cfg.eval.sample_strategy == "stratified"


def test_model_type_mapping():
    assert model_type_from_config(load_config("configs/vlm/smolvlm2_2b_fp32.yaml")) == "vlm"
    assert model_type_from_config(load_config("configs/vlm/qwen3vl_4b_gptq_local.yaml")) == "vlm"
    assert model_type_from_config(load_config("configs/vla/pi05_libero_plan.yaml")) == "vla"


def test_vlm_benchmark_requires_vllm_backend():
    cfg = load_config("configs/vlm/qwen3vl_4b_awq_local.yaml")
    cfg.runtime.backend = "transformers"
    try:
        vlm.run_benchmark(cfg)
    except ValueError as exc:
        assert "runtime.backend=vllm" in str(exc)
    else:
        raise AssertionError("Expected VLM benchmark to reject non-vLLM backend")


def test_family_backend_compatibility():
    pi05_cfg = load_config("configs/vla/pi05_libero_plan.yaml")
    pi05_cfg.runtime.backend = "vllm"
    try:
        validate_config(pi05_cfg)
    except ValueError as exc:
        assert "pi05" in str(exc).lower()
    else:
        raise AssertionError("Expected Pi0.5 to reject vLLM backend")
