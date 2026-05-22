from mm_edge_infer_accel import vla_lerobot, vlm
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
    assert model_type_from_config(load_config("configs/vla/pi05_libero.yaml")) == "vla"


def test_pi05_libero_config_runs_action_inference():
    cfg = load_config("configs/vla/pi05_libero.yaml")

    assert cfg.eval.dataset == "HuggingFaceVLA/libero"
    assert cfg.eval.episodes == [0, 1, 2]
    assert cfg.eval.sample_count == 100
    assert cfg.eval.mode == "queue"
    assert cfg.profile.warmup == 3
    assert cfg.runtime.enable_prefix_kv_cache is True


def test_enable_prefix_kv_cache_defaults_to_true():
    cfg = load_config("configs/vlm/qwen3vl_4b_bf16.yaml")
    assert cfg.runtime.enable_prefix_kv_cache is True


def test_enable_prefix_kv_cache_can_be_disabled_via_yaml():
    cfg = load_config("configs/vla/pi05_libero.yaml")
    cfg.runtime.enable_prefix_kv_cache = False
    assert cfg.runtime.enable_prefix_kv_cache is False


def test_pi05_lerobot_benchmark_dispatches_libero_action_inference(monkeypatch):
    cfg = load_config("configs/vla/pi05_libero.yaml")
    received = {}

    def fake_run_libero_action_inference(**kwargs):
        received.update(kwargs)
        return {"source": "libero"}

    monkeypatch.setattr(vla_lerobot, "run_libero_action_inference", fake_run_libero_action_inference)

    assert vla_lerobot.run_benchmark(cfg, output="outputs/test.json") == {"source": "libero"}
    assert received == {
        "model_id": "lerobot/pi05_libero_finetuned_v044",
        "dataset_id": "HuggingFaceVLA/libero",
        "episodes": [0, 1, 2],
        "sample_count": 100,
        "mode": "queue",
        "warmup": 3,
        "output": "outputs/test.json",
        "enable_prefix_kv_cache": True,
    }


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
    pi05_cfg = load_config("configs/vla/pi05_libero.yaml")
    pi05_cfg.runtime.backend = "vllm"
    try:
        validate_config(pi05_cfg)
    except ValueError as exc:
        assert "pi05" in str(exc).lower()
    else:
        raise AssertionError("Expected Pi0.5 to reject vLLM backend")
