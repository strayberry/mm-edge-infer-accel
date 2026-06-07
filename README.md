# Multimodal Edge Inference Acceleration

This repository collects edge-inference experiments for multimodal LLM and VLA workloads.
The current focus is practical latency, memory, and accuracy trade-offs on constrained GPUs
such as RTX 3080 Ti and Jetson AGX Orin.

## Current Tracks

| Track | Scope | Status |
| --- | --- | --- |
| Qwen3-VL-4B VLM | vLLM BF16 / AWQ / GPTQ on OCRBench, concurrency curve, latency and memory | Mostly complete |
| Pi0.5 LeRobot reference | LIBERO action inference, reset vs queue, prefix KV cache, PyTorch/Nsight profiling | Mostly complete |
| Pi0.5 FlashRT / Orin | BF16, cache2, INT8, vitpack ablations, 300-frame action similarity | Complete |
| Pi0.5 closed-loop | LIBERO env success rate, control Hz, episode length, policy/env latency | Current evaluation track |
| TensorRT side tests | Qwen3-VL visual+projector export and Qwen3-0.6B TensorRT-LLM text-only test | Side track, not the main project path |

## Repository Layout

```text
mm_edge_infer_accel/
├── cli.py                 CLI dispatch for benchmark, quantize, profile, env-check
├── config.py              YAML loading, experiment dataclasses, validation
├── common.py              quantization_plan() and profile_command()
├── datasets.py            OCRBench loading and stratified sampling
├── env.py                 GPU/CUDA/package environment inspection
├── metrics.py             answer matching and latency metrics
├── profiling.py           timing, NVTX ranges, GPU memory snapshots
├── vlm.py                 VLM benchmark orchestration through vLLM
├── vla.py                 VLA dispatch and action metrics
├── vla_lerobot.py         LeRobot-backed VLA dispatch
├── pi05_runtime.py        Pi0.5 policy loading and LIBERO action inference
├── pi05_optimizations.py  Prefix KV cache optimization for Pi0.5 denoise loop
├── runners/
│   └── vllm_runner.py     VLLMRunner load, prompt build, generate, batch generate
└── quantization/
    ├── qwen3vl_calibration.py
    └── qwen3vl_llmcompressor.py
```

Main experiment configs live under `configs/vlm/` and `configs/vla/`. Historical and ablation
configs live under `configs/archive/`.

## Environments

Use separate environments for normal repo work, vLLM, and TensorRT-LLM side tests.

| Purpose | Environment |
| --- | --- |
| General tests and Pi0.5 LeRobot runs | `/root/autodl-tmp/envs/pi05` when present |
| vLLM benchmarks | `/root/autodl-tmp/envs/mm-edge-infer-accel-vllm` |
| TensorRT-LLM side tests | `/root/autodl-tmp/envs/qwen3vl-trtllm` |

The old `/root/miniconda3/envs/mm-edge-infer-accel` environment has been removed. If the Pi0.5
environment is not present on a host, recreate it before running LeRobot/Pi0.5 commands. Always set
`VLLM_USE_FLASHINFER_SAMPLER=0` for vLLM processes in this project.

## CLI

```bash
python -m mm_edge_infer_accel.cli <command>   [--config <yaml>]   [--concurrency <N>]   [--sample-count <N>]   [--sample-strategy first|stratified]   [--max-new-tokens <N>]   [--max-pixels <N>]   [--mode reset|queue]   [--episode <N> ...]   [--run]   [--dry-run]   [--output <json>]
```

Commands:

- `benchmark`: run an experiment only when `--run` is passed; otherwise print the plan.
- `quantize`: print the config-driven plan. Use `scripts/quant_qwen3vl4b_llmcompressor.py` for the retained AWQ/GPTQ path.
- `profile`: generate an `nsys`/`ncu` command string; it does not run the profiler.
- `env-check`: print local GPU/CUDA/package information.

Config validation maps model families to experiment type and backend:

| Family | Type | Backend |
| --- | --- | --- |
| `qwen3-vl`, `smolvlm2` | `vlm` | `vllm` |
| `pi0_fast`, `pi05` | `vla` | `lerobot` |

Using the wrong backend raises `ValueError` during config validation.

## Qwen3-VL Benchmarks

Canonical vLLM command:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 conda run --no-capture-output -p /root/autodl-tmp/envs/mm-edge-infer-accel-vllm   python -m mm_edge_infer_accel.cli benchmark   --config configs/vlm/qwen3vl_4b_awq_local.yaml   --concurrency 8   --run   --output outputs/qwen3vl_4b_awq_c8.json
```

Current Qwen3-VL-4B constraints and defaults:

- RTX 3080 Ti 12GB uses `max_model_len: 1024` for all 4B comparison runs.
- `mm_processor_kwargs.truncation: false` and `model.max_pixels: 602112` are required for image token count consistency.
- Main completed curve: BF16 / AWQ / GPTQ at concurrency `1, 2, 4, 8, 16, 32`.
- Current deployment-style default: AWQ or GPTQ at `concurrency=8`. BF16 is the quality baseline.

## Pi0.5 Benchmarks

Pi0.5 real LIBERO action inference is available through both the script and CLI.

Script form:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05   python scripts/run_pi05_action_inference.py   --model-id /root/autodl-tmp/hf_cache/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/<snapshot>   --source libero   --mode reset   --episode 0   --output outputs/pi05_libero_episode0_reset.json
```

CLI form:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05   python -m mm_edge_infer_accel.cli benchmark   --config configs/vla/pi05_libero.yaml   --mode reset   --episode 0   --run   --output outputs/pi05_libero_episode0_reset.json
```

Useful Pi0.5 environment variables:

| Variable | Purpose |
| --- | --- |
| `MM_EDGE_PI05_COMPILE` | Optional `torch.compile` path; default false and previously showed no benefit |
| `MM_EDGE_PI05_COMPILE_MODE` | Compile mode, default `reduce-overhead` |
| `MM_EDGE_PI05_NUM_INFERENCE_STEPS` | Override denoising step count |
| `MM_EDGE_PI05_TF32` | Legacy flag; Pi0.5 is bf16, so this does not change inference behavior |

The prefix KV cache optimization is enabled by default through `runtime.enable_prefix_kv_cache`.
It replaces `model.sample_actions` and caches the visual+text prefix KV cache once, so denoising
steps only process the action/noise suffix.

## TensorRT Side Tests

These scripts are exploratory and are not the main Qwen3-VL project path.

Qwen3-VL visual + merger/projector ONNX export:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm   python scripts/export_qwen3vl_vit_projector_onnx.py --help
```

Qwen3-0.6B text-only TensorRT-LLM engine side test:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm   python scripts/export_qwen3_0_6b_trtllm_engine.py --help

conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm   python scripts/run_qwen3_0_6b_trtllm_benchmark.py --help
```

For Qwen3-VL, the retained practical route is vLLM for the decoder plus optional TensorRT for the
visual+merger/projector module. Full TensorRT-LLM engine backend for Qwen3-VL is not the current
mainline in this repo.

## Profiling

The code contains NVTX ranges for the major runtime stages:

- VLM: `vlm_vllm_load_model`, `vlm_vllm_ocrbench_loop`, `vlm_vllm_warmup`, `preprocess`, `generate`, `decode`
- Pi0.5: `pi05_load_config_processors`, `pi05_load_policy`, `pi05_warmup`, `pi05_dataset_getitem`, `pi05_preprocess`, `pi05_policy_reset`, `pi05_select_action`, `pi05_postprocess`, `pi05_action_metrics`

Generate profiler command strings with:

```bash
python -m mm_edge_infer_accel.cli profile --config <yaml>
```

For Pi0.5 torch profiling, use:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05   python scripts/profile_pi05_torch.py --help
```

## Reports

Key reports:

- `reports/qwen3vl_4b_vllm_bf16_comparison.md`
- `reports/qwen3vl_4b_vllm_concurrency_curve.md`
- `reports/qwen3vl_2b_bf16_nsys_stratified100.md`
- `reports/pi05_lerobot_reference_profiling_rtx3080ti.md`
- `reports/pi05_prefix_kv_cache_optimization_rtx3080ti.md`
- `reports/pi05_orin_flashrt_experiment_report.md`
- `reports/qwen3vl_vit_projector_tensorrt_result.md`
- `reports/qwen3_0_6b_tensorrt_llm_result.md`

Current takeaways:

- Qwen3-VL-4B on RTX 3080 Ti should use AWQ/GPTQ with vLLM for serving-style runs.
- Qwen3-VL visual TensorRT is useful as a side optimization target, but the decoder remains vLLM in the practical path.
- Pi0.5 LeRobot reference benefits from prefix KV cache by reducing repeated prefix encoding inside the denoise loop.
- Pi0.5 FlashRT/Orin `cache2` is the retained runtime optimization; vitpack/token pooling is not retained as a correctness-preserving path.

## Testing

Run unit tests with the available project environment:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/mm-edge-infer-accel-vllm   python -m pytest tests/
```

If the Pi0.5/general environment is present, it can also run the test suite:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05   python -m pytest tests/
```

Expected coverage is lightweight and CPU-compatible for normal unit tests. GPU-heavy benchmark
outputs go under `outputs/`, which is gitignored.
