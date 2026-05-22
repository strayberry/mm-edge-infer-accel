# Multimodal Edge Inference Acceleration and Benchmarking

This repo studies **edge-oriented multimodal inference acceleration and benchmarking**. It focuses on how VLM and VLA models behave under constrained GPU memory and deployment-oriented latency budgets, including quantization, concurrent serving, throughput-latency tradeoffs, and robot action inference stability.

The current implementation is an early-stage research/engineering toolkit, not a production server. The experiments are split into two tracks:

- **VLM online inference**: Qwen3-VL OCRBench benchmarking through vLLM, with BF16/AWQ/GPTQ comparison, quantization helpers, concurrency analysis, and throughput-latency tradeoffs.
- **VLA edge inference**: Pi0.5/LeRobot LIBERO action inference and profiling for reset-vs-queue stability analysis, followed by Jetson-oriented low-latency deployment experiments.

The VLA track started with Pi0.5 debugging on an RTX 3080 Ti 12GB before the target edge device was available. It then moved to Jetson AGX Orin 32G (`SM count: 14`) for edge profiling and optimization. Those Orin experiments use [FlashRT](https://github.com/LiangSu8899/FlashRT), a high-performance real-time AI inference runtime designed for low-latency, small-batch edge workloads. This project has already contributed FlashRT Pi0.5/Orin fixes and optimizations that were accepted upstream:

- [FlashRT PR #31](https://github.com/LiangSu8899/FlashRT/pull/31)
- [FlashRT PR #40](https://github.com/LiangSu8899/FlashRT/pull/40)
- [FlashRT PR #42](https://github.com/LiangSu8899/FlashRT/pull/42)

Future VLA edge optimizations may continue to build on this open-source runtime where a dedicated low-latency inference path is more appropriate than the reference LeRobot path.

## Repository Layout

```text
configs/
  vlm/                         Main VLM benchmark YAML configs
  vla/                         Main VLA / LeRobot benchmark YAML configs

mm_edge_infer_accel/
  cli.py                       Command-line entrypoint
  config.py                    YAML loading and validation
  common.py                    Shared config/result helpers
  datasets.py                  OCRBench dataset sampling utilities
  env.py                       Environment/system inspection
  metrics.py                   Shared metric helpers
  vlm.py                       VLM benchmark orchestration
  vla.py                       VLA benchmark dispatch and action metrics
  vla_lerobot.py               LeRobot-backed VLA load-only path
  pi05_runtime.py              Pi0.5 policy loading and LIBERO action inference
  profiling.py                 NVTX helpers and nsys/ncu command generation
  runners/
    vllm_runner.py             vLLM model loading and batched generation
  quantization/
    qwen3vl_calibration.py     Calibration sample preparation
    qwen3vl_llmcompressor.py   AWQ/GPTQ quantization logic

scripts/
  analyze_qwen3vl4b_maxnew64_by_category.py
                               Offline Qwen3-VL category analysis helper
  export_libero_npz.py         Export LeRobot LIBERO frames for offline eval
  profile_pi05_torch.py        Pi0.5 torch.profiler entrypoint
  quant_qwen3vl4b_llmcompressor.py
                               Thin AWQ/GPTQ quantization entrypoint
  run_pi05_action_inference.py Thin Pi0.5 synthetic/LIBERO action runner
  sweep_pi05_inference_steps.py
                               Pi0.5 denoising-step sweep helper

tests/                         CPU-friendly tests for config, CLI, datasets,
                               quantization helpers, analysis, and action metrics

reports/                       Experiment reports
outputs/                       Local benchmark JSON outputs; gitignored
profiling/                     Local profiler artifacts; gitignored
```

## Core Concepts

Experiments are described by YAML configs under `configs/vlm/` and `configs/vla/`. Each config declares a model family and backend. Validation enforces the expected pairing:

| Model family           | Workload | Backend   |
| ---------------------- | -------- | --------- |
| `qwen3-vl`, `smolvlm2` | VLM      | `vllm`    |
| `pi0_fast`, `pi05`     | VLA      | `lerobot` |

Main configs are intentionally kept small. Use CLI overrides for common experiment variants such as sample count, sample strategy, max new tokens, max pixels, and concurrency.

The main CLI is:

```bash
python -m mm_edge_infer_accel.cli <command> [--config <yaml>] [--concurrency <N>] [--sample-count <N>] [--sample-strategy first|stratified] [--max-new-tokens <N>] [--max-pixels <N>] [--run] [--dry-run] [--output <json>]
```

Commands:

- `benchmark`: load a config and optionally execute it with `--run`.
- `quantize`: print a config-driven quantization plan. The retained Qwen3-VL-4B quantization path uses `scripts/quant_qwen3vl4b_llmcompressor.py`.
- `profile`: generate an `nsys` or `ncu` command string; it does not run the profiler.
- `env-check`: print environment and package information.

Main configs currently kept in the active config directories:

```text
configs/vlm/qwen3vl_4b_bf16.yaml
configs/vlm/qwen3vl_4b_awq_local.yaml
configs/vlm/qwen3vl_4b_gptq_local.yaml
configs/vlm/smolvlm2_2b_fp32.yaml
configs/vla/pi05_libero.yaml
```

## Implemented Workloads

### VLM: OCRBench with vLLM

The VLM path currently supports `echo840/OCRBench` through the `datasets` library. Sampling supports:

- `first`: first `N` examples.
- `stratified`: stratified by `question_type`.

The vLLM runner supports:

- BF16 and local quantized checkpoints.
- Runtime concurrency through `--concurrency`.
- Request-level latency, throughput, token throughput, TTFT, and failure metrics.
- Per-sample outputs plus aggregate metrics in JSON.

For Qwen3-VL-4B on RTX 3080 Ti 12GB, the current fair-comparison constraint is:

```yaml
runtime:
  max_model_len: 1024
  mm_processor_kwargs:
    truncation: false
model:
  max_pixels: 602112
```

`truncation: false` avoids image-token mismatch caused by local tokenizer truncation settings. BF16 cannot start with `max_model_len: 2048` on the 12GB card.

### VLA: Pi0.5 LIBERO Action Inference

The VLA path uses native LeRobot for Pi0.5 instead of forcing it through vLLM. The implemented pieces are:

- Pi0.5 policy load-only check through the CLI.
- Synthetic action sanity check.
- Real LIBERO action inference through `scripts/run_pi05_action_inference.py`.
- `reset` mode: clear the action queue before each frame and force a new chunk prediction.
- `queue` mode: keep Pi0.5's internal action queue and measure realistic control-loop action output.
- Action MAE, cosine similarity, latency, loop Hz, and chunk prediction count.

Pi0.5 real LIBERO action inference is intentionally script-backed for now; it is not yet wired into `python -m mm_edge_infer_accel.cli benchmark`.

## Installation

Use separate environments. The VLM/vLLM stack, quantization stack, and Pi0.5/LeRobot stack have different dependency constraints.

General development and tests:

```bash
python -m pip install -e ".[dev]"
```

Quantization helpers:

```bash
python -m pip install -e ".[quant]"
```

vLLM benchmark environment:

```bash
python -m pip install -e ".[vllm]"
```

Optional dependency groups are defined in `pyproject.toml`:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest and Ruff |
| `quant` | Transformers, datasets, LLM Compressor, bitsandbytes, Qwen-VL utilities |
| `vllm` | vLLM and VLM benchmark dependencies |

Pi0.5 should be installed in a separate LeRobot-compatible environment. The current validated stack uses Python 3.12 and a newer Transformers/OpenPI-compatible LeRobot path.

## Common Commands

Always disable the FlashInfer sampler for current vLLM runs:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Print a benchmark plan without executing:

```bash
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_awq_local.yaml
```

Run a VLM benchmark:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_awq_local.yaml \
  --concurrency 8 \
  --run \
  --output outputs/qwen3vl_4b_awq_vllm_ocrbench_stratified100_c8.json
```

Run a full first-1000 OCRBench evaluation without a separate YAML:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_awq_local.yaml \
  --sample-count 1000 \
  --sample-strategy first \
  --concurrency 8 \
  --run \
  --output outputs/qwen3vl_4b_awq_vllm_ocrbench_first1000_c8.json
```

Run a max-token or max-pixel ablation without a separate YAML:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_bf16.yaml \
  --sample-count 100 \
  --sample-strategy stratified \
  --max-new-tokens 64 \
  --max-pixels 501760 \
  --run \
  --output outputs/qwen3vl_4b_bf16_maxnew64_maxpixels501760.json
```

Generate an Nsight Systems command:

```bash
python -m mm_edge_infer_accel.cli profile \
  --tool nsys \
  --config configs/vlm/qwen3vl_4b_awq_local.yaml
```

Run Qwen3-VL-4B LLM Compressor quantization:

```bash
python scripts/quant_qwen3vl4b_llmcompressor.py \
  --method gptq \
  --calib-source docvqa \
  --docvqa-dataset-id lmms-lab/DocVQA \
  --docvqa-config DocVQA \
  --docvqa-split validation \
  --max-calib-samples 128 \
  --max-calib-seq-len 1024 \
  --calib-max-pixels 602112 \
  --output /path/to/Qwen3-VL-4B-Instruct-GPTQ-local
```

Run Pi0.5 LIBERO action inference:

```bash
HF_HUB_DISABLE_XET=1 \
python scripts/run_pi05_action_inference.py \
  --source libero \
  --episode 0 \
  --sample-count 100 \
  --mode queue \
  --warmup 3 \
  --output outputs/pi05_libero_action_inference_100_queue.json
```

Run tests:

```bash
python -m pytest tests/
```

## Reports

Reports summarize completed experiments. The most useful entry points are:

- `reports/qwen3vl_4b_vllm_bf16_comparison.md`
- `reports/qwen3vl_4b_vllm_concurrency_curve.md`
- `reports/pi05_orin_flashrt_experiment_report.md`
- `reports/pi05_lerobot_reference_profiling_rtx3080ti.md`
- `reports/pi05_prefix_kv_cache_optimization_rtx3080ti.md`

`outputs/` and `profiling/` are local artifacts and are gitignored.

## Current Takeaways

- Qwen3-VL-4B BF16 remains the quality baseline on OCRBench, but it is not the preferred 12GB serving default under high concurrency.
- AWQ/GPTQ are the practical deployment candidates for Qwen3-VL-4B vLLM serving. The current default recommendation for serving-style runs is `concurrency=8`.
- Pi0.5 should use its action queue for control-loop execution. Per-frame reset mode is useful as a conservative baseline, but it is too slow for 10 fps LIBERO control.

## Development Notes

- Use structured YAML configs for experiments instead of hard-coded runner logic.
- Keep VLM serving experiments and VLA control-loop experiments separate; their runtime stacks and success metrics are different.
- Treat TensorRT artifacts built on RTX 3080 Ti as local validation only. Jetson deployment requires rebuilding under the target JetPack/TensorRT version.
- Add new reports only after the underlying experiment has been run and the supporting output or profiling artifact is available.
