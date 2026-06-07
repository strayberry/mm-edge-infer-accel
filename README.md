# Multimodal Edge Inference Acceleration

This repository is a research and engineering toolkit for edge-oriented multimodal inference. It benchmarks how VLM and VLA models behave under constrained GPU memory, low-latency deployment requirements, and robotics control-loop constraints.

The project is not a production server. It is organized around reproducible experiments, reports, and thin scripts that make before/after optimization results explicit.

## Current Tracks

| Track | Runtime | Status | Main output |
| --- | --- | --- | --- |
| Qwen3-VL VLM serving | vLLM | Active baseline complete | OCRBench accuracy, latency, concurrency |
| Pi0.5 LeRobot reference | LeRobot / PyTorch | Reference profiling complete | LIBERO action latency, reset vs queue, prefix KV cache |
| Pi0.5 FlashRT / Orin | FlashRT | Orin validation complete | BF16/cache2/INT8/vitpack action-similarity results |
| TensorRT side tests | TensorRT / TensorRT-LLM | Side experiments only | Visual-module and small text-only feasibility checks |

Main rule: keep VLM serving experiments and VLA control-loop experiments separate. They use different runtimes, datasets, metrics, and success criteria.

## Repository Layout

```text
configs/
  vlm/                         Active VLM benchmark YAML configs
  vla/                         Active VLA / LeRobot benchmark YAML configs

mm_edge_infer_accel/
  cli.py                       CLI dispatch for benchmark/quantize/profile/env-check
  config.py                    YAML loading and validation
  common.py                    Shared quantization/profile helpers
  datasets.py                  OCRBench loading and sampling helpers
  env.py                       Environment and package inspection
  metrics.py                   Shared latency and accuracy helpers
  profiling.py                 NVTX helpers and profiler command generation
  vlm.py                       VLM benchmark orchestration
  vla.py                       VLA dispatch and action metrics
  vla_lerobot.py               LeRobot-backed Pi0.5 benchmark path
  pi05_runtime.py              Pi0.5 policy loading and LIBERO action inference
  pi05_optimizations.py        Pi0.5 prefix KV cache optimization patch
  runners/vllm_runner.py       vLLM model loading and generation wrapper
  quantization/                Qwen3-VL calibration and LLM Compressor helpers

scripts/
  quant_qwen3vl4b_llmcompressor.py
  run_pi05_action_inference.py
  export_libero_npz.py
  profile_pi05_torch.py
  sweep_pi05_inference_steps.py
  analyze_qwen3vl4b_maxnew64_by_category.py

tests/                         CPU-friendly unit tests
reports/                       Completed experiment reports
outputs/                       Local benchmark JSON outputs; gitignored
profiling/                     Local profiler artifacts; gitignored
```

## Config Model

Experiments are YAML-driven. `model.family` determines workload type and allowed backend:

| Model family | Workload | Backend |
| --- | --- | --- |
| `qwen3-vl`, `smolvlm2` | VLM | `vllm` |
| `pi0_fast`, `pi05` | VLA | `lerobot` |

Active configs:

```text
configs/vlm/qwen3vl_4b_bf16.yaml
configs/vlm/qwen3vl_4b_awq_local.yaml
configs/vlm/qwen3vl_4b_gptq_local.yaml
configs/vlm/smolvlm2_2b_fp32.yaml
configs/vla/pi05_libero.yaml
```

Historical and ablation configs are intentionally kept out of the active config directories. Prefer CLI overrides for sample count, max new tokens, max pixels, concurrency, Pi0.5 mode, and episodes.

## CLI

```bash
python -m mm_edge_infer_accel.cli <command> [options]
```

Commands:

- `benchmark`: print or run a config-driven benchmark.
- `quantize`: print the config-driven quantization plan.
- `profile`: generate an `nsys` or `ncu` command string; it does not run the profiler.
- `env-check`: print system, GPU, CUDA, and package information.

Common benchmark options:

```text
--config <yaml>
--run
--output <json>
--concurrency <N>
--sample-count <N>
--sample-strategy first|stratified
--max-new-tokens <N>
--max-pixels <N>
--mode reset|queue
--episode <N>
```

## Environment Notes

Use separate environments for incompatible runtime stacks.

Recommended local environments on this machine:

| Purpose | Environment |
| --- | --- |
| vLLM benchmarks and current unit tests | `/root/autodl-tmp/envs/mm-edge-infer-accel-vllm` |
| TensorRT-LLM side tests | `/root/autodl-tmp/envs/qwen3vl-trtllm` |
| Pi0.5 / LeRobot reference | Create or use a LeRobot-compatible env on the target machine |

The older `/root/miniconda3/envs/mm-edge-infer-accel` env is no longer used. The previously documented `/root/autodl-tmp/envs/pi05` env may need to be recreated before Pi0.5 LeRobot runs on this host.

Always disable the FlashInfer sampler for current vLLM runs:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Install editable extras as needed:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[vllm]"
python -m pip install -e ".[quant]"
```

## Qwen3-VL VLM Benchmarks

The VLM path currently supports `echo840/OCRBench` through the `datasets` library.

Supported sampling modes:

- `first`: first `N` samples.
- `stratified`: round-robin by `question_type`.

Current Qwen3-VL-4B constraint on RTX 3080 Ti 12GB:

```yaml
runtime:
  max_model_len: 1024
  mm_processor_kwargs:
    truncation: false
model:
  max_pixels: 602112
```

`max_model_len: 2048` does not fit BF16 reliably on the 12GB card. `truncation: false` avoids local processor truncation changing the image-token count.

Run a Qwen3-VL benchmark:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python -m mm_edge_infer_accel.cli benchmark   --config configs/vlm/qwen3vl_4b_awq_local.yaml   --sample-count 100   --sample-strategy stratified   --concurrency 8   --run   --output outputs/qwen3vl_4b_awq_stratified100_c8.json
```

Run a full first-1000 accuracy pass:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python -m mm_edge_infer_accel.cli benchmark   --config configs/vlm/qwen3vl_4b_gptq_local.yaml   --sample-count 1000   --sample-strategy first   --concurrency 8   --run   --output outputs/qwen3vl_4b_gptq_first1000_c8.json
```

Run Qwen3-VL-4B LLM Compressor quantization:

```bash
python scripts/quant_qwen3vl4b_llmcompressor.py   --method gptq   --calib-source docvqa   --docvqa-dataset-id lmms-lab/DocVQA   --docvqa-config DocVQA   --docvqa-split validation   --max-calib-samples 128   --max-calib-seq-len 1024   --calib-max-pixels 602112   --output /path/to/Qwen3-VL-4B-Instruct-GPTQ-local
```

## Pi0.5 / LIBERO Benchmarks

Pi0.5 uses native LeRobot/PyTorch, not vLLM.

Implemented paths:

- CLI benchmark through `configs/vla/pi05_libero.yaml`.
- Thin script entrypoint through `scripts/run_pi05_action_inference.py`.
- `reset` mode: reset the action queue and force a new chunk prediction per frame.
- `queue` mode: keep the internal action queue and measure realistic control-loop output.
- Prefix KV cache optimization controlled by `runtime.enable_prefix_kv_cache`.

Run Pi0.5 LIBERO through the CLI:

```bash
HF_HUB_DISABLE_XET=1 python -m mm_edge_infer_accel.cli benchmark   --config configs/vla/pi05_libero.yaml   --mode queue   --episode 0   --sample-count 100   --run   --output outputs/pi05_libero_ep0_queue_100.json
```

Run the thin script directly:

```bash
HF_HUB_DISABLE_XET=1 python scripts/run_pi05_action_inference.py   --model-id /path/to/pi05_libero_finetuned_v044   --source libero   --episode 0   --sample-count 100   --mode queue   --warmup 3   --output outputs/pi05_libero_action_inference_100_queue.json
```

## Profiling

Generate an Nsight Systems command:

```bash
python -m mm_edge_infer_accel.cli profile   --tool nsys   --config configs/vlm/qwen3vl_4b_awq_local.yaml
```

Generate an Nsight Compute command:

```bash
python -m mm_edge_infer_accel.cli profile   --tool ncu   --config configs/vlm/qwen3vl_4b_awq_local.yaml
```

The CLI only prints profiler commands. Run them manually after checking permissions and target-device support.

## Reports

Useful report entry points:

| Report | Purpose |
| --- | --- |
| `reports/qwen3vl_4b_vllm_bf16_comparison.md` | Qwen3-VL-4B BF16 comparison notes |
| `reports/qwen3vl_4b_vllm_concurrency_curve.md` | BF16/AWQ/GPTQ concurrency curve |
| `reports/pi05_lerobot_reference_profiling_rtx3080ti.md` | Pi0.5 reference profiling on RTX 3080 Ti |
| `reports/pi05_prefix_kv_cache_optimization_rtx3080ti.md` | Pi0.5 prefix KV cache optimization result |
| `reports/pi05_orin_flashrt_experiment_report.md` | Pi0.5 FlashRT / Jetson AGX Orin validation |

Add reports only after the underlying experiment has been run and the output/profiling artifact exists. `outputs/` and `profiling/` are local artifacts and are gitignored.

## Current Takeaways

- Qwen3-VL-4B BF16 is the quality baseline, but AWQ/GPTQ are the practical 12GB deployment candidates.
- For Qwen3-VL-4B serving-style runs on RTX 3080 Ti, the current default comparison point is `concurrency=8`.
- Pi0.5 reference inference benefits from prefix KV cache, but its PyTorch/LeRobot path is still not the low-latency edge deployment path.
- Pi0.5 FlashRT / Orin validation showed `cache2` is the main retained optimization direction; token pooling/vitpack-style spatial compression is not the mainline.
- TensorRT artifacts built on RTX should be treated as local validation only. Jetson deployment requires rebuilding under the target JetPack/TensorRT version.

## Testing

Current available test environment on this machine:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/mm-edge-infer-accel-vllm   python -m pytest tests/
```

Generic command if dependencies are installed in the active environment:

```bash
python -m pytest tests/
```
