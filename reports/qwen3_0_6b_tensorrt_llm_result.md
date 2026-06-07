# Qwen3-0.6B TensorRT-LLM 实验结果

日期：2026-06-07

## 实验范围

本轮只验证 text-only `Qwen/Qwen3-0.6B`，不包含 Qwen3-VL、多模态 projector、OCRBench 或端到端 VLM。

目标是先确认小型 Qwen3 CausalLM 是否可以走 TensorRT-LLM TensorRT engine backend，并和同机 vLLM BF16 baseline 做 latency 对比。

## 模型与环境

- 模型：`/root/autodl-tmp/models/Qwen3-0.6B`
- HF architecture：`Qwen3ForCausalLM`
- HF dtype：`bfloat16`
- TensorRT-LLM：`1.2.1`
- TensorRT-LLM 环境：`/root/autodl-tmp/envs/qwen3vl-trtllm`
- vLLM 环境：`/root/autodl-tmp/envs/mm-edge-infer-accel-vllm`
- 对比口径：8 条英文 prompt，单条顺序生成，`max_new_tokens=64`

## 产物

side-test 脚本：

```text
scripts/export_qwen3_0_6b_trtllm_engine.py
scripts/run_qwen3_0_6b_vllm_benchmark.py
scripts/run_qwen3_0_6b_trtllm_benchmark.py
```

TensorRT-LLM checkpoint：

```text
outputs/tensorrt/qwen3_0_6b_trtllm_bf16_ckpt/
```

大小：

```text
1.5G
```

TensorRT-LLM BF16 engine：

```text
outputs/tensorrt/qwen3_0_6b_trtllm_bf16_engine/
```

大小：

```text
1.5G
```

TensorRT-LLM W8A16 checkpoint / engine：

```text
outputs/tensorrt/qwen3_0_6b_trtllm_w8a16_ckpt/
outputs/tensorrt/qwen3_0_6b_trtllm_w8a16_engine/
```

大小：

```text
1.1G
```

benchmark 输出：

```text
outputs/qwen3_0_6b_vllm_bf16_baseline.json
outputs/qwen3_0_6b_vllm_bf16_optimized.json
outputs/qwen3_0_6b_trtllm_bf16_engine_baseline.json
outputs/qwen3_0_6b_trtllm_w8a16_engine_baseline.json
```

## 构建结果

HF 权重已成功转换为 TensorRT-LLM checkpoint。

BF16 导出与构建命令：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm \
  python scripts/export_qwen3_0_6b_trtllm_engine.py
```

W8A16 / INT8 weight-only 导出与构建命令：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm \
  python scripts/export_qwen3_0_6b_trtllm_engine.py \
  --quant-algo w8a16 \
  --checkpoint-dir outputs/tensorrt/qwen3_0_6b_trtllm_w8a16_ckpt \
  --engine-dir outputs/tensorrt/qwen3_0_6b_trtllm_w8a16_engine
```

engine build 配置：

```text
max_batch_size=1
max_input_len=1024
max_seq_len=1088
max_num_tokens=1024
gpt_attention_plugin=bfloat16
gemm_plugin=bfloat16
context_fmha=enable
remove_input_padding=enable
kv_cache_type=paged
```

构建结果：

| item | value |
| --- | ---: |
| TensorRT engine generation | 12.6 s |
| TensorRT-LLM total build time | 18 s |
| TRT weights memory | 1524 MB |
| TRT max scratch memory | 37.8 MB |
| Build phase peak memory | 5799 MB |
| engine size | 1.5G |

W8A16 构建结果：

| item | value |
| --- | ---: |
| TensorRT engine generation | 11.8 s |
| TensorRT-LLM total build time | 16 s |
| TRT weights memory | 1085 MB |
| TRT max scratch memory | 37.8 MB |
| Build phase peak memory | 4628 MB |
| engine size | 1.1G |

构建时 `use_fp8_context_fmha` 被自动关闭，因为当前不是 FP8 quantization workflow。这是预期行为。

## Latency 对比

vLLM eager benchmark 命令：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
conda run --no-capture-output -p /root/autodl-tmp/envs/mm-edge-infer-accel-vllm \
  python scripts/run_qwen3_0_6b_vllm_benchmark.py
```

vLLM optimized benchmark 命令：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
conda run --no-capture-output -p /root/autodl-tmp/envs/mm-edge-infer-accel-vllm \
  python scripts/run_qwen3_0_6b_vllm_benchmark.py \
  --disable-enforce-eager \
  --output outputs/qwen3_0_6b_vllm_bf16_optimized.json
```

TensorRT-LLM BF16 engine benchmark 命令：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm \
  python scripts/run_qwen3_0_6b_trtllm_benchmark.py
```

TensorRT-LLM W8A16 engine benchmark 命令：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/qwen3vl-trtllm \
  python scripts/run_qwen3_0_6b_trtllm_benchmark.py \
  --engine-dir outputs/tensorrt/qwen3_0_6b_trtllm_w8a16_engine \
  --quant-algo w8a16 \
  --output outputs/qwen3_0_6b_trtllm_w8a16_engine_baseline.json
```

| backend | mode | dtype | load time | mean latency | p50 latency | p95 latency | output tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | eager | BF16 | 11.36 s | 1424.21 ms | 1428.68 ms | 1460.66 ms | 44.94 |
| vLLM | compile/CUDA Graph | BF16 | 58.76 s | 170.39 ms | 170.46 ms | 172.00 ms | 375.61 |
| TensorRT-LLM engine | TensorRT engine | BF16 | 3.30 s | 223.42 ms | 220.60 ms | 231.07 ms | 286.45 |
| TensorRT-LLM engine | TensorRT engine | W8A16 | 2.91 s | 201.43 ms | 203.99 ms | 205.43 ms | 317.73 |

按 mean latency 计算：

```text
TensorRT-LLM W8A16 vs vLLM eager: 1424.21 / 201.43 = 7.07x faster
vLLM optimized vs TensorRT-LLM W8A16: 201.43 / 170.39 = 1.18x faster
TensorRT-LLM W8A16 vs TensorRT-LLM BF16: 223.42 / 201.43 = 1.11x faster
```

## 结论

- `Qwen3-0.6B` text-only 路线可以走 TensorRT-LLM TensorRT engine backend。
- TensorRT-LLM BF16/W8A16 engine 明显快于 vLLM eager baseline，但不快于本轮 vLLM optimized baseline。
- vLLM optimized 的单请求 latency 最好，但首次 load/compile/CUDA Graph capture 成本明显更高：`58.76 s`。
- TensorRT-LLM W8A16 相比 BF16 engine 有小幅 latency 收益，并把 checkpoint/engine 体积从 `1.5G` 降到 `1.1G`。
- TensorRT-LLM engine 的优势是启动更快、engine 路径确定、部署形态更接近 TensorRT；当前 W8A16 latency 仍不是最优。
- 该结论不能直接外推到 Qwen3-VL；Qwen3-VL 的 TensorRT-LLM TensorRT engine backend 当前仍不是已跑通路径。

## 已知注意事项

- 当前 benchmark 是 side-test 脚本跑出的固定 8 prompt microbenchmark，不接入项目 CLI/config 主线。
- vLLM eager baseline 使用 `enforce_eager=True`，日志显示 CUDA Graph/torch.compile 没启用；它只代表保守 baseline。
- vLLM optimized baseline 使用 `--disable-enforce-eager`，日志确认 `torch.compile` 和 CUDA Graph capture 生效。
- side-test 脚本会在进程内补 TensorRT-LLM 环境的 `LD_LIBRARY_PATH`，避免找不到 `libcublasLt.so.13`。
