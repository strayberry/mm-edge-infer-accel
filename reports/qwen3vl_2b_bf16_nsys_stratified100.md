# Qwen3-VL-2B BF16 Stratified100 Profiling 报告

## 范围

本文分析 Qwen3-VL-2B BF16 在 `echo840/OCRBench` 分层抽样 100 条样本上的推理性能。

本报告合并两类数据：

```text
Nsight Systems:
~/autodl-tmp/profiling/qwen3vl_2b_bf16_nsys_stratified100.nsys-rep
~/autodl-tmp/profiling/qwen3vl_2b_bf16_nsys_stratified100.sqlite

Nsight 对应 benchmark:
outputs/qwen3vl_2b_bf16_ocrbench_stratified100_nsys.json

Generate 阶段拆分:
outputs/qwen3vl_2b_bf16_generate_breakdown_stratified100.json
```

运行配置：

```text
模型: Qwen/Qwen3-VL-2B-Instruct
加载精度: BF16
数据集: echo840/OCRBench
样本数: 100
采样方式: 按 question_type 分层抽样
Batch size: 1
max_new_tokens: 128
设备: RTX 3080 Ti 12GB
```

注意：Nsight Systems 结果用于观察系统级时间线、CUDA API、kernel 和 memcpy。`profile_generate` 结果用于拆分 `generate()` 内部的 prefill / decode。两者的绝对耗时不应完全等同，因为 Nsight tracing 会引入额外开销。

## 总览

| 指标 | Nsight benchmark | profile_generate |
| --- | ---: | ---: |
| Accuracy | 0.89 | 0.89 |
| Correct | 89 / 100 | 89 / 100 |
| Input tokens mean | 201.25 | 201.25 |
| Input tokens max | 607 | 607 |
| Generated tokens mean | 19.26 | 19.26 |
| Generated tokens max | 128 | 128 |
| Peak allocated memory | 4198.18 MB | 4198.18 MB |
| Peak reserved memory | 4280.0 MB | 4280.0 MB |
| Model load memory delta | 4186.0 MB | 4186.0 MB |
| Request latency mean | 893.65 ms | 669.39 ms |
| Request latency p50 | 263.73 ms | 200.65 ms |
| Request latency p90 | 2646.85 ms | 1988.60 ms |
| Request latency p95 | 3307.99 ms | 2446.97 ms |

两次运行的 accuracy、token 数和显存对齐，说明样本和生成结果一致。Nsight 下 latency 更高，主要是 profiling/tracing 带来的额外开销，因此阶段比例分析更依赖 `profile_generate`。

## Generate 阶段拆分

`profile_generate` 通过 monkey-patch `model.forward` 捕获原生 `model.generate()` 内部每次前向传播：

```text
prefill_forward: prompt + image 的第一次 forward
decode: 后续逐 token forward
generation_overhead: generate 总耗时 - forward 总耗时
```

分类校验结果：

| 指标 | 数值 |
| --- | ---: |
| forward_classification_warning_count | 0 |
| prefill_forward_ms_mean | 73.20 ms |
| prefill_forward_ms_p50 | 62.72 ms |
| prefill_forward_ms_p90 | 103.70 ms |
| decode_ms_mean | 541.31 ms |
| decode_ms_p50 | 118.04 ms |
| decode_ms_p90 | 1810.99 ms |
| decode_ms_p95 | 2311.87 ms |
| generate_ms_mean | 637.77 ms |
| generation_overhead_ms_mean | 23.27 ms |
| tpot_ms_mean | 29.71 ms |
| tpot_ms_p90 | 30.92 ms |

核心结论：

```text
当前瓶颈主要是 decode，不是 prefill。
TPOT 很稳定，尾延迟主要由 generated_tokens 数量决定。
```

平均每个 decode token 约 29-31 ms。慢样本不是单 token 变慢，而是输出 token 变多。

## Nsight Systems 视角

NVTX 汇总：

| Range | Instances | 总耗时 | 平均耗时 |
| --- | ---: | ---: | ---: |
| `vlm_ocrbench_loop` | 1 | 100.43 s | 100.43 s |
| `vlm_generate` | 100 | 86.02 s | 860.24 ms |
| `vlm_ttft` | 100 | 9.68 s | 96.83 ms |
| `vlm_load_model` | 1 | 4.87 s | 4.87 s |
| `vlm_preprocess` | 100 | 3.30 s | 33.00 ms |
| `vlm_warmup` | 1 | 1.07 s | 1.07 s |
| `vlm_decode` | 100 | 0.024 s | 0.24 ms |

Nsight 的系统级结论与 `profile_generate` 一致：`vlm_generate` 是主耗时，预处理和文本 decode 都不是当前主瓶颈。

CUDA API 汇总：

| API | Calls | Total Time |
| --- | ---: | ---: |
| `cudaLaunchKernel` | 3,500,147 | 22.40 s |
| `cudaMemcpyAsync` | 37,576 | 6.69 s |
| `cudaStreamSynchronize` | 26,007 | 0.48 s |
| `cudaMemsetAsync` | 57,696 | 0.40 s |

`cudaLaunchKernel` 次数很高，这是自回归 decode 的典型形态。每生成一个 token 都会触发大量小 kernel，因此长输出样本会带来明显尾延迟。

## CUDA Kernel 和 MemOps

GPU kernel 主要由以下类型构成：

```text
BF16 GEMM / GEMV
Flash Attention
elementwise / reduce
cat / copy
LayerNorm / activation
```

耗时最高的 kernel 仍然是 BF16 GEMM/GEMV 与 PyTorch 小算子组合。Nsight 中可以看到 Flash Attention kernel，说明 attention 路径没有退化到 naive attention。

MemOps 汇总：

| Operation | Count | Total Time |
| --- | ---: | ---: |
| Host-to-Device memcpy | 2361 | 1.42 s |
| CUDA memset | 57696 | 0.03 s |
| Device-to-Device memcpy | 11569 | 0.028 s |
| Device-to-Host memcpy | 23646 | 0.025 s |

H2D copy 不是总体主瓶颈，但存在长尾，最大 H2D copy 约 181 ms。后续如果做摄像头输入或 Jetson 部署，需要继续关注 CPU preprocessing、H2D copy 和 stream overlap。

## 按题型拆分

下面使用 `profile_generate` 的 100 sample 结果，按 OCRBench `question_type` 聚合。

| Question Type | N | Acc | Input Mean | Output Mean | Prefill Mean | Decode Mean | Generate Mean | Request Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Artistic Text Recognition | 10 | 0.90 | 137.4 | 2.7 | 88.9 ms | 51.4 ms | 164.8 ms | 180.0 ms |
| Digit String Recognition | 10 | 0.80 | 89.1 | 5.2 | 62.5 ms | 123.6 ms | 198.6 ms | 206.4 ms |
| Doc-oriented VQA | 10 | 0.90 | 579.7 | 62.8 | 103.1 ms | 1830.3 ms | 1984.0 ms | 2037.7 ms |
| Handwriting Recognition | 10 | 0.90 | 92.6 | 2.6 | 61.7 ms | 47.5 ms | 119.9 ms | 128.2 ms |
| Handwritten Mathematical Expression Recognition | 10 | 0.90 | 105.9 | 34.1 | 62.5 ms | 982.4 ms | 1077.5 ms | 1085.6 ms |
| Irregular Text Recognition | 10 | 1.00 | 89.5 | 2.3 | 62.1 ms | 39.7 ms | 112.4 ms | 120.3 ms |
| Key Information Extraction | 10 | 1.00 | 593.9 | 15.1 | 104.2 ms | 419.4 ms | 542.8 ms | 733.2 ms |
| Non-Semantic Text Recognition | 10 | 1.00 | 91.3 | 4.5 | 61.4 ms | 104.1 ms | 177.3 ms | 185.3 ms |
| Regular Text Recognition | 10 | 1.00 | 90.4 | 2.1 | 62.4 ms | 32.8 ms | 105.9 ms | 113.7 ms |
| Scene Text-centric VQA | 10 | 0.50 | 142.7 | 61.2 | 63.3 ms | 1781.8 ms | 1894.5 ms | 1903.5 ms |

主要观察：

1. **Doc-oriented VQA 和 Scene Text-centric VQA 主导尾延迟。**
   它们平均输出约 60 个 token，decode 接近 1.8 s。

2. **KIE 的输入 token 很多，但延迟没有 Doc VQA 高。**
   KIE 平均输入 593.9 token，prefill 约 104 ms，但输出只有 15.1 token，因此总延迟明显低于 Doc VQA。

3. **普通 OCR 类任务很快。**
   Regular、Irregular、Handwriting 等类别输出只有 2-3 token，请求平均延迟约 110-130 ms。

4. **Scene Text-centric VQA 准确率最低。**
   该类别 accuracy 只有 0.50，同时输出长，说明它既慢又不稳定，后续需要单独看 prompt 和评测规则。

## 最慢样本

| Type | Request | Output Tokens | Input Tokens | Correct |
| --- | ---: | ---: | ---: | --- |
| Doc-oriented VQA | 3979.4 ms | 124 | 577 | true |
| Scene Text-centric VQA | 3960.3 ms | 128 | 209 | true |
| Doc-oriented VQA | 3889.3 ms | 128 | 593 | false |
| Doc-oriented VQA | 3265.2 ms | 103 | 577 | true |
| Handwritten Mathematical Expression Recognition | 2731.8 ms | 88 | 141 | true |
| Scene Text-centric VQA | 2447.0 ms | 78 | 85 | false |
| Scene Text-centric VQA | 2175.8 ms | 70 | 181 | false |
| Doc-oriented VQA | 2158.7 ms | 67 | 574 | true |

慢样本几乎都由长输出造成。多个样本接近或达到 `max_new_tokens=128`，这会线性增加 decode 时间。

## 结论

当前 Qwen3-VL-2B BF16 在 12GB 单卡上显存压力不大：

```text
模型加载显存增量: 4186 MB
peak allocated: 4198 MB
peak reserved: 4280 MB
```

真正限制 latency 的不是显存，也不是 prefill，而是长答案 decode。

最关键结论：

1. `generate()` 是主耗时来源。
2. `prefill_forward` 平均只有 73 ms，p90 约 104 ms。
3. `decode` 平均 541 ms，p90 约 1811 ms，是尾延迟主要来源。
4. TPOT 稳定在约 30 ms，说明慢样本主要是输出 token 多。
5. Doc VQA、Scene VQA、公式识别是主要慢类别。
6. Nsight 显示 kernel launch 数量很高，符合自回归逐 token decode 的特征。

## 下一步优化方向优先级

基于本次 profiling，后续优化应围绕 Qwen3-VL-2B 当前 decode 瓶颈展开，并且不通过降低 `max_new_tokens` 或改短答 prompt 来规避长输出场景。

| Priority | 方向 | 目的 | 主要观察指标 |
| --- | --- | --- | --- |
| P0 | Speculative Decoding | 不改变输出长度，直接针对长输出 decode 加速 | acceptance rate、TPOT、request latency、额外显存 |
| P1 | KV Cache 量化 / 管理 | 面向长上下文和长输出的 cache 显存/带宽优化 | KV cache memory、peak memory、TPOT、输出一致性 |
| P1 | TensorRT / 推理引擎优化 | 做局部模块或 vision encoder 的部署可行性验证 | ONNX/TRT 可导出性、模块 latency、精度偏差 |
| P2 | max_pixels / visual token control | 主要优化 prefill，作为后续边缘部署准备 | input tokens、prefill、accuracy、H2D/preprocess |

后续已经完成 `Qwen3-VL-4B BF16` 的 vLLM 对比报告，见
`reports/qwen3vl_4b_vllm_bf16_comparison.md`。旧的 bnb4 结果来自
Transformers/bitsandbytes 路径，不再作为当前 vLLM 主对比口径。下一步优化重点应回到
decode 加速、KV cache 管理、视觉 token 预算 sweep，以及 VLA baseline。
