# Qwen3-VL-4B vLLM BF16 对比报告

## 一句话结论

在当前 RTX 3080 Ti 12GB、vLLM、OCRBench、batch size 1 的设置下，**BF16 是最好的主
baseline**。它在 full OCRBench 1000 上 accuracy 最高，端到端 latency 和 tokens/s 也最好。

AWQ/GPTQ 当前更适合作为部署候选，而不是 BF16 的速度替代：

```text
BF16: accuracy 0.863, request mean 1230.7 ms, tokens/s 28.56
AWQ : accuracy 0.855, request mean 1337.4 ms, tokens/s 28.29
GPTQ: accuracy 0.854, request mean 1324.4 ms, tokens/s 28.24
```

## 应该怎么读这份结果

本报告有两个层级：

1. **Full OCRBench 1000** 是最终判断依据。
   它覆盖完整 test split，样本数足够大，更适合比较 accuracy 和端到端 latency。

2. **Stratified-100** 只是快速检查。
   它用于确认配置是否正常、快速观察趋势，但由于每类只有约 10 条样本，accuracy 会有波动。

因此，虽然 GPTQ 在 stratified100 上出现过 0.90 accuracy，高于 BF16 的 0.87，但在完整
1000 条样本上 BF16 仍然最高。最终结论以 1000-sample 为准。

## 实验口径

本文比较 Qwen3-VL-4B 在 vLLM 后端下的三种配置：

| 配置 | 含义 | 当前用途 |
| --- | --- | --- |
| BF16 | 原始 BF16 checkpoint | 主 baseline |
| AWQ | vLLM 加载的本地 AWQ checkpoint | 低比特部署候选 |
| GPTQ | DocVQA validation calibration 后的 GPTQ checkpoint | 低比特部署候选 |

三组 1000-sample 运行使用相同评测配置：

```text
数据集: echo840/OCRBench test split
样本数: 1000
Runtime: vLLM
设备: RTX 3080 Ti 12GB
batch size: 1
model.max_pixels: 602112
runtime.mm_processor_kwargs.truncation: false
runtime.max_model_len: 1024
max_new_tokens: 128
temperature: 0.0
profile.warmup: 1
```

注意：

1. Full OCRBench 1000 是最终判断依据；stratified100 只用于快速检查配置和趋势。
2. Stratified100 是固定采样：按 `question_type` 分组、类别名排序、每类轮流取样，没有随机种子。
3. Full 1000 结果来自历史输出，当时 `warmup: 0`。当前 config 已改为 `warmup: 1`；paired comparison 显示 warmup 不改变输出。
4. Stratified100 已在新增 vLLM internal metrics 后重跑；这些输出包含外部 wall-clock latency 和 vLLM queue / prefill / decode / inference 拆分。

主配置和结果文件：

```text
configs/vlm/qwen3vl_4b_bf16.yaml
configs/vlm/qwen3vl_4b_awq_local.yaml
configs/vlm/qwen3vl_4b_gptq_local.yaml

outputs/qwen3vl_4b_bf16_vllm_ocrbench_1000.json
outputs/qwen3vl_4b_awq_vllm_ocrbench_1000.json
outputs/qwen3vl_4b_gptq_vllm_ocrbench_1000.json
outputs/qwen3vl_4b_bf16_maxnew64_ocrbench_1000.json

outputs/qwen3vl_4b_bf16_vllm_ocrbench_stratified100.json
outputs/qwen3vl_4b_awq_vllm_ocrbench_stratified100.json
outputs/qwen3vl_4b_gptq_vllm_ocrbench_stratified100.json
```

## 为什么统一用 max_model_len 1024

BF16 在 RTX 3080 Ti 12GB 上无法用 `max_model_len: 2048` 启动。vLLM 报告：

```text
Available KV cache memory: 0.15 GiB
Estimated maximum model length: 1088
```

所以三组公平对比统一使用：

```text
runtime.max_model_len: 1024
```

量化模型有时可以启动更大的 `max_model_len`，但那属于部署能力差异。如果用 BF16=1024、
AWQ/GPTQ=2048 来比较 accuracy 或 latency，就会混入不同 KV-cache budget 和上下文设置，
不适合作为公平对比。

## 为什么必须设置 truncation false

本地 AWQ/GPTQ tokenizer 文件中包含后端 truncation：

```text
tokenizer.json:
  truncation.max_length: 512
```

当 `model.max_pixels: 602112` 时，较大的 DocVQA 图像会展开为 567 个 image tokens。
如果不关闭 tokenizer truncation，processor 路径只会保留 511 个 image token ids，并报错：

```text
Mismatch in image token count
Got ids=[511] and text=[567]
```

因此三组主实验都使用：

```text
model.max_pixels: 602112
runtime.mm_processor_kwargs.truncation: false
```

这个设置的目的不是提高分辨率，而是避免 AWQ/GPTQ tokenizer 的 512 truncation 破坏图像
token 数量。

## Full OCRBench 1000 结果

| Metric | BF16 | AWQ | GPTQ |
| --- | ---: | ---: | ---: |
| Samples | 1000 | 1000 | 1000 |
| max_model_len | 1024 | 1024 | 1024 |
| max_pixels | 602112 | 602112 | 602112 |
| processor truncation | false | false | false |
| max_new_tokens | 128 | 128 | 128 |
| Accuracy | 0.863 | 0.855 | 0.854 |
| Correct | 863 / 1000 | 855 / 1000 | 854 / 1000 |
| Failed | 0 | 0 | 0 |
| Request mean | 1230.7 ms | 1337.4 ms | 1324.4 ms |
| Request p50 | 439.7 ms | 470.9 ms | 434.1 ms |
| Request p90 | 4002.3 ms | 4109.8 ms | 4132.6 ms |
| Request p95 | 4120.2 ms | 4242.1 ms | 4275.3 ms |
| Tokens/s | 28.56 | 28.29 | 28.24 |
| Sample tokens/s mean | 23.90 | 23.55 | 23.57 |
| Input tokens mean | 23.478 | 23.478 | 23.478 |
| Input tokens max | 52 | 52 | 52 |
| Generated tokens mean | 35.11 | 37.80 | 37.36 |
| Generated tokens max | 128 | 128 | 128 |
| GPU memory after load | 10413.5 MB | 10535.6 MB | 10523.5 MB |
| Model load memory delta | 10155.3 MB | 10277.3 MB | 10265.3 MB |
| Load time | 39.29 s | 39.40 s | 39.13 s |
| Total workload time | 1230.7 s | 1337.4 s | 1324.4 s |

主要观察：

1. **BF16 accuracy 最高。**
   BF16 是 0.863，AWQ 是 0.855，GPTQ 是 0.854。

2. **BF16 端到端 latency 最低。**
   BF16 的 request mean 是 1230.7 ms，低于 AWQ/GPTQ。

3. **AWQ/GPTQ 没有降低 benchmark JSON 里的 vLLM engine 总显存。**
   `gpu_memory_after_load` 三者都在 10.4-10.5 GB 左右。这个指标统计的是 vLLM engine
   初始化后的整体 GPU used memory，不只是 checkpoint weight size；它还包含 KV cache
   预留、runtime buffer、CUDA/PyTorch allocator 预留、vision/multimodal 相关 buffer 等。
   因此这不代表 AWQ/GPTQ 没有省权重显存，而是省下的空间主要被 vLLM 转成更大的 KV cache
   capacity / concurrency budget。AWQ kernel check 日志里 raw model loading memory 是
   3.64 GiB，同时 KV cache capacity 达到 38,256 tokens。

4. **量化没有带来 tokens/s 收益。**
   三者 tokens/s 很接近，BF16 仍略高。

## BF16 max_new_tokens=64 Full 1000

根据 stratified100 的 latency 结果，额外跑了 BF16 `max_new_tokens=64` 的 full OCRBench 1000：

```text
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_bf16.yaml \
  --sample-count 1000 \
  --sample-strategy first \
  --max-new-tokens 64 \
  --run \
  --output outputs/qwen3vl_4b_bf16_maxnew64_ocrbench_1000.json

outputs/qwen3vl_4b_bf16_maxnew64_ocrbench_1000.json
```

| Metric | BF16 max_new_tokens=128 | BF16 max_new_tokens=64 |
| --- | ---: | ---: |
| Samples | 1000 | 1000 |
| Accuracy | 0.863 | 0.840 |
| Correct | 863 / 1000 | 840 / 1000 |
| Request mean | 1230.7 ms | 906.2 ms |
| Request p50 | 439.7 ms | 438.9 ms |
| Request p90 | 4002.3 ms | 2165.2 ms |
| Request p95 | 4120.2 ms | 2199.0 ms |
| Tokens/s | 28.56 | 27.57 |
| Generated tokens mean | 35.11 | 24.95 |
| Generated tokens max | 128 | 64 |
| vLLM TTFT mean | N/A | 134.6 ms |
| vLLM prefill mean | N/A | 100.9 ms |
| vLLM decode mean | N/A | 769.5 ms |
| vLLM queue mean | N/A | 0.032 ms |

`max_new_tokens=64` 在 full 1000 上把 mean latency 降低约 26%，p90/p95 大幅降低；但 accuracy
从 0.863 降到 0.840，下降 2.3 个百分点。它可以作为 latency-oriented setting，但不建议替代
accuracy 主 baseline。这个结果也说明 100-sample 上只下降 1 个点偏乐观，最终 accuracy 判断仍应
以 full 1000 为准。

进一步的 per-category analysis 显示，accuracy 下降主要来自 Doc-oriented VQA：

| question_type | n | acc128 | acc64 | delta | regressions | trunc64 | gen64 mean | decode64 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Doc-oriented VQA | 200 | 0.835 | 0.725 | -0.110 | 22 | 75.5% | 54.54 | 1704.0 ms |
| Scene Text-centric VQA | 200 | 0.900 | 0.895 | -0.005 | 1 | 42.5% | 41.60 | 1295.3 ms |
| Key Information Extraction | 200 | 0.875 | 0.875 | 0.000 | 0 | 0.5% | 9.19 | 271.3 ms |
| Handwritten Mathematical Expression Recognition | 100 | 0.680 | 0.680 | 0.000 | 0 | 3.0% | 28.36 | 906.5 ms |
| Text recognition categories | 350 | unchanged | unchanged | 0.000 | 0 | 0.0% | 2.4-6.8 | 49-187 ms |

因此，full1000 的结论不是“全局 64 足够好”，而是：短文本识别、KIE、公式类可以用较小
token budget；Doc-oriented VQA 需要更大预算，Scene Text-centric VQA 也需要谨慎。下一步更合理的
方向是按 `question_type` 动态设置 `max_new_tokens`，而不是全局 64。

按类别看，`max_new_tokens=64` 的 latency 收益集中在长输出类别：

| question_type | n | req128 ms | req64 ms | delta ms | reduction | gen128 | gen64 | trunc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Doc-oriented VQA | 200 | 2956.2 | 1875.4 | -1080.8 | 36.6% | 89.15 | 54.54 | 75.5% |
| Scene Text-centric VQA | 200 | 1939.5 | 1427.0 | -512.5 | 26.4% | 57.57 | 41.60 | 42.5% |
| Regular Text Recognition | 50 | 444.3 | 172.7 | -271.6 | 61.1% | 2.40 | 2.40 | 0.0% |
| Key Information Extraction | 200 | 418.2 | 424.7 | 6.5 | -1.6% | 9.24 | 9.19 | 0.5% |
| Digit String Recognition | 50 | 288.4 | 296.0 | 7.7 | -2.7% | 6.84 | 6.84 | 0.0% |
| Artistic Text Recognition | 50 | 161.2 | 171.5 | 10.3 | -6.4% | 2.68 | 2.68 | 0.0% |
| Non-Semantic Text Recognition | 50 | 198.1 | 208.9 | 10.8 | -5.4% | 4.18 | 4.18 | 0.0% |
| Irregular Text Recognition | 50 | 149.3 | 164.6 | 15.2 | -10.2% | 2.42 | 2.42 | 0.0% |
| Handwriting Recognition | 50 | 150.2 | 167.0 | 16.8 | -11.2% | 2.48 | 2.48 | 0.0% |
| Handwritten Mathematical Expression Recognition | 100 | 983.2 | 1017.4 | 34.2 | -3.5% | 28.72 | 28.36 | 3.0% |

这里 `trunc64` 表示生成长度达到 64 token 上限的比例。Doc-oriented VQA 的 `trunc64`
达到 75.5%，accuracy 同时下降 11 个点，说明这类样本不能使用全局 64 token 预算。
Scene Text-centric VQA 也有 42.5% 的截断率，但 accuracy 只下降 0.5 个点，说明其中一部分
长输出不是评分所需的关键信息。

更具体地说，64-token 方案的高截断类别如下：

| question_type | n | acc128 | acc64 | delta | regressions | trunc64 | gen64 mean | decode64 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Doc-oriented VQA | 200 | 0.835 | 0.725 | -0.110 | 22 | 75.5% | 54.54 | 1704.0 ms |
| Scene Text-centric VQA | 200 | 0.900 | 0.895 | -0.005 | 1 | 42.5% | 41.60 | 1295.3 ms |
| Handwritten Mathematical Expression Recognition | 100 | 0.680 | 0.680 | 0.000 | 0 | 3.0% | 28.36 | 906.5 ms |
| Key Information Extraction | 200 | 0.875 | 0.875 | 0.000 | 0 | 0.5% | 9.19 | 271.3 ms |

因此，较合理的动态 token budget 是：

| 类别 | 建议 |
| --- | --- |
| Doc-oriented VQA | 保持 128，不能降到 64 |
| Scene Text-centric VQA | 谨慎，可测试 96 |
| Key Information Extraction | 64 足够 |
| Text recognition 类别 | 可以进一步测试 16 或 32 |
| Handwritten Mathematical Expression Recognition | 64 不降 accuracy，但 latency 不降，需单独看输出格式 |

## Stratified-100 快速检查

| Metric | BF16 | AWQ | GPTQ |
| --- | ---: | ---: | ---: |
| Accuracy | 0.87 | 0.85 | 0.90 |
| Correct | 87 / 100 | 85 / 100 | 90 / 100 |
| Request mean | 827.4 ms | 941.3 ms | 943.4 ms |
| Request p50 | 246.8 ms | 242.5 ms | 258.8 ms |
| Request p90 | 2698.9 ms | 3145.5 ms | 3402.2 ms |
| Tokens/s | 25.42 | 26.84 | 25.75 |
| vLLM TTFT mean | 142.0 ms | 147.8 ms | 140.2 ms |
| vLLM prefill mean | 102.9 ms | 100.4 ms | 102.3 ms |
| vLLM decode mean | 683.4 ms | 791.3 ms | 801.4 ms |
| vLLM inference mean | 786.2 ms | 891.6 ms | 903.7 ms |

这组结果只说明 GPTQ 在这 100 条分层样本上碰巧更高，不说明 GPTQ 整体超过 BF16。
完整 1000 条样本中，GPTQ 是 0.854，BF16 是 0.863。

## 为什么量化没有变快

直觉上，AWQ/GPTQ 的 4-bit 权重更小，应该更快。但本实验测的是 OCRBench 的端到端请求，
耗时不只来自权重矩阵乘法。

主要耗时来源包括：

1. vision encoder；
2. multimodal preprocessing；
3. attention / KV cache 访问；
4. vLLM scheduler 和 runtime overhead；
5. int4 dequantization overhead；
6. 输出 token 数带来的自回归 decode 时间。

因此，4-bit checkpoint 文件更小，不一定能在这个单请求 OCRBench 场景中转化为更低
request latency。当前 AWQ/GPTQ 更像是为更紧显存、更大 batch、更高并发、更长 context
或边缘设备准备的部署候选，而不是当前 BF16 baseline 的速度替代。

### AWQ kernel check

为了确认 AWQ 没有变快是否因为没有走硬件友好的 INT4/W4A16 kernel，额外运行了一次
AWQ kernel check：

```text
VLLM_USE_FLASHINFER_SAMPLER=0
configs/vlm/qwen3vl_4b_awq_local.yaml
output: /tmp/awq_kernel_check.json
```

vLLM 启动日志显示：

```text
quantization=compressed-tensors
Using MarlinLinearKernel for CompressedTensorsWNA16
Model loading took 3.64 GiB memory
Available KV cache memory: 5.26 GiB
GPU KV cache size: 38,256 tokens
```

这说明当前 AWQ 已经启用了 vLLM 的 Marlin W4A16 路径，不是 BitsAndBytes 那种通用在线
4-bit 加载路径。因此，AWQ 没有超过 BF16 的原因不是“没有用到 INT4/W4A16 kernel”。

本次 check 的指标和主 AWQ stratified100 基本一致：

| Metric | BF16 eager warmup1 | AWQ Marlin W4A16 |
| --- | ---: | ---: |
| Accuracy | 0.87 | 0.85 |
| Correct | 87 / 100 | 85 / 100 |
| Request mean | 827.4 ms | 941.3 ms |
| Request p50 | 246.8 ms | 242.5 ms |
| Request p90 | 2698.9 ms | 3145.5 ms |
| Tokens/s | 25.42 | 26.84 |
| Generated tokens mean | 21.01 | 25.23 |
| vLLM decode mean | 683.4 ms | 791.3 ms |

这里 AWQ 的 aggregate tokens/s 和 BF16 接近，甚至略高；但 AWQ 平均生成 token 更多，
长输出样本更多触发自回归 decode 成本，所以 request mean 更高。换句话说，当前瓶颈不是
单纯的 LLM linear GEMM，而是 batch size 1 下的端到端请求路径和输出长度分布。

## GPTQ 和 AWQ 怎么理解

当前保留的 GPTQ checkpoint 使用 DocVQA validation calibration，已经替换旧的 text-only
GPTQ checkpoint。替换后，GPTQ 在 full 1000 上达到 0.854，和 AWQ 的 0.855 很接近。

但从当前数据看：

```text
BF16 > AWQ ≈ GPTQ
```

这里的比较指 full OCRBench 1000 的 accuracy 和端到端 latency。AWQ/GPTQ 仍然有部署价值，
但它们不是当前 12GB 单卡、batch size 1、OCRBench 评测下的最优选择。

## 已测试优化

### 当前主配置已启用

```text
vLLM backend
FlashAttention v2 for LLM / ViT / MMEncoderAttention
PagedAttention-style KV cache block management
max_model_len: 1024
max_pixels: 602112
mm_processor_kwargs.truncation: false
enforce_eager: true
warmup: 1
VLLM_USE_FLASHINFER_SAMPLER=0
```

FlashAttention 和 PagedAttention 不是二选一。FlashAttention 优化 attention kernel；
PagedAttention-style block manager 优化 KV cache 管理。当前 vLLM 日志已经确认 LLM、ViT、
MMEncoderAttention 都使用 FlashAttention。

### Batch Size 1 Latency

| 方法 | 结果 | 结论 |
| --- | --- | --- |
| `warmup: 1` | BF16 accuracy 不变，并移除首条 cold-start spike | 保留 |
| `max_new_tokens=64` | stratified100 mean 从 827.4 ms 到 625.3 ms；full 1000 mean 从 1230.7 ms 到 906.2 ms | 可作为 latency-oriented setting |
| `max_new_tokens=32` | mean 到 461.7 ms，accuracy 到 0.81 | 过度截断 |
| `max_pixels=501760/401408` | mean 818.3 / 804.0 ms | 没有稳定 latency 收益 |
| `enforce_eager: false` | mean 397.3 ms，但 accuracy 0.77 | 暂不作为 baseline |
| ngram speculative | mean 899.9 ms 到 859.4 ms，accuracy 0.87 到 0.85 | 收益不足 |

`warmup: 1` 的 paired comparison 显示 BF16/AWQ/GPTQ 均为：

```text
changed_predictions: 0
regressions: 0
improvements: 0
```

`max_new_tokens` sweep 是目前最清晰的 latency 优化：

| Metric | 128 baseline | 64 | 32 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.87 | 0.86 | 0.81 |
| Request mean | 827.4 ms | 625.3 ms | 461.7 ms |
| Request p90 | 2698.9 ms | 2103.7 ms | 1056.7 ms |
| Generated tokens mean | 21.01 | 15.94 | 11.48 |
| vLLM decode mean | 683.4 ms | 486.1 ms | 326.2 ms |

vLLM internal metrics 显示，batch size 1 的 queue 基本可以忽略，主耗时来自 decode：

```text
BF16 queue mean: 0.038 ms
BF16 prefill mean: 102.9 ms
BF16 decode mean: 683.4 ms
BF16 inference mean: 786.2 ms
```

因此 `max_new_tokens` 有效，是因为它直接减少生成 token 数和自回归 decode 时间；而不是因为
改变了 queue 或 prefill。

`max_pixels` sweep 暂时不是优化重点：

| Metric | 602112 baseline | 501760 | 401408 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.87 | 0.88 | 0.86 |
| Request mean | 827.4 ms | 818.3 ms | 804.0 ms |
| Request p90 | 2698.9 ms | 2715.3 ms | 2613.6 ms |
| vLLM prefill mean | 102.9 ms | 92.6 ms | 90.6 ms |
| vLLM decode mean | 683.4 ms | 686.0 ms | 674.9 ms |

### 部署侧实验

这些方法释放显存或增加 KV cache capacity，但没有改善当前 batch size 1 latency。

| 方法 | 关键结果 | 当前判断 |
| --- | --- | --- |
| AWQ Marlin W4A16 | `Using MarlinLinearKernel for CompressedTensorsWNA16`，KV cache 38,256 tokens | 已走 W4A16 kernel，但端到端不快于 BF16 |
| FP8 KV cache | KV capacity 约 2048 到 3072 tokens，accuracy 0.87 到 0.85 | 适合并发/长 context，非当前 baseline |
| BitsAndBytes 4-bit | raw model loading memory 3.24 GiB，KV cache 40,752 tokens；mean 1064.6 ms | 省显存但 batch size 1 更慢 |
| GPTQ DocVQA calibration | full 1000 accuracy 0.854 | 接近 AWQ，但不超过 BF16 |

BitsAndBytes 当前只保留标准入口：

```text
quant.method: bitsandbytes
```

旧的 `bnb4` 结果来自 Transformers/bitsandbytes 路径，不作为当前 vLLM 主对比口径。

### 暂不采用

CUDA Graph / compiled path 显著提速，但当前输出不稳定：

| Metric | BF16 eager warmup1 | BF16 compile/CUDA graph |
| --- | ---: | ---: |
| Accuracy | 0.87 | 0.77 |
| Request mean | 827.4 ms | 397.3 ms |
| Tokens/s | 25.42 | 62.20 |

这里不应简单归因于“CUDA Graph 改变结果”。更准确地说，`enforce_eager: false` 触发了完整
vLLM compiled execution path；当前这条路径对 Qwen3-VL-4B BF16 输出不够稳定。

draft-model speculative 也暂不采用，vLLM 0.21.0 对 Qwen3-VL M-RoPE 报：

```text
Speculative Decoding with draft models or parallel drafting does not support M-RoPE yet
```

## 下一步

1. **测试 short-answer prompt。**
   很多长尾 latency 来自 DocVQA/STVQA 输出解释段落。先在 stratified100 上测试“只输出最短答案、
   不解释”的 prompt。

2. **测试 `warmup: 2/3`。**
   当前 `warmup: 1` 后仍可能有未覆盖 shape 触发 Triton JIT，例如 `_bilinear_pos_embed_kernel`。

3. **测试按 `question_type` 动态设置 `max_new_tokens`。**
   根据 full1000 category analysis，文本识别/数字识别可用 16 或 32；KIE 可用 32 或 64；
   HME 可用 64；STVQA 可用 96；DocVQA 应保持 128 或测试 96。

4. **部署侧再评估 AWQ/GPTQ/BitsAndBytes/FP8 KV。**
   这些方法不适合替代当前 BF16 batch size 1 baseline，但适合更高并发、更大 batch、更长 context
   或更紧显存场景。
