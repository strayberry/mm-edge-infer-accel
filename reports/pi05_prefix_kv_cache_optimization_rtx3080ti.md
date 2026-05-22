# Pi0.5 prefix KV cache 优化效果报告

## 测试配置

- **硬件**: NVIDIA GeForce RTX 3080 Ti (12GB)
- **模型**: `lerobot/pi05_libero_finetuned_v044`
- **数据集**: `HuggingFaceVLA/libero`，episodes `[0, 1, 2]`
- **样本数**: 300 frames（每个 episode 100 frame）
- **模式**: `reset`（每帧清空 action queue）
- **warmup**: 3 frames

## 测试结果

| 指标 | 无优化 | prefix KV cache | 提升 |
|------|:------:|:----------------:|:----:|
| action_mean | 0.4248s | **0.3750s** | **-11.7%** |
| loop_hz | 2.30 Hz | **2.61 Hz** | **+13.2%** |
| MAE | 0.01546 | 0.01548 | ~持平 |
| Cosine sim | 0.997 | 0.997 | ~持平 |
| GPU memory | ~8922 MB | ~8922 MB | 无差异 |

## 优化原理

### Pi0.5 denoising 流程（原始）

Pi0.5 的动作生成是一个扩散去噪过程（在 RTX 3080 Ti 上验证），包含 `num_inference_steps`（默认 10）步。每一步都需要模型 forward pass 来预测噪声 velocity `v_t`。每次 forward 的输入由两部分拼接而成：

```
input = [prefix_tokens | noisy_action_tokens]
         ↑                    ↑
      视觉+文本编码           当前去噪步的动作噪声
```

**原始实现的问题**：在每步 denoising 中，`embed_prefix`（视觉编码 + 文本编码）都会重新执行，包括：
- Vision tower forward（PaliGemma 视觉编码器）
- 文本 token embedding
- Prefix 的 self-attention 计算

这些计算在每步之间是**完全相同的**（prefix 不会变化），但原始代码没有利用这一事实，导致 `num_steps` 步中有 `num_steps - 1` 步的 prefix 计算是冗余的。

### 优化方案：prefix KV cache

核心思路：prefix 的内容在整个 denoising 过程中不变，因此其 KV cache 只需计算一次，后续步骤复用。

```
原始（每步独立）:
  step 0: [prefix forward] → KV_0 → [suffix forward] → v_0
  step 1: [prefix forward] → KV_1 → [suffix forward] → v_1   ← KV_1 == KV_0，但重新算了
  step 2: [prefix forward] → KV_2 → [suffix forward] → v_2   ← 同样冗余
  ...

优化后（KV cache）:
  [prefix forward] → KV_cache  ← 只跑一次
  step 0: [suffix forward with KV_cache] → v_0
  step 1: [suffix forward with KV_cache] → v_1
  step 2: [suffix forward with KV_cache] → v_2
  ...
```

具体改动：

1. **前置 prefix forward**：在 denoising 循环开始前，执行 `embed_prefix()` 获取 prefix 的 embeddings 和 masks，然后通过一次完整的 `paligemma_with_expert.forward(use_cache=True)` 获取 `past_key_values`
2. **逐步复用**：每个 denoising step 中，`_denoise_step_cached()` 只对 suffix（noisy action）部分做 forward，通过 `past_key_values=copy.deepcopy(past_key_values)` 传入 cached prefix KV
3. **后缀编码**：`_embed_suffix_fast()` 处理 noisy action 加上时间步 embedding，与 cached prefix KV 拼接后送入 transformer
4. **deepcopy 必要性**：模型 forward 在计算 attention 时会原地修改 `past_key_values` 中的张量（维度扩展），因此每步必须 `deepcopy`，否则下一步会拿到上一步膨胀后的 KV。Pi0.5 的 action chunk 较小（49 tokens），deepcopy 开销可忽略

### 代码结构

```
mm_edge_infer_accel/pi05_optimizations.py
├── apply_pi05_optimizations()     # 入口：替换 model.sample_actions
├── _optimized_sample_actions()    # 替换后的主函数
│   ├── embed_prefix()             # 编码 visual + text prefix
│   ├── paligemma forward          # 计算 prefix KV cache
│   ├── _make_suffix_context()     # 预计算 suffix attention mask
│   └── 去噪循环 (×num_steps)
│       ├── _denoise_step_cached() # suffix forward with cached KV
│       └── x_t.add(v_t, dt)      # Euler 更新
└── _embed_suffix_fast()           # suffix embedding + time MLP
```

## 控制方式

默认启用，通过 YAML 配置关闭：

```yaml
runtime:
  enable_prefix_kv_cache: false
```
