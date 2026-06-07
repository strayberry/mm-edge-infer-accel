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

## Closed-loop LIBERO 阶段性结果

### 评测口径

本节记录 LeRobot reference closed-loop 评测，不再读取固定 `.npz` 帧，而是直接运行 LIBERO env。

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-06-05 |
| 环境 | `/root/autodl-tmp/envs/pi05` |
| 模型 | `lerobot/pi05_libero_finetuned_v044` |
| LIBERO suite | `libero_spatial` |
| Task ids | `0, 1, 2` |
| Episodes | 每个 task 3 个，共 9 episodes |
| Max steps | 280 |
| Rendering | OSMesa |
| 对比项 | prefix KV cache on / off |

输出文件：

```text
outputs/pi05_closed_loop_libero_spatial_tasks0_1_2_ep3_prefix_on.json
outputs/pi05_closed_loop_libero_spatial_tasks0_1_2_ep3_prefix_off.json
```

### 汇总结果

| config | success | steps mean | steps std | episode Hz mean | action p50 | action mean | control e2e p50 | env.step p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix KV cache on | 9/9 | 134.4 | 60.9 | 2.65 Hz | 4.2 ms | 14.3 ms | 312.9 ms | 303.5 ms |
| prefix KV cache off | 9/9 | 104.7 | 31.2 | 2.60 Hz | 4.4 ms | 15.8 ms | 308.2 ms | 298.7 ms |

按 task 拆分：

| config | task 0 success / steps | task 1 success / steps | task 2 success / steps |
| --- | --- | --- | --- |
| prefix KV cache on | 3/3, `[80, 78, 170]` | 3/3, `[117, 202, 266]` | 3/3, `[95, 101, 101]` |
| prefix KV cache off | 3/3, `[78, 77, 77]` | 3/3, `[110, 115, 184]` | 3/3, `[102, 96, 103]` |

### 结论

- 本轮 closed-loop 评测中，prefix KV cache on/off 均为 `9/9` success，未观察到成功率退化。
- prefix KV cache 对 action chunk 推理有小幅收益：action mean 从 `15.8 ms` 降到 `14.3 ms`。
- 总控制循环主要受 `env.step` 影响，`env.step p50` 约 `300 ms`，因此 prefix KV cache 对整体 control Hz 的影响有限。
- episode steps 差异较大，尤其 task 0/1；当前不能把 steps 差异直接归因于 prefix KV cache，需要更多 task/seed 扩展验证。
- 当前 closed-loop 结果可作为 TensorRT/FlashRT 优化前的 LeRobot reference baseline 之一，但还不是完整 LIBERO suite 统计。
