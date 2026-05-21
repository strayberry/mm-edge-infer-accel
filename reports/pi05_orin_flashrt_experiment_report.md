# Pi0.5 FlashRT Orin Experiment Report

## 摘要

本报告汇总 Jetson AGX Orin 32G 上 FlashRT Pi0.5 的完整实验与 profiling 结果，
覆盖 BF16 和 INT8 两种精度路径、多种 runtime 优化（temporal cache、spatial
pooling、vitpack）和 Nsight Systems 深度 profiling。

关键结论：

- **BF16 baseline**: 244.3 ms (4.09 Hz)。cuBLASLt BF16 GEMM 占 ~70% GPU 时间，
  是绝对瓶颈。
- **INT8 baseline**: 163.5 ms (6.12 Hz)。CUTLASS INT8 GEMM 替换 encoder/decoder
  FFN，单次 GEMM 快 2-7×，但引入 quant/dequant 开销，总 kernel 调用翻倍。
- **当前最佳部署候选**: `BF16 cache2` (p50 152.4 ms, 6.56 Hz, cos_mean=0.983)。
  temporal K/V reuse 速度接近 pool2 但动作偏差明显更小。
- **最快路线**: `INT8 cache2` (p50 104.4 ms, 9.58 Hz)，但 action similarity 不足
  (cos_mean=0.881)，不能视为 lossless。
- **Attention 不是瓶颈**: FlashAttention splitkv 仅占 ~4.5% GPU 时间。
- **INT8 下仍有 ~9% BF16 GEMM 残留**（attention QKV/O 投影），全部量化预期可再省 ~15ms。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 设备 | Jetson AGX Orin 32G |
| Machine | KST Jetson AGX Orin AVSAI |
| SoC | tegra234 |
| 架构 | aarch64 |
| L4T | R36.4.7 |
| Ubuntu | 22.04.5 LTS |
| Kernel | 5.15.148-tegra |
| CUDA Toolkit | 12.6.68 |
| Python | 3.10.12 |
| PyTorch | 2.8.0 |
| Torch CUDA | 12.6 |
| GPU capability | SM87 / compute capability 8.7 |
| 可见 GPU memory | 29.98 GB |
| SM count | 14 |
| GPU clock | 930.75 MHz (MAXN + jetson_clocks) |
| Power mode | MAXN |

FlashRT 编译参数：

```bash
cmake -B build -S . \
  -DGPU_ARCH=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_HDIMS='96;128;256' \
  -DFA2_DTYPES='bf16' \
  -DPython3_EXECUTABLE=$(which python)
```

## 评测口径

模型与数据：

- checkpoint: `/root/models/pi05_libero_finetuned_v044`
- dataset: `/root/pi05_eval/libero_episodes0_1_2_100frames_each.npz`
- 数据内容：3 个 LIBERO episode，每个 episode 100 帧，共 300 帧
- 输入：2 views, image resize/crop 到 `(224, 224, 3)`
- denoising steps: 10（BF16/INT8 baseline）
- baseline: `pool=1, cache_frames=1`
- cache2: `pool=1, cache_frames=2`

### NPZ 评测数据生成

NPZ 文件从 HuggingFace 的 LIBERO LeRobot dataset 导出，导出脚本：

```bash
python scripts/export_libero_npz.py \
  --dataset-id "HuggingFaceVLA/libero" \
  --episodes 0,1,2 \
  --sample-count 100 \
  --output outputs/libero_episodes0_1_2_100frames_each.npz
```

该脚本会：

1. 用 `LeRobotDataset` 加载指定 episode。
2. 取前 `sample-count` 帧（或整个 episode 的较小者）。
3. 提取 `observation.images.image`（主相机）、`observation.images.image2`（wrist camera）、
   `observation.state`、`task`、`action` 等字段。
4. 图像统一转成 `uint8 HWC` 格式，动作/状态存为 `float32`。
5. 输出为 `np.savez_compressed` 格式，包含 `images`、`wrist_images`、`states`、`tasks`、
   `frame_indices`、`episode_indices`、`reference_actions` 等数组。

NPZ 文件随后复制到 Orin 上供 `eval_libero.py` 使用。

### 评测脚本

```bash
python examples/orin/eval_libero.py \
  --npz /root/pi05_eval/libero_episodes0_1_2_100frames_each.npz \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --frames 300 \
  --warmup 3 \
  --configs baseline,cache2 \
  --out <output.npz> \
  --bad-frame-dir <bad-frame-dir> \
  --bad-frame-topk 30
```

重要修正：

- 多 episode NPZ 中 task 会变化，评测脚本需要按 `episode_index + task` 分段调用 `set_prompt()`。
- `cache_frames=2` 的 warmup 会污染 temporal KV cache，warmup 后必须 reset prompt/frame counter。
- 所有结论以 prompt fix + warmup reset 后的 300 帧结果为准。

## 300 帧结果

### 统一对比：以 BF16 baseline 为参考

所有 action similarity 都以 `BF16 baseline` 的 action chunk 为参考。

| config | p50 | Hz | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 244.3 ms | 4.09 | 0 | 0 | 1.000000 | 1.000000 | 0 |
| BF16 cache2 | 152.4 ms | 6.56 | 0.010613 | 0.049782 | 0.982723 | 0.478394 | 1.859494 |
| BF16 pool2 | 154.6 ms | 6.47 | 0.067040 | 0.170755 | 0.918667 | 0.098733 | 1.859494 |
| BF16 pool4 | 123.3 ms | 8.11 | 0.135037 | 0.315950 | 0.726205 | -0.768505 | 1.866992 |
| INT8 baseline | 163.5 ms | 6.12 | 0.094451 | 0.173171 | 0.875164 | 0.150117 | 1.919478 |
| INT8 cache2 | 104.4 ms | 9.58 | 0.094369 | 0.180585 | 0.881481 | 0.149363 | 1.919478 |
| BF16 vitpack12_f2 | 140.6 ms | 7.11 | 0.196959 | 0.381647 | 0.588616 | -0.864485 | 1.870741 |
| BF16 vitpack18_f2 | 146.2 ms | 6.84 | 0.110307 | 0.260673 | 0.794643 | -0.813983 | 1.866992 |
| INT8 vitpack12_f2 | 97.6 ms | 10.24 | 0.259306 | 0.581693 | 0.356749 | -0.876018 | 1.919478 |
| INT8 vitpack18_f2 | 102.3 ms | 9.77 | 0.187953 | 0.316793 | 0.646263 | -0.639782 | 1.919478 |

统一口径结论：

- 速度最快的是 `INT8 vitpack12_f2`，但 `cos_mean=0.356749`，不可用。
- 接近 10 Hz 且相对最好的高速方案是 `INT8 cache2`，但相对 BF16 baseline 的 `cos_mean=0.881481`，仍不能视为无损。
- `BF16 pool2` 速度接近 `BF16 cache2`，但 `cos_mean=0.918667`，最坏帧 `cos_min=0.098733`，明显不如 cache2。
- correctness-preserving 角度，目前只有 `BF16 cache2` 保持较高均值相似度：`cos_mean=0.982723`。
- 所有 vitpack 方案都有负 `cos_min`，最坏帧动作方向反转，不能作为主线。

### BF16 细分

| config | p50 | Hz | p95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 244.3 ms | 4.09 | 245.0 ms | 242.2 ms | 245.5 ms |
| cache2 | 152.4 ms | 6.56 | 244.7 ms | 61.8 ms | 246.1 ms |

BF16 `cache2` vs BF16 baseline:

| mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.010613 | 0.049782 | 0.982723 | 0.478394 | 1.859494 |

### INT8 细分

| config | p50 | Hz | p95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 163.5 ms | 6.12 | 163.8 ms | 163.0 ms | 168.0 ms |
| cache2 | 104.4 ms | 9.58 | 165.3 ms | 44.5 ms | 168.1 ms |

INT8 `cache2` vs INT8 baseline:

| mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.018609 | 0.062768 | 0.976375 | 0.337565 | 1.919478 |

### 与 Dataset Reference Actions 的关系

| config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.087353 | 0.132249 | 0.912225 | -0.897470 | 0.472012 |
| BF16 cache2 | 0.081298 | 0.125228 | 0.938563 | -0.897470 | 0.410405 |
| INT8 baseline | 0.133107 | 0.226115 | 0.840911 | -0.965416 | 0.785405 |
| INT8 cache2 | 0.131475 | 0.212014 | 0.847293 | -0.979253 | 0.705048 |

注意：baseline 自己相对 `reference_actions` 的最低 cosine 也为负，说明 dataset action
与 policy 输出之间可能存在 stochasticity、chunk alignment、frame/action offset、
normalization 或专家动作分布差异。它更适合做辅助趋势判断，而不是最终 success rate 指标。

## 精度与算子路径

### BF16

启用方式：

```bash
unset FVK_PI05_RTX_FORCE_INT8
unset FVK_PI05_RTX_INT8_ENCODER_ONLY
```

主要路径：

- Vision encoder GEMM: BF16 GEMM, `gemm.bf16_nn`
- Decoder/DiT GEMM: BF16 GEMM
- Attention: FlashRT Pi0.5 attention/MHA CUDA kernel
- RoPE: FlashRT RoPE CUDA kernel
- Norm/activation/elementwise: FlashRT CUDA kernels
- Graph: full pipeline CUDA Graph + decoder-only CUDA Graph
- GEMM autotune: 每个 shape 测试 8 个候选算法

特点：

- 当前 correctness baseline。
- cache2 的 p50 从 244.3 ms 降到 152.4 ms。
- 300 帧上均值相似度较好，但最坏帧仍会出现明显 stale-cache 偏差。

### INT8

启用方式：

```bash
export FVK_PI05_RTX_FORCE_INT8=1
unset FVK_PI05_RTX_INT8_ENCODER_ONLY
```

日志确认：

```text
FVK_PI05_RTX_FORCE_INT8/INT8_ENCODER_ONLY set: INT8 encoder+decoder
INT8 quantized 90 decoder GEMM weights
INT8 quantized 90 encoder GEMM weights
Skipping cuBLASLt INT8 autotune: decoder INT8 uses CUTLASS fused path
```

主要路径：

- Encoder GEMM weights: INT8 quantized
- Decoder GEMM weights: INT8 quantized
- Decoder INT8 GEMM: CUTLASS fused path
- INT8 不走 cuBLASLt INT8 autotune
- Attention/RoPE/norm/activation/image preprocessing 等非 GEMM 部分仍主要是 BF16/FP32 辅助路径
- Graph: full pipeline CUDA Graph + decoder-only CUDA Graph

特点：

- 当前最快路线。
- baseline p50 163.5 ms，cache2 p50 104.4 ms，cache2 接近 10 Hz。
- 相比 BF16，INT8 的 action 漂移更明显，不能直接作为 lossless correctness baseline。

### FP16

Orin 上尝试过 upstream FP16 路径，但当前构建没有编译 FP16 FlashAttention entry：

```text
fvk_attention_fa2: fp16 entry was not compiled.
Rebuild with -DFA2_DTYPES="fp16;bf16" to enable it.
```

因此本轮没有 FP16 300 帧结果。

## Nsight Systems Profiling 分析

### 评测设置

工具：NVIDIA Nsight Systems 2024.5.4
方法：`nsys profile --trace=cuda`，捕获 5 帧推理（BF16 和 INT8 各一次）。
Profiling 文件（*.nsys-rep）已下载到本地，可用 Nsight Systems GUI 打开查看完整 timeline。

### BF16 baseline GPU kernel 分布

```
 Rank  Kernel                                              占比      总时间     调用次数   单次平均
 ────  ──────────────────────────────────────────────────  ─────    ────────  ────────  ───────
  1    ampere_bf16_s16816gemm_bf16_128x128 (主力 GEMM)     41.1%     608 ms      993     254 µs
  2    ampere_bf16_s16816gemm_bf16_64x64 (小 GEMM)        11.0%     163 ms     2520      65 µs
  3    ampere_bf16_s16816gemm_bf16_128x64_4stages          9.7%     144 ms     2520      57 µs
  4    ampere_bf16_s16816gemm_bf16_128x64_3stages          7.9%     116 ms     1437      39 µs
  5    gate_silu_mul (融合激活)                             3.4%      50 ms     1328       7 µs
  6    ampere_bf16_s16816gemm_bf16_128x256                 3.3%      49 ms      234     210 µs
  7    FlashAttention splitkv                               3.1%      46 ms     1260      36 µs
  8    cutlass_bf16_gemm+relu (encoder)                    2.8%      41 ms      544      75 µs
  9    bias_res_kernel                                     1.6%      23 ms      468      50 µs
 10    qkv_split_rope                                      1.2%      18 ms     1332      13 µs
 11    ada_rms_norm                                        1.0%      15 ms     2590       6 µs
 12    FlashAttention hdim=256                              0.9%      14 ms       68     204 µs
 13    FlashAttention hdim=96                               0.5%       8 ms      108      72 µs
     其他 memory-bound (gate_mul_res, res_add, layer_norm) ~3-4%     ~25 ms      多      4-50 µs
```

**BF16 结论：GEMM 占 ~70% GPU 时间，attention 只占 ~4.5%。** 单个 `128x128` GEMM
变体就占 41%，这是 decoder 10 步扩散 × 18 层 × 每层 2-3 个 GEMM 累加的结果。
Orin SM87 的 14 SM bf16 tensor core 吞吐有限，GEMM 是绝对瓶颈。

### INT8 baseline GPU kernel 分布

```
 Rank  Kernel                                              占比      总时间     调用次数   单次平均
 ────  ──────────────────────────────────────────────────  ─────    ────────  ────────  ───────
  1    elementwise_kernel (权重量化, 校准阶段)              45.1%    1538 ms      180    8542 µs
  2    CUTLASS INT8 GEMM (大 tile)                          5.3%     181 ms      204     889 µs
  3    CUTLASS INT8 GEMM (小 tile, 主力)                    5.1%     174 ms     5112      34 µs
  4    CUTLASS INT8 GEMM (中 tile)                          4.4%     149 ms     1328     112 µs
  5    ampere_bf16_128x128 (未量化的 BF16 GEMM)             4.3%     146 ms      721     236 µs
  6    FlashAttention splitkv                               1.3%      45 ms     1260      36 µs
  7    quantize_int8_rowwise                                1.6%      55 ms     2656      21 µs
  8    ampere_bf16_128x256 (未量化 BF16)                    1.4%      49 ms      234     210 µs
  9    gate_residual_ada_norm_int8 (融合)                   0.7%      24 ms     2450      10 µs
 10    qkv_split_rope                                       0.5%      16 ms     1332      12 µs
     其他 (bias_res, add_bias, rms_norm_int8 等)          ~2-3%     ~70 ms      多      5-120 µs
```

**INT8 结论：推理阶段 CUTLASS INT8 GEMM ~15% + 遗留 BF16 GEMM ~9% = ~24% GEMM 时间。**
Rank 1 的 45% 属于模型加载阶段权重量化，不属于每帧推理开销。INT8 GEMM 单次 34-112µs
比 BF16 快 2-7×，但总 kernel 调用次数从 ~12,000 增加到 ~25,000（quant/dequant 来回转换）。

### BF16 vs INT8 并排对比（每帧推理部分）

| 类别 | BF16 (242ms) | INT8 (166ms) | INT8 变化 |
|------|:-----------:|:-----------:|:----------|
| cuBLASLt BF16 GEMM | ~70% (171ms) | ~9% (15ms) | attention QKV/O 等未量化路径 |
| CUTLASS INT8 GEMM | — | ~15% (25ms) | 新增，替换 encoder/decoder FFN |
| FlashAttention | ~4.5% (11ms) | ~1.3% (2ms) | 一致，非瓶颈 |
| memory-bound 融合 (norm/activation) | ~10% (24ms) | ~2% (3ms) | 被 INT8 融合版本替代 |
| INT8 quant/dequant 开销 | — | ~3% (5ms) | 新增 overhead |
| 总 GPU kernel 调用/帧 | ~12,000 | ~25,000 | 翻倍 |

### 关键洞察

1. **BF16 的瓶颈非常明确：cuBLASLt GEMM 占 ~70%。** 仅 `128x128` 一个 GEMM 变体
   就占 41%。Orin 的 14 SM 在 bf16 GEMM 上的吞吐远低于 Thor 的 80+ SM。
2. **INT8 用 CUTLASS rowwise INT8 GEMM 替换后，单次 GEMM 延迟降低 2-7×**，但引入
   quant/dequant 来回转换开销（每次 GEMM 前后需要 quantize_int8_rowwise / dequantize），
   总 kernel 调用次数翻倍。
3. **INT8 下仍有 ~9% 的 BF16 GEMM 未被量化**（attention QKV/O 投影等），这些仍走
   cuBLASLt bf16。如果全部 INT8 化，预期可再省 ~15ms。
4. **Attention 不是瓶颈** — FlashAttention splitkv 在两种模式下都是单次 36µs、
   每帧 ~210 次，只占总 GPU 时间的 ~4.5%。优化 attention 对 Orin 收益很小。
5. **INT8 融合 kernel 效果不错** — `gate_residual_ada_norm_int8` 单次 10µs，替代了
   BF16 中三个分离 kernel 的 63µs。

### 与 Thor SM110 的对比

相同 Pi0.5 模型在 Thor 上纯 graph replay ~44ms（2-view），Orin BF16 baseline 是
242ms（5.5× 慢）。倍数主要由两个因素决定：
- **SM 数量**：Thor 80+ SM vs Orin 14 SM（~5.7×）
- **memory bandwidth**：Thor 204 GB/s vs Orin ~200 GB/s（接近持平）
- 实际差距 5.5× 基本符合 SM 数差距，说明 GEMM 已经接近线性 scaling。

## cache2 的含义

`cache2` 是 temporal cache 调度优化，不是单个新算子：

- key frame 跑完整 vision encoder + decoder。
- reuse frame 复用上一帧视觉/encoder cache，只走 decoder-only graph。
- 因此 latency 出现交替：完整帧约 240/160 ms，复用帧约 60/44 ms。
- p50 代表交替序列的中位数，不等同于所有帧都稳定在该延迟。

## 已证伪或暂不保留的实验

已证伪方向：

- post-ViT pooling `pool2/pool4`: 速度提升明显，但 action 漂移过大。
- `vitpack12_f2`: layer 12 后做 2x2 vision token pack，单帧速度好，但 300 帧大面积漂移。
- RMS/Norm-gated soft packer: 没有改善 `vitpack12_f2`，说明 token RMS 不能代表控制重要性。
- main-only vision pack: 更慢，且最坏帧更差。
- stage refresh: 强制在 `align_grasp/lift_move` 刷新质量很好，但 p50 回到接近 baseline，
  性价比不如 cache2。

### 完整 ViT 后 pooling 复测

实现方式：

- 完整运行 SigLIP ViT 27 层，保持原始 `2 views x 16 x 16 = 512` visual tokens。
- ViT 完成后，对每个 view 的 `16x16` token grid 做规则 average pooling。
- `pool2`: 每 `2x2` token 平均成 1 个 token，`512 -> 128`。
- `pool4`: 每 `4x4` token 平均成 1 个 token，`512 -> 32`。
- pooling 后再进入 vision final norm、multi-modal projector、Gemma encoder 和 decoder。
- 该方法不减少 ViT 27 层本身的计算，只减少 projector/Gemma encoder/decoder 的 token 长度。

BF16 300 帧：

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 243.1 ms | 4.11 | - | - | - | - |
| pool2 | 154.6 ms | 6.47 | 0.067040 | 0.170755 | 0.918667 | 0.098733 |
| pool4 | 123.3 ms | 8.11 | 0.135037 | 0.315950 | 0.726205 | -0.768505 |

结论：

- `pool2` 的速度和 `BF16 cache2` 接近，但 action similarity 明显更差。
- `pool4` 速度更快，但已经出现负 `cos_min`，不可用。
- 这说明即便不破坏 ViT 内部计算，只要把 projector/Gemma 看到的视觉 token 做规则平均压缩，
  Pi0.5 的动作输出仍会明显漂移。

### ViT 中间层 2x2 token pack 复测

- `vitpack12_f2`: SigLIP layer 12 后把两个 view 都做规则 2x2 average pack，vision tokens 从 512 降到 128。
- `vitpack18_f2`: SigLIP layer 18 后做同样的 2x2 average pack。

BF16 300 帧：

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 243.9 ms | 4.10 | - | - | - | - |
| vitpack12_f2 | 140.6 ms | 7.11 | 0.196959 | 0.381647 | 0.588616 | -0.864485 |
| vitpack18_f2 | 146.2 ms | 6.84 | 0.110307 | 0.260673 | 0.794643 | -0.813983 |

INT8 300 帧：

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 164.8 ms | 6.07 | - | - | - | - |
| vitpack12_f2 | 97.6 ms | 10.24 | 0.235052 | 0.587194 | 0.427604 | -0.873032 |
| vitpack18_f2 | 102.3 ms | 9.77 | 0.144822 | 0.310017 | 0.751075 | -0.632650 |

结论：

- `vitpack18_f2` 明显优于 `vitpack12_f2`，说明越晚压缩越能保留部分动作相关视觉特征。
- 但 300 帧下两者都出现负 `cos_min`，不能作为 correctness-preserving 优化。
- 当前主线仍应优先保留 `cache2`，后续如继续优化，应围绕 adaptive refresh 或更保守的
  temporal reuse，而不是中间层空间压缩。

## 综合推荐排序

| Rank | 优化项 | p50 收益 | 综合判断 |
| ---: | --- | ---: | --- |
| 1 | `BF16 cache2` | -37.0% | 当前最佳部署候选；cos_mean=0.983，需坏帧分析 |
| 2 | `INT8 baseline` | -33.1% | 速度好但 cos_mean=0.875，不能直接替代 baseline |
| 3 | `INT8 cache2` | -57.3% | 最快路线 9.58 Hz，但 action 偏差最大 |
| 4 | `BF16 pool2` | -36.9% | 速度强但 cos_mean=0.919，不如 cache2 |
| 5 | `BF16 pool4` | -49.8% | 速度最快但 cos_min 为负，基本不可用 |
| 6 | 全部 INT8 化 (QKV/O) | 预期~15ms | 未实现，attention 投影转为 INT8 可再省 ~10% |

## 后续建议

1. **完善 `cache_frames=2` 坏帧分析。** 300 帧结果确认 cache2 平均质量好但仍有
   stale-cache 坏帧。应定位低 cosine frame 是否集中在物体或夹爪快速变化阶段，
   并设计 adaptive refresh 策略。

2. **将剩余 ~9% BF16 GEMM 转为 INT8。** 当前 attention QKV/O 投影仍走 cuBLASLt
   bf16（~15ms）。用 CUTLASS INT8 覆盖后可再降 ~10% 延迟。

3. **继续以 `BF16 cache2` 作为 correctness-preserving 部署候选。** 在更多
   LIBERO 场景或真机上验证 roll-out success rate。

4. **FP16 路径待补全。** Orin 的 FA2 需要重新编译以包含 `fp16` dtype，之后可
   补充 FP16 baseline 对比。

5. **保持 `steps=10` 作为 reference 口径。** `steps=5` 收益较小（-12.8%），
   且改变 denoising schedule，应作为 ablation 而不是默认优化项。

6. **停止 vitpack/pooling 系列实验。** 300 帧 BF16 和 INT8 都已确认这些方案
   不适合 correctness-preserving 部署，不应继续投入。
