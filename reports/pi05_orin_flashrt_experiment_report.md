# Pi0.5 Orin FlashRT Experiment Report

日期：2026-05-26

本报告保留 Jetson AGX Orin 32G 上 Pi0.5 FlashRT 优化实验的完整记录，覆盖 300 帧 offline action similarity、closed-loop LIBERO、INT8 归因、cache/adaptive refresh、token pooling/vitpack 负结果、INT8 crash 验证和 Nsight Systems profiling。

## 评测口径

设备与环境：

- Jetson AGX Orin 32G, aarch64
- L4T R36.4.7, Ubuntu 22.04.5 LTS
- CUDA Toolkit 12.6.68
- Python 3.10.12
- FlashRT Pi0.5 Orin runtime

模型与数据：

- checkpoint: `/root/models/pi05_libero_finetuned_v044`
- dataset: `/root/pi05_eval/libero_episodes0_1_2_100frames_each.npz`
- 数据内容：3 个 LIBERO episode，每个 episode 100 帧，共 300 帧
- 输入：2 views, image resize/crop 到 `(224, 224, 3)`
- denoising steps: 10
- baseline: `pool=1, cache_frames=1`
- cache2: `pool=1, cache_frames=2`

NPZ 数据由本项目脚本从 LeRobot LIBERO dataset 导出：

```bash
python scripts/export_libero_npz.py \
  --dataset-id HuggingFaceVLA/libero \
  --episodes 0,1,2 \
  --sample-count 100 \
  --output outputs/libero_episodes0_1_2_100frames_each.npz
```

导出的 NPZ 再复制到 Jetson 的 `/root/pi05_eval/`，供 FlashRT Orin eval 脚本读取。

FlashRT 历史评测脚本（在 Jetson 的 `/root/FlashRT` checkout 中运行；本仓库不包含 `examples/orin`）。下文所有 `examples/orin/...` 和 `examples/thor/...` 命令同样指 FlashRT checkout 内的脚本，不是 mm-edge-infer-accel 的本地脚本：

```bash
cd /root/FlashRT
source /root/pi0.5/bin/activate
python examples/orin/eval_libero_offline.py \
  --npz /root/pi05_eval/libero_episodes0_1_2_100frames_each.npz \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --frames 300 \
  --warmup 3 \
  --configs bf16_baseline,bf16_cache2 \
  --out <output.npz> \
  --bad-frame-dir <bad-frame-dir> \
  --bad-frame-topk 30
```

重要修正：

- 多 episode NPZ 中 task 会变化，评测脚本需要按 `episode_index + task` 分段调用 `set_prompt()`。
- `cache_frames=2` 的 warmup 会污染 temporal KV cache，warmup 后必须 reset prompt/frame counter。
- 当前结论以 prompt fix + warmup reset 后的 300 帧结果为准。

## 当前 300 帧结果

### 统一对比：以 BF16 baseline 为参考

下表把 300 帧结果放在同一 action-similarity 口径下比较；所有 action similarity 都以 `BF16 baseline` 的 action chunk 为参考。BF16 `baseline/cache2` 行已更新为补齐 encoder/decoder BF16 GEMM autotune 后的当前结果。`pool`/`vitpack` 行保留失败 ablation 的 300 帧结果用于质量比较，其中 BF16 latency 没有在 autotune 后重跑，不能和当前 BF16 baseline latency 当成严格 A/B。

| config | p50 | Hz | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 217.8 ms | 4.59 | 0 | 0 | 1.000000 | 1.000000 | 0 |
| BF16 cache2 | 136.6 ms | 7.32 | 0.010644 | 0.049523 | 0.982405 | 0.478944 | 1.859494 |
| BF16 pool2 | 154.6 ms | 6.47 | 0.067040 | 0.170755 | 0.918667 | 0.098733 | 1.859494 |
| BF16 pool4 | 123.3 ms | 8.11 | 0.135037 | 0.315950 | 0.726205 | -0.768505 | 1.866992 |
| INT8 baseline | 164.2 ms | 6.09 | 0.094458 | 0.173249 | 0.875155 | 0.150065 | 1.919478 |
| INT8 cache2 | 104.5 ms | 9.57 | 0.094373 | 0.180533 | 0.881476 | 0.149197 | 1.919478 |
| BF16 vitpack12_f2 | 140.6 ms | 7.11 | 0.196959 | 0.381647 | 0.588616 | -0.864485 | 1.870741 |
| BF16 vitpack18_f2 | 146.2 ms | 6.84 | 0.110307 | 0.260673 | 0.794643 | -0.813983 | 1.866992 |
| INT8 vitpack12_f2 | 97.6 ms | 10.24 | 0.259306 | 0.581693 | 0.356749 | -0.876018 | 1.919478 |
| INT8 vitpack18_f2 | 102.3 ms | 9.77 | 0.187953 | 0.316793 | 0.646263 | -0.639782 | 1.919478 |

### BF16 GEMM autotune 前后对照

上面的统一总表只保留当前 BF16 baseline 口径。BF16 encoder/decoder GEMM autotune 前后的 300 帧结果如下保留用于 A/B 对照。

Latency:

| result | config | mean | p50 | Hz | p95 | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before autotune | BF16 baseline | 244.2 ms | 244.3 ms | 4.09 | 245.0 ms | 242.2 ms | 245.5 ms |
| after autotune | BF16 baseline | 217.7 ms | 217.8 ms | 4.59 | 219.6 ms | 215.6 ms | 220.9 ms |
| before autotune | BF16 cache2 | 153.1 ms | 152.4 ms | 6.56 | 244.7 ms | 61.8 ms | 246.1 ms |
| after autotune | BF16 cache2 | 137.1 ms | 136.6 ms | 7.32 | 218.7 ms | 55.1 ms | 219.8 ms |

Latency delta:

| config | mean delta | p50 delta | p95 delta | min delta | max delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | -26.6 ms | -26.5 ms | -25.4 ms | -26.6 ms | -24.6 ms |
| BF16 cache2 | -16.0 ms | -15.8 ms | -26.0 ms | -6.7 ms | -26.3 ms |

`cache2` 相对同次 BF16 baseline 的 action similarity 基本保持原口径：

| result | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before autotune | 0.010613 | 0.049782 | 0.982723 | 0.478394 | 1.859494 |
| after autotune | 0.010644 | 0.049523 | 0.982405 | 0.478944 | 1.859494 |

First action vs dataset `reference_actions`:

| result | config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| before autotune | BF16 baseline | 0.087353 | 0.132249 | 0.912225 | -0.897470 | 0.472012 |
| after autotune | BF16 baseline | 0.087326 | 0.132257 | 0.912264 | -0.897294 | 0.472012 |
| before autotune | BF16 cache2 | 0.081298 | 0.125228 | 0.938563 | -0.897470 | 0.410405 |
| after autotune | BF16 cache2 | 0.081274 | 0.125360 | 0.938584 | -0.897294 | 0.410405 |

统一口径结论：

- 速度最快的是 `INT8 vitpack12_f2`，但 `cos_mean=0.356749`，不可用。
- 接近 10 Hz 且相对最好的高速方案是 `INT8 cache2`，但相对 BF16 baseline 的 `cos_mean=0.881476`，仍不能视为无损。
- `BF16 pool2` 速度接近 `BF16 cache2`，但 `cos_mean=0.918667`，最坏帧 `cos_min=0.098733`，明显不如 cache2。
- 当前候选里 `BF16 cache2` 的均值相似度最好：`cos_mean=0.982405`；但 `cos_min` 仍说明它有 stale-cache 尾部坏帧，不能把离线相似度直接等同于 rollout success。
- 所有 vitpack 方案都有负 `cos_min`，最坏帧动作方向反转，不能作为主线。

BF16:

| config | p50 | Hz | p95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 217.8 ms | 4.59 | 219.6 ms | 215.6 ms | 220.9 ms |
| cache2 | 136.6 ms | 7.32 | 218.7 ms | 55.1 ms | 219.8 ms |

BF16 `cache2` vs BF16 baseline:

| mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.010644 | 0.049523 | 0.982405 | 0.478944 | 1.859494 |

INT8:

| config | p50 | Hz | p95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 164.2 ms | 6.09 | 164.4 ms | 163.6 ms | 166.1 ms |
| cache2 | 104.5 ms | 9.57 | 164.7 ms | 44.4 ms | 165.9 ms |

INT8 `cache2` vs INT8 baseline:

| mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.018609 | 0.062768 | 0.976375 | 0.337565 | 1.919478 |

First action vs dataset `reference_actions`:

| config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.087326 | 0.132257 | 0.912264 | -0.897294 | 0.472012 |
| BF16 cache2 | 0.081274 | 0.125360 | 0.938584 | -0.897294 | 0.410405 |
| INT8 baseline | 0.133107 | 0.226115 | 0.840911 | -0.965416 | 0.785405 |
| INT8 cache2 | 0.131475 | 0.212014 | 0.847293 | -0.979253 | 0.705048 |

### INT8 correctness 拆分实验

目的：先定位 INT8 相对 BF16 baseline 的 action 偏差来源，再决定是否继续写 INT8 kernel 或 tile 优化。主指标是每个 config 的 action chunk 与同次 `BF16 baseline` action chunk 的 300 帧相似度；dataset `reference_actions` 只作为辅助 sanity check。

历史实验命令：

以下 INT8 split configs 来自 FlashRT 本地诊断分支，结论已保留在文档中；FlashRT 当前主线 eval 脚本已删除这些测试 config，下面命令不再用于当前主线复现。

```bash
unset FVK_PI05_RTX_FORCE_INT8
unset FVK_PI05_RTX_INT8_ENCODER_ONLY
unset FVK_PI05_RTX_INT8_DECODER_ONLY
unset FVK_PI05_RTX_INT8_VISION

python examples/orin/eval_libero_offline.py \
  --npz /root/pi05_eval/libero_episodes0_1_2_100frames_each.npz \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --frames 300 \
  --warmup 3 \
  --configs baseline,int8_decoder_only,int8_encoder_only,int8_encoder_decoder,int8_cache2 \
  --out /root/pi05_eval/int8_split_300f.npz \
  --bad-frame-dir /root/pi05_eval/bad_frames_int8_split_300f \
  --bad-frame-topk 30
```

结果：

| config | p50 | Hz | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 | cos<0.9 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 216.4 ms | 4.62 | 0 | 0 | 1.000000 | 1.000000 | 0 | 0 |
| INT8 decoder only | 202.0 ms | 4.95 | 0.039251 | 0.064065 | 0.980933 | 0.802171 | 1.859494 | 24 |
| INT8 encoder only | 179.8 ms | 5.56 | 0.068800 | 0.141983 | 0.896222 | 0.132738 | 1.870741 | 78 |
| INT8 encoder+decoder | 164.2 ms | 6.09 | 0.094458 | 0.173249 | 0.875155 | 0.150065 | 1.919478 | 85 |
| INT8 cache2 | 104.5 ms | 9.57 | 0.094373 | 0.180533 | 0.881476 | 0.149197 | 1.919478 | 80 |

First action vs dataset `reference_actions`:

| config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.087336 | 0.132249 | 0.912257 | -0.897294 | 0.472012 |
| INT8 decoder only | 0.107688 | 0.165162 | 0.891191 | -0.871255 | 0.533029 |
| INT8 encoder only | 0.116096 | 0.185381 | 0.854122 | -0.991827 | 0.785405 |
| INT8 encoder+decoder | 0.133107 | 0.226115 | 0.840911 | -0.965416 | 0.785405 |
| INT8 cache2 | 0.131475 | 0.212014 | 0.847293 | -0.979253 | 0.705048 |

Per-dimension first-action MAE vs BF16 baseline:

| config | dim0 | dim1 | dim2 | dim3 | dim4 | dim5 | dim6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 decoder only | 0.047942 | 0.061696 | 0.085997 | 0.006263 | 0.011283 | 0.008810 | 0.046537 |
| INT8 encoder only | 0.115210 | 0.073252 | 0.140224 | 0.017615 | 0.031042 | 0.012792 | 0.067007 |
| INT8 encoder+decoder | 0.144303 | 0.115029 | 0.198835 | 0.020624 | 0.038786 | 0.017214 | 0.090963 |
| INT8 cache2 | 0.142498 | 0.112248 | 0.205619 | 0.020073 | 0.037806 | 0.017161 | 0.097736 |

结论：

- INT8 action 偏差主因是 `encoder INT8`，不是单独的 decoder INT8。`decoder only` 的 `cos_mean=0.980933`，但只省约 14 ms；`encoder only` 省约 36 ms，但 `cos_mean` 掉到 `0.896222`，最坏帧 `cos_min=0.132738`。
- `encoder+decoder` 的偏差基本继承 encoder 问题，再叠加 decoder 偏差；`cache2` 没有显著加重均值偏差，说明当前 INT8 correctness 主问题不是 stale cache，而是 encoder INT8 本身。
- 坏帧主要集中在 episode 2 后段，且 action dim0/dim2/dim6 的尾部误差较大。后续若继续 INT8，应先拆 encoder 内部 QKV/O/FFN 哪一类 GEMM 导致偏差，再考虑 tile 或 kernel 性能优化。
- 因此短期不建议先做 INT8 tile 提速；应该先做 encoder INT8 correctness 归因。

### INT8 encoder scope 归因实验

目的：进一步拆开 `encoder INT8`，确认 action drift 来自 attention 投影还是 FFN 投影。实验分支增加了本地诊断开关 `FVK_PI05_RTX_INT8_ENCODER_SCOPE`，支持 `qkv`、`o`、`attn`、`ffn`、`down` 和逗号组合。该开关只用于实验归因，不是当前主线优化接口。

历史实验命令：

以下 encoder-scope configs 和 `FVK_PI05_RTX_INT8_ENCODER_SCOPE` 来自 FlashRT 本地诊断分支，当前主线代码不再保留这组归因开关。

```bash
unset FVK_PI05_RTX_FORCE_INT8
unset FVK_PI05_RTX_INT8_ENCODER_ONLY
unset FVK_PI05_RTX_INT8_DECODER_ONLY
unset FVK_PI05_RTX_INT8_ENCODER_SCOPE
unset FVK_PI05_RTX_INT8_VISION

python examples/orin/eval_libero_offline.py \
  --npz /root/pi05_eval/libero_episodes0_1_2_100frames_each.npz \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --frames 300 \
  --warmup 3 \
  --configs baseline,int8_encoder_qkv,int8_encoder_o,int8_encoder_attn,int8_encoder_ffn,int8_encoder_down,int8_encoder_only \
  --out /root/pi05_eval/int8_encoder_scope_300f.npz \
  --bad-frame-dir /root/pi05_eval/bad_frames_int8_encoder_scope_300f \
  --bad-frame-topk 30
```

结果：

| config | p50 | Hz | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 | cos<0.9 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 244.1 ms | 4.10 | 0 | 0 | 1.000000 | 1.000000 | 0 | 0 |
| encoder qkv INT8 | 239.5 ms | 4.17 | 0.010818 | 0.035439 | 0.990626 | 0.812421 | 1.829502 | 17 |
| encoder o INT8 | 241.0 ms | 4.15 | 0.000488 | 0.000751 | 0.999997 | 0.999973 | 0.006171 | 0 |
| encoder attn INT8 | 236.0 ms | 4.24 | 0.011051 | 0.034274 | 0.990247 | 0.734902 | 1.822005 | 17 |
| encoder ffn INT8 | 191.1 ms | 5.23 | 0.063173 | 0.131493 | 0.907241 | 0.138730 | 1.866992 | 73 |
| encoder down INT8 | 228.0 ms | 4.39 | 0.002166 | 0.003398 | 0.999512 | 0.865920 | 0.026741 | 1 |
| encoder all INT8 | 185.1 ms | 5.40 | 0.068790 | 0.142144 | 0.896229 | 0.132443 | 1.866992 | 78 |

First action vs dataset `reference_actions`:

| config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.087353 | 0.132249 | 0.912225 | -0.897470 | 0.472012 |
| encoder qkv INT8 | 0.088421 | 0.137317 | 0.912632 | -0.898141 | 0.483635 |
| encoder o INT8 | 0.087346 | 0.132115 | 0.912229 | -0.897590 | 0.472012 |
| encoder attn INT8 | 0.087096 | 0.136567 | 0.918239 | -0.895578 | 0.490762 |
| encoder ffn INT8 | 0.112987 | 0.183768 | 0.858649 | -0.997624 | 0.736035 |
| encoder down INT8 | 0.087262 | 0.133950 | 0.912252 | -0.900488 | 0.472012 |
| encoder all INT8 | 0.116103 | 0.185271 | 0.854114 | -0.991952 | 0.785405 |

Per-dimension first-action MAE vs BF16 baseline:

| config | dim0 | dim1 | dim2 | dim3 | dim4 | dim5 | dim6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| encoder qkv INT8 | 0.018367 | 0.014984 | 0.021843 | 0.002429 | 0.003722 | 0.002099 | 0.017833 |
| encoder o INT8 | 0.000904 | 0.000859 | 0.000882 | 0.000112 | 0.000186 | 0.000125 | 0.000487 |
| encoder attn INT8 | 0.018168 | 0.014810 | 0.024410 | 0.002427 | 0.003919 | 0.002092 | 0.024206 |
| encoder ffn INT8 | 0.112372 | 0.068518 | 0.134918 | 0.015704 | 0.032056 | 0.010548 | 0.054023 |
| encoder down INT8 | 0.003612 | 0.003207 | 0.004846 | 0.000444 | 0.000860 | 0.000473 | 0.001375 |
| encoder all INT8 | 0.115117 | 0.073155 | 0.140501 | 0.017597 | 0.031029 | 0.012784 | 0.066844 |

结论：

- `encoder FFN INT8` 是 full encoder INT8 action drift 的主要来源。它贡献主要速度收益，也贡献主要错误：`cos_min=0.138730`，73/300 帧低于 0.9。
- `encoder o INT8` 几乎无损，但速度收益很小。
- `encoder down INT8` 基本安全，只有 1/300 帧低于 0.9，但速度收益有限。
- `encoder qkv/attn INT8` 均值可接受，但有少量尾部坏帧，不能视为完全无损。
- 因此 full encoder INT8 的 correctness 问题不是 attention 优先，而是 FFN gate/up 路径优先。

### INT8 encoder safe mixed precision 实验

目的：验证“保留 FFN gate/up BF16，只量化相对安全的 qkv/o/down”是否能保住质量并获得可用速度收益。配置：

```text
int8_encoder_safe = qkv,o,down INT8 + FFN gate/up BF16
```

历史实验命令：

以下 `int8_encoder_safe` 配置同样来自 FlashRT 本地诊断分支，当前主线只保留实验结论，不保留该 config。

```bash
python examples/orin/eval_libero_offline.py \
  --npz /root/pi05_eval/libero_episodes0_1_2_100frames_each.npz \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --frames 300 \
  --warmup 3 \
  --configs baseline,int8_encoder_safe,int8_encoder_qkv,int8_encoder_down,int8_encoder_only \
  --out /root/pi05_eval/int8_encoder_safe_300f.npz \
  --bad-frame-dir /root/pi05_eval/bad_frames_int8_encoder_safe_300f \
  --bad-frame-topk 30
```

结果：

| config | p50 | Hz | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 | cos<0.9 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 243.0 ms | 4.12 | 0 | 0 | 1.000000 | 1.000000 | 0 | 0 |
| int8_encoder_safe | 222.6 ms | 4.49 | 0.010809 | 0.027237 | 0.993071 | 0.816462 | 0.180501 | 12 |
| int8_encoder_qkv | 237.2 ms | 4.22 | 0.010818 | 0.035439 | 0.990626 | 0.812421 | 1.829502 | 17 |
| int8_encoder_down | 228.0 ms | 4.39 | 0.002166 | 0.003398 | 0.999512 | 0.865920 | 0.026741 | 1 |
| int8_encoder_only | 185.0 ms | 5.41 | 0.068790 | 0.142144 | 0.896229 | 0.132443 | 1.866992 | 78 |

First action vs dataset `reference_actions`:

| config | mae_mean | mae_p95 | cos_mean | cos_min | max_abs_p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 0.087353 | 0.132249 | 0.912225 | -0.897470 | 0.472012 |
| int8_encoder_safe | 0.087372 | 0.133818 | 0.917144 | -0.893025 | 0.472012 |
| int8_encoder_qkv | 0.088421 | 0.137317 | 0.912632 | -0.898141 | 0.483635 |
| int8_encoder_down | 0.087262 | 0.133950 | 0.912252 | -0.900488 | 0.472012 |
| int8_encoder_only | 0.116103 | 0.185271 | 0.854114 | -0.991952 | 0.785405 |

结论：

- `int8_encoder_safe` 比 qkv-only 快约 14.6 ms，质量也略好：`cos_mean=0.993071`，bad frames 从 17 降到 12。
- 但它相对 BF16 baseline 只快约 20 ms，单帧仍是 222.6 ms，不接近 10 Hz；因此它不是最终性能主线。
- 该实验的价值主要是诊断：排除 FFN gate/up 后，INT8 action drift 大幅缓解，进一步证明 FFN gate/up 是主要问题。
- 当前不建议继续把 `int8_encoder_safe` 当成产品化优化；如果继续 INT8，只应研究 FFN gate/up 的 correctness，否则应转回 BF16 cache2 / adaptive cache。

相关 BF16/INT8 总表输出文件：

```bash
/root/pi05_eval/int8_encoder_scope_300f.npz
/root/pi05_eval/int8_encoder_scope_300f_per_frame.csv
/root/pi05_eval/bad_frames_int8_encoder_scope_300f

/root/pi05_eval/int8_encoder_safe_300f.npz
/root/pi05_eval/int8_encoder_safe_300f_per_frame.csv
/root/pi05_eval/bad_frames_int8_encoder_safe_300f
```

输出文件：

```bash
/root/pi05_eval/flashrt_bf16_autotune_300f.npz
/root/pi05_eval/flashrt_bf16_autotune_300f_per_frame.csv
/root/pi05_eval/bad_frames_bf16_autotune_300f

/root/pi05_eval/flashrt_int8_after_upstream_300f.npz
/root/pi05_eval/flashrt_int8_after_upstream_300f_per_frame.csv
/root/pi05_eval/bad_frames_int8_after_upstream_300f

/root/pi05_eval/int8_split_300f.npz
/root/pi05_eval/int8_split_300f_per_frame.csv
/root/pi05_eval/bad_frames_int8_split_300f
```

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
- 补齐 encoder/decoder BF16 GEMM autotune 后，baseline p50 为 217.8 ms；cache2 进一步降到 136.6 ms。
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

## cache2 的含义

`cache2` 是 temporal cache 调度优化，不是单个新算子：

- key frame 跑完整 vision encoder + decoder。
- reuse frame 复用上一帧视觉/encoder cache，只走 decoder-only graph。
- 因此 latency 出现交替：当前 BF16 autotune 后完整帧约 218 ms、复用帧约 55 ms；当前 INT8 完整帧约 164 ms、复用帧约 44 ms。
- p50 代表交替序列的中位数，不等同于所有帧都稳定在该延迟。

### cache 全对比：固定窗口 vs 自适应 refresh

2026-05-22 新增实验：在 FlashRT 历史实验脚本 `eval_cache_libero.py` 中使用同一数据、同次 baseline 口径，对比 cache2/3/4 的固定窗口和 adaptive pixel-MAE refresh（threshold=6.5）版本。300 帧 BF16，同次 run。

| config | p50 | mean | forced_full | cos_mean | cos_min | cos<0.9 |
|---|---|---|---|---|---|---|
| BF16 cache2 | 135.5 ms | 135.6 ms | — | 0.982403 | 0.478313 | 4 |
| BF16 cache3 | 56.6 ms | 110.3 ms | — | 0.960540 | 0.122606 | 24 |
| BF16 cache4 | 55.9 ms | 95.8 ms | — | 0.945754 | -0.058609 | 30 |
| BF16 adaptive_cache2 | 216.4 ms | 146.2 ms | 34/300 | 0.983152 | 0.478313 | 4 |
| BF16 adaptive_cache3 | 55.6 ms | 131.5 ms | 89/300 | 0.974735 | 0.434654 | 10 |
| BF16 adaptive_cache4 | 56.6 ms | 128.0 ms | 106/300 | 0.964295 | -0.058609 | 26 |

固定窗口结论：

- **cache2 → cache3：p50 大幅下降**（136→57ms），因为 median 从全帧切到 reuse 帧。
- **cache3 → cache4：p50 几乎不变**（56.6→55.9ms），因为 median 已经在 reuse 区间，增加更多 reuse 不会进一步拉 p50；但 `cos_min` 从 0.123 掉到 -0.059，尾部坏帧明显恶化。
- **cache4 不做主线**：速度不再提升，质量反而更差。

自适应 vs 固定对比结论：

- **adaptive_cache2：不可用**。cache=2 只有 1 个 reuse 槽位，额外 34 次 forced_full 就让 full 帧超过一半，p50 跳到 full-frame 水平。`cos_min` 与 cache2 相同，说明触发刷新的帧不是最坏帧。
- **adaptive_cache3：唯一显著改善的配对**。p50 保持 56ms（与 cache3 一致），`cos_mean` 从 0.961 提升到 0.975，`cos_min` 从 0.123 提升到 0.435，cos<0.9 帧数从 24 降到 10。这说明 adaptive refresh 能在 cache3 窗口内有效捕获需要提前刷新的场景变化。
- **adaptive_cache4：均值改善，尾部无改善**。`cos_min=-0.059` 与 cache4 一致，说明 window=4 时累积漂移太大，adaptive refresh 不足以挽救。

当前保留路线：
- **bf16_cache2**：最稳的 correctness baseline，cos_mean 最高（0.982），适合需要高保真的场景。
- **bf16_adaptive_cache3**：最佳速度/质量交换，p50=56ms + cos_mean=0.975。适合延迟敏感、容忍少量 action drift 的场景。
- **bf16_cache3/4 和 adaptive_cache2/4 不保留为主赛道**。

实验脚本：FlashRT 历史本地脚本 `examples/orin/eval_cache_libero.py`，只含 cache 相关配置，剥离 pool/steps/int8 等非 cache 实验；该脚本不是本仓库脚本。

输出文件：
```bash
/root/pi05_eval/cache_fixed_compare.npz     # baseline + cache2/3/4
/root/pi05_eval/cache_adaptive_compare.npz   # baseline + adaptive_cache2/3/4
```

## Closed-loop LIBERO 结果

### BF16 cache/adaptive full sweep

这组结果来自 FlashRT Thor-style Orin closed-loop eval，尽量对齐 FlashRT `examples/thor/eval_libero.py` 的协议，只替换为 Orin/Pi0.5 frontend 和 cache 设置。它使用 `OffScreenRenderEnv`、LIBERO task init states、task-language prompt、`model.predict(images=[agentview, wrist], prompt=task_description)`、`replan_steps=5`，并按 config/task 做 subprocess isolation。

范围：`libero_spatial` 全 10 个 tasks，每个 task 3 trials，共 30 trials/config。

| config | success | used_full | forced_full | p50 |
| --- | ---: | ---: | ---: | ---: |
| BF16 cache=1 | 24/30 = 80.0% | 807/807 | — | ~219 ms |
| BF16 cache=2 | 22/30 = 73.3% | 446/884 | — | ~186 ms |
| BF16 cache=3 | 11/30 = 36.7% | 402/1176 | — | ~57 ms |
| BF16 adaptive cache=3 | 21/30 = 70.0% | 864/867 | 834/867 | ~220 ms |

字段含义：

- `used_full`: 实际执行 full vision/encoder path 的 replan 次数。固定窗口的 scheduled full frame 也计入这里。
- `forced_full`: adaptive policy 主动提前刷新触发的次数，不包含固定窗口 scheduled full。

结论：

- BF16 cache=2 是 closed-loop 中较保守的 cache 路线：success 接近 cache=1，但 speedup 不大，因为大约一半 replans 仍跑 full pipeline。
- BF16 cache=3 很快，但 success 从 24/30 掉到 11/30，说明 fixed temporal cache reuse 有明显 task-level cliff。
- BF16 adaptive cache=3 能恢复部分 success（11/30 -> 21/30），但 forced_full 过多，几乎退回 full pipeline latency。因此 frame-level pixel-MAE refresh 不适合做默认策略。

### INT8 quick sweep（临时 binding fix 后）

这组结果来自 2026-05-26 的快速检查，用于回答“修复 INT8 calibration crash 后，INT8 closed-loop 是否可用”。它只覆盖 `libero_spatial` tasks `0,1,2`，每个 task 3 trials，共 9 trials/config。实验在临时 re-apply FlashRT PR #40 binding fix 的 Jetson 工作树上运行，不代表 FlashRT upstream/main 原样结果。

| config | success | used_full | p50 | Hz |
| --- | ---: | ---: | ---: | ---: |
| BF16 baseline | 7/9 = 77.8% | 228/228 | 220.0 ms | 4.5 |
| INT8 baseline | 6/9 = 66.7% | 259/259 | 166.8 ms | 6.0 |
| INT8 cache2 | 4/9 = 44.4% | 153/305 | 125.9 ms | 7.9 |

分 task：

| config | task 0 | task 1 | task 2 |
| --- | ---: | ---: | ---: |
| BF16 baseline | 3/3 | 1/3 | 3/3 |
| INT8 baseline | 2/3 | 1/3 | 3/3 |
| INT8 cache2 | 1/3 | 1/3 | 2/3 |

结论：

- 临时 binding fix 后，INT8 closed-loop 能完整跑完，不再在 calibration 阶段 crash。
- INT8 baseline 有明确速度收益：closed-loop p50 从约 `220 ms` 降到 `167 ms`，但 success 从 `7/9` 降到 `6/9`。
- INT8 cache2 接近 `8 Hz`，但 success 进一步降到 `4/9`，说明 `INT8 + temporal cache` 的 task-level 风险明显高于 BF16 baseline。
- 该 quick sweep 与 300 帧 offline cosine 结论一致：INT8 的速度收益是真实的，但 action drift / task degradation 不能忽略。

## 已证伪或暂不保留的实验

早期做过 30/100 帧和多种 ablation，结论已经被 300 帧结果覆盖，不再作为主结论保留。

已证伪方向：

- post-ViT pooling `pool2/pool4`: 速度提升明显，但 action 漂移过大。
- `vitpack12_f2`: layer 12 后做 2x2 vision token pack，单帧速度好，但 300 帧大面积漂移。
- RMS/Norm-gated soft packer: 没有改善 `vitpack12_f2`，说明 token RMS 不能代表控制重要性。
- main-only vision pack: 更慢，且最坏帧更差。
- stage refresh: 强制在 `align_grasp/lift_move` 刷新质量很好，但 p50 回到接近 baseline，性价比不如 cache2。
- cache4、patch/edge/shallow patch-embed refresh 初筛: 相对 `cache3 + pixel MAE` 没有更好的质量/延迟交换，代码中不继续保留这些分支。

这些实验相关代码不应继续作为主线提交；保留报告结论即可。

### 完整 ViT 后 pooling 复测

实现方式：

- 完整运行 SigLIP ViT 27 层，保持原始 `2 views x 16 x 16 = 512` visual tokens。
- ViT 完成后，对每个 view 的 `16x16` token grid 做规则 average pooling。
- `pool2`: 每 `2x2` token 平均成 1 个 token，`512 -> 128`。
- `pool4`: 每 `4x4` token 平均成 1 个 token，`512 -> 32`。
- pooling 后再进入 vision final norm、multi-modal projector、Gemma encoder 和 decoder。
- 该方法不减少 ViT 27 层本身的计算，只减少 projector/Gemma encoder/decoder 的 token 长度。

BF16 300 帧：

这组 post-ViT pooling 复测发生在 BF16 encoder/decoder autotune 前；pooling 的 action drift 结论仍有效，但表内 BF16 latency 不是当前 autotune 后 latency。

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 243.1 ms | 4.11 | - | - | - | - |
| pool2 | 154.6 ms | 6.47 | 0.067040 | 0.170755 | 0.918667 | 0.098733 |
| pool4 | 123.3 ms | 8.11 | 0.135037 | 0.315950 | 0.726205 | -0.768505 |

输出文件：

```bash
/root/pi05_eval/flashrt_postvit_pool_bf16_300f.npz
/root/pi05_eval/flashrt_postvit_pool_bf16_300f_per_frame.csv
/root/pi05_eval/bad_frames_postvit_pool_bf16_300f
```

结论：

- `pool2` 的旧 p50 与 cache 路线同量级，但当前 autotune 后 `BF16 cache2` 更快，且 action similarity 明显更好。
- `pool4` 速度更快，但已经出现负 `cos_min`，不可用。
- 完整 ViT 后 pooling 比中间层 `vitpack` 更稳，但仍不是 correctness-preserving。
- 这说明即便不破坏 ViT 内部计算，只要把 projector/Gemma 看到的视觉 token 做规则平均压缩，Pi0.5 的动作输出仍会明显漂移。

### ViT 中间层 2x2 token pack 复测

复测配置：

- `vitpack12_f2`: SigLIP layer 12 后把两个 view 都做规则 2x2 average pack，vision tokens 从 512 降到 128。
- `vitpack18_f2`: SigLIP layer 18 后做同样的 2x2 average pack。
- 两者都不是输入端 pooling，而是在 ViT 中间层压缩后继续跑剩余 vision layers、projector、Gemma encoder/decoder。

实现方式：

- 在 `Pi05Pipeline.vision_encoder()` 中，先按原始 512 visual tokens 运行前 N 层 SigLIP。
- 当 `vision_pack_layer=N` 时，对每个 view 的 `16x16` token grid 做规则 `2x2 average pooling`。
- token 数从 `2 views x 256 = 512` 降到 `2 views x 64 = 128`。
- pooled tokens 被拷回 vision residual buffer 的前 128 行，后续 SigLIP layer、vision final norm、multi-modal projector、Gemma encoder/decoder 都只处理 128 个 visual tokens。
- attention backend 对 SigLIP attention 的 `q_seq` 做了动态切片，使后续 vision attention 可以在每 view 64 tokens 上运行。
- 该实验只使用规则 average pack，没有 soft gating、top-k、非规则 gather，也没有重新训练模型。

BF16 300 帧：

这组 BF16 vitpack 复测也发生在 BF16 encoder/decoder autotune 前；表内 latency 用于说明 token pack 的速度收益，不是当前 BF16 baseline 的严格 A/B。

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 243.9 ms | 4.10 | - | - | - | - |
| vitpack12_f2 | 140.6 ms | 7.11 | 0.196959 | 0.381647 | 0.588616 | -0.864485 |
| vitpack18_f2 | 146.2 ms | 6.84 | 0.110307 | 0.260673 | 0.794643 | -0.813983 |

INT8 300 帧：

本表 INT8 action similarity 是相对同次 INT8 baseline；统一总表中的 INT8 vitpack 行是相对 BF16 baseline，因此两处数值口径不同。

| config | p50 | Hz | mae_mean vs baseline | mae_p95 | cos_mean | cos_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 164.8 ms | 6.07 | - | - | - | - |
| vitpack12_f2 | 97.6 ms | 10.24 | 0.235052 | 0.587194 | 0.427604 | -0.873032 |
| vitpack18_f2 | 102.3 ms | 9.77 | 0.144822 | 0.310017 | 0.751075 | -0.632650 |

输出文件：

```bash
/root/pi05_eval/flashrt_vitpack12_18_bf16_300f.npz
/root/pi05_eval/flashrt_vitpack12_18_bf16_300f_per_frame.csv
/root/pi05_eval/bad_frames_vitpack12_18_bf16_300f

/root/pi05_eval/flashrt_vitpack12_18_int8_300f.npz
/root/pi05_eval/flashrt_vitpack12_18_int8_300f_per_frame.csv
/root/pi05_eval/bad_frames_vitpack12_18_int8_300f
```

结论：

- `vitpack18_f2` 明显优于 `vitpack12_f2`，说明越晚压缩越能保留部分动作相关视觉特征。
- 但 300 帧下两者都出现负 `cos_min`，动作最坏帧方向已经反了，不能作为 correctness-preserving 优化。
- INT8 + `vitpack18_f2` p50 达到 102.3 ms，接近 10 Hz，但质量明显不可接受。
- 这进一步确认：Pi0.5/LIBERO 当前瓶颈不能靠规则 2x2 vision token average pack 解决；速度收益是真实的，但 action similarity 代价过大。
- 当前主线仍应优先保留 `cache2`，后续如继续优化，应围绕 adaptive refresh 或更保守的 temporal reuse，而不是中间层空间压缩。

不可行原因：

- `2x2 average pooling` 会把夹爪边缘、杯子/盘子边界、接触点和背景 token 混合，VLA action 对这些局部几何比分类任务敏感得多。
- 主视图和 wrist view 都承载控制信息：主视图给全局相对位置，wrist view 给近场对准。对两个 view 同时压缩会同时破坏全局和局部几何。
- `vitpack18_f2` 比 `vitpack12_f2` 好，说明“压得晚”确实能缓解问题；但仍然出现负 `cos_min`，说明根因不是单纯压缩时机，而是 average pack 本身改变了视觉 token 分布。
- 原 Pi0.5 后续 projector/Gemma encoder 没有见过 128 个 average-pooled visual tokens 的输入分布，也没有针对新的 token layout/位置序列重训。
- INT8 本身已经相对 BF16 有明显 action drift，再叠加 vitpack 后误差进一步放大。

删除原因：

- `vitpack` 代码是为了复现实验临时恢复的研究分支，不是主线 runtime 优化。
- 300 帧 BF16/INT8 都证明它不是 correctness-preserving 方法。
- 保留这类开关会增加 pipeline、attention backend 和 eval config 的维护复杂度，并容易让后续 benchmark 误用不可行配置。
- 因此主分支只保留实验记录和结果，不保留 `vision_pack_layer` / `vitpack*_f2` 代码路径。

## INT8 crash 复现与临时修复验证

2026-05-26 在最新 FlashRT `upstream/main` 上重新测试 full INT8，`bench_pi05.py` 和 closed-loop eval 都会在 calibration 阶段失败：

```text
FVK_PI05_RTX_FORCE_INT8/INT8_ENCODER_ONLY set: INT8 encoder+decoder
Loading checkpoint: /root/models/pi05_libero_finetuned_v044
Calibrating...
RuntimeError: cuBLAS error at /root/FlashRT/csrc/gemm/gemm_runner.cu:675 code=13
```

栈位置是 `bench_pi05.py -> calibrate_with_real_data() -> pipeline.run_pipeline() -> vision_encoder() -> _vision_layer() -> gemm.bf16_nn()`。因此这不是 closed-loop 脚本的问题，而是 INT8 路径在校准/图捕获前已经触发底层错误。

临时重新应用 PR #40 的 2 行 binding fix 后，INT8 benchmark 可以跑通。该 fix 修改 `csrc/bindings.cpp` 中 `bias_gelu_bf16` / `bias_gelu_bf16_strict` 的 pybind alias：

```cpp
bias_gelu_inplace_bf16(..., seq_len * dim, dim, ...)
```

C++ 签名需要 `(M, N)`，Python 已经传入 `(seq_len, dim)`，binding 再乘一次 `dim` 会让 kernel 处理 `seq_len * dim * dim` 个元素，导致越界。修复为：

```cpp
bias_gelu_inplace_bf16(..., seq_len, dim, ...)
```

验证命令：

```bash
cd /root/FlashRT
source /root/pi0.5/bin/activate
export FVK_PI05_RTX_FORCE_INT8=1
python examples/orin/bench_pi05.py \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --num-views 2 --pool 1 --layers 27 --steps 10 --cache-frames 1 \
  --warmup 2 --reps 5
```

临时 fix 后结果：

```text
Config : num_views=2 pool=1 layers=27 steps=10 cache_frames=1
p50    : 163.9 ms -> 6.10 Hz
p95    : 164.1 ms
min    : 163.6 ms -> 6.11 Hz
```

状态说明：

- PR #40 `fix(pi05): fix bf16 bias gelu binding shape` 曾合入，但随后被 #41 revert。
- 当前 FlashRT `upstream/main` 不含该 fix，所以 full INT8 会重新触发 calibration crash。
- Jetson 上的验证是在临时分支 `tmp/reapply-pi05-bias-gelu-binding-test` 上进行，只用于确认 crash 根因；不代表 FlashRT upstream/main 原样结果。
- 该问题是实际越界 bug，不是性能实验开关。若继续使用 INT8 路径，应先重新处理这个 binding shape 问题。

## INT8 closed-loop quick sweep（临时 binding fix 后）

目的：确认 full INT8 在修复 calibration crash 后，closed-loop LIBERO 是否仍有可用的 task-level 表现。该实验使用临时 re-apply FlashRT PR #40 binding fix 的 Jetson 工作树，不是 FlashRT upstream/main 原样结果。结果表已经整理在上面的 “Closed-loop LIBERO 结果 / INT8 quick sweep” 小节。

命令：

```bash
python examples/orin/eval_libero_int8_closed_loop.py \
  --checkpoint /root/models/pi05_libero_finetuned_v044 \
  --task_suite libero_spatial \
  --quick \
  --num_trials 3 \
  --configs bf16_baseline,int8_baseline,int8_cache2 \
  --output /root/pi05_eval/orin_closed_loop_int8_quick_3tasks_after_biasfix.json
```

若要作为严肃结论，需要在修复 INT8 crash 后跑 full `libero_spatial` 10 tasks × 3 trials。

## Nsight Systems Profiling 分析（2026-05-22）

### 评测设置

设备：Jetson AGX Orin (SM87, 14 SM, 930.75 MHz GPU clock, MAXN 模式)
工具：NVIDIA Nsight Systems 2024.5.4
方法：

- BF16：在 FlashRT checkout 中用 `examples/orin/eval_libero_offline.py --cuda-profiler-range` 配合 `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node`，只捕获 autotune 后 BF16 baseline 的 30 帧 measured CUDA graph replay。
- INT8：本节保留 2026-05-21 的旧 INT8 profile 结论；它与新的 BF16 profile 不是同一份 replay-only trace，后续若继续 INT8 profiling 应按 FlashRT `eval_libero_offline.py` 重新抓同口径报告。

profile 文件保留在本地 FlashRT 工作区或本机实验目录，不作为本仓库 tracked artifact：
- `pi05_bf16_baseline_30f.nsys-rep` — autotune 后 BF16 baseline replay-only profile，30 帧 p50 `220.5 ms`
- `pi05_int8_baseline_5f.nsys-rep` — 旧 INT8 baseline profile，p50 约 `166 ms`

在本地用 "NVIDIA Nsight Systems" 打开即可查看 CUDA kernel 时间线。

### BF16 baseline GPU kernel 分布

```
 Rank  Kernel                                              占比      总时间     调用次数   单次平均
 ────  ──────────────────────────────────────────────────  ─────    ────────  ────────  ───────
  1    ampere_bf16_s16816gemm_bf16_128x128                 44.3%    2898 ms     5340     543 µs
  2    ampere_bf16_s16816gemm_bf16_256x128                 13.8%     902 ms      510    1768 µs
  3    ampere_bf16_s16816gemm_bf16_128x64_32x6             9.5%      623 ms    10800      58 µs
  4    ampere_bf16_s16816gemm_bf16_128x64_64x4             7.1%      467 ms    10800      43 µs
  5    gate_silu_mul                                        5.1%      332 ms     5910      56 µs
  6    ampere_bf16_s16816gemm_bf16_128x64_64x3             3.1%      202 ms     5400      37 µs
  7    FlashAttention splitkv                               2.8%      181 ms     5400      34 µs
  8    bias_res_kernel                                      2.5%      166 ms     3060      54 µs
  9    add_bias_bf16                                        1.6%      102 ms      840     121 µs
 10    FlashAttention hdim=256                              1.6%      102 ms      510     200 µs
 11    qkv_split_rope                                       1.4%       93 ms     5940      16 µs
 12    qkv_split                                            1.3%       88 ms      810     109 µs
 13    gelu                                                 1.2%       79 ms      810      97 µs
```

**BF16 结论：补齐 encoder/decoder cuBLASLt autotune 后，热点仍然是 BF16 GEMM。** 单个 `128x128` GEMM family 仍占 `44.3%`；`256x128` encoder large shape 和 `128x64` decoder small-M family 继续占主要剩余时间。FlashAttention splitkv 只有 `2.8%`，attention 仍不是第一优化目标。

新 BF16 replay-only `.nsys-rep` 的 `cuda_gpu_kern_sum` 里最重的 kernel family 是：

- `ampere_bf16_s16816gemm_bf16_128x128...`：`44.3%`
- `ampere_bf16_s16816gemm_bf16_256x128...`：`13.8%`
- `ampere_bf16_s16816gemm_bf16_128x64...`：`9.5% + 7.1% + 3.1%`
- `gate_silu_mul`：`5.1%`
- `FlashAttention splitkv`：`2.8%`

这意味着 BF16 后续优化应继续围绕 GEMM shape class 选择，而不是优先改 attention。单独优化 activation/residual 类 kernel 的上限也更低；它们更适合作为 GEMM 调度或 epilogue 融合机会的一部分评估。

### INT8 baseline GPU kernel 分布

```
 Rank  Kernel                                              占比      总时间     调用次数   单次平均
 ────  ──────────────────────────────────────────────────  ─────    ────────  ────────  ───────
  1    elementwise_kernel (dequant/辅助, 校准阶段)          45.1%    1538 ms      180    8542 µs
  2    CUTLASS INT8 GEMM (大 tile)                          5.3%     181 ms      204     889 µs
  3    CUTLASS INT8 GEMM (小 tile, 主力)                    5.1%     174 ms     5112      34 µs
  4    CUTLASS INT8 GEMM (中 tile)                          4.4%     149 ms     1328     112 µs
  5    ampere_bf16_128x128 (未量化的 BF16 GEMM)             4.3%     146 ms      721     236 µs
  6    FlashAttention splitkv                               1.3%      45 ms     1260      36 µs
  7    quantize_int8_rowwise                                1.6%      55 ms     2656      21 µs
  8    ampere_bf16_128x256 (未量化 BF16)                    1.4%      49 ms      234     210 µs
  9    gate_residual_ada_norm_int8 (融合)                   0.7%      24 ms     2450      10 µs
 10    qkv_split_rope                                       0.5%      16 ms     1332      12 µs
     其他 (bias_res, add_bias, rms_norm_int8 等)          ~2-3%     ~70 ms     多       5-120 µs
```

**INT8 结论：CUTLASS INT8 GEMM ~15% + 遗留 BF16 GEMM ~9% = ~24% GEMM 时间。** 注意 INT8 profile 包含了权重量化的 GPU 时间（45% 的 elementwise kernel 来自模型加载阶段的量化），这不属于每帧推理开销。INT8 GEMM 单次 34-112µs 比 BF16 快 2-7×，但总 kernel 调用次数从 ~12,000 增加到 ~25,000（quant/dequant 来回转换）。

Jetson 上直接重跑原始 INT8 `.nsys-rep` 后，主运行时的前几项 kernel 也一致：

- `CUTLASS INT8 GEMM` 仍是主干算子
- `ampere_bf16_128x128...` 和 `ampere_bf16_128x256...` 仍有残留
- `quantize_int8_rowwise_kernel`、`unrolled_elementwise_kernel`、`elementwise_kernel` 构成了明显的量化/搬运成本
- `FlashAttention splitkv` 仍然只占很小比例

所以 INT8 的进一步收益主要来自补齐剩余 BF16 GEMM 和减少 quant/dequant 相关开销，而不是改 attention。

#### INT8 replay-only 残留 BF16 归因

为了排除 load、权重量化、calibration 和 CUDA Graph capture 的噪声，历史诊断时曾在 FlashRT profiling 入口只包住 measured loop 的 CUDA profiler start/stop。Jetson 上用 `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node` 抓默认 INT8 baseline 的 5 次 full-frame graph replay。当前文档保留归因结论；后续若重抓 INT8，优先沿用 FlashRT eval 脚本的 `--cuda-profiler-range` 口径。

默认 INT8 replay-only profile 中，最重的残留 BF16 GEMM 是：

| Kernel | 5-frame calls | Calls/frame | 5-frame time | 归因 |
|---|---:|---:|---:|---|
| `ampere_bf16_*_128x128_*` | 545 | 109 | 108.2 ms | SigLIP `27 x 4` layer GEMM + vision projector |
| BF16 `relu_256x64` | 50 | 10 | 0.93 ms | decoder action input projection |
| BF16 `64x64` | 50 | 10 | 0.60 ms | decoder action output projection |
| BF16 `128x128` small CUTLASS | 5 | 1 | 0.43 ms | vision patch embedding |

调用数和 Pi0.5 代码路径能直接对上：

- 默认 full INT8 已覆盖 Gemma encoder/decoder 的主 QKV/O/FFN GEMM。
- 默认仍保留 BF16 的 vision 路径包含 SigLIP 每层 QKV/O/FFN up/down，`27 x 4 = 108` 次 full-frame GEMM。
- 再加一次 vision projector，正好是 profile 里的 `109` 次每帧 BF16 热 GEMM。
- 之前混合 profile 中看到的 `ampere_bf16_128x256` 不是 clean graph replay 的主要残留热项。

打开已有 `FVK_PI05_RTX_INT8_VISION=1` 后再抓同口径 replay-only profile：

- 默认 profile 的 `545` 次 BF16 vision 热 GEMM 消失。
- rowwise INT8 quantize 从 `1970` 次增到 `2510` 次，正好每帧新增 `108` 次，与 SigLIP layer GEMM 数一致。
- INT8 CUTLASS GEMM 接走这些 vision GEMM；残留 BF16 只剩每帧一次 projector、每帧一次 patch embedding，以及 decoder action in/out 的小 GEMM。

这把方向 1 的问题收窄了：默认 INT8 的大块 BF16 残留不是未知 Gemma attention 投影，而是 vision encoder 默认仍走 BF16。vision INT8 已有现成路径，但端到端只再省约 `2-4 ms`，因为它把 vision BF16 GEMM 换成了 INT8 GEMM 加 rowwise quantize，而整体时间仍被 decoder/encoder INT8 GEMM 主干和量化边界支配。

#### Decoder static INT8 rowwise scale 实验

clean replay profile 还显示默认 INT8 每个 full frame 有 `394` 次 `quantize_int8_rowwise`：

- encoder `attn_o + ffn_down`: `17 + 17 = 34/frame`
- decoder `attn_o + ffn_down`: `18 layers x 10 steps x 2 = 360/frame`

因此做过一个最小方向 2 实验：复用已有 `quantize_int8_rowwise_static`，将 decoder `_int8_gemm()` 的 post-attn 和 FFN-down 激活 scales 在单样本 calibration 后冻结。它只替换这 `360/frame` 个 decoder dynamic rowwise quantize，不改 INT8 GEMM 权重或其他算子。

Jetson 同树 15-rep latency：

| Config | p50 | 结论 |
|---|---:|---|
| dynamic decoder INT8 | 163.4 ms | 当前 baseline |
| static decoder rowwise scales | 163.3 ms | 噪声级收益 |

30 帧 LIBERO action 输出与已存动态 INT8 直接比较：

| Output | MAE mean | cosine mean | cosine min | max abs p95 |
|---|---:|---:|---:|---:|
| static baseline vs dynamic baseline | 0.002711 | 0.999933 | 0.999607 | 0.036774 |
| static cache2 vs dynamic cache2 | 0.003062 | 0.999898 | 0.999484 | 0.047526 |

结论：冻结 decoder rowwise scales 在 30 帧 smoke 范围内没有明显质量风险，但也没有端到端速度收益。这说明“把 dynamic quantize 换成 static quantize”不足以解决方向 2；后续应看更重的 INT8 GEMM 邻接边界和融合机会，而不是保留这个单独开关。

#### Vision INT8 / projector INT8 补齐实验

代码复查后，默认 full INT8 已覆盖 Gemma encoder/decoder 的 QKV、O、FFN GEMM；默认残留 BF16 GEMM 的主要来源是 vision 侧，因为 `FVK_PI05_RTX_INT8_VISION=1` 仍是 opt-in。先打开现有 vision INT8 路径重新 profile，再把 `vision_projector_w` 纳入同一条 rowwise INT8 路径：

- 历史 smoke benchmark 中 vision INT8 打开后 baseline p50 约 `161-162 ms`，相对默认 INT8 baseline 约 `163-166 ms` 只有小收益。
- projector INT8 补齐后，10 次 smoke benchmark 为 `p50=161.2 ms`；30 帧 LIBERO action-similarity 为 baseline `p50=161.4 ms`、cache2 `p50=102.7 ms`。
- 30 帧 cache2 vs 同次 INT8 baseline 为 `mae_mean=0.013028`、`cos_mean=0.996498`、`cos_min=0.962295`，说明 projector 补齐至少没有在 smoke 范围内破坏 cache2 相对一致性。
- 新的 vision INT8 profile 里，剩余 BF16 GEMM 已退到尾部；更显眼的是权重量化阶段的 PyTorch elementwise 开销和推理期的 `quantize_int8_rowwise_kernel`。

收益判断：

- `vision_projector` 每个 full frame 只执行一次，不在 decoder `18 layers x 10 denoise steps` 的主热循环里。
- 它只影响 full refresh frame；`cache2` 的 reuse frame 主要绕过 vision/encoder 侧，因此看 cache2 p50 更难放大 projector 收益。
- projector BF16 原路径本来就是规则 cuBLASLt GEMM；换成 INT8 还要付出一次 BF16 activation rowwise quantize，单点节省会被量化开销部分抵消。
- 打开 vision INT8 后，该 BF16 缺口在 profile 中已不是热项。`161-162 ms` 的端到端 p50 与补 projector 后的 `161.2-161.4 ms` 基本重合，属于噪声级收益。

结论：projector INT8 证明“补一个残留 BF16 GEMM”不等于能带来可见提速。这次代码实验不保留；若继续推进 INT8，应先区分“加载/校准期量化开销”和“graph replay 每帧开销”，再优先处理重复调用、高占比、量化边界低成本的残留热项，或减少 runtime rowwise quantize。

#### BF16 encoder/decoder GEMM autotune 实验

profiling 显示 BF16 路径的主要瓶颈仍是 cuBLASLt GEMM。代码复查后发现 Pi0.5 BF16 只显式 autotune 了 vision BF16 shape；Gemma encoder/decoder 的 BF16 plain GEMM 仍依赖 cuBLASLt 首次 heuristic top-1。补齐 BF16 encoder/decoder shape autotune 后，在 Orin 上得到可见收益：

| 配置 | p50 | p95 | 说明 |
|---|---:|---:|---|
| 原 BF16 baseline | ~242.6-243 ms | ~244 ms | 旧基线 |
| 仅 decoder BF16 autotune | 237.1 ms | 238.1 ms | 补 action/QKV/O/FFN/down/out shape |
| encoder + decoder BF16 autotune | 215.9 ms | 217.1 ms | 补齐 encoder QKV/O/FFN shape |

对应 tuned replay-only profile（FlashRT `eval_libero.py`，30 frame，CUDA graph node 展开）：

| Kernel / 类别 | 占比 | 30-frame time | 调用数 | 说明 |
|---|---:|---:|---:|---|
| `ampere_bf16_*_128x128_*` | 44.3% | 2897.9 ms | 5340 | 仍是最大 GEMM 热项 |
| `ampere_bf16_*_256x128_*` | 13.8% | 901.7 ms | 510 | encoder large GEMM family |
| `ampere_bf16_*_128x64_*` | 9.5% + 7.1% + 3.1% | 1291.9 ms | 27000 | decoder small-M GEMM family |
| `gate_silu_mul` | 5.1% | 332.1 ms | 5910 | 最大的非 GEMM kernel family |
| FlashAttention splitkv | 2.8% | 181.2 ms | 5400 | 仍非主要瓶颈 |

结论：

- 这是目前 BF16 路径最有效、最低风险的 GEMM shape 优化；无需改 kernel，只补齐已有 cuBLASLt autotune 覆盖。
- tuned 后 BF16 baseline 从约 `4.12 Hz` 提升到约 `4.63 Hz`。
- 继续做专用 kernel 时，不能只盯一个 `128x128` 名称；现在热点分成 encoder 大 shape（如 `256x128`）和 decoder small-M shape（`128x64` 系列）两类。

同时试过 BF16 decoder `gate`/`up` 合并 GEMM：把两个同输入 `N=4096` GEMM 合成一个 `N=8192` GEMM。A/B 结果：

| 配置 | p50 | 结论 |
|---|---:|---|
| gate/up 不合并 | 216.0 ms | 保留 |
| gate/up 合并 | 217.3 ms | 略慢 |

合并虽然减少了 launch 数，但单个 `N=8192` GEMM autotune 后约 `109-111 us`，与两个 `~60 us` GEMM 的成本基本持平，端到端没有收益。因此 gate/up 合并实验代码不保留。

#### BF16 decoder small-M CUTLASS 实验

在 BF16 encoder/decoder cuBLASLt autotune 后，profile 里仍能看到大量 decoder small-M GEMM（`M=10`，例如 `128x64` family）。为验证“专用 small-M CUTLASS kernel 是否还能继续挤出收益”，曾临时新增本地 microbench；该脚本在实验结束后已删除，只保留结论。

实验过程里发现两个关键问题：

1. `bf16_nn` 的 B 权重布局是 row-major `(K, N)`；早期 microbench 误分配成 `(N, K)`，早期 CUTLASS kernel 也误按 `ColumnMajor` 读取，导致 pipeline action 完全漂移。
2. 修正为 RowMajor B 后，随机 GEMM correctness 显示部分 shape 与 cuBLASLt bitwise 一致，但 `dec_attn_o` 和 `dec_ffn_down` 因累加/舍入路径不同会产生可见 BF16 输出差异；在扩散循环里会被放大。因此不能把所有 decoder GEMM 都替换成自定义 CUTLASS。

保守接入策略（已验证后删除 runtime 代码）：

- 使用 CUTLASS small-M 只替换与 cuBLASLt bitwise 一致的 `action_in`、`attn_qkv`、`ffn_gate`、`ffn_up`、`action_out`。
- `attn_o` 和 `ffn_down` 保持 cuBLASLt BF16 路径。
- 曾通过环境变量 opt-in：`FVK_PI05_BF16_SMALLM_CUTLASS=1`，默认关闭；由于收益太小，相关 CMake/bindings/kernel/pipeline runtime 代码已删除，不保留到主线。

30 帧 default BF16 vs safe small-M CUTLASS 直接 action 对比：

| 对比 | MAE mean | MAE p95 | cosine mean | cosine min | max abs p95 |
|---|---:|---:|---:|---:|---:|
| baseline vs baseline | 0.000027 | 0.000243 | 1.000000 | 0.999998 | 0.003749 |
| cache2 vs cache2 | 0.000041 | 0.000243 | 1.000000 | 0.999998 | 0.003749 |

300 帧 safe small-M CUTLASS 结果：

| Config | p50 | p95 | min | vs same-run baseline |
|---|---:|---:|---:|---|
| baseline | 214.4 ms | 216.2 ms | 213.2 ms | reference |
| cache2 | 133.7 ms | 216.0 ms | 53.0 ms | `cos_mean=0.982399`, `cos_min=0.478313` |

相对当前 cuBLASLt-tuned BF16（约 baseline `218.2 ms`、cache2 `137.3 ms` 的 30 帧对照），safe small-M CUTLASS 只节省约 `3-4 ms`。它说明 decoder small-M 专用 kernel 方向可行但收益有限；更激进地替换 `attn_o/down` 会破坏 action correctness。当前结论是：不保留该优化代码，只保留实验记录。后续若重启这个方向，应先做逐 shape correctness 测试，再考虑更小范围的 kernel 接入。

### BF16 vs INT8 并排对比（profiling 粗对照）

| 类别 | BF16 autotuned replay (`221 ms`) | INT8 legacy profile (`166 ms`) | INT8 变化 |
|------|:-----------:|:-----------:|:----------|
| cuBLASLt BF16 GEMM | 主要热点，top BF16 GEMM families >75% | ~9% (15ms) | 默认 INT8 保留的 vision 等 BF16 路径 |
| CUTLASS INT8 GEMM | — | ~15% (25ms) | 新增，替换 encoder/decoder FFN |
| FlashAttention | ~5% (~12ms) | ~1.3% (2ms) | 一致，非瓶颈 |
| memory-bound 融合 (norm/activation) | ~10% (24ms) | ~2% (3ms) | 被 INT8 融合版本替代 |
| INT8 quant/dequant 开销 | — | ~3% (5ms) | 新增 overhead |
| 总 GPU kernel 调用/帧 | ~12,000 | ~25,000 | 翻倍 |

### 关键洞察

1. **BF16 autotune 后瓶颈仍然非常明确：cuBLASLt GEMM 是主耗时。** `128x128` 一个 GEMM family 仍占 `44.3%`；`256x128` encoder large shape 和 `128x64` decoder small-M family 是下一层热点。Orin 的 14 SM 在 bf16 GEMM 上的吞吐远低于 Thor 的 80+ SM。
2. **INT8 用 CUTLASS rowwise INT8 GEMM 替换后，单次 GEMM 延迟降低 2-7×**，但引入了 quant/dequant 来回转换的开销（每次 GEMM 前后需要 quantize_int8_rowwise / dequantize），总 kernel 调用次数翻倍。
3. **默认 INT8 下仍有 ~9% 的 BF16 GEMM 未被量化**，其中 vision 侧是明确来源；打开 vision INT8 后这部分显著下降，但端到端 p50 只再省约 2-4 ms，说明后续不该把“小 BF16 投影补齐”当成唯一主线。
4. **Attention 不是瓶颈** — 新 BF16 replay-only profile 中 FlashAttention splitkv 占 `2.8%`；即使把其他 attention kernel 加上，它也远小于 GEMM 主体。优化 attention 对 Orin 收益很小。
5. **INT8 融合 kernel 效果不错** — `gate_residual_ada_norm_int8` 单次 10µs，替代了 BF16 中三个分离 kernel 的 63µs。

### 与 Thor SM110 的对比

相同 Pi0.5 模型在 Thor 上纯 graph replay ~44ms（2-view），当前 autotune 后 Orin BF16 baseline replay 是约 221ms（约 5.0× 慢；旧 BF16 baseline 约 242ms）。倍数主要由两个因素决定：
- **SM 数量**：Thor 80+ SM vs Orin 14 SM（~5.7×）
- **memory bandwidth**：Thor 204 GB/s vs Orin ~200 GB/s（接近持平）
- 实际差距仍主要跟随 SM 数差距，说明 GEMM 吞吐仍是平台差异的主要来源。

### Profiling 数据文件

两个 `.nsys-rep` 文件保留为本地实验 artifact，可以用 NVIDIA Nsight Systems GUI 打开查看完整 CUDA kernel 时间线：

```bash
# 本地打开
nsys-ui /path/to/pi05_bf16_baseline_30f.nsys-rep
# 或双击文件
```

BF16 文件是 autotune 后的 measured replay-only trace；INT8 文件仍是较早的完整 trace，包含加载/校准噪声。在 GUI 中可以：
- 按 kernel 名称/类型过滤
- 测量任意时间范围的 kernel 耗时
- 查看 CUDA Graph replay 内部调度
- 分析 kernel launch 延迟和 GPU 利用率

## 当前结论

- BF16 cache2 是当前最稳的 runtime 优化基线：p50 136.6 ms，约 7.32 Hz。
- INT8 cache2 是 offline action-similarity 表中最快路线：p50 104.4 ms，约 9.58 Hz，但最坏帧和整体 action 偏差更大。
- 当前 FlashRT `upstream/main` 的 full INT8 会在 calibration 阶段 crash；临时 re-apply #40 binding fix 后，INT8 baseline bench 可跑到 p50 163.9 ms。
- 临时 fix 后的 closed-loop quick sweep 显示：INT8 baseline 有速度收益但 success 从 BF16 的 7/9 降到 6/9；INT8 cache2 p50 125.9 ms 但 success 只有 4/9。因此 INT8/cache2 不能只按 offline latency 判断可用性。
- 300 帧结果显示 cache2 的主要问题不是平均质量，而是少数 stale-cache 坏帧。
- 下一步如果继续优化，应优先做更强的 adaptive refresh 信号，而不是继续做 vision token pack。
