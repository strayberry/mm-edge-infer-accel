# Pi0.5 LeRobot Reference Profiling on RTX 3080 Ti

## 摘要

本报告保留 Pi0.5 LIBERO LeRobot/reference action inference 在 RTX 3080 Ti 上的 profiling 结论、已实施优化、撤回项，以及当时对迁移到 Jetson AGX Orin 32G 的可行性判断。它用于记录 reference PyTorch 路径的瓶颈和优化背景，不应与 Jetson 上的 FlashRT Pi0.5 latency/profiling 数值直接比较；当前 Orin FlashRT 实验结论见 `reports/pi05_orin_flashrt_experiment_report.md`。

当前 reference 口径保持 checkpoint 默认 `10` denoising steps，不把降低 denoising steps 作为优化项。

关键结论：

- 模型为 bf16 精度，TF32 对其无影响。
- `torch.inference_mode()` 已默认启用。
- prefix KV cache 优化实测有效：300 frames reset 模式下 action mean 从 0.4254s 降到 0.3924s（-7.8%）。
- `torch.compile` 实测无收益。
- C++/CUDA `fused_denoise_update` 已移除（小 tensor 上无收益）。

## 实验口径

| 项目      | 值                                   |
| --------- | ------------------------------------ |
| Model     | `lerobot/pi05_libero_finetuned_v044` |
| Dataset   | `HuggingFaceVLA/libero`              |
| Episode   | `0`                                  |
| Frames    | `5`                                  |
| Mode      | `reset`                              |
| Warmup    | `1` frame                            |
| GPU       | RTX 3080 Ti 12GB                     |
| Pi0.5 env | `/root/autodl-tmp/envs/pi05`         |

`reset` 模式表示每帧调用前执行 `policy.reset()`，强制每帧重新预测完整 action chunk。该模式适合定位完整模型推理瓶颈。部署时还需要参考 `queue` 模式，因为 policy 会复用 action queue，控制循环频率不能只看 `reset`。

## Profiling 结论

Nsight Systems 的 NVTX 阶段汇总显示，稳态耗时集中在 `pi05_select_action`：

| 阶段                          | 次数 | 总耗时 ms | 平均耗时 ms |
| ----------------------------- | ---: | --------: | ----------: |
| `pi05_load_policy`            |    1 |  154969.2 |    154969.2 |
| `pi05_load_config_processors` |    1 |    3151.6 |      3151.6 |
| `pi05_select_action`          |    5 |    2866.2 |       573.2 |
| `pi05_warmup_select_action`   |    1 |    1103.2 |      1103.2 |
| `pi05_load_dataset`           |    1 |     834.1 |       834.1 |
| `pi05_dataset_getitem`        |    5 |      33.0 |         6.6 |
| `pi05_preprocess`             |    5 |       9.2 |         1.8 |
| `pi05_postprocess`            |    5 |       4.1 |         0.8 |

`pi05_select_action` 内部 CUDA 活动：

| 类别                                 |  次数 |     总量 |
| ------------------------------------ | ----: | -------: |
| CUDA kernels                         | 66820 | 945.2 ms |
| CUDA memcpy                          |  2030 |   4.8 ms |
| CUDA runtime `cudaLaunchKernel`      | 61050 | 447.0 ms |
| CUDA runtime `cudaStreamSynchronize` |   105 | 287.3 ms |

Memcpy 拆分：

| 类别             | 次数 |   总耗时 |     总大小 |
| ---------------- | ---: | -------: | ---------: |
| Device-to-Device | 1925 | 4.772 ms | 939.830 MB |
| Host-to-Device   |  105 | 0.040 ms |   0.015 MB |

结论：输入 H2D copy 可以忽略，主要瓶颈在模型计算和 kernel launch 调度。

## PyTorch Profiler

由于当前 Docker/driver 权限阻止 NCU counters，使用 PyTorch profiler 做 operator 级分析。Profiler 会引入额外开销，因此只看调用结构和比例，不直接用 profiler wall time 做部署结论。

原始 profiler 热点：

| Operator / Kernel                    | Calls | CUDA total |
| ------------------------------------ | ----: | ---------: |
| `aten::mm`                           |  4158 |   247.8 ms |
| `aten::addmm`                        |  2697 |   128.5 ms |
| Cutlass BF16 GEMM group              |   108 |   110.4 ms |
| Cutlass FP32/TF32 GEMM group         |  1215 |    84.0 ms |
| `aten::mul`                          |  7383 |    33.3 ms |
| `aten::bmm`                          |  1221 |    24.3 ms |
| `aten::add`                          |  7086 |    23.4 ms |
| `aten::copy_`                        |  6921 |    22.4 ms |
| GEMV group                           |  1110 |    21.8 ms |
| `aten::_to_copy`                     |  4737 |    14.7 ms |
| `aten::scaled_dot_product_attention` |   243 |    13.9 ms |
| `aten::cat`                          |  2535 |     8.9 ms |

prefix KV cache 引入后的 profiler 对比：

| Operator                             | 原始 Calls | 优化后 Calls | 变化 |
| ------------------------------------ | ---------: | ------------: | ---: |
| `aten::mm`                           |       4158 |          4158 |    0 |
| `aten::addmm`                        |       2697 |          2697 |    0 |
| `aten::scaled_dot_product_attention` |        243 |           243 |    0 |
| `aten::mul`                          |       7383 |          7329 |  -54 |
| `aten::add`                          |       7086 |          7062 |  -24 |
| `aten::copy_`                        |       6921 |          6807 | -114 |
| `aten::_to_copy`                     |       4737 |          4623 | -114 |
| `aten::cat`                          |       2535 |          2448 |  -87 |

该结果说明 prefix KV cache 没有改变 GEMM 和 attention 主干计算，收益主要来自 denoise loop 内部 mask/timestep 构造、临时 tensor 和 Python/operator 调度减少。

## 性能对比

当前测得数据（10 frames reset mode，RTX 3080 Ti 12GB）：

| 版本 | Action mean | Loop Hz |
| --- | --: | --: |
| 纯 FP32 + prefix KV cache | 414.3 ms | 2.41 |
| TF32=1（无 prefix KV cache） | 424.8 ms | 2.35 |
| TF32=1 + prefix KV cache（当前默认） | **375.0 ms** | **2.67** |

prefix KV cache 对比（300 frames reset mode，当前默认配置，模型为 bf16）：

| 配置 | Action mean | Loop Hz | p50 | p90 | 降幅 |
| --- | --: | --: | --: | --: | --: |
| 无 KV cache | 0.4254s | 2.30 | 0.4097s | 0.4771s | — |
| 有 KV cache（**当前默认**） | **0.3924s** | **2.49** | **0.3804s** | **0.4280s** | **-7.8%** |



## 当前优化项

| ID | 优化项 | 状态 | 开关/代码位置 | 说明 |
| --- | --- | --- | --- | --- |
| 优化项 | 状态 | 开关/代码位置 | 说明 |
| --- | --- | --- | --- |
| `torch.inference_mode()` | 默认启用 | `pi05_runtime.py` | 关闭 autograd/version counter 开销。 |
| `torch.compile` | 已接入，默认关闭 | `MM_EDGE_PI05_COMPILE=1` | 实测无收益：Pi0.5 denoising loop 存在 graph break，小 tensor 上 Triton kernel 无法超越 eager PyTorch。 |
| prefix KV cache | **默认启用** | `runtime.enable_prefix_kv_cache`（YAML） | 缓存 prefix KV，避免每步重复编码视觉+文本。 |
| C++/CUDA fused_denoise_update | 已移除 | — | 小 tensor 上无收益，已删除。 |

保持 checkpoint 默认 `10` denoising steps。

## 撤回项

降低 denoising steps 到 `5` 已从当前优化项中撤回。原因：

- 这是算法质量/速度折中，不是纯 runtime 优化。
- 现有数据只覆盖 episode 0 的小样本，不能证明控制质量等价。
- MAE 不能充分代表真实 rollout 成功率、动作稳定性和长期误差累积。
- 当前 profiling/优化报告需要保持 reference 口径干净，因此保留 checkpoint 默认 `10` steps。

相关实验结果可以作为历史消融参考，但不纳入当前推荐路径。

## Nsight Compute 状态

当前 Docker/驱动权限阻止读取 GPU performance counters，`ncu` 会报：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

因此目前不能在本容器内完成 occupancy / roofline 指标采集。需要宿主机放开 NVIDIA performance counter 权限，或用带 profiler 权限的 Docker 容器重新启动后再运行 NCU。

## FlashRT 参考

FlashRT 的方向与本项目 profiling 结论一致：small-batch realtime VLA 的主要问题不是 dataset 或 preprocess，而是 decoder 小 batch 下的 GEMM、attention、小 op 和 kernel launch。

已参考或适合迁移的思路：

- 静态化 shape、timestep、mask，减少 denoise loop 内重复构造。
- CUDA Graph capture，减少 Python 和 kernel dispatch overhead。
- fused norm / activation / residual / gated FFN，减少 memory-bound 小 op 和中间 tensor。
- static quantization / calibration，避免动态 quantize/amax/descale。
- cache_frames / temporal KV reuse，降低连续控制帧上的重复 vision/encoder 成本。
- INT8 W8A8 GEMM / CUTLASS / cuBLASLt 路线，尤其适合 Orin 这类边缘部署目标。

当前不建议直接复刻的方向：

- 在 RTX 3080 Ti 上直接做 FlashRT 风格 FP8/NVFP4 全栈。SM86 缺少新卡上的更强 FP8/NVFP4 路径。
- 为单个很小的 elementwise op 写独立 CUDA kernel。`x_t + dt * v_t` 只有约 `1600` elements，launch overhead 会吃掉收益。

本地参考仓库：`/root/FlashRT`。

## Jetson AGX Orin 32G 适配性

目标设备：

| 项目         | 值                        |
| ------------ | ------------------------- |
| 设备         | Jetson AGX Orin 32G       |
| Machine      | KST Jetson AGX Orin AVSAI |
| SoC          | tegra234                  |
| 架构         | aarch64                   |
| L4T          | R36.4.7                   |
| Ubuntu       | 22.04.5 LTS               |
| Kernel       | 5.15.148-tegra            |
| CUDA Toolkit | 12.6.68                   |
| Python       | 3.10.12                   |
| GCC/G++      | 11.4.0                    |
| RAM          | 30698 MB                  |
| SWAP         | 15349 MB                  |
| Power mode   | MAXN                      |

适配判断：

| 优化项 | Orin 可行性 | 建议 |
| --- | --- | --- |
| `torch.inference_mode()` | 可行 | 平台无关，建议启用。 |
| `torch.compile` | 不建议优先 | Jetson/aarch64 上 Inductor/Triton 生态和稳定性不如 x86，先不要作为主线。 |
| prefix KV cache | 可行 | 默认启用（`runtime.enable_prefix_kv_cache: true`），RTX 3080 Ti 上已验证有效。 |
| 降低 denoising steps | 不纳入 | 保持默认 `10` steps，避免质量变量混入。 |

Orin 上推荐先使用：

prefix KV cache 通过 YAML 配置自动启用，无需环境变量。

## Orin 下一步计划

1. 在 Orin 上复现 prefix KV cache 优化效果（reset mode, 300 frames）。
2. 同一配置跑 queue mode，确认真实控制循环下 action queue 复用后的频率和稳定性。
3. 对 `enable_prefix_kv_cache` 跑 episode `0, 1, 2`、每个 episode 100 帧的 reset/queue 稳定性复测。
4. 参考 `/root/FlashRT` 的 Orin/SM87 路线，评估 cache_frames、vision token pooling、INT8 W8A8 GEMM、fused RMSNorm/SiLU/gated FFN。
5. 如果能获得 profiler 权限，在 Orin 上运行 NCU，采集 top GEMM / attention / memory-bound 小 kernel 的 occupancy 和 roofline。

## 复现命令

### 当前 CLI 方式

```bash
# prefix KV cache 默认启用，直接运行
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python -m mm_edge_infer_accel.cli benchmark \
    --config configs/vla/pi05_libero.yaml \
    --sample-count 5 --episode 0 --mode reset --run
```

### 脚本方式

```bash
# baseline
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python scripts/run_pi05_action_inference.py \
    --source libero --episode 0 --sample-count 5 --mode reset --warmup 1

# 关闭 prefix KV cache 做对比
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python scripts/run_pi05_action_inference.py \
    --disable-prefix-kv-cache \
    --source libero --episode 0 --sample-count 5 --mode reset --warmup 1
```

### Nsight Systems

```bash
nsys profile --sample=none --trace=cuda,nvtx,osrt --stats=true \
  --force-overwrite true -o profiling/pi05_libero_reset5_nsys \
  conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
    python -m mm_edge_infer_accel.cli benchmark \
      --config configs/vla/pi05_libero.yaml \
      --sample-count 5 --episode 0 --mode reset --run
```

### torch.profiler

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python scripts/profile_pi05_torch.py \
    --sample-count 3 --warmup 1 --mode reset \
    --trace-output profiling/pi05_torch_profile_reset3.json \
    --table-output profiling/pi05_torch_profile_reset3_table.txt \
    --summary-output outputs/pi05_torch_profile_reset3_summary.json
```
