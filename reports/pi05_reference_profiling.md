# Pi0.5 Reference Profiling

## 摘要

本报告汇总 Pi0.5 LIBERO reference action inference 的 profiling 结论、已实施优化、撤回项，以及迁移到 Jetson AGX Orin 32G 的可行性。当前主线口径保持 checkpoint 默认 `10` denoising steps，不把降低 denoising steps 作为优化项。

关键结论：

- 主瓶颈是 `policy.select_action()` 的完整 action chunk prediction；dataset 读取、preprocess、postprocess、H2D copy 都不是主耗时。
- `select_action()` 内部主要由 GEMM / attention 计算、大量小 op 和 CUDA kernel launch 调度共同主导。
- TF32 + `torch.inference_mode()` 已带来稳定收益：action mean 从 `573.3 ms` 降到 `470.8 ms`。
- Pi0.5 denoise-loop patch 继续减少 loop 内重复 mask/timestep 构造：action mean 降到 `359.5 ms`。
- C++/CUDA `fused_denoise_update` 已实现，但当前 update tensor 太小，强制 CUDA 比 torch fallback 慢，默认 `auto` 会回退 torch。
- Jetson AGX Orin 32G 上建议优先验证 TF32、`inference_mode()` 和 denoise-loop patch；`torch.compile` 和手写小 CUDA kernel 不应作为第一优先级。

## 实验口径

| 项目 | 值 |
| --- | --- |
| Model | `lerobot/pi05_libero_finetuned_v044` |
| Dataset | `HuggingFaceVLA/libero` |
| Episode | `0` |
| Frames | `5` |
| Mode | `reset` |
| Warmup | `1` frame |
| GPU | RTX 3080 Ti 12GB |
| Pi0.5 env | `/root/autodl-tmp/envs/pi05` |

`reset` 模式表示每帧调用前执行 `policy.reset()`，强制每帧重新预测完整 action chunk。该模式适合定位完整模型推理瓶颈。部署时还需要参考 `queue` 模式，因为 policy 会复用 action queue，控制循环频率不能只看 `reset`。

## Profiling 结论

Nsight Systems 的 NVTX 阶段汇总显示，稳态耗时集中在 `pi05_select_action`：

| 阶段 | 次数 | 总耗时 ms | 平均耗时 ms |
| --- | ---: | ---: | ---: |
| `pi05_load_policy` | 1 | 154969.2 | 154969.2 |
| `pi05_load_config_processors` | 1 | 3151.6 | 3151.6 |
| `pi05_select_action` | 5 | 2866.2 | 573.2 |
| `pi05_warmup_select_action` | 1 | 1103.2 | 1103.2 |
| `pi05_load_dataset` | 1 | 834.1 | 834.1 |
| `pi05_dataset_getitem` | 5 | 33.0 | 6.6 |
| `pi05_preprocess` | 5 | 9.2 | 1.8 |
| `pi05_postprocess` | 5 | 4.1 | 0.8 |

`pi05_select_action` 内部 CUDA 活动：

| 类别 | 次数 | 总量 |
| --- | ---: | ---: |
| CUDA kernels | 66820 | 945.2 ms |
| CUDA memcpy | 2030 | 4.8 ms |
| CUDA runtime `cudaLaunchKernel` | 61050 | 447.0 ms |
| CUDA runtime `cudaStreamSynchronize` | 105 | 287.3 ms |

Memcpy 拆分：

| 类别 | 次数 | 总耗时 | 总大小 |
| --- | ---: | ---: | ---: |
| Device-to-Device | 1925 | 4.772 ms | 939.830 MB |
| Host-to-Device | 105 | 0.040 ms | 0.015 MB |

结论：输入 H2D copy 可以忽略，主要瓶颈在模型计算和 kernel launch 调度。

## PyTorch Profiler

由于当前 Docker/driver 权限阻止 NCU counters，使用 PyTorch profiler 做 operator 级分析。Profiler 会引入额外开销，因此只看调用结构和比例，不直接用 profiler wall time 做部署结论。

原始 profiler 热点：

| Operator / Kernel | Calls | CUDA total |
| --- | ---: | ---: |
| `aten::mm` | 4158 | 247.8 ms |
| `aten::addmm` | 2697 | 128.5 ms |
| Cutlass BF16 GEMM group | 108 | 110.4 ms |
| Cutlass FP32/TF32 GEMM group | 1215 | 84.0 ms |
| `aten::mul` | 7383 | 33.3 ms |
| `aten::bmm` | 1221 | 24.3 ms |
| `aten::add` | 7086 | 23.4 ms |
| `aten::copy_` | 6921 | 22.4 ms |
| GEMV group | 1110 | 21.8 ms |
| `aten::_to_copy` | 4737 | 14.7 ms |
| `aten::scaled_dot_product_attention` | 243 | 13.9 ms |
| `aten::cat` | 2535 | 8.9 ms |

OPT-005 patch 后的 profiler 对比：

| Operator | 原始 Calls | OPT-005 Calls | 变化 |
| --- | ---: | ---: | ---: |
| `aten::mm` | 4158 | 4158 | 0 |
| `aten::addmm` | 2697 | 2697 | 0 |
| `aten::scaled_dot_product_attention` | 243 | 243 | 0 |
| `aten::mul` | 7383 | 7329 | -54 |
| `aten::add` | 7086 | 7062 | -24 |
| `aten::copy_` | 6921 | 6807 | -114 |
| `aten::_to_copy` | 4737 | 4623 | -114 |
| `aten::cat` | 2535 | 2448 | -87 |

该结果说明 OPT-005 没有改变 GEMM 和 attention 主干计算，收益主要来自 denoise loop 内部 mask/timestep 构造、临时 tensor 和 Python/operator 调度减少。

## 性能对比

| 版本 | Action mean | Action p50 | End-to-end mean | Loop Hz | Action MAE mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 573.3 ms | 569.5 ms | 576.0 ms | 1.71 | 0.012560 |
| OPT-001 + OPT-002 | 470.8 ms | 464.2 ms | 473.1 ms | 2.08 | 0.013664 |
| OPT-001 + OPT-002 + OPT-005 | 359.5 ms | 357.6 ms | 361.5 ms | 2.71 | 0.013821 |
| OPT-005 + forced CUDA update | 374.4 ms | 371.9 ms | 376.6 ms | 2.60 | 0.013805 |

相对 baseline：

| 版本 | Action mean 变化 | Loop Hz 变化 |
| --- | ---: | ---: |
| OPT-001 + OPT-002 | -17.9% | +21.6% |
| OPT-001 + OPT-002 + OPT-005 | -37.3% | +58.5% |

## 当前优化项

| ID | 优化项 | 状态 | 开关/代码位置 | 说明 |
| --- | --- | --- | --- | --- |
| OPT-001 | TF32 matmul / cuDNN | 默认启用 | `MM_EDGE_PI05_TF32=1`，`mm_edge_infer_accel/pi05_runtime.py::pi05_inference_optimizations` | 让 Ampere FP32 matmul/conv 可走 TF32 Tensor Core。LeRobot 默认没有显式打开。 |
| OPT-002 | `torch.inference_mode()` | 默认启用 | `mm_edge_infer_accel/pi05_runtime.py::_warmup_policy`、`run_libero_action_inference` | 关闭 autograd/version counter 等纯推理不需要的开销。 |
| OPT-003 | `torch.compile` 接入 | 已接入，默认关闭 | `MM_EDGE_PI05_COMPILE=1`，`MM_EDGE_PI05_COMPILE_MODE=reduce-overhead` | 还没有作为有效收益记录，需要单独测首次编译、graph break、显存峰值和 steady-state。 |
| OPT-005 | Pi0.5 denoise-loop patch | 已接入，默认关闭 | `MM_EDGE_PI05_PATCH_SAMPLE_ACTIONS=1`，`mm_edge_infer_accel/pi05_optimizations.py` | 缓存 timestep、suffix attention mask、position ids，减少 loop 内 `cat/to/copy_/cumsum/tensor` 等小 op。 |
| OPT-006 | C++/CUDA `fused_denoise_update` | 已实现，默认 auto 回退 torch | `MM_EDGE_PI05_FUSED_UPDATE_BACKEND=auto|torch|cuda`，`csrc/pi05_ops/denoise_update_kernel.cu` | 单独实现 `x_t + dt * v_t`。当前小 tensor 上 forced CUDA 更慢，不建议强制启用。 |

当前推荐配置：

```bash
MM_EDGE_PI05_TF32=1
MM_EDGE_PI05_PATCH_SAMPLE_ACTIONS=1
MM_EDGE_PI05_FUSED_UPDATE_BACKEND=auto
```

保持 checkpoint 默认 `10` denoising steps，不设置 `MM_EDGE_PI05_NUM_INFERENCE_STEPS`。

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
| GCC/G++ | 11.4.0 |
| RAM | 30698 MB |
| SWAP | 15349 MB |
| Power mode | MAXN |

适配判断：

| 优化 | Orin 可行性 | 建议 |
| --- | --- | --- |
| OPT-001 TF32 | 可行 | Orin 是 Ampere SM87，PyTorch 可启用 TF32；收益需在 Orin 上实测。 |
| OPT-002 `inference_mode()` | 可行 | 平台无关，建议启用。 |
| OPT-003 `torch.compile` | 不建议优先 | Jetson/aarch64 上 Inductor/Triton 生态和稳定性不如 x86，先不要作为主线。 |
| OPT-005 denoise-loop patch | 可行 | 最值得优先迁移验证，主要减少 Python/PyTorch 小 op 和重复 mask 构造。 |
| OPT-006 C++/CUDA update | 可编译但不建议强制 | 需要 `TORCH_CUDA_ARCH_LIST=8.7`；当前小 tensor 预计仍不如 torch fallback，保持 `auto`。 |
| 降低 denoising steps | 不纳入 | 保持默认 `10` steps，避免质量变量混入。 |

Orin 上推荐先使用：

```bash
export TORCH_CUDA_ARCH_LIST=8.7
export MM_EDGE_PI05_TF32=1
export MM_EDGE_PI05_PATCH_SAMPLE_ACTIONS=1
export MM_EDGE_PI05_FUSED_UPDATE_BACKEND=auto
```

## Orin 下一步计划

1. 在 Orin 上复现三组 reset benchmark：baseline、TF32 + `inference_mode()`、TF32 + `inference_mode()` + OPT-005。
2. 同一配置跑 queue mode，确认真实控制循环下 action queue 复用后的频率和稳定性。
3. 对 OPT-001 和 OPT-005 跑 episode `0, 1, 2`、每个 episode 100 帧的 reset/queue 稳定性复测。
4. 如果 Orin 上 kernel launch 仍是主要问题，优先做 CUDA Graph capture，而不是继续单独优化小 elementwise kernel。
5. 参考 `/root/FlashRT` 的 Orin/SM87 路线，评估 cache_frames、vision token pooling、INT8 W8A8 GEMM、fused RMSNorm/SiLU/gated FFN。
6. 如果能获得 profiler 权限，在 Orin 上运行 NCU，采集 top GEMM / attention / memory-bound 小 kernel 的 occupancy 和 roofline。

## 复现命令

Baseline Nsight Systems：

```bash
nsys profile \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite true \
  -o profiling/pi05_libero_reset5_nsys \
  /root/autodl-tmp/envs/pi05/bin/python -m scripts.run_pi05_action_inference \
    --source libero \
    --episode 0 \
    --sample-count 5 \
    --mode reset \
    --warmup 1 \
    --output outputs/pi05_libero_action_inference_profile_reset5.json
```

TF32 + `inference_mode()`：

```bash
MM_EDGE_PI05_TF32=1 \
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python -m scripts.run_pi05_action_inference \
    --source libero \
    --episode 0 \
    --sample-count 5 \
    --mode reset \
    --warmup 1 \
    --output outputs/pi05_libero_action_inference_reset5_tf32.json
```

OPT-005 denoise-loop patch：

```bash
MM_EDGE_PI05_TF32=1 \
MM_EDGE_PI05_PATCH_SAMPLE_ACTIONS=1 \
MM_EDGE_PI05_FUSED_UPDATE_BACKEND=auto \
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python -m scripts.run_pi05_action_inference \
    --source libero \
    --episode 0 \
    --sample-count 5 \
    --mode reset \
    --warmup 1 \
    --output outputs/pi05_libero_action_inference_reset5_tf32_patch.json
```

OPT-006 强制 C++/CUDA fused update，只用于实验对比：

```bash
MM_EDGE_PI05_TF32=1 \
MM_EDGE_PI05_PATCH_SAMPLE_ACTIONS=1 \
MM_EDGE_PI05_FUSED_UPDATE_BACKEND=cuda \
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python -m scripts.run_pi05_action_inference \
    --source libero \
    --episode 0 \
    --sample-count 5 \
    --mode reset \
    --warmup 1 \
    --output outputs/pi05_libero_action_inference_reset5_tf32_patch_cuda_update.json
```

PyTorch profiler：

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python -m scripts.profile_pi05_torch \
    --sample-count 3 \
    --warmup 1 \
    --mode reset \
    --trace-output profiling/pi05_torch_profile_reset3.json \
    --table-output profiling/pi05_torch_profile_reset3_table.txt \
    --summary-output outputs/pi05_torch_profile_reset3_summary.json
```
