# Pi0.5 LeRobot LIBERO Action Stability Sweep on RTX 3080 Ti

本报告保留 RTX 3080 Ti 上 LeRobot/reference Pi0.5 的多 episode `reset`/`queue` action stability sweep。它用于记录 reference action-queue 行为，不应与 Jetson 上的 FlashRT Pi0.5 latency/profiling 数值直接比较；当前 Orin FlashRT 实验结论见 `reports/pi05_orin_flashrt_experiment_report.md`。

## 结论

本轮把 Pi0.5 LIBERO action inference 从单 episode 扩展到 episode `0, 1, 2`，每个 episode 跑前 100 帧，并分别测试 `reset` 和 `queue` 两种模式。

主要结论：

- `reset` 模式每帧都重新预测 action chunk，3 个 episode 平均约 `2.10 Hz`，低于 LIBERO dataset 的 `10 fps`。
- `queue` 模式保留 Pi0.5 内部 action queue，3 个 episode 平均约 `49.97 Hz`，能满足 10 fps 控制循环的动作输出频率。
- `queue` 模式每 100 帧只触发约 `2` 次 chunk prediction，其余帧从缓存 action queue 取动作。
- 3 个 episode 的 action MAE 均值很接近：reset 平均 `0.013673`，queue 平均 `0.013205`。这说明 queue 模式没有明显放大与 dataset action 的逐帧误差。
- queue 与 reset 在同一帧的输出并不完全相同，因为 queue 使用 chunk 内后续动作；这正是 Pi0.5 action-chunk 策略的实际部署形态。

因此，在 Pi0.5 / LeRobot 的 LIBERO action inference 中，`queue` 模式显著提升动作输出频率，从约 `2.10 Hz` 提升到约 `49.97 Hz`，同时没有观察到相对 dataset action 的系统性误差放大。后续部署和稳定性实验应以 `queue` 作为默认执行模式，`reset` 仅保留为逐帧重算 baseline。

## 实验口径

| Item               | Value                                |
| ------------------ | ------------------------------------ |
| Model              | `lerobot/pi05_libero_finetuned_v044` |
| Dataset            | `HuggingFaceVLA/libero`              |
| Episodes           | `0, 1, 2`                            |
| Frames per episode | `100`                                |
| Modes              | `reset`, `queue`                     |
| Warmup             | `3` frames                           |
| Environment        | `/root/autodl-tmp/envs/pi05`         |
| GPU                | RTX 3080 Ti 12GB                     |

`reset` 表示每帧调用前 `policy.reset()`，强制重新预测一个 action chunk。`queue` 表示保留 Pi0.5 内部 action queue，只有 queue 为空时才重新预测 chunk。

当前 Pi0.5 checkpoint 的 action chunk 长度约为 50 步。实验输出中，queue 模式在第 0 帧 `queue_len_before=0`、`queue_len_after=49`，说明一次 chunk prediction 生成约 50 个动作：当前帧立即执行第 1 个动作，剩余 49 个动作进入内部 queue。因此 100 帧 queue 实验通常只触发约 2 次 chunk prediction。报告中的 `raw_action_shape` 和 `final_action_shape` 为 `[1, 7]`，表示每次 `select_action()` 返回给控制循环的当前一步 7 维动作，不是完整 chunk 的形状。

## Episode Summary

| Episode | Mode | Frames | Loop Hz | Chunk predictions | Chunk Hz | Action mean ms | Action p50 ms | E2E mean ms | MAE mean | Cosine mean |
| --: | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | reset | 100 | 2.22 | 100 | 2.22 | 441.6 | 425.9 | 443.3 | 0.012122 | 0.998833 |
| 0 | queue | 100 | 51.77 | 2 | 1.04 | 11.7 | 3.4 | 13.1 | 0.014442 | 0.981202 |
| 1 | reset | 100 | 2.01 | 100 | 2.01 | 487.2 | 506.3 | 489.0 | 0.017064 | 0.962339 |
| 1 | queue | 100 | 50.38 | 2 | 1.01 | 12.2 | 3.6 | 13.6 | 0.013095 | 0.980527 |
| 2 | reset | 100 | 2.08 | 100 | 2.08 | 473.9 | 498.2 | 475.5 | 0.011834 | 0.999037 |
| 2 | queue | 100 | 47.76 | 2 | 0.96 | 13.6 | 3.7 | 15.1 | 0.012078 | 0.998824 |

## Queue Cached-Action Latency

下表只统计 queue 模式中 `queue_len_before > 0` 的 cached-action 帧，不包含第 0 和第 50 帧左右的 chunk prediction。

| Episode | Cached frames | Cached action mean ms | Cached action p50 ms | Cached E2E mean ms | Cached E2E p50 ms |
| --: | --: | --: | --: | --: | --: |
| 0 | 98 | 3.36 | 3.30 | 4.71 | 4.70 |
| 1 | 98 | 3.53 | 3.60 | 4.94 | 5.00 |
| 2 | 98 | 3.64 | 3.70 | 5.13 | 5.20 |

## Queue vs Reset Same-Frame Difference

下表比较同一 episode、同一 frame 下 queue 输出动作和 reset 输出动作的差异。该指标不是和 dataset action 比，而是衡量 action queue 策略相对逐帧重算策略的输出差异。

| Episode | Paired frames | Queue vs reset MAE mean | Queue vs reset MAE p50 | Queue vs reset cosine mean |
| --: | --: | --: | --: | --: |
| 0 | 100 | 0.017139 | 0.011922 | 0.980283 |
| 1 | 100 | 0.022227 | 0.011799 | 0.944053 |
| 2 | 100 | 0.012936 | 0.011839 | 0.998675 |

## 解释

`reset` 模式适合作为最保守的逐帧推理 baseline，但速度只有约 2 Hz，不能直接满足 10 fps 控制循环。`queue` 模式才是 Pi0.5 更自然的执行方式：一次 chunk prediction 后连续输出缓存动作，cached-action 帧的 end-to-end p50 约 5 ms，可以轻松满足 10 fps。

从稳定性看，episode 0/1/2 的 queue MAE 与 reset MAE 在同一量级，没有看到 queue 模式带来系统性的误差放大。后续如果要更严格验证控制质量，应继续扩大 episode 数，并按任务阶段或动作维度拆分 MAE/drift。

## Source Files

```text
outputs/pi05_libero_action_inference_100_reset.json
outputs/pi05_libero_action_inference_100_queue.json
outputs/pi05_libero_action_inference_ep1_100_reset.json
outputs/pi05_libero_action_inference_ep1_100_queue.json
outputs/pi05_libero_action_inference_ep2_100_reset.json
outputs/pi05_libero_action_inference_ep2_100_queue.json
```
