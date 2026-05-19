# Pi0.5 LIBERO Action Inference 报告

## 一句话结论

Pi0.5 已经在本机跑通从模型加载到真实 LIBERO 小样本 action inference 的完整路径。

当前结论：

- `lerobot/pi05_libero_finetuned_v044` 可以在 RTX 3080 Ti 12GB 上加载。
- 需要使用独立 LeRobot 环境 `/root/autodl-tmp/envs/pi05`。
- 需要 Hugging Face gated access 才能加载 `google/paligemma-3b-pt-224` tokenizer。
- 真实 `HuggingFaceVLA/libero` episode 0 的 3 个样本可以完成 action prediction。
- 输出 action shape 为 `[1, 7]`，和 dataset action 维度一致。
- 100-frame Hz 测试显示：每帧重新预测 action chunk 约 2.22 Hz；使用内部 action queue
  时，整体 loop 约 51.8 Hz，缓存动作输出的 end-to-end 均值约 4.7 ms。

## 实验口径

| Item | Value |
| --- | --- |
| Model | `lerobot/pi05_libero_finetuned_v044` |
| Dataset | `HuggingFaceVLA/libero` |
| Backend | LeRobot |
| Env | `/root/autodl-tmp/envs/pi05` |
| Python | 3.12.13 |
| Torch | 2.11.0+cu130 |
| Transformers | 5.5.4 |
| LeRobot | 0.5.2 |
| GPU | RTX 3080 Ti 12GB |

主项目环境 `/root/miniconda3/envs/mm-edge-infer-accel` 不适合直接跑 Pi0.5：它是 Python 3.10，而当前 LeRobot Pi0.5 路径需要
Python 3.12 和新版 Transformers/OpenPI 兼容栈。

## 模型加载结果

load-only 输出：

```text
outputs/pi05_libero_load_only.json
```

| Metric | Value |
| --- | ---: |
| Load time | 159.00 s |
| GPU memory before load | 258.19 MB |
| GPU memory after load | 9408.19 MB |
| Model load memory delta | 9150.00 MB |

第一次下载模型时，默认 Xet 路径会卡在 `huggingface_hub.file_download.xet_get`。
后续命令都使用：

```bash
HF_HUB_DISABLE_XET=1
```

模型缓存位置：

```text
/root/autodl-tmp/hf_cache/hub/models--lerobot--pi05_libero_finetuned_v044
```

大小约 7.0 GB。

## Synthetic Action Check

在接真实 dataset 前，先用 synthetic LIBERO-style observation 验证了完整路径：

```text
checkpoint preprocessor -> PI05Policy.select_action -> checkpoint postprocessor
```

输出：

```text
outputs/pi05_synthetic_action.json
```

输入经过 checkpoint preprocessor 后的 batch：

| Field | Shape |
| --- | --- |
| `observation.images.image` | `[1, 3, 256, 256]` |
| `observation.images.image2` | `[1, 3, 256, 256]` |
| `observation.state` | `[1, 8]` |
| `observation.language.tokens` | `[1, 200]` |
| `observation.language.attention_mask` | `[1, 200]` |

| Metric | Value |
| --- | ---: |
| Preprocess time | 0.0047 s |
| `select_action` time | 0.8559 s |
| Postprocess time | 0.0027 s |
| Raw action shape | `[1, 7]` |
| Final action shape | `[1, 7]` |
| CUDA reserved after action | 9408.00 MB |

这一步依赖 `google/paligemma-3b-pt-224` tokenizer。未获得 gated access 时，
preprocessor 会在 tokenizer 加载阶段返回 403。

## 真实 LIBERO 小样本结果

真实样本输出：

```text
outputs/pi05_libero_action_inference.json
```

运行命令：

```bash
HF_HOME=/root/autodl-tmp/hf_cache \
HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub \
HF_HUB_DISABLE_XET=1 \
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python scripts/run_pi05_action_inference.py \
  --source libero \
  --sample-count 3 \
  --output outputs/pi05_libero_action_inference.json
```

数据设置：

| Item | Value |
| --- | --- |
| Dataset | `HuggingFaceVLA/libero` |
| Episode | 0 |
| Episode frames | 214 |
| Sample count | 3 |

只通过 `LeRobotDataset(..., episodes=[0])` 请求 episode 0，避免下载完整 LIBERO。

输入特征：

| Field | Shape | dtype |
| --- | --- | --- |
| `observation.images.image` | `[256, 256, 3]` | image |
| `observation.images.image2` | `[256, 256, 3]` | image |
| `observation.state` | `[8]` | float32 |
| `action` | `[7]` | float32 |

整体结果：

| Metric | Value |
| --- | ---: |
| Dataset init time | 0.9654 s |
| Policy load time | 151.6527 s |
| Action latency mean | 0.5628 s |
| Action MAE mean vs dataset action | 0.015632 |
| Output action shape | `[1, 7]` |
| CUDA reserved during action | 9408 MB |

逐样本结果：

| Frame | Action latency | MAE | Cosine |
| ---: | ---: | ---: | ---: |
| 0 | 0.8040 s | 0.015247 | 0.997514 |
| 1 | 0.4442 s | 0.014873 | 0.997080 |
| 2 | 0.4402 s | 0.016777 | 0.996287 |

第一个 action call 更慢，主要包含加载后的首次模型/kernel 初始化。后续真实样本的
`select_action` latency 约 0.44 s。

## Hz Measurement

进一步在 `HuggingFaceVLA/libero` episode 0 前 100 帧上测试两种模式：

- `reset`: 每帧调用前 `policy.reset()`，强制重新预测一个 action chunk。
- `queue`: 保留 Pi0.5 内部 action queue；第一次调用预测 chunk，后续帧从 queue 取 action。

输出：

```text
outputs/pi05_libero_action_inference_20_reset.json
outputs/pi05_libero_action_inference_20_queue.json
outputs/pi05_libero_action_inference_100_reset.json
outputs/pi05_libero_action_inference_100_queue.json
```

运行命令：

```bash
PYTHONPATH=/mm-edge-infer-accel \
HF_HOME=/root/autodl-tmp/hf_cache \
HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub \
HF_HUB_DISABLE_XET=1 \
conda run --no-capture-output -p /root/autodl-tmp/envs/pi05 \
  python scripts/run_pi05_action_inference.py \
  --source libero \
  --episode 0 \
  --sample-count 100 \
  --mode reset \
  --warmup 3 \
  --output outputs/pi05_libero_action_inference_100_reset.json
```

`queue` 模式只需把 `--mode reset` 和输出文件名改成 `queue`。

整体结果：

| Metric | reset | queue |
| --- | ---: | ---: |
| Frames | 100 | 100 |
| Warmup frames | 3 | 3 |
| Policy load time | 145.10 s | 144.76 s |
| Chunk predictions | 100 | 2 |
| Loop Hz | 2.22 | 51.77 |
| Chunk prediction Hz | 2.22 | 1.04 |
| Action latency mean | 441.6 ms | 11.7 ms |
| Action latency p50 | 425.9 ms | 3.35 ms |
| Action latency p90 | 494.9 ms | 3.6 ms |
| End-to-end mean | 443.3 ms | 13.1 ms |
| End-to-end p50 | 428.0 ms | 4.7 ms |
| End-to-end p90 | 496.7 ms | 5.1 ms |
| MAE mean vs dataset action | 0.012122 | 0.014442 |

`queue` 模式的整体 mean 会被两次 chunk prediction 拉高。只看 98 个 cached-action
frames：

| Cached-action Metric | Value |
| --- | ---: |
| Cached frames | 98 |
| Action latency mean | 3.36 ms |
| Action latency p50 | 3.3 ms |
| End-to-end mean | 4.71 ms |
| End-to-end p50 | 4.7 ms |
| Action-only Hz | 297.24 |
| End-to-end Hz | 212.40 |

100-frame 测试覆盖了两次 chunk prediction，比 20-frame 初测更能代表 queue 模式的周期性开销。
20-frame 初测结果保留在 JSON 中作为 sanity check，但报告结论以 100-frame 为准。

因此 Pi0.5 在当前 RTX 3080 Ti 上有两个不同的速度口径：

- 如果每一帧都闭环重算 chunk，约 2.2 Hz，达不到 LIBERO dataset 的 10 fps。
- 如果按 Pi0.5 原本的 action chunk / queue 方式执行，chunk 预测之后的 action 输出远高于
  10 fps；瓶颈变成每隔一个 chunk 重新预测时的约 0.42-0.44 s 计算成本。

## 如何理解 MAE

这里的 MAE/cosine 是和 dataset 当前帧 action 做逐帧比较，用来验证 action 输出是否在合理范围内。
它不能直接等价为 LIBERO 任务成功率。

完整任务成功率还需要：

- 接 LIBERO environment；
- 执行动作序列；
- 做 episode-level success 判断；
- 统计不同 task 的 rollout 成功率。

## 当前限制

- 当前只是小样本 action inference，不是完整 rollout。
- Hz 测试是 offline dataset frame inference，不包含真实机器人或 LIBERO simulator step time。
- batch size 必须保持 1；模型加载后显存余量有限。
- `google/paligemma-3b-pt-224` 是 gated repo，需要 HF access。
- Pi0.5 环境和模型缓存都放在 `/root/autodl-tmp`，避免占系统盘。

## 下一步

下一步应该接近真实评估口径：

- 跑 3-5 个 episode，每个 episode 取 100 帧以内，确认 Hz 和 MAE 是否稳定。
- 接 LIBERO environment rollout，统计 episode-level success。
- 如果要优化 latency，优先优化 chunk prediction 路径，而不是 cached action 输出路径。
