from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EPISODES = [0, 1, 2]


def result_path(episode: int, mode: str) -> Path:
    if episode == 0:
        return OUTPUTS / f"pi05_libero_action_inference_100_{mode}.json"
    return OUTPUTS / f"pi05_libero_action_inference_ep{episode}_100_{mode}.json"


def load_result(episode: int, mode: str) -> dict:
    with result_path(episode, mode).open() as f:
        return json.load(f)


def flatten_action(sample: dict, key: str = "final_action") -> list[float]:
    value = sample[key]
    if value and isinstance(value[0], list):
        value = value[0]
    return [float(v) for v in value]


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def mean_metric(result: dict, metric: str) -> float:
    values = [
        sample["action_metrics"][metric]
        for sample in result["samples"]
        if metric in sample.get("action_metrics", {})
    ]
    return fmean(values)


def cached_summary(result: dict) -> dict:
    cached = [s for s in result["samples"] if s["queue_len_before"] > 0]
    if not cached:
        return {"count": 0, "action_p50_ms": None, "e2e_p50_ms": None, "action_mean_ms": None}
    action_ms = [s["action_seconds"] * 1000 for s in cached]
    e2e_ms = [s["end_to_end_seconds"] * 1000 for s in cached]
    return {
        "count": len(cached),
        "action_p50_ms": median(action_ms),
        "action_mean_ms": fmean(action_ms),
        "e2e_p50_ms": median(e2e_ms),
        "e2e_mean_ms": fmean(e2e_ms),
    }


def paired_reset_queue(episode: int) -> dict:
    reset = load_result(episode, "reset")
    queue = load_result(episode, "queue")
    reset_by_frame = {sample["frame_index"]: sample for sample in reset["samples"]}
    paired = []
    for sample in queue["samples"]:
        frame = sample["frame_index"]
        if frame not in reset_by_frame:
            continue
        paired.append((reset_by_frame[frame], sample))
    maes = [
        mae(flatten_action(reset_sample), flatten_action(queue_sample))
        for reset_sample, queue_sample in paired
    ]
    cosines = [
        cosine(flatten_action(reset_sample), flatten_action(queue_sample))
        for reset_sample, queue_sample in paired
    ]
    return {
        "count": len(paired),
        "mae_mean": fmean(maes) if maes else None,
        "mae_p50": median(maes) if maes else None,
        "cosine_mean": fmean(cosines) if cosines else None,
    }


def mode_table() -> str:
    lines = [
        "| Episode | Mode | Frames | Loop Hz | Chunk predictions | Chunk Hz | Action mean ms | Action p50 ms | E2E mean ms | MAE mean | Cosine mean |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in EPISODES:
        for mode in ["reset", "queue"]:
            result = load_result(episode, mode)
            action = result["action_time_summary"]
            e2e = result["end_to_end_time_summary"]
            lines.append(
                "| {episode} | {mode} | {frames} | {loop_hz} | {chunks} | {chunk_hz} | "
                "{action_mean} | {action_p50} | {e2e_mean} | {mae_mean} | {cosine_mean} |".format(
                    episode=episode,
                    mode=mode,
                    frames=result["sample_count"],
                    loop_hz=fmt(result["loop_hz"], 2),
                    chunks=result["chunk_predict_count"],
                    chunk_hz=fmt(result["chunk_predict_hz"], 2),
                    action_mean=fmt(action["mean"] * 1000, 1),
                    action_p50=fmt(action["p50"] * 1000, 1),
                    e2e_mean=fmt(e2e["mean"] * 1000, 1),
                    mae_mean=fmt(result["action_mae_mean"], 6),
                    cosine_mean=fmt(mean_metric(result, "cosine_similarity"), 6),
                )
            )
    return "\n".join(lines)


def cached_table() -> str:
    lines = [
        "| Episode | Cached frames | Cached action mean ms | Cached action p50 ms | Cached E2E mean ms | Cached E2E p50 ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in EPISODES:
        summary = cached_summary(load_result(episode, "queue"))
        lines.append(
            "| {episode} | {count} | {action_mean} | {action_p50} | {e2e_mean} | {e2e_p50} |".format(
                episode=episode,
                count=summary["count"],
                action_mean=fmt(summary["action_mean_ms"], 2),
                action_p50=fmt(summary["action_p50_ms"], 2),
                e2e_mean=fmt(summary["e2e_mean_ms"], 2),
                e2e_p50=fmt(summary["e2e_p50_ms"], 2),
            )
        )
    return "\n".join(lines)


def paired_table() -> str:
    lines = [
        "| Episode | Paired frames | Queue vs reset MAE mean | Queue vs reset MAE p50 | Queue vs reset cosine mean |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode in EPISODES:
        paired = paired_reset_queue(episode)
        lines.append(
            "| {episode} | {count} | {mae_mean} | {mae_p50} | {cosine_mean} |".format(
                episode=episode,
                count=paired["count"],
                mae_mean=fmt(paired["mae_mean"], 6),
                mae_p50=fmt(paired["mae_p50"], 6),
                cosine_mean=fmt(paired["cosine_mean"], 6),
            )
        )
    return "\n".join(lines)


def aggregate() -> dict:
    reset = [load_result(ep, "reset") for ep in EPISODES]
    queue = [load_result(ep, "queue") for ep in EPISODES]
    return {
        "reset_loop_hz_mean": fmean(r["loop_hz"] for r in reset),
        "queue_loop_hz_mean": fmean(r["loop_hz"] for r in queue),
        "reset_mae_mean": fmean(r["action_mae_mean"] for r in reset),
        "queue_mae_mean": fmean(r["action_mae_mean"] for r in queue),
        "queue_chunk_count_mean": fmean(r["chunk_predict_count"] for r in queue),
        "reset_chunk_count_mean": fmean(r["chunk_predict_count"] for r in reset),
    }


def main() -> int:
    agg = aggregate()
    report = f"""# Pi0.5 LIBERO Action Stability Sweep

## 结论

本轮把 Pi0.5 LIBERO action inference 从单 episode 扩展到 episode `0, 1, 2`，每个 episode
跑前 100 帧，并分别测试 `reset` 和 `queue` 两种模式。

主要结论：

- `reset` 模式每帧都重新预测 action chunk，3 个 episode 平均约 `{agg['reset_loop_hz_mean']:.2f} Hz`，低于 LIBERO dataset 的 `10 fps`。
- `queue` 模式保留 Pi0.5 内部 action queue，3 个 episode 平均约 `{agg['queue_loop_hz_mean']:.2f} Hz`，能满足 10 fps 控制循环的动作输出频率。
- `queue` 模式每 100 帧只触发约 `{agg['queue_chunk_count_mean']:.0f}` 次 chunk prediction，其余帧从缓存 action queue 取动作。
- 3 个 episode 的 action MAE 均值很接近：reset 平均 `{agg['reset_mae_mean']:.6f}`，queue 平均 `{agg['queue_mae_mean']:.6f}`。这说明 queue 模式没有明显放大与 dataset action 的逐帧误差。
- queue 与 reset 在同一帧的输出并不完全相同，因为 queue 使用 chunk 内后续动作；这正是 Pi0.5 action-chunk 策略的实际部署形态。

## 实验口径

| Item | Value |
| --- | --- |
| Model | `lerobot/pi05_libero_finetuned_v044` |
| Dataset | `HuggingFaceVLA/libero` |
| Episodes | `0, 1, 2` |
| Frames per episode | `100` |
| Modes | `reset`, `queue` |
| Warmup | `3` frames |
| Environment | `/root/autodl-tmp/envs/pi05` |
| GPU | RTX 3080 Ti 12GB |

`reset` 表示每帧调用前 `policy.reset()`，强制重新预测一个 action chunk。`queue` 表示保留
Pi0.5 内部 action queue，只有 queue 为空时才重新预测 chunk。

## Episode Summary

{mode_table()}

## Queue Cached-Action Latency

下表只统计 queue 模式中 `queue_len_before > 0` 的 cached-action 帧，不包含第 0 和第 50 帧左右的 chunk prediction。

{cached_table()}

## Queue vs Reset Same-Frame Difference

下表比较同一 episode、同一 frame 下 queue 输出动作和 reset 输出动作的差异。该指标不是和 dataset action 比，而是衡量 action queue 策略相对逐帧重算策略的输出差异。

{paired_table()}

## 解释

`reset` 模式适合作为最保守的逐帧推理 baseline，但速度只有约 2 Hz，不能直接满足 10 fps
控制循环。`queue` 模式才是 Pi0.5 更自然的执行方式：一次 chunk prediction 后连续输出缓存动作，
cached-action 帧的 end-to-end p50 约 5 ms，可以轻松满足 10 fps。

从稳定性看，episode 0/1/2 的 queue MAE 与 reset MAE 在同一量级，没有看到 queue 模式带来系统性的误差放大。
后续如果要更严格验证控制质量，应继续扩大 episode 数，并按任务阶段或动作维度拆分 MAE/drift。

## Source Files

```text
outputs/pi05_libero_action_inference_100_reset.json
outputs/pi05_libero_action_inference_100_queue.json
outputs/pi05_libero_action_inference_ep1_100_reset.json
outputs/pi05_libero_action_inference_ep1_100_queue.json
outputs/pi05_libero_action_inference_ep2_100_reset.json
outputs/pi05_libero_action_inference_ep2_100_queue.json
```
"""
    (REPORTS / "pi05_libero_action_stability_sweep.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
