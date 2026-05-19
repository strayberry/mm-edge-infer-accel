from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"

CONCURRENCY = [1, 2, 4, 8, 16, 32]
MODELS = {
    "BF16 baseline": "bf16",
    "AWQ": "awq",
    "GPTQ": "gptq",
}


def load_metrics(model_slug: str, concurrency: int) -> dict:
    path = OUTPUTS / (
        f"qwen3vl_4b_{model_slug}_vllm_ocrbench_stratified100_c{concurrency}.json"
    )
    with path.open() as f:
        return json.load(f)["metrics"]


def load_first1000_metrics(model_slug: str, concurrency: int) -> dict:
    path = OUTPUTS / (
        f"qwen3vl_4b_{model_slug}_vllm_ocrbench_first1000_c{concurrency}.json"
    )
    with path.open() as f:
        return json.load(f)["metrics"]


def collect_curve() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for model_name, slug in MODELS.items():
        rows = []
        for concurrency in CONCURRENCY:
            metrics = load_metrics(slug, concurrency)
            rows.append(
                {
                    "model": model_name,
                    "concurrency": concurrency,
                    "rps": metrics["requests_per_second"],
                    "serving_tokens_per_second": metrics["serving_tokens_per_second"],
                    "p50_ms": metrics["request_latency_ms_p50"],
                    "p95_ms": metrics["request_latency_ms_p95"],
                    "ttft_mean_ms": metrics["vllm_first_token_latency_ms_mean"],
                    "failure_rate": metrics["failure_rate"],
                    "corrupted": metrics["vllm_corrupted_count"],
                }
            )
        data[model_name] = rows
    return data


def setup_axes(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CONCURRENCY)
    ax.grid(True, alpha=0.25)
    return fig, ax


def save_rps_plot(data: dict[str, list[dict]]) -> str:
    fig, ax = setup_axes("Request throughput vs concurrency", "RPS")
    for model, rows in data.items():
        ax.plot(
            [row["concurrency"] for row in rows],
            [row["rps"] for row in rows],
            marker="o",
            linewidth=2,
            label=model,
        )
    ax.legend()
    path = ASSETS / "qwen3vl_4b_concurrency_rps.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPORTS))


def save_token_plot(data: dict[str, list[dict]]) -> str:
    fig, ax = setup_axes("Serving token throughput vs concurrency", "Generated tokens / second")
    for model, rows in data.items():
        ax.plot(
            [row["concurrency"] for row in rows],
            [row["serving_tokens_per_second"] for row in rows],
            marker="o",
            linewidth=2,
            label=model,
        )
    ax.legend()
    path = ASSETS / "qwen3vl_4b_concurrency_tokens.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPORTS))


def save_latency_plot(data: dict[str, list[dict]]) -> str:
    fig, ax = setup_axes("Latency vs concurrency", "Milliseconds")
    styles = {
        "p50_ms": ("P50", "-"),
        "p95_ms": ("P95", "--"),
        "ttft_mean_ms": ("TTFT mean", ":"),
    }
    for model, rows in data.items():
        for key, (label, linestyle) in styles.items():
            ax.plot(
                [row["concurrency"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                linewidth=2,
                linestyle=linestyle,
                label=f"{model} {label}",
            )
    ax.legend(ncol=2, fontsize=8)
    path = ASSETS / "qwen3vl_4b_concurrency_latency.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPORTS))


def markdown_table(data: dict[str, list[dict]]) -> str:
    lines = [
        "| 模型 | 并发 | RPS | 生成 tok/s | P50 ms | P95 ms | TTFT mean ms | 失败率 | 异常输出 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for row in data[model]:
            lines.append(
                "| {model} | {concurrency} | {rps:.4f} | {serving_tokens_per_second:.4f} | "
                "{p50_ms:.1f} | {p95_ms:.1f} | {ttft_mean_ms:.1f} | "
                "{failure_rate:.1f} | {corrupted} |".format(**row)
            )
    return "\n".join(lines)


def first1000_table() -> str:
    rows = []
    for model_name, slug in MODELS.items():
        for concurrency in [4, 8]:
            metrics = load_first1000_metrics(slug, concurrency)
            rows.append(
                "| {model} | {concurrency} | {accuracy:.3f} | {correct}/1000 | "
                "{rps:.4f} | {tok:.4f} |".format(
                    model=model_name,
                    concurrency=concurrency,
                    accuracy=metrics["accuracy"],
                    correct=metrics["correct"],
                    rps=metrics["requests_per_second"],
                    tok=metrics["serving_tokens_per_second"],
                )
            )
    return "\n".join(
        [
            "| 模型 | 并发 | Accuracy | Correct | RPS | 生成 tok/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def source_file_list(kind: str) -> str:
    if kind == "curve":
        paths = [
            f"outputs/qwen3vl_4b_{slug}_vllm_ocrbench_stratified100_c{concurrency}.json"
            for slug in MODELS.values()
            for concurrency in CONCURRENCY
        ]
    else:
        paths = [
            f"outputs/qwen3vl_4b_{slug}_vllm_ocrbench_first1000_c{concurrency}.json"
            for slug in MODELS.values()
            for concurrency in [4, 8]
        ]
    return "\n".join(paths)


def write_report(data: dict[str, list[dict]], rps_img: str, token_img: str, latency_img: str):
    report = f"""# Qwen3-VL-4B vLLM 并发曲线报告

## 结论

本报告使用 OCRBench `stratified100` 结果分析并发性能曲线。注意：**100 条 stratified
结果只用于观察吞吐和延迟趋势，不用于比较模型准确率**。模型准确率判断应以 1000 条结果为准。

本轮已加入 BF16 baseline。结论变得更明确：BF16 可以作为质量基线，但在 RTX 3080 Ti
12GB 上显存占用明显更高，vLLM 启动后只剩约 `0.21 GiB` KV cache，日志给出的
`Maximum concurrency for 1,024 tokens per request` 只有约 `1.50x`。虽然本数据集仍能跑到
高并发，但 c16/c32 主要靠排队消化请求，TTFT 和请求延迟明显放大，不适合作为默认服务形态。

如果只能选一个默认值，建议使用：

```text
concurrency = 8
模型 = AWQ 或 GPTQ，不建议 BF16 作为 12GB 单卡默认部署模型
```

## 实验口径

性能曲线：

- 数据集：`echo840/OCRBench`
- 采样方式：`stratified`
- 样本数：`100`
- 后端：`vLLM`
- 模型：Qwen3-VL-4B BF16 baseline / AWQ local / GPTQ local
- 并发点：`1, 2, 4, 8, 16, 32`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `runtime.max_model_len: 1024`
- `model.max_pixels: 602112`
- `runtime.mm_processor_kwargs.truncation: false`

准确率参考：

- 数据集：`echo840/OCRBench`
- 样本数：`1000`
- 采样方式：`first`
- 已跑并发点：`4, 8`

100 条 stratified 曲线适合观察吞吐和延迟形状，但不适合作为准确率比较依据，因为 1000 条结果呈现了不同的准确率结论。

## 请求吞吐曲线

![请求吞吐曲线]({rps_img})

请求吞吐随并发持续上升，到 `c32` 仍未完全饱和。BF16 的吞吐低于 AWQ/GPTQ，尤其在高并发下差距明显。

- BF16 baseline：从 c1 的 `1.01 RPS` 提升到 c32 的 `3.38 RPS`。
- AWQ：从 c1 的 `1.01 RPS` 提升到 c32 的 `4.86 RPS`。
- GPTQ：从 c1 的 `1.02 RPS` 提升到 c32 的 `4.70 RPS`。

## Token 吞吐曲线

![Token 吞吐曲线]({token_img})

生成 token 吞吐同样随并发上升。量化模型的高并发 token 吞吐明显好于 BF16 baseline。

- BF16 baseline：c32 为 `74.26 tok/s`。
- AWQ：c32 为 `131.70 tok/s`。
- GPTQ：c32 为 `120.00 tok/s`。

## 延迟曲线

![延迟曲线]({latency_img})

并发提高后，吞吐上升，但 P50/P95/TTFT 都会上升。BF16 在 c16/c32 的 TTFT 抬升最明显，说明它在 12GB 单卡上更容易进入排队状态。

- BF16 P50 从 c1 的 `258.7 ms` 上升到 c32 的 `8220.2 ms`。
- AWQ P50 从 c1 的 `249.6 ms` 上升到 c32 的 `5572.3 ms`。
- GPTQ P50 从 c1 的 `256.3 ms` 上升到 c32 的 `5622.7 ms`。
- BF16 c32 的 TTFT mean 为 `2766.4 ms`，显著高于 AWQ/GPTQ 的约 `676-686 ms`。

## Stratified100 性能数据

{markdown_table(data)}

## 1000 条准确率参考

{first1000_table()}

1000 条结果显示，BF16 baseline 的准确率略高于 AWQ/GPTQ，但吞吐和延迟不占优。AWQ/GPTQ 的准确率基本持平，因此 100 条 stratified
结果不应作为模型质量结论依据。

## 结果解读

BF16 baseline 的价值是作为质量上限和精度参考，不适合作为当前 12GB 单卡的默认部署模型。它在 c32 仍能完成实验，但 TTFT 和 P50 延迟已经明显高于量化模型，说明高并发收益主要来自排队批处理，而不是健康的在线服务延迟。

AWQ/GPTQ 更适合本项目当前的边缘推理部署实验。AWQ 在 c16/c32 的吞吐更好；GPTQ 在 c1-c8 的请求吞吐略好。1000 条准确率参考显示二者质量差距很小，实际选择应优先看吞吐、延迟和部署兼容性。

## 具体建议

| 场景 | 推荐模型 | 推荐并发 | 说明 |
| --- | --- | ---: | --- |
| 默认部署 / 报告主推荐 | AWQ 或 GPTQ | 8 | 吞吐相比 c4 明显提高，延迟低于 c16/c32 |
| 低延迟优先 | AWQ/GPTQ，必要时 BF16 | 1-2 | 适合 demo、交互式问答、低请求量场景 |
| 吞吐优先 | AWQ 优先，其次 GPTQ | 16 | 适合后台批处理或可接受数秒级延迟的服务 |
| 压测上限 / 曲线右端点 | AWQ/GPTQ | 32 | 吞吐最高，但延迟最高，不建议默认使用 |
| 质量基线 | BF16 baseline | 4 或 8 | 用于和量化模型做质量对照，不建议作为 12GB 默认服务模型 |

最终推荐：

```text
默认并发: 8
默认部署模型: AWQ 或 GPTQ
质量基线: BF16 baseline
低延迟配置: 1 或 2
吞吐优先配置: 16
压测上限配置: 32
```

如果项目里只保留一个默认 benchmark/serving 配置，建议使用 `concurrency=8`；如果要展示量化收益，报告主线应写成 BF16 baseline vs AWQ/GPTQ。

## 源文件

性能曲线 JSON：

```text
{source_file_list("curve")}
```

准确率参考 JSON：

```text
{source_file_list("accuracy")}
```
"""
    (REPORTS / "qwen3vl_4b_vllm_concurrency_curve.md").write_text(report)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = collect_curve()
    rps_img = save_rps_plot(data)
    token_img = save_token_plot(data)
    latency_img = save_latency_plot(data)
    write_report(data, rps_img, token_img, latency_img)


if __name__ == "__main__":
    main()
