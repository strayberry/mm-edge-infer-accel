from __future__ import annotations


def load_ocrbench(sample_count: int, sample_strategy: str = "first"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "OCRBench requires `datasets`. Install it with `pip install datasets`."
        ) from exc

    ds = load_dataset("echo840/OCRBench", split="test")
    if sample_strategy == "first" or sample_count >= len(ds):
        return ds.select(range(min(sample_count, len(ds))))
    if sample_strategy == "stratified":
        return stratified_select(ds, sample_count, key="question_type")
    raise ValueError(f"Unsupported OCRBench sample strategy: {sample_strategy}")


def stratified_select(ds, sample_count: int, key: str):
    groups: dict[str, list[int]] = {}
    for idx, item in enumerate(ds):
        groups.setdefault(str(item.get(key, "unknown")), []).append(idx)

    selected: list[int] = []
    ordered_groups = sorted(groups.items(), key=lambda item: item[0])
    while len(selected) < sample_count:
        added = False
        for _, indices in ordered_groups:
            if indices:
                selected.append(indices.pop(0))
                added = True
                if len(selected) == sample_count:
                    break
        if not added:
            break
    return ds.select(selected)
