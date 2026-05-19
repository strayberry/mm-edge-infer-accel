from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def synchronize() -> None:
    if cuda_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> dict:
    if not cuda_available():
        return {
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
    }


def gpu_memory_snapshot_mb() -> dict:
    if not cuda_available():
        return {
            "gpu_total_memory_mb": None,
            "gpu_used_memory_mb": None,
            "gpu_free_memory_mb": None,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_mb = total_bytes / 1024**2
    free_mb = free_bytes / 1024**2
    return {
        "gpu_total_memory_mb": round(total_mb, 2),
        "gpu_used_memory_mb": round(total_mb - free_mb, 2),
        "gpu_free_memory_mb": round(free_mb, 2),
    }


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if cuda_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if cuda_available():
            torch.cuda.nvtx.range_pop()


@contextmanager
def timed_stage(name: str, sink: dict) -> Iterator[None]:
    synchronize()
    started = time.perf_counter()
    with nvtx_range(name):
        yield
    synchronize()
    sink[f"{name}_seconds"] = time.perf_counter() - started


def sum_stage_seconds(samples: list[dict], stage: str) -> float:
    return sum(item.get(f"{stage}_seconds", 0.0) for item in samples)
