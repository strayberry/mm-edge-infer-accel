from __future__ import annotations

import os
from pathlib import Path

import torch

_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Exception | None = None


def fused_denoise_update(x_t, v_t, dt: float):
    """Fused denoising Euler update: ``x_t + dt * v_t``.

    Uses the local C++/CUDA extension when it is available and falls back to
    torch otherwise.
    """

    if _should_use_cuda(x_t, v_t):
        extension = _load_cuda_extension()
        if extension is not None:
            return extension.denoise_update(x_t, v_t, float(dt))
    return x_t.add(v_t, alpha=dt)


def fused_denoise_update_backend(x_t=None, v_t=None) -> str:
    requested = _requested_backend()
    if requested == "torch":
        return "torch"
    if requested == "auto" and x_t is None:
        return "auto"
    if x_t is not None and v_t is not None and not _is_cuda_compatible(x_t, v_t):
        return "torch"
    if requested == "auto" and x_t is not None and x_t.numel() < _cuda_min_elements():
        return "torch"
    if _load_cuda_extension() is None:
        return "torch"
    return "cuda"


def _should_use_cuda(x_t, v_t) -> bool:
    requested = _requested_backend()
    if requested == "torch":
        return False
    if not _is_cuda_compatible(x_t, v_t):
        return False
    if requested == "auto":
        return x_t.numel() >= _cuda_min_elements()
    return True


def _requested_backend() -> str:
    requested = os.environ.get("MM_EDGE_PI05_FUSED_UPDATE_BACKEND", "auto").strip().lower()
    if requested not in {"cuda", "auto", "torch"}:
        raise ValueError("MM_EDGE_PI05_FUSED_UPDATE_BACKEND must be one of: cuda, auto, torch")
    return requested


def _cuda_min_elements() -> int:
    return int(os.environ.get("MM_EDGE_PI05_FUSED_UPDATE_CUDA_MIN_ELEMENTS", "32768"))


def _is_cuda_compatible(x_t, v_t) -> bool:
    return (
        isinstance(x_t, torch.Tensor)
        and isinstance(v_t, torch.Tensor)
        and x_t.is_cuda
        and v_t.is_cuda
        and x_t.dtype == torch.float32
        and v_t.dtype == torch.float32
        and x_t.shape == v_t.shape
        and x_t.is_contiguous()
        and v_t.is_contiguous()
    )


def _load_cuda_extension():
    global _CUDA_EXTENSION, _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ERROR is not None:
        return None

    try:
        from torch.utils.cpp_extension import load

        repo_root = Path(__file__).resolve().parents[2]
        sources = [
            repo_root / "csrc" / "pi05_ops" / "denoise_update.cpp",
            repo_root / "csrc" / "pi05_ops" / "denoise_update_kernel.cu",
        ]
        _CUDA_EXTENSION = load(
            name="mm_edge_pi05_ops",
            sources=[str(source) for source in sources],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return _CUDA_EXTENSION
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        _CUDA_EXTENSION_ERROR = exc
        return None
