from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def _run(cmd: List[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return f"unavailable: {exc}"


def _module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return "installed"


def collect_environment() -> Dict[str, Any]:
    gpu = "nvidia-smi not found"
    if shutil.which("nvidia-smi"):
        gpu = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "gpu": gpu,
        "cuda_compiler": _run(["nvcc", "--version"]) if shutil.which("nvcc") else "nvcc not found",
        "tensorrt": _run(["trtexec", "--version"])
        if shutil.which("trtexec")
        else "trtexec not found",
        "packages": {
            "torch": _module_version("torch"),
            "transformers": _module_version("transformers"),
            "bitsandbytes": _module_version("bitsandbytes"),
            "yaml": _module_version("yaml"),
        },
    }
