from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _prepend_existing_library_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing + ([current] if current else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _ensure_trtllm_library_path() -> None:
    if os.environ.get("MM_EDGE_TRTLLM_LD_READY") == "1":
        return
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_dir / "site-packages"
    library_paths = [
        site_packages / "nvidia" / "cu13" / "lib",
        site_packages / "tensorrt_libs",
    ]
    _prepend_existing_library_paths(library_paths)
    os.environ["MM_EDGE_TRTLLM_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-test helper: export text-only Qwen3-0.6B to a TensorRT-LLM engine"
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-0.6B",
        help="Local HuggingFace Qwen3-0.6B model directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/tensorrt/qwen3_0_6b_trtllm_bf16_ckpt",
        help="Output TensorRT-LLM checkpoint directory",
    )
    parser.add_argument(
        "--engine-dir",
        default="outputs/tensorrt/qwen3_0_6b_trtllm_bf16_engine",
        help="Output TensorRT-LLM engine directory",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--quant-algo",
        default="none",
        choices=["none", "w8a16"],
        help="Optional TensorRT-LLM quantization mode for this side test",
    )
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=1088)
    parser.add_argument("--max-num-tokens", type=int, default=1024)
    parser.add_argument(
        "--skip-checkpoint", action="store_true", help="Reuse an existing checkpoint"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Only export the TensorRT-LLM checkpoint"
    )
    return parser.parse_args()


def _quant_config(quant_algo: str):
    if quant_algo == "none":
        return None
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    if quant_algo == "w8a16":
        return QuantConfig(quant_algo=QuantAlgo.W8A16)
    raise ValueError(f"unsupported quant_algo: {quant_algo}")


def export_checkpoint(model_path: Path, checkpoint_dir: Path, dtype: str, quant_algo: str) -> None:
    from tensorrt_llm.models.automodel import AutoModelForCausalLM

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    quant_config = _quant_config(quant_algo)
    print(f"loading and converting {model_path}")
    print(f"quant_algo={quant_algo}")
    model = AutoModelForCausalLM.from_hugging_face(
        str(model_path),
        dtype=dtype,
        quant_config=quant_config,
        load_model_on_cpu=True,
    )
    print(f"saving TensorRT-LLM checkpoint to {checkpoint_dir}")
    model.save_checkpoint(str(checkpoint_dir))


def build_engine(args: argparse.Namespace) -> None:
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    trtllm_build = Path(sys.prefix) / "bin" / "trtllm-build"
    timing_cache = engine_dir / "timing.cache"
    command = [
        str(trtllm_build),
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--output_dir",
        args.engine_dir,
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_input_len",
        str(args.max_input_len),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_num_tokens",
        str(args.max_num_tokens),
        "--gpt_attention_plugin",
        args.dtype,
        "--gemm_plugin",
        args.dtype,
        "--context_fmha",
        "enable",
        "--remove_input_padding",
        "enable",
        "--kv_cache_type",
        "paged",
        "--output_timing_cache",
        str(timing_cache),
        "--log_level",
        "info",
    ]
    print("running:", " ".join(command))
    subprocess.run(command, check=True, env=os.environ.copy())


def main() -> int:
    args = parse_args()
    _ensure_trtllm_library_path()

    model_path = Path(args.model_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not args.skip_checkpoint:
        export_checkpoint(model_path, checkpoint_dir, args.dtype, args.quant_algo)

    if not args.skip_build:
        build_engine(args)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
