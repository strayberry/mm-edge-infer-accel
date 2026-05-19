from __future__ import annotations

import argparse

from mm_edge_infer_accel.quantization.qwen3vl_llmcompressor import (
    DEFAULT_OUTPUTS,
    DEFAULT_SOURCE,
    Qwen3VLLLMCompressorArgs,
    quantize_qwen3vl4b,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-VL-4B decoder Linear layers with LLM Compressor."
    )
    parser.add_argument("--method", choices=("awq", "gptq"), required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Defaults to "
            f"{DEFAULT_OUTPUTS['awq']} for AWQ or {DEFAULT_OUTPUTS['gptq']} for GPTQ."
        ),
    )
    parser.add_argument("--max-calib-samples", type=int, default=128)
    parser.add_argument("--max-calib-seq-len", type=int, default=1024)
    parser.add_argument("--calib-max-pixels", type=int, default=602112)
    parser.add_argument(
        "--calib-source",
        choices=("text", "docvqa", "ocrbench-doc"),
        default="text",
        help="Calibration data source. 'text' preserves the original text-only path.",
    )
    parser.add_argument("--docvqa-dataset-id", default="lmms-lab/DocVQA")
    parser.add_argument("--docvqa-config", default="DocVQA")
    parser.add_argument("--docvqa-split", default="validation")
    parser.add_argument("--sequential-targets", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize_qwen3vl4b(
        Qwen3VLLLMCompressorArgs(
            method=args.method,
            source=args.source,
            output=args.output,
            max_calib_samples=args.max_calib_samples,
            max_calib_seq_len=args.max_calib_seq_len,
            calib_max_pixels=args.calib_max_pixels,
            calib_source=args.calib_source,
            docvqa_dataset_id=args.docvqa_dataset_id,
            docvqa_config=args.docvqa_config,
            docvqa_split=args.docvqa_split,
            sequential_targets=args.sequential_targets,
            device_map=args.device_map,
            dtype=args.dtype,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
