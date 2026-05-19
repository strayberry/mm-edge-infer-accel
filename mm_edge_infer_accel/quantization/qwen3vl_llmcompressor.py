from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Literal

import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from .qwen3vl_calibration import (
    TEXT_CALIB_DATA,
    load_docvqa_calibration,
    load_ocrbench_doc_calibration,
    multimodal_data_collator,
)


DEFAULT_SOURCE = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
DEFAULT_OUTPUTS = {
    "awq": "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct-AWQ-local",
    "gptq": "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct-GPTQ-local",
}

TARGET_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

QuantMethod = Literal["awq", "gptq"]


@dataclass
class Qwen3VLLLMCompressorArgs:
    method: QuantMethod
    source: str = DEFAULT_SOURCE
    output: str | None = None
    max_calib_samples: int = 128
    max_calib_seq_len: int = 1024
    calib_max_pixels: int = 602112
    calib_source: str = "text"
    docvqa_dataset_id: str = "lmms-lab/DocVQA"
    docvqa_config: str | None = "DocVQA"
    docvqa_split: str = "validation"
    sequential_targets: str | None = None
    device_map: str = "auto"
    dtype: str = "bfloat16"
    dry_run: bool = False

    @property
    def output_dir(self) -> str:
        return self.output or DEFAULT_OUTPUTS[self.method]


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def require_llmcompressor() -> None:
    if importlib.util.find_spec("llmcompressor") is None:
        raise RuntimeError(
            "LLM Compressor is not installed. Install it in the main project environment first:\n"
            "  pip install llmcompressor"
        )


def patch_transformers_for_llmcompressor() -> None:
    from transformers import modeling_utils

    if hasattr(modeling_utils, "TORCH_INIT_FUNCTIONS"):
        return
    modeling_utils.TORCH_INIT_FUNCTIONS = _torch_init_functions()


def patch_compressed_tensors_for_llmcompressor() -> None:
    from compressed_tensors.utils import match

    if not hasattr(match, "_match_name"):
        match._match_name = match.match_name


def _torch_init_functions() -> dict:
    return {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
        "zeros_": torch.nn.init.zeros_,
        "ones_": torch.nn.init.ones_,
    }


def import_llmcompressor_modifier(method: QuantMethod):
    require_llmcompressor()
    patch_transformers_for_llmcompressor()
    patch_compressed_tensors_for_llmcompressor()
    try:
        from llmcompressor import oneshot
    except ImportError:
        from llmcompressor.transformers import oneshot

    if method == "awq":
        try:
            from llmcompressor.modifiers.awq import AWQModifier
        except ImportError:
            from llmcompressor.modifiers.transform.awq import AWQModifier

        return oneshot, AWQModifier

    try:
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError:
        from llmcompressor.modifiers.quantization import GPTQModifier

    return oneshot, GPTQModifier


def is_decoder_target(name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    return (
        name.startswith("model.language_model.layers.")
        and name.split(".")[-1] in TARGET_LINEAR_SUFFIXES
    )


def collect_decoder_targets(model: torch.nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if is_decoder_target(name, module)]


def build_calibration_dataset(args: Qwen3VLLLMCompressorArgs, processor):
    if args.calib_source == "text":
        return Dataset.from_dict({"text": TEXT_CALIB_DATA[: args.max_calib_samples]}), None, "text"
    if args.calib_source == "docvqa":
        return (
            load_docvqa_calibration(
                args.docvqa_dataset_id,
                args.docvqa_config,
                args.docvqa_split,
                args.max_calib_samples,
                processor,
                args.max_calib_seq_len,
                max_pixels=args.calib_max_pixels,
            ),
            multimodal_data_collator,
            None,
        )
    if args.calib_source == "ocrbench-doc":
        return (
            load_ocrbench_doc_calibration(
                args.max_calib_samples,
                processor,
                args.max_calib_seq_len,
                max_pixels=args.calib_max_pixels,
            ),
            multimodal_data_collator,
            None,
        )
    raise ValueError("calib_source must be one of: text, docvqa, ocrbench-doc")


def build_recipe(
    method: QuantMethod,
    modifier_cls,
    targets: list[str],
    sequential_targets: str | None,
):
    if method == "awq":
        kwargs = {}
        if sequential_targets:
            kwargs["sequential_targets"] = sequential_targets
        return [
            modifier_cls(
                targets=targets,
                scheme="W4A16_ASYM",
                ignore=["lm_head", "model.visual"],
                **kwargs,
            )
        ]
    return [modifier_cls(targets=targets, scheme="W4A16")]


def quantize_qwen3vl4b(args: Qwen3VLLLMCompressorArgs) -> None:
    if args.method not in {"awq", "gptq"}:
        raise ValueError("method must be one of: awq, gptq")

    model = AutoModelForImageTextToText.from_pretrained(
        args.source,
        dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    targets = collect_decoder_targets(model)
    processor = AutoProcessor.from_pretrained(
        args.source,
        trust_remote_code=True,
        max_pixels=args.calib_max_pixels,
    )
    calib_data, data_collator, text_column = build_calibration_dataset(args, processor)

    scheme = "W4A16_ASYM AWQ" if args.method == "awq" else "W4A16 GPTQ"
    print("source:", args.source)
    print("output:", args.output_dir)
    print("dtype:", args.dtype)
    print("method:", args.method)
    print("scheme:", scheme)
    print("calib_source:", args.calib_source)
    print("calib_max_pixels:", args.calib_max_pixels)
    print("target_count:", len(targets))
    print("calib_samples:", len(calib_data))
    print("max_calib_seq_len:", args.max_calib_seq_len)
    print("first_targets:", targets[:10])
    if args.dry_run:
        return

    oneshot, modifier_cls = import_llmcompressor_modifier(args.method)
    recipe = build_recipe(args.method, modifier_cls, targets, args.sequential_targets)
    kwargs = {}
    if text_column is not None:
        kwargs["text_column"] = text_column
    if data_collator is not None:
        kwargs["data_collator"] = data_collator

    oneshot(
        model=model,
        processor=processor,
        dataset=calib_data,
        recipe=recipe,
        num_calibration_samples=len(calib_data),
        max_seq_length=args.max_calib_seq_len,
        output_dir=args.output_dir,
        save_compressed=True,
        trust_remote_code_model=True,
        **kwargs,
    )
    processor.save_pretrained(args.output_dir)
    print(f"Saved {args.method.upper()} model to: {args.output_dir}")
