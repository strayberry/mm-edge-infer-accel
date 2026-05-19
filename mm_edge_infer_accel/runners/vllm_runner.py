from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..config import ExperimentConfig, model_load_path


@dataclass
class GenerationResult:
    text: str
    token_count: int
    vllm_metrics: dict[str, float | int | bool | None]


def _positive_delta(end: float, start: float) -> float | None:
    if end <= 0 or start <= 0:
        return None
    return max(0.0, end - start)


def _extract_vllm_metrics(metrics: Any) -> dict[str, float | int | bool | None]:
    if metrics is None:
        return {}

    queued_ts = float(getattr(metrics, "queued_ts", 0.0) or 0.0)
    scheduled_ts = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
    first_token_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_token_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)

    return {
        "vllm_num_generation_tokens": getattr(metrics, "num_generation_tokens", None),
        "vllm_queue_seconds": _positive_delta(scheduled_ts, queued_ts),
        "vllm_prefill_seconds": _positive_delta(first_token_ts, scheduled_ts),
        "vllm_decode_seconds": _positive_delta(last_token_ts, first_token_ts),
        "vllm_inference_seconds": _positive_delta(last_token_ts, scheduled_ts),
        "vllm_first_token_latency_seconds": first_token_latency or None,
        "vllm_is_corrupted": getattr(metrics, "is_corrupted", None),
    }


def _generation_result_from_request(request_output: Any, token_counter) -> GenerationResult:
    output = request_output.outputs[0]
    token_ids = getattr(output, "token_ids", None)
    token_count = len(token_ids) if token_ids is not None else token_counter(output.text)
    return GenerationResult(
        text=output.text,
        token_count=token_count,
        vllm_metrics=_extract_vllm_metrics(getattr(request_output, "metrics", None)),
    )


def _patch_transformers_tokenization_compat() -> None:
    """Compatibility for older remote processors under Transformers 5.x."""
    import transformers.tokenization_utils as tokenization_utils
    import transformers.tokenization_utils_base as tokenization_utils_base

    for name in (
        "PaddingStrategy",
        "PreTokenizedInput",
        "TextInput",
        "TruncationStrategy",
    ):
        if not hasattr(tokenization_utils, name) and hasattr(tokenization_utils_base, name):
            setattr(tokenization_utils, name, getattr(tokenization_utils_base, name))


class VLLMRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        if cfg.runtime.disable_flashinfer_sampler:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from transformers import AutoProcessor
        from vllm import LLM

        _patch_transformers_tokenization_compat()
        self.cfg = cfg
        self.load_path = model_load_path(cfg)
        processor_kwargs: dict[str, Any] = {"trust_remote_code": cfg.model.trust_remote_code}
        if cfg.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = cfg.model.max_pixels
        self.processor = AutoProcessor.from_pretrained(self.load_path, **processor_kwargs)
        mm_processor_kwargs: dict[str, Any] = {}
        if cfg.model.max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = cfg.model.max_pixels
        mm_processor_kwargs.update(cfg.runtime.mm_processor_kwargs)
        kv_cache_dtype = (
            "auto" if cfg.runtime.kv_cache_dtype == "fp16" else cfg.runtime.kv_cache_dtype
        )
        llm_kwargs: dict[str, Any] = {
            "model": self.load_path,
            "trust_remote_code": cfg.model.trust_remote_code,
            "dtype": cfg.model.dtype,
            "max_model_len": cfg.runtime.max_model_len,
            "gpu_memory_utilization": cfg.runtime.gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": 1},
            "enforce_eager": cfg.runtime.enforce_eager,
            "kv_cache_dtype": kv_cache_dtype,
            "mm_processor_kwargs": mm_processor_kwargs or None,
            "disable_log_stats": False,
        }
        if cfg.quant.method == "bitsandbytes":
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
        if cfg.runtime.speculative_config:
            llm_kwargs["speculative_config"] = deepcopy(cfg.runtime.speculative_config)
        self.llm = LLM(**llm_kwargs)

    def build_prompt(self, image, text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def token_count(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        encoded = tokenizer(text, add_special_tokens=False)
        return len(encoded["input_ids"])

    def generate(self, prompt: str, image, max_tokens: int) -> GenerationResult:
        from vllm import SamplingParams

        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            SamplingParams(max_tokens=max_tokens, temperature=0.0),
            use_tqdm=False,
        )
        return _generation_result_from_request(outputs[0], self.token_count)

    def generate_batch(
        self,
        prompts_and_images: list[tuple[str, Any]],
        max_tokens: int,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        outputs = self.llm.generate(
            [
                {"prompt": prompt, "multi_modal_data": {"image": image}}
                for prompt, image in prompts_and_images
            ],
            SamplingParams(max_tokens=max_tokens, temperature=0.0),
            use_tqdm=False,
        )
        return [
            _generation_result_from_request(request_output, self.token_count)
            for request_output in outputs
        ]
