from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from datasets import Dataset, load_dataset


TEXT_CALIB_DATA = [
    "Describe this image in detail.",
    "Extract all visible text from the image.",
    "Answer the question based on the image.",
    "Summarize the document screenshot.",
    "Analyze the chart and explain the trend.",
    "What is the spatial relationship between the objects?",
    "Read the table and answer the question.",
    "Identify the object being manipulated by the robot.",
] * 32


OCRBENCH_DOC_TYPES = {
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Scene Text-centric VQA",
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
}


def build_qwen3vl_prompt(processor, image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def multimodal_data_collator(batch: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("Calibration data_collator expects batch size 1.")
    return {key: _as_tensor(value) for key, value in batch[0].items()}


def load_text_calibration(max_samples: int):
    return Dataset.from_dict({"text": TEXT_CALIB_DATA[:max_samples]})


def load_docvqa_calibration(
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    max_samples: int,
    processor,
    max_seq_len: int,
    max_pixels: int | None = None,
) -> Dataset:
    if dataset_config:
        raw = load_dataset(dataset_id, dataset_config, split=f"{split}[:{max_samples}]")
    else:
        raw = load_dataset(dataset_id, split=f"{split}[:{max_samples}]")
    samples = []
    for item in raw:
        image = _first_present(item, ("image", "doc_image", "page_image"))
        question = _first_present(item, ("question", "query"))
        if image is None or question is None:
            raise KeyError(
                "DocVQA calibration expects image and question columns. "
                f"Available columns: {list(item.keys())}"
            )
        samples.append(
            _encode_vision_sample(
                processor,
                image,
                str(question),
                max_seq_len,
                max_pixels=max_pixels,
            )
        )
    return Dataset.from_list(samples)


def load_ocrbench_doc_calibration(
    max_samples: int,
    processor,
    max_seq_len: int,
    max_pixels: int | None = None,
) -> Dataset:
    raw = load_dataset("echo840/OCRBench", split="test")
    samples = []
    for item in raw:
        question_type = item.get("question_type")
        if question_type not in OCRBENCH_DOC_TYPES:
            continue
        samples.append(
            _encode_vision_sample(
                processor,
                item["image"],
                item["question"],
                max_seq_len,
                max_pixels=max_pixels,
            )
        )
        if len(samples) >= max_samples:
            break
    return Dataset.from_list(samples)


def _encode_vision_sample(
    processor,
    image,
    question: str,
    max_seq_len: int,
    max_pixels: int | None = None,
) -> dict[str, Any]:
    image = _resize_to_max_pixels(image.convert("RGB"), max_pixels)
    prompt = build_qwen3vl_prompt(processor, image, question)
    encoded = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    if encoded["input_ids"].shape[-1] > max_seq_len:
        raise ValueError(
            f"Encoded calibration sample has {encoded['input_ids'].shape[-1]} tokens, "
            f"which exceeds max_seq_len={max_seq_len}. Lower --calib-max-pixels or "
            "increase --max-calib-seq-len."
        )
    return {key: value.tolist() for key, value in encoded.items()}


def _resize_to_max_pixels(image, max_pixels: int | None):
    if max_pixels is None:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height))


def _first_present(item: Mapping[str, Any], keys: tuple[str, ...]):
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None
