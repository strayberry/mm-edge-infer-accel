from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLViTProjectorWrapper(torch.nn.Module):
    """Wraps Qwen3-VL model.visual: ViT encoder plus merger/projector outputs.

    This is not the full Qwen3-VL model and does not include tokenizer,
    text embeddings, LLM prefill/decode, KV cache, or sampling.
    """
    def __init__(self, visual: torch.nn.Module, image_grid_thw: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        self.register_buffer("image_grid_thw", image_grid_thw.clone())

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.visual(pixel_values, self.image_grid_thw)
        result = [outputs.last_hidden_state, outputs.pooler_output]
        deepstack_features = getattr(outputs, "deepstack_features", None)
        if deepstack_features is not None:
            result.extend(deepstack_features)
        return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL ViT + projector/merger visual module to ONNX"
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct",
        help="Local Qwen3-VL model directory",
    )
    parser.add_argument(
        "--output",
        default="outputs/onnx/qwen3vl_4b_visual.onnx",
        help="Output ONNX path for the completed ViT + projector/merger visual-module export",
    )
    parser.add_argument(
        "--sample-dir",
        default="outputs/onnx/qwen3vl_4b_visual_sample",
        help="Directory for completed ViT + projector/merger sample input/output arrays",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true", help="Run onnx.checker after export")
    return parser.parse_args()


def build_sample_inputs(
    processor: AutoProcessor, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is in the image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs["pixel_values"].to(dtype=torch.float16), inputs["image_grid_thw"]


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    output = Path(args.output)
    sample_dir = Path(args.sample_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    pixel_values, image_grid_thw = build_sample_inputs(processor, args.image_size)
    wrapper = Qwen3VLViTProjectorWrapper(model.model.visual, image_grid_thw).eval()
    with torch.inference_mode():
        torch_outputs = wrapper(pixel_values)

    output_names = ["last_hidden_state", "pooler_output"] + [
        f"deepstack_{idx}" for idx in range(max(0, len(torch_outputs) - 2))
    ]
    print("pixel_values", tuple(pixel_values.shape), pixel_values.dtype)
    print("image_grid_thw", tuple(image_grid_thw.shape), image_grid_thw.dtype)
    for name, value in zip(output_names, torch_outputs, strict=True):
        print(name, tuple(value.shape), value.dtype)

    np.save(sample_dir / "pixel_values.npy", pixel_values.cpu().numpy())
    np.save(sample_dir / "image_grid_thw.npy", image_grid_thw.cpu().numpy())
    for name, value in zip(output_names, torch_outputs, strict=True):
        np.save(sample_dir / f"{name}.npy", value.cpu().numpy())

    torch.onnx.export(
        wrapper,
        (pixel_values,),
        output,
        input_names=["pixel_values"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    print(f"wrote {output}")
    print(f"wrote samples under {sample_dir}")

    if args.check:
        import onnx

        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model)
        print("onnx.checker passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
