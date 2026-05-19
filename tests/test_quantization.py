import pytest
import torch

from mm_edge_infer_accel.quantization.qwen3vl_llmcompressor import (
    Qwen3VLLLMCompressorArgs,
    build_calibration_dataset,
    collect_decoder_targets,
    torch_dtype,
)


def test_torch_dtype_maps_supported_names():
    assert torch_dtype("bfloat16") is torch.bfloat16
    assert torch_dtype("float16") is torch.float16

    with pytest.raises(ValueError, match="Unsupported dtype"):
        torch_dtype("float32")


def test_collect_decoder_targets_only_includes_language_model_linear_layers():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "q_proj": torch.nn.Linear(2, 2),
                            "not_target": torch.nn.Linear(2, 2),
                        }
                    )
                ]
            )
            self.model.visual = torch.nn.Module()
            self.model.visual.q_proj = torch.nn.Linear(2, 2)

    assert collect_decoder_targets(TinyModel()) == ["model.language_model.layers.0.q_proj"]


def test_text_calibration_dataset_uses_requested_sample_count():
    args = Qwen3VLLLMCompressorArgs(method="gptq", max_calib_samples=3, calib_source="text")

    dataset, data_collator, text_column = build_calibration_dataset(args, processor=None)

    assert len(dataset) == 3
    assert data_collator is None
    assert text_column == "text"


def test_calibration_source_validation():
    args = Qwen3VLLLMCompressorArgs(method="gptq", calib_source="bad")

    with pytest.raises(ValueError, match="calib_source"):
        build_calibration_dataset(args, processor=None)
