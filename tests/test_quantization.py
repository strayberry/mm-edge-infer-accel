import pytest
import torch

from mm_edge_infer_accel.quantization.qwen3vl_llmcompressor import (
    Qwen3VLLLMCompressorArgs,
    build_calibration_dataset,
    build_recipe,
    collect_decoder_targets,
    collect_quantization_targets,
    collect_visual_targets,
    torch_dtype,
)


def test_torch_dtype_maps_supported_names():
    assert torch_dtype("bfloat16") is torch.bfloat16
    assert torch_dtype("float16") is torch.float16

    with pytest.raises(ValueError, match="Unsupported dtype"):
        torch_dtype("float32")


class TinyQwen3VLModel(torch.nn.Module):
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
        self.model.visual.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "qkv": torch.nn.Linear(2, 2),
                                "proj": torch.nn.Linear(2, 2),
                            }
                        ),
                        "mlp": torch.nn.ModuleDict(
                            {
                                "linear_fc1": torch.nn.Linear(2, 2),
                                "linear_fc2": torch.nn.Linear(2, 2),
                            }
                        ),
                    }
                )
            ]
        )
        self.model.visual.merger = torch.nn.ModuleDict(
            {
                "linear_fc1": torch.nn.Linear(2, 2),
                "linear_fc2": torch.nn.Linear(2, 2),
                "not_target": torch.nn.Linear(2, 2),
            }
        )


def test_collect_decoder_targets_only_includes_language_model_linear_layers():
    assert collect_decoder_targets(TinyQwen3VLModel()) == [
        "model.language_model.layers.0.q_proj"
    ]


def test_collect_visual_targets_includes_qwen3vl_visual_linear_layers():
    assert collect_visual_targets(TinyQwen3VLModel()) == [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ]


def test_collect_quantization_targets_supports_decoder_and_visual_decoder_scopes():
    model = TinyQwen3VLModel()

    assert collect_quantization_targets(model, "decoder") == [
        "model.language_model.layers.0.q_proj"
    ]
    assert collect_quantization_targets(model, "visual_decoder") == [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "model.language_model.layers.0.q_proj",
    ]


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


def test_smoothquant_default_output_dir():
    args = Qwen3VLLLMCompressorArgs(method="smoothquant")

    assert args.output_dir.endswith("Qwen3-VL-4B-Instruct-SmoothQuant-local")


def test_visual_decoder_smoothquant_default_output_dir():
    args = Qwen3VLLLMCompressorArgs(method="smoothquant", target_scope="visual_decoder")

    assert args.output_dir.endswith("Qwen3-VL-4B-Instruct-SmoothQuant-VisualDecoder-local")


def test_smoothquant_recipe_combines_smoothing_and_w8a8_quantization():
    class FakeSmoothQuantModifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeQuantizationModifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    recipe = build_recipe(
        "smoothquant",
        (FakeSmoothQuantModifier, FakeQuantizationModifier),
        ["model.language_model.layers.0.q_proj"],
        sequential_targets=None,
    )

    assert len(recipe) == 2
    assert isinstance(recipe[0], FakeSmoothQuantModifier)
    assert recipe[0].kwargs == {"smoothing_strength": 0.5, "ignore": ["model.visual"]}
    assert isinstance(recipe[1], FakeQuantizationModifier)
    assert recipe[1].kwargs == {
        "targets": ["model.language_model.layers.0.q_proj"],
        "scheme": "W8A8",
        "ignore": ["lm_head", "model.visual"],
    }


def test_visual_decoder_smoothquant_recipe_keeps_visual_targets():
    class FakeSmoothQuantModifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeQuantizationModifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    recipe = build_recipe(
        "smoothquant",
        (FakeSmoothQuantModifier, FakeQuantizationModifier),
        [
            "model.visual.blocks.0.attn.qkv",
            "model.language_model.layers.0.q_proj",
        ],
        sequential_targets=None,
        target_scope="visual_decoder",
    )

    assert len(recipe) == 2
    assert "ignore" not in recipe[0].kwargs
    assert recipe[0].kwargs["smoothing_strength"] == 0.5
    assert recipe[0].kwargs["mappings"]
    assert recipe[1].kwargs == {
        "targets": [
            "model.visual.blocks.0.attn.qkv",
            "model.language_model.layers.0.q_proj",
        ],
        "scheme": "W8A8",
        "ignore": ["lm_head"],
    }


def test_visual_decoder_scope_is_rejected_for_awq():
    class FakeAWQModifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    with pytest.raises(ValueError, match="visual_decoder"):
        build_recipe(
            "awq",
            FakeAWQModifier,
            ["model.visual.blocks.0.attn.qkv"],
            sequential_targets=None,
            target_scope="visual_decoder",
        )
