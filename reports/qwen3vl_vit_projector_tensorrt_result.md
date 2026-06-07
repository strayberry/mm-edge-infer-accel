# Qwen3-VL-4B TensorRT ViT + Projector Result

## Scope

This report records the completed TensorRT experiment only: exporting and benchmarking the Qwen3-VL-4B visual module path from `Qwen3VLForConditionalGeneration.model.visual`.

The exported module includes:

- ViT patch embedding and vision transformer blocks
- final `merger` / projector output (`pooler_output`)
- `deepstack_merger_list` outputs (`deepstack_0/1/2`)
- fixed-shape position/rotary logic for the 224x224 sample

It does not include:

- tokenizer or processor runtime
- text embedding
- image embedding injection into the text sequence
- MRoPE metadata construction outside the visual module
- LLM prefill/decode
- KV cache
- sampling
- full Qwen3-VL end-to-end inference

## Artifacts

Completed artifacts:

```text
outputs/onnx/qwen3vl_4b_visual.onnx
outputs/onnx/qwen3vl_4b_visual_sample/
outputs/tensorrt/qwen3vl_4b_visual_fp16.engine
outputs/tensorrt/qwen3vl_4b_visual_fp16_times.json
```

Note: the filenames still use `visual`, but the actual scope is ViT + merger/projector, because the exported object is `model.model.visual`.

Export script:

```text
scripts/export_qwen3vl_vit_projector_onnx.py
```

## Fixed Shape

The completed run uses a fixed 224x224 single-image sample. `image_grid_thw` is captured as a constant buffer in the wrapper, so this artifact is not a dynamic-resolution engine.

| input/output | shape | dtype | meaning |
| --- | ---: | --- | --- |
| `pixel_values` | `[256, 1536]` | FP16 input sample | flattened visual patch input |
| `last_hidden_state` | `[256, 1024]` | FP16 | final ViT hidden state before final merger |
| `pooler_output` | `[64, 2560]` | FP16 | final merger/projector output used as image embeddings |
| `deepstack_0/1/2` | `[64, 2560]` | FP16 | intermediate deepstack merger outputs |

## Performance

Environment: x86 / RTX 3080 Ti, TensorRT 8.6.1, fixed 224x224 single image.

| backend | scope | mean | p50 | p95 | throughput | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PyTorch eager FP16 | ViT + projector only | 23.04 ms | 22.33 ms | 26.60 ms | 43.40 qps | CUDA event, 200 iterations |
| TensorRT FP16 | ViT + projector only | 3.43 ms | 3.44 ms | 3.47 ms | 290.98 qps | CUDA Graph on, no H2D/D2H |

Observed speedup:

```text
23.04 ms / 3.43 ms = 6.7x
```

## TensorRT Build Notes

- ONNX parser passed with no unsupported op.
- Engine build time: 340.7 s.
- Engine size: 800 MiB.
- Build peak TensorRT GPU allocator: 1610 MiB.
- Runtime TensorRT allocation after context creation: about 803 MiB.
- Warnings observed:
  - ONNX INT64 constants cast to INT32.
  - 165 FP16 subnormal weights affected during conversion.

## Conclusion

The completed TensorRT result is a visual-module-only acceleration for Qwen3-VL-4B. It proves that the ViT + merger/projector path can be exported and accelerated with TensorRT at fixed 224x224 shape.

This result cannot be counted as full Qwen3-VL end-to-end acceleration. End-to-end latency still includes processor work, text embedding, image embedding injection, LLM prefill/decode, KV cache, sampling, and serving runtime overhead.
