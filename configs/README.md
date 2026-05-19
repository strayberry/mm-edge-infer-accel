# Configs

Active configs are kept intentionally small. They define stable model/backend
baselines; common experiment variants should be passed through CLI overrides
instead of creating new YAML files.

## Active configs

```text
vlm/qwen3vl_4b_bf16.yaml
vlm/qwen3vl_4b_awq_local.yaml
vlm/qwen3vl_4b_gptq_local.yaml
vlm/smolvlm2_2b_fp32.yaml
vla/pi05_libero_plan.yaml
```

Use these overrides for routine variations:

```bash
python -m mm_edge_infer_accel.cli benchmark \
  --config configs/vlm/qwen3vl_4b_awq_local.yaml \
  --sample-count 1000 \
  --sample-strategy first \
  --concurrency 8 \
  --run \
  --output outputs/result.json
```

Supported benchmark overrides:

- `--concurrency`
- `--sample-count`
- `--sample-strategy`
- `--max-new-tokens`
- `--max-pixels`
