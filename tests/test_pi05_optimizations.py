from __future__ import annotations

import torch

from mm_edge_infer_accel.pi05_kernels import fused_denoise_update
from mm_edge_infer_accel.pi05_optimizations import apply_pi05_optimizations


def test_fused_denoise_update_matches_euler_update():
    x_t = torch.tensor([[1.0, 2.0]])
    v_t = torch.tensor([[0.5, -1.0]])

    actual = fused_denoise_update(x_t, v_t, -0.2)

    torch.testing.assert_close(actual, x_t + (-0.2 * v_t))


def test_apply_pi05_optimizations_disabled_records_reason():
    result = apply_pi05_optimizations(object(), enabled=False)

    assert result.as_dict() == {
        "sample_actions_patched": False,
        "fused_denoise_update": False,
        "fused_denoise_update_backend": "torch",
        "cached_timesteps": False,
        "cached_suffix_masks": False,
        "reason": "disabled",
    }
