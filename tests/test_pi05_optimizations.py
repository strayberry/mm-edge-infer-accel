from __future__ import annotations

from types import SimpleNamespace

from mm_edge_infer_accel.pi05_optimizations import apply_pi05_optimizations


def test_apply_pi05_optimizations_disabled_records_reason():
    result = apply_pi05_optimizations(object(), enabled=False)

    assert result.as_dict() == {
        "sample_actions_patched": False,
        "reason": "disabled",
    }


def test_apply_pi05_optimizations_enabled_no_model():
    policy = SimpleNamespace()
    result = apply_pi05_optimizations(policy, enabled=True)

    assert result.sample_actions_patched is False
    assert "no model" in result.reason


class _FakeModel:
    embed_prefix = None
    paligemma_with_expert = None
    _prepare_attention_masks_4d = None
    action_in_proj = None
    time_mlp_in = None
    time_mlp_out = None
    action_out_proj = None
    sample_noise = None
    sample_actions = None


def test_apply_pi05_optimizations_enabled_with_model():
    policy = SimpleNamespace(model=_FakeModel())
    result = apply_pi05_optimizations(policy, enabled=True)

    assert result.sample_actions_patched is True
    assert result.reason is None
    assert hasattr(policy.model, "_mm_edge_original_sample_actions")
    assert hasattr(policy.model, "sample_actions")


def test_apply_pi05_optimizations_preserves_original():
    model = _FakeModel()
    original = SimpleNamespace()
    model.sample_actions = original
    policy = SimpleNamespace(model=model)

    apply_pi05_optimizations(policy, enabled=True)

    assert policy.model._mm_edge_original_sample_actions is original


def test_apply_pi05_optimizations_skips_if_already_patched():
    model = _FakeModel()
    model._mm_edge_original_sample_actions = "already_saved"
    policy = SimpleNamespace(model=model)

    apply_pi05_optimizations(policy, enabled=True)

    assert policy.model._mm_edge_original_sample_actions == "already_saved"


def test_apply_pi05_optimizations_missing_attributes():
    class _PartialModel:
        embed_prefix = None
        paligemma_with_expert = None

    policy = SimpleNamespace(model=_PartialModel())
    result = apply_pi05_optimizations(policy, enabled=True)

    assert result.sample_actions_patched is False
    assert "missing" in result.reason
