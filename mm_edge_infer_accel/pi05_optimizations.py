from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class Pi05PatchResult:
    sample_actions_patched: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_actions_patched": self.sample_actions_patched,
            "reason": self.reason,
        }


def apply_pi05_optimizations(policy, *, enabled: bool) -> Pi05PatchResult:
    if not enabled:
        return Pi05PatchResult(False, "disabled")

    model = getattr(policy, "model", None)
    if model is None:
        return Pi05PatchResult(False, "policy has no model")

    required = (
        "embed_prefix",
        "paligemma_with_expert",
        "_prepare_attention_masks_4d",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
        "sample_noise",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        return Pi05PatchResult(False, f"missing attributes: {missing}")

    if getattr(model, "_mm_edge_original_sample_actions", None) is None:
        model._mm_edge_original_sample_actions = model.sample_actions
    model.sample_actions = types.MethodType(_optimized_sample_actions, model)

    return Pi05PatchResult(True)


@torch.no_grad()
def _optimized_sample_actions(
    self,
    images,
    img_masks,
    tokens,
    masks,
    noise=None,
    num_steps=None,
    **kwargs,
):
    if self._rtc_enabled():
        return self._mm_edge_original_sample_actions(
            images, img_masks, tokens, masks,
            noise=noise, num_steps=num_steps, **kwargs,
        )

    if num_steps is None:
        num_steps = self.config.num_inference_steps

    bsize = tokens.shape[0]
    device = tokens.device

    if noise is None:
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )

    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    suffix_context = _make_suffix_context(self, prefix_pad_masks)
    dt = -1.0 / num_steps

    x_t = noise
    for step in range(num_steps):
        timestep = torch.full((bsize,), 1.0 + step * dt, device=device)
        v_t = _denoise_step_cached(
            self,
        past_key_values=copy.deepcopy(past_key_values),
            x_t=x_t,
            timestep=timestep,
            suffix_context=suffix_context,
        )
        x_t = x_t.add(v_t, alpha=dt)

        if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
            self.rtc_processor.track(time=float(timestep[0].item()), x_t=x_t, v_t=v_t)

    return x_t


def _make_suffix_context(model, prefix_pad_masks):
    suffix_len = model.config.chunk_size
    batch_size = prefix_pad_masks.shape[0]
    device = prefix_pad_masks.device

    suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
    suffix_att_masks = torch.zeros(batch_size, suffix_len, dtype=torch.float32, device=device)
    suffix_att_masks[:, 0] = 1

    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)

    return {
        "attention_mask": full_att_2d_masks_4d,
        "position_ids": position_ids,
    }


def _denoise_step_cached(model, *, past_key_values, x_t, timestep, suffix_context):
    suffix_embs, adarms_cond = _embed_suffix_fast(model, x_t, timestep)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=suffix_context["attention_mask"],
        position_ids=suffix_context["position_ids"],
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -model.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def _embed_suffix_fast(model, noisy_actions, timestep):
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        model.action_in_proj.out_features,
        min_period=model.config.min_period,
        max_period=model.config.max_period,
        device=timestep.device,
    )
    time_emb = time_emb.type(dtype=timestep.dtype)

    action_emb = model._apply_checkpoint(model.action_in_proj, noisy_actions)

    def time_mlp_func(time_emb):
        x = model.time_mlp_in(time_emb)
        x = F.silu(x)
        x = model.time_mlp_out(x)
        return F.silu(x)

    adarms_cond = model._apply_checkpoint(time_mlp_func, time_emb)
    return action_emb, adarms_cond


def _make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
