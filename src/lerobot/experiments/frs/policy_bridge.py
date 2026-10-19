#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Bridge between FRS experiment scripts and flow-matching policy checkpoints.

Handles the per-policy plumbing that `sample_actions(reference_actions=...)`
and `reverse_actions(...)` need but that differs between pi0 / pi05 / smolvla:
observation preparation, state handling and the padded-action format. Also
recovers the dataset statistics baked into a checkpoint's pre-processor so raw
robot-space reference actions can be normalized the same way as training data.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.frs import build_reference_chunk, normalize_action_chunk
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def load_policy_with_processors(
    policy_path: str | Path,
    device: str = "cuda",
) -> tuple[PreTrainedPolicy, object, object, PreTrainedConfig]:
    """Load a policy checkpoint plus its serialized pre/post processors.

    Returns `(policy, preprocessor, postprocessor, policy_config)`.
    """
    policy_path = Path(policy_path)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = device
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, policy_config


def get_dataset_stats(preprocessor) -> dict:
    """Extract the dataset statistics from a pre-processor pipeline.

    The stats are serialized inside the pipeline's NormalizerProcessorStep when
    a training run saves its processors to the checkpoint directory.
    """
    for step in getattr(preprocessor, "steps", []):
        if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
            return step.stats
    raise ValueError(
        "Could not find a (un)normalizer step carrying dataset stats in the "
        f"pre-processor pipeline ({preprocessor}). Was the checkpoint saved by "
        "lerobot-train?"
    )


def build_normalized_reference(
    raw_reference: Tensor,
    policy: PreTrainedPolicy,
    dataset_stats: dict,
    mode: str = "tile",
) -> Tensor:
    """Raw robot-space reference actions -> normalized, padded policy-format chunk.

    `raw_reference` is `(action_dim,)`, `(B, action_dim)` or `(B, ref_len, action_dim)`
    in the same units as the training actions (e.g. absolute joint positions).
    """
    if raw_reference.dim() == 1:
        raw_reference = raw_reference[None, None, :]
    normalized = normalize_action_chunk(raw_reference, policy.config, dataset_stats)
    return build_reference_chunk(
        normalized,
        chunk_size=policy.config.chunk_size,
        max_action_dim=policy.config.max_action_dim,
        mode=mode,
    ).to(policy.config.device)


@torch.no_grad()
def reverse_actions_for_policy(
    policy: PreTrainedPolicy,
    batch: dict[str, Tensor],
    padded_actions: Tensor,
    num_steps: int,
) -> Tensor:
    """Dispatch `model.reverse_actions` with the per-policy observation plumbing.

    `batch` is an already pre-processed batch (normalized state, tokens, images)
    as produced by the policy's pre-processor; `padded_actions` is a normalized
    `(B, chunk_size, max_action_dim)` chunk.
    """
    model = policy.model

    policy_type = policy.config.type
    if policy_type == "pi05":
        images, img_masks = policy._preprocess_images(batch)
        states, state_masks = policy._prepare_memory_states(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return model.reverse_actions(
            images, img_masks, tokens, masks, padded_actions,
            states=states, state_masks=state_masks, num_steps=num_steps,
        )
    if policy_type == "smolvla":
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return model.reverse_actions(
            images, img_masks, lang_tokens, lang_masks, state, padded_actions,
            num_steps=num_steps,
        )
    if policy_type == "pi0":
        images, img_masks = policy._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = policy.prepare_state(batch)
        return model.reverse_actions(
            images, img_masks, lang_tokens, lang_masks, state, padded_actions,
            num_steps=num_steps,
        )
    raise ValueError(
        f"FRS is implemented for pi0 / pi05 / smolvla flow-matching policies, got {policy_type!r}"
    )


def pad_action_chunk(normalized_actions: Tensor, policy: PreTrainedPolicy) -> Tensor:
    """Pad a normalized (B, chunk, action_dim) chunk to (B, chunk, max_action_dim)."""
    return pad_vector(normalized_actions.to(policy.config.device), policy.config.max_action_dim)


def action_dim_of(policy: PreTrainedPolicy) -> int:
    """Robot-side action dimensionality of the policy (before padding)."""
    return policy.config.output_features[ACTION].shape[0]
