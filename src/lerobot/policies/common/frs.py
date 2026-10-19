#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for Flow Reversal Steering (FRS).

Tang et al., "Improving Robotic Generalist Policies via Flow Reversal Steering"
(arXiv:2606.13675). These utilities prepare action / noise tensors in the padded
policy format expected by ``sample_actions`` / ``reverse_integrate``:

- a coarse reference chunk shorter than the policy chunk size is tiled (or
  averaged then tiled) along the chunk axis — the paper's Appendix C.4 shows
  this keeps performance while making the recovered noise easier to model;
- dims beyond the robot's action dim (the openpi padding dims) get fresh
  N(0, I) noise on the noise side, which the paper found to have no effect.
"""

import torch
from torch import Tensor

from lerobot.configs import FeatureType
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION


def make_action_normalizer(policy_config, dataset_stats) -> NormalizerProcessorStep:
    """Build a standalone ACTION normalizer from a policy config and dataset stats.

    Mirrors the normalizer construction in
    `lerobot.processor.factory.make_default_policy_processor_steps`, restricted to
    what is needed to normalize action chunks (FRS reference chunks / DSBC noise
    targets) exactly like the policy's own pre-processor would.
    """
    return NormalizerProcessorStep(
        features={**policy_config.input_features, **policy_config.output_features},
        norm_map=policy_config.normalization_mapping,
        stats=dataset_stats,
    )


def normalize_action_chunk(
    actions: Tensor,
    policy_config,
    dataset_stats,
) -> Tensor:
    """Normalize raw actions with the policy's ACTION stats (no padding).

    The inverse of the policy post-processor's unnormalization: e.g. for
    MEAN_STD, `(action - mean) / std`. Use this on raw (robot-space) reference
    actions before :func:`build_reference_chunk`.
    """
    normalizer = make_action_normalizer(policy_config, dataset_stats)
    # `_apply_transform` is the shared core of Normalizer/UnnormalizerProcessorStep
    # (their public __call__ require a full EnvTransition, which we don't have).
    return normalizer._apply_transform(  # noqa: SLF001
        torch.as_tensor(actions, dtype=torch.float32), ACTION, FeatureType.ACTION, inverse=False
    )


def build_reference_chunk(
    reference: Tensor,
    chunk_size: int,
    max_action_dim: int,
    mode: str = "tile",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Expand a reference action into a full padded policy-format chunk.

    Args:
        reference: Reference actions of shape ``(batch, action_dim)``
            (a single coarse action, e.g. one IK joint target) or
            ``(batch, ref_len, action_dim)`` (a short chunk, e.g. an
            action repeated ``action_repeats`` times as in the paper).
        chunk_size: Policy chunk size to expand to along axis 1.
        max_action_dim: Policy padded action dim (``config.max_action_dim``);
            ``action_dim`` may be smaller, remaining dims are zero-padded
            (their noise side is handled by :func:`expand_noise`).
        mode: ``"tile"`` repeats the reference along the chunk axis;
            ``"mean"`` first averages a multi-step reference along the chunk
            axis into a single action, then tiles it (paper Appendix C.4).
        generator: Optional torch generator for reproducible padding noise
            (unused here; kept for API symmetry with :func:`expand_noise`).

    Returns:
        ``(batch, chunk_size, max_action_dim)`` float32 tensor.
    """
    if reference.dim() == 2:
        reference = reference[:, None, :]
    if reference.dim() != 3:
        raise ValueError(f"reference must be (B, action_dim) or (B, ref_len, action_dim), got {tuple(reference.shape)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    batch, ref_len, action_dim = reference.shape
    if mode == "mean":
        if ref_len > 1:
            reference = reference.mean(dim=1, keepdim=True)
            ref_len = 1
    elif mode != "tile":
        raise ValueError(f"mode must be 'tile' or 'mean', got {mode!r}")

    if ref_len > chunk_size:
        reference = reference[:, :chunk_size, :]
        ref_len = chunk_size

    # Tile by repeating the (possibly averaged) reference until chunk_size is filled
    tile_count = (chunk_size + ref_len - 1) // ref_len
    reference = reference.repeat(1, tile_count, 1)[:, :chunk_size, :]

    if action_dim > max_action_dim:
        raise ValueError(
            f"reference action_dim ({action_dim}) exceeds max_action_dim ({max_action_dim}); "
            "the policy would truncate it — check the checkpoint's output features."
        )
    reference = torch.nn.functional.pad(reference, (0, max_action_dim - action_dim))

    return reference.to(dtype=torch.float32)


def expand_noise(
    noise_vec: Tensor,
    chunk_size: int,
    max_action_dim: int,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Expand a per-step noise action into a full padded policy-format chunk.

    Used at DSBC inference time: the noise policy predicts a single
    ``(batch, action_dim)`` noise vector â0 (the chunk axis was averaged away
    during training), which is tiled along the chunk axis, and the padding dims
    beyond ``action_dim`` are filled with fresh ``N(0, sigma^2 I)`` samples
    (paper: replacing padding-dim noise with N(0, I) has no effect).

    Args:
        noise_vec: ``(batch, action_dim)`` or ``(batch, chunk_size, action_dim)``
            predicted noise. A 2D input is tiled to ``chunk_size``.
        chunk_size: Policy chunk size along axis 1.
        max_action_dim: Policy padded action dim; dims beyond ``action_dim``
            are sampled from N(0, sigma^2).
        sigma: Std of the padding-dim noise (paper Appendix C.3 also explores
            adding sigma-scaled Gaussian noise to the recovered noise itself).
        generator: Optional torch generator for reproducible sampling.

    Returns:
        ``(batch, chunk_size, max_action_dim)`` float32 tensor.
    """
    if noise_vec.dim() == 2:
        noise_vec = noise_vec[:, None, :]
    if noise_vec.dim() != 3:
        raise ValueError(f"noise_vec must be (B, action_dim) or (B, k, action_dim), got {tuple(noise_vec.shape)}")

    batch, ref_len, action_dim = noise_vec.shape
    tile_count = (chunk_size + ref_len - 1) // ref_len
    noise_vec = noise_vec.repeat(1, tile_count, 1)[:, :chunk_size, :]

    if action_dim > max_action_dim:
        raise ValueError(
            f"noise action_dim ({action_dim}) exceeds max_action_dim ({max_action_dim})"
        )
    pad_shape = (batch, chunk_size, max_action_dim - action_dim)
    if pad_shape[-1] > 0:
        pad_noise = torch.normal(
            mean=0.0, std=sigma, size=pad_shape, dtype=torch.float32, device=noise_vec.device, generator=generator
        )
        noise_vec = torch.cat([noise_vec.to(torch.float32), pad_noise], dim=-1)
    else:
        noise_vec = noise_vec.to(torch.float32)

    return noise_vec
