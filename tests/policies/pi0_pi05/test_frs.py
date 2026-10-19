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

"""Flow Reversal Steering (FRS, arXiv:2606.13675) integration tests.

Verifies on random-init policies (GPU, mirroring the RTC test pattern) that:
- `reference_actions=` steering moves the sampled chunk toward the reference;
- `reverse_actions` composes with `noise=` exactly like the combined path;
- reference validation / noise-conflict errors fire.

Heavy: each policy holds a full VLM; requires CUDA and an HF token (PaliGemma
config download). Run with:
    pytest tests/policies/pi0_pi05/test_frs.py -v
"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.frs import build_reference_chunk  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda, require_hf_token  # noqa: E402

CHUNK_SIZE = 10
ACTION_DIM = 7
STATE_DIM = 14


@pytest.fixture(autouse=True)
def _free_cuda_memory():
    """Release the previous policy's VRAM between the heavyweight tests."""
    yield
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_features(config):
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }


def _make_stats():
    return {
        "observation.state": {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        "action": {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }


def _make_batch(device):
    return {
        OBS_STATE: torch.randn(1, STATE_DIM, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }


def _rand_reference(device):
    # A short (3-step) coarse reference, tiled to the policy chunk like the
    # paper's `action_repeats` reference chunks. Stats are identity, so the
    # model-space reference equals the raw one.
    return build_reference_chunk(
        torch.randn(1, 3, ACTION_DIM, device=device),
        chunk_size=CHUNK_SIZE,
        max_action_dim=ACTION_DIM,
    )


# --------------------------------------------------------------------------
# pi0
# --------------------------------------------------------------------------


def _setup_pi0():
    from lerobot.policies.pi0 import PI0Config, PI0Policy, make_pi0_pre_post_processors

    set_seed(42)
    config = PI0Config(
        max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM, chunk_size=CHUNK_SIZE, dtype="float32"
    )
    _make_features(config)
    policy = PI0Policy(config)
    policy.eval()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=_make_stats())
    batch = preprocessor(_make_batch(config.device))
    return policy, batch


@require_cuda
@require_hf_token
def test_pi0_frs_steering_pulls_actions_toward_reference():
    policy, batch = _setup_pi0()
    device = policy.config.device

    reference = _rand_reference(device)
    fixed_noise = policy.model.sample_noise((1, CHUNK_SIZE, ACTION_DIM), device)

    with torch.no_grad():
        steered = policy.predict_action_chunk(
            batch, reference_actions=reference.clone(), num_reverse_steps=50
        )
        unsteered = policy.predict_action_chunk(batch, noise=fixed_noise.clone())

    dist_steered = (steered - reference[..., :ACTION_DIM]).norm().item()
    dist_unsteered = (unsteered - reference[..., :ACTION_DIM]).norm().item()
    assert dist_steered < dist_unsteered, (
        f"FRS output ({dist_steered:.3f}) should be closer to the reference than "
        f"an unsteered sample ({dist_unsteered:.3f})"
    )


@require_cuda
@require_hf_token
def test_pi0_frs_tracks_its_own_reference():
    policy, batch = _setup_pi0()
    device = policy.config.device

    ref_a = _rand_reference(device)
    ref_b = -ref_a.clone()  # opposite reference

    with torch.no_grad():
        out_a = policy.predict_action_chunk(batch, reference_actions=ref_a, num_reverse_steps=50)
        out_b = policy.predict_action_chunk(batch, reference_actions=ref_b, num_reverse_steps=50)

    assert not torch.allclose(out_a, out_b, rtol=1e-4)
    dist_a_to_a = (out_a - ref_a[..., :ACTION_DIM]).norm().item()
    dist_b_to_a = (out_b - ref_a[..., :ACTION_DIM]).norm().item()
    assert dist_a_to_a < dist_b_to_a


@require_cuda
@require_hf_token
def test_pi0_frs_reverse_actions_matches_reference_kwarg_path():
    policy, batch = _setup_pi0()
    device = policy.config.device

    reference = _rand_reference(device)

    with torch.no_grad():
        images, img_masks = policy._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = policy.prepare_state(batch)
        noise_hat = policy.model.reverse_actions(
            images, img_masks, lang_tokens, lang_masks, state, reference, num_steps=10
        )
        manual = policy.predict_action_chunk(batch, noise=noise_hat.clone())
        combined = policy.predict_action_chunk(batch, reference_actions=reference.clone(), num_reverse_steps=10)

    torch.testing.assert_close(manual, combined, rtol=1e-4, atol=1e-5)


@require_cuda
@require_hf_token
def test_pi0_frs_validation_errors():
    policy, batch = _setup_pi0()
    device = policy.config.device

    reference = _rand_reference(device)
    fixed_noise = policy.model.sample_noise((1, CHUNK_SIZE, ACTION_DIM), device)

    with torch.no_grad():
        with pytest.raises(ValueError, match="not both"):
            policy.predict_action_chunk(
                batch, noise=fixed_noise.clone(), reference_actions=reference.clone()
            )
        with pytest.raises(ValueError, match="must have shape"):
            policy.predict_action_chunk(
                batch, reference_actions=torch.randn(1, 3, ACTION_DIM, device=device)
            )


# --------------------------------------------------------------------------
# pi05
# --------------------------------------------------------------------------


def _setup_pi05():
    from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors

    set_seed(42)
    config = PI05Config(
        max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM, chunk_size=CHUNK_SIZE, dtype="float32"
    )
    _make_features(config)
    policy = PI05Policy(config)
    policy.eval()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=_make_stats())
    batch = preprocessor(_make_batch(config.device))
    return policy, batch


@require_cuda
@require_hf_token
def test_pi05_frs_steering_pulls_actions_toward_reference():
    policy, batch = _setup_pi05()
    device = policy.config.device

    reference = _rand_reference(device)
    fixed_noise = policy.model.sample_noise((1, CHUNK_SIZE, ACTION_DIM), device)

    with torch.no_grad():
        steered = policy.predict_action_chunk(
            batch, reference_actions=reference.clone(), num_reverse_steps=50
        )
        unsteered = policy.predict_action_chunk(batch, noise=fixed_noise.clone())

    dist_steered = (steered - reference[..., :ACTION_DIM]).norm().item()
    dist_unsteered = (unsteered - reference[..., :ACTION_DIM]).norm().item()
    assert dist_steered < dist_unsteered


@require_cuda
@require_hf_token
def test_pi05_frs_reverse_actions_matches_reference_kwarg_path():
    policy, batch = _setup_pi05()
    device = policy.config.device

    reference = _rand_reference(device)

    with torch.no_grad():
        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        states, state_masks = policy._prepare_memory_states(batch)
        noise_hat = policy.model.reverse_actions(
            images, img_masks, tokens, masks, reference, states=states, state_masks=state_masks, num_steps=10
        )
        manual = policy.predict_action_chunk(batch, noise=noise_hat.clone())
        combined = policy.predict_action_chunk(batch, reference_actions=reference.clone(), num_reverse_steps=10)

    torch.testing.assert_close(manual, combined, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# smolvla
# --------------------------------------------------------------------------


def _setup_smolvla():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

    set_seed(42)
    config = SmolVLAConfig(max_action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE)
    _make_features(config)
    policy = SmolVLAPolicy(config)
    policy.to(config.device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=None, dataset_stats=_make_stats()
    )
    batch = preprocessor(_make_batch(config.device))
    return policy, batch


@require_cuda
def test_smolvla_frs_steering_pulls_actions_toward_reference():
    policy, batch = _setup_smolvla()
    device = policy.config.device

    reference = _rand_reference(device)
    fixed_noise = policy.model.sample_noise((1, CHUNK_SIZE, ACTION_DIM), device)

    with torch.no_grad():
        steered = policy.predict_action_chunk(
            batch, reference_actions=reference.clone(), num_reverse_steps=50
        )
        unsteered = policy.predict_action_chunk(batch, noise=fixed_noise.clone())

    dist_steered = (steered - reference[..., :ACTION_DIM]).norm().item()
    dist_unsteered = (unsteered - reference[..., :ACTION_DIM]).norm().item()
    assert dist_steered < dist_unsteered


@require_cuda
def test_smolvla_frs_reverse_actions_round_trip():
    policy, batch = _setup_smolvla()
    device = policy.config.device

    reference = _rand_reference(device)

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        noise_hat = policy.model.reverse_actions(
            images, img_masks, lang_tokens, lang_masks, state, reference, num_steps=50
        )
        out = policy.predict_action_chunk(batch, noise=noise_hat.clone())
        unsteered = policy.predict_action_chunk(
            batch, noise=policy.model.sample_noise((1, CHUNK_SIZE, ACTION_DIM), device)
        )

    dist_frs = (out - reference[..., :ACTION_DIM]).norm().item()
    dist_unsteered = (unsteered - reference[..., :ACTION_DIM]).norm().item()
    assert dist_frs < dist_unsteered
