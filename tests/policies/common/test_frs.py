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

"""CPU tests for the Flow Reversal Steering helpers (arXiv:2606.13675)."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.common.frs import (
    build_reference_chunk,
    expand_noise,
    make_action_normalizer,
    normalize_action_chunk,
)


class TestBuildReferenceChunk:
    def test_single_action_is_tiled(self):
        ref = torch.tensor([[1.0, 2.0, 3.0]])
        chunk = build_reference_chunk(ref, chunk_size=4, max_action_dim=5)
        assert chunk.shape == (1, 4, 5)
        assert chunk.dtype == torch.float32
        for t in range(4):
            assert torch.equal(chunk[0, t, :3], ref[0])
        # Padding dims beyond action_dim are zero.
        assert torch.equal(chunk[0, :, 3:], torch.zeros(4, 2))

    def test_short_chunk_is_tiled(self):
        ref = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        chunk = build_reference_chunk(ref, chunk_size=5, max_action_dim=3)
        assert chunk.shape == (1, 5, 3)
        expected = torch.cat([ref[0], ref[0], ref[0, :1]], dim=0)
        assert torch.equal(chunk[0], expected)

    def test_mean_mode_averages_then_tiles(self):
        ref = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]])
        chunk = build_reference_chunk(ref, chunk_size=3, max_action_dim=3, mode="mean")
        assert chunk.shape == (1, 3, 3)
        assert torch.equal(chunk[0], torch.full((3, 3), 2.0))

    def test_reference_longer_than_chunk_is_truncated(self):
        ref = torch.randn(2, 7, 3)
        chunk = build_reference_chunk(ref, chunk_size=4, max_action_dim=3)
        assert chunk.shape == (2, 4, 3)
        assert torch.equal(chunk, ref[:, :4, :])

    def test_action_dim_exceeding_max_raises(self):
        with pytest.raises(ValueError, match="exceeds max_action_dim"):
            build_reference_chunk(torch.randn(1, 3), chunk_size=4, max_action_dim=2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            build_reference_chunk(torch.randn(1, 3), chunk_size=4, max_action_dim=3, mode="bogus")

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="must be"):
            build_reference_chunk(torch.randn(3), chunk_size=4, max_action_dim=3)


class TestExpandNoise:
    def test_2d_noise_is_tiled_and_padded(self):
        torch.manual_seed(0)
        noise_vec = torch.randn(2, 3)
        chunk = expand_noise(noise_vec, chunk_size=4, max_action_dim=6)
        assert chunk.shape == (2, 4, 6)
        for t in range(4):
            assert torch.equal(chunk[:, t, :3], noise_vec)
        # Padding dims are fresh N(0, 1) samples, not copies of the predicted dims.
        assert not torch.allclose(chunk[:, :, 3:], torch.zeros(2, 4, 3))
        assert 0.7 < chunk[:, :, 3:].std() < 1.3

    def test_sigma_scales_padding_noise(self):
        torch.manual_seed(1)
        chunk = expand_noise(torch.randn(2, 3), chunk_size=8, max_action_dim=7, sigma=2.0)
        assert 1.5 < chunk[:, :, 3:].std() < 2.6

    def test_full_width_noise_gets_no_padding(self):
        noise_vec = torch.randn(2, 4, 3)
        chunk = expand_noise(noise_vec, chunk_size=4, max_action_dim=3)
        assert chunk.shape == (2, 4, 3)
        assert torch.equal(chunk, noise_vec)

    def test_generator_makes_sampling_reproducible(self):
        g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
        c1 = expand_noise(torch.randn(2, 3), chunk_size=4, max_action_dim=6, generator=g1)
        torch.randn(2, 3)  # advance the global stream between the two calls
        c2 = expand_noise(torch.randn(2, 3), chunk_size=4, max_action_dim=6, generator=g2)
        assert torch.equal(c1[:, :, 3:], c2[:, :, 3:])


def _fake_policy_config(action_dim: int = 3):
    return SimpleNamespace(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
    )


class TestNormalizeActionChunk:
    def test_mean_std_math(self):
        mean = torch.tensor([0.0, 10.0, -5.0])
        std = torch.tensor([1.0, 2.0, 0.5])
        stats = {"action": {"mean": mean, "std": std}}
        actions = torch.tensor([[1.0, 12.0, -4.0]])
        out = normalize_action_chunk(actions, _fake_policy_config(), stats)
        torch.testing.assert_close(out, (actions - mean) / (std + 1e-8))

    def test_identity_when_stats_missing(self):
        config = _fake_policy_config()
        config.normalization_mapping = {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        }
        actions = torch.randn(2, 4, 3)
        out = normalize_action_chunk(actions, config, {"action": {}})
        assert torch.equal(out, actions)

    def test_make_action_normalizer_uses_action_stats(self):
        stats = {"action": {"mean": torch.zeros(3), "std": torch.ones(3)}}
        step = make_action_normalizer(_fake_policy_config(), stats)
        out = step._apply_transform(  # noqa: SLF001
            torch.tensor([[2.0, 2.0, 2.0]]), "action", FeatureType.ACTION, inverse=False
        )
        torch.testing.assert_close(out, torch.full((1, 3), 2.0))
