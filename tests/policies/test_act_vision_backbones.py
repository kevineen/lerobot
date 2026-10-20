#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for ACT vision backbones (torchvision ResNet, Hugging Face DINOv2/DINOv3, timm)."""

import os

import pytest

pytest.importorskip("transformers", reason="transformers is required to test HF vision backbones")

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.common.vision_backbones import build_vision_backbone
from lerobot.utils.import_utils import is_package_available

ACTION_DIM = 6
STATE_DIM = 6


def make_act_config(image_shape: tuple, vision_backbone: str = "resnet18", **kwargs) -> ACTConfig:
    config = ACTConfig(
        chunk_size=8,
        n_action_steps=8,
        dim_model=64,
        n_encoder_layers=2,
        dim_feedforward=128,
        vision_backbone=vision_backbone,
        **kwargs,
    )
    config.input_features = {
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return config


def make_dummy_batch(config: ACTConfig, batch_size: int = 2) -> dict:
    batch = {key: torch.rand(batch_size, *ft.shape) for key, ft in config.input_features.items()}
    batch["action"] = torch.rand(batch_size, config.chunk_size, ACTION_DIM)
    batch["action_is_pad"] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
    return batch


@pytest.fixture(scope="session")
def tiny_dinov2_path(tmp_path_factory) -> str:
    """A randomly-initialized tiny DINOv2 saved locally, so no hub download is needed."""
    from transformers import Dinov2Config, Dinov2Model

    model = Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=14,
            image_size=112,
        )
    )
    path = tmp_path_factory.mktemp("hf_models") / "tiny-dinov2"
    model.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_dinov3_path(tmp_path_factory) -> str:
    """A randomly-initialized tiny DINOv3 (with register tokens) saved locally."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    model = DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=16,
            image_size=64,
            num_register_tokens=2,
        )
    )
    path = tmp_path_factory.mktemp("hf_models") / "tiny-dinov3"
    model.save_pretrained(path)
    return str(path)


def test_default_resnet_backbone_unchanged():
    """The default torchvision ResNet path must keep its original structure."""
    config = make_act_config((3, 84, 84))
    policy = ACTPolicy(config)

    assert policy.model.backbone.feature_channels == 512
    assert policy.model.encoder_img_feat_input_proj.in_channels == 512

    loss, loss_dict = policy.forward(make_dummy_batch(config))
    loss.backward()
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict


@pytest.mark.parametrize("freeze", [False, True])
def test_dinov2_backbone(tiny_dinov2_path, freeze: bool):
    config = make_act_config((3, 56, 84), vision_backbone=tiny_dinov2_path, freeze_vision_backbone=freeze)
    policy = ACTPolicy(config)

    assert policy.model.backbone.feature_channels == 32
    assert policy.model.encoder_img_feat_input_proj.in_channels == 32

    loss, _ = policy.forward(make_dummy_batch(config))
    loss.backward()
    assert torch.isfinite(loss)

    backbone_grads = [p.grad for p in policy.model.backbone.parameters()]
    optimizer_groups = policy.get_optim_params()
    if freeze:
        assert all(not p.requires_grad for p in policy.model.backbone.parameters())
        assert all(g is None for g in backbone_grads)
        # Frozen backbone parameters must be excluded from the optimizer's backbone group.
        assert len(optimizer_groups[1]["params"]) == 0
    else:
        assert any(g is not None for g in backbone_grads)
        backbone_param_ids = {id(p) for p in policy.model.backbone.parameters()}
        assert len(optimizer_groups[1]["params"]) > 0
        assert all(id(p) in backbone_param_ids for p in optimizer_groups[1]["params"])


def test_dinov3_backbone_strips_register_tokens(tiny_dinov3_path):
    """DINOv3 outputs cls + register + patch tokens; only the patch tokens may reach the encoder."""
    config = make_act_config((3, 64, 80), vision_backbone=tiny_dinov3_path)
    policy = ACTPolicy(config)

    assert policy.model.backbone.feature_channels == 32

    feature_map = policy.model.backbone(torch.rand(2, 3, 64, 80))
    assert feature_map.shape == (2, 32, 4, 5)

    loss, _ = policy.forward(make_dummy_batch(config))
    loss.backward()
    assert torch.isfinite(loss)


def test_hf_backbone_pads_input_to_patch_multiple(tiny_dinov2_path, tiny_dinov3_path):
    """Images whose size is not a multiple of the patch size are padded instead of truncated."""
    config = make_act_config((3, 56, 84), vision_backbone=tiny_dinov2_path)
    policy = ACTPolicy(config)
    # 50x62 is not divisible by the patch size 14: padded to 56x70 -> 4x5 patch grid.
    feature_map = policy.model.backbone(torch.rand(1, 3, 50, 62))
    assert feature_map.shape == (1, 32, 4, 5)

    config = make_act_config((3, 64, 80), vision_backbone=tiny_dinov3_path)
    policy = ACTPolicy(config)
    # 50x62 is not divisible by the patch size 16: padded to 64x64 -> 4x4 patch grid.
    feature_map = policy.model.backbone(torch.rand(1, 3, 50, 62))
    assert feature_map.shape == (1, 32, 4, 4)


def test_hf_backbone_random_init(tiny_dinov2_path):
    """`pretrained_backbone_weights=None`/\"random\" builds the architecture without weights."""
    for weights in (None, "random"):
        config = make_act_config(
            (3, 56, 84), vision_backbone=tiny_dinov2_path, pretrained_backbone_weights=weights
        )
        policy = ACTPolicy(config)
        loss, _ = policy.forward(make_dummy_batch(config))
        assert torch.isfinite(loss)


def test_hf_backbone_ignores_stale_torchvision_default(tiny_dinov2_path):
    """A torchvision weights enum (e.g. the config default) must not be passed to transformers."""
    config = make_act_config(
        (3, 56, 84),
        vision_backbone=tiny_dinov2_path,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    )
    policy = ACTPolicy(config)
    loss, _ = policy.forward(make_dummy_batch(config))
    assert torch.isfinite(loss)


def test_unknown_torchvision_backbone_raises():
    with pytest.raises(ValueError, match="Unknown torchvision backbone"):
        build_vision_backbone("resnet999", pretrained_backbone_weights=None)


@pytest.mark.skipif(is_package_available("timm"), reason="timm is installed")
def test_timm_prefix_without_timm_installed():
    with pytest.raises(ImportError, match="requires timm"):
        build_vision_backbone("timm:vit_base_patch14_dinov2.lvd142m", pretrained_backbone_weights=None)


@pytest.mark.skipif(
    not os.environ.get("LEROBOT_TEST_HF_HUB"),
    reason="Set LEROBOT_TEST_HF_HUB=1 to run this hub-download test",
)
def test_dinov2_pretrained_from_hub():
    """End-to-end check with real pretrained DINOv2 weights from the hub."""
    config = make_act_config((3, 224, 224), vision_backbone="facebook/dinov2-small")
    policy = ACTPolicy(config)
    assert policy.model.backbone.feature_channels == 384
    loss, _ = policy.forward(make_dummy_batch(config))
    assert torch.isfinite(loss)
