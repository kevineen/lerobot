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
"""Vision backbone factory.

Provides a common interface for extracting a spatial feature map (B, C, H, W) from a batch of
images (B, 3, H, W), regardless of the underlying architecture (torchvision ResNets, Hugging Face
models such as DINOv2 / DINOv3, or timm models).

Use `build_vision_backbone` to instantiate one. The returned module exposes a `feature_channels`
attribute (number of channels of the output feature map) so callers can build their projections
without knowing the architecture.
"""

import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

_RANDOM_WEIGHTS = "random"
# Pattern of torchvision weights enums (e.g. "ResNet18_Weights.IMAGENET1K_V1"), used to detect a
# stale torchvision-style default when a non-torchvision backbone is requested.
_TORCHVISION_WEIGHTS_PATTERN = "_Weights."


def build_vision_backbone(
    vision_backbone: str,
    pretrained_backbone_weights: str | None = None,
    replace_final_stride_with_dilation: bool = False,
) -> nn.Module:
    """Build a vision backbone returning a (B, C, H, W) feature map.

    Args:
        vision_backbone: Identifier of the backbone. Supported forms:
            - "resnet18", "resnet50", ...: torchvision ResNet variants.
            - "timm:<name>": any timm model with feature extraction (e.g.
              "timm:vit_base_patch14_dinov2.lvd142m"). Requires `timm` to be installed.
            - anything else: a Hugging Face model id or local path (e.g. "facebook/dinov2-base",
              "facebook/dinov3-vitb16-pretrain-lvd1689m" — the latter is a gated repo requiring a
              HF token and license acceptance). Requires `transformers` to be installed.
        pretrained_backbone_weights: Weights source.
            - For torchvision backbones: a torchvision weights enum name, or None for random init.
            - For timm backbones: a timm pretrained tag, "random"/None for random init.
            - For Hugging Face backbones: a model id or local path to load weights from; "random"
              or None builds the architecture with random weights from the config of
              `vision_backbone`. A torchvision weights enum name (e.g. left-over default
              "ResNet18_Weights.IMAGENET1K_V1") is ignored and weights are loaded from
              `vision_backbone` itself.
        replace_final_stride_with_dilation: torchvision ResNet only. Whether to replace the final
            stride with dilation to keep a larger feature map.
    """
    if vision_backbone.startswith("timm:"):
        return TimmBackbone(
            vision_backbone.removeprefix("timm:"),
            pretrained_backbone_weights=pretrained_backbone_weights,
        )
    if vision_backbone.startswith("resnet"):
        return TorchvisionResNetBackbone(
            vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
        )
    return HFVisionBackbone(
        vision_backbone,
        pretrained_backbone_weights=pretrained_backbone_weights,
    )


class TorchvisionResNetBackbone(nn.Module):
    """torchvision ResNet returning the layer4 feature map, with frozen batch norm as in ACT."""

    def __init__(
        self,
        vision_backbone: str,
        pretrained_backbone_weights: str | None,
        replace_final_stride_with_dilation: bool,
    ):
        super().__init__()
        try:
            backbone_model = getattr(torchvision.models, vision_backbone)(
                replace_stride_with_dilation=[False, False, replace_final_stride_with_dilation],
                weights=pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
        except AttributeError as e:
            available = [m for m in dir(torchvision.models) if not m.startswith("_") and m.islower()]
            raise ValueError(
                f"Unknown torchvision backbone '{vision_backbone}'. ResNet variants are supported "
                f"(e.g. resnet18, resnet34, resnet50). For other architectures, use a Hugging Face "
                f"model id or a 'timm:<name>' backbone. Available torchvision models: {available}"
            ) from e
        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.feature_extractor = IntermediateLayerGetter(
            backbone_model, return_layers={"layer4": "feature_map"}
        )
        self.feature_channels = backbone_model.fc.in_features

    def forward(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)["feature_map"]


class HFVisionBackbone(nn.Module):
    """Hugging Face vision backbone (e.g. DINOv2, DINOv3) returning a (B, C, H, W) feature map.

    Token-based (ViT) models: input images are padded to a multiple of the patch size, then the
    patch tokens (cls and register tokens stripped) are reshaped into a spatial feature map.
    Convolutional models (e.g. DINOv3 ConvNeXt): the 4D output of the model is returned as is.
    """

    def __init__(self, vision_backbone: str, pretrained_backbone_weights: str | None):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError(
                f"Backbone '{vision_backbone}' requires Hugging Face transformers. "
                "Install with: pip install -e '.[transformers]'"
            ) from e

        weights = pretrained_backbone_weights
        # A torchvision weights enum name can only be a stale default here; fall back to loading
        # from `vision_backbone` itself instead of passing it to transformers.
        if weights is not None and _TORCHVISION_WEIGHTS_PATTERN in weights:
            load_pretrained, source = True, vision_backbone
        elif weights is None or weights == _RANDOM_WEIGHTS:
            load_pretrained, source = False, vision_backbone
        else:
            load_pretrained, source = True, weights

        config = AutoConfig.from_pretrained(source)
        if load_pretrained:
            self.feature_extractor = AutoModel.from_pretrained(source, config=config)
        else:
            self.feature_extractor = AutoModel.from_config(config)

        hf_config = self.feature_extractor.config
        self.patch_size = getattr(hf_config, "patch_size", None)
        if self.patch_size is not None:
            # Token-based (ViT) backbone.
            self.feature_channels = hf_config.hidden_size
        else:
            hidden_sizes = getattr(hf_config, "hidden_sizes", None)
            if not hidden_sizes:
                raise ValueError(
                    f"Cannot infer the number of output channels of backbone '{vision_backbone}' "
                    "(no `patch_size` nor `hidden_sizes` in its config). It may not be usable as a "
                    "vision backbone."
                )
            self.feature_channels = hidden_sizes[-1]

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        if self.patch_size is not None:
            x = _pad_to_patch_multiple(x, self.patch_size)
        out = self.feature_extractor(pixel_values=x)
        feature_map = out.last_hidden_state
        if feature_map.dim() == 3:
            # (B, num_tokens, D) -> keep patch tokens only -> (B, D, H', W').
            padded_h = x.shape[-2] // self.patch_size
            padded_w = x.shape[-1] // self.patch_size
            num_patch_tokens = padded_h * padded_w
            num_prefix_tokens = feature_map.shape[1] - num_patch_tokens
            if num_prefix_tokens < 0:
                raise RuntimeError(
                    f"Backbone returned {feature_map.shape[1]} tokens but the input image yields "
                    f"{num_patch_tokens} patches. Check `vision_backbone` and the image size."
                )
            # Strip leading special tokens (cls, and registers for models like DINOv3).
            feature_map = feature_map[:, num_prefix_tokens:]
            feature_map = feature_map.transpose(1, 2).reshape(batch_size, -1, padded_h, padded_w)
        return feature_map


class TimmBackbone(nn.Module):
    """timm backbone with feature extraction, returning its last (B, C, H, W) feature map."""

    def __init__(self, model_name: str, pretrained_backbone_weights: str | None):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                f"Backbone 'timm:{model_name}' requires timm. Install with: pip install -e '.[timm]'"
            ) from e

        weights = pretrained_backbone_weights
        # Note: timm resolves its own default pretrained tag when `pretrained=True`; passing a
        # torchvision weights enum name here (stale default) would be ignored.
        pretrained = weights not in (None, _RANDOM_WEIGHTS) and _TORCHVISION_WEIGHTS_PATTERN not in weights
        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_channels = self.feature_extractor.feature_info.channels()[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)[-1]


def _pad_to_patch_multiple(x: Tensor, patch_size: int) -> Tensor:
    """Pad (B, C, H, W) with replicate padding up to a multiple of `patch_size` (in place dims)."""
    pad_h = -x.shape[-2] % patch_size
    pad_w = -x.shape[-1] % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x
