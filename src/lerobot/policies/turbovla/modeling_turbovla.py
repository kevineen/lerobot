#!/usr/bin/env python

# TurboVLA policy for the lerobot fork (route B in so101-lab/docs/20_turbovla.md).
# Wraps the vendored upstream network (./turbovla/) so that training and inference
# ride the standard lerobot pipelines: min-max normalized actions ([-1, 1], the
# same convention as the upstream RoboTwin recipe), ImageNet normalization done
# inside the model, instruction string taken from the batch's "task" key.

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_turbovla import TurboVLAConfig
from .turbovla.models import TurboVLA
from .turbovla.models.configuration import (
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig as NetConfig,
    VisionEncoderConfig,
)

# DINOv3 ViT-B image processor defaults (facebook/dinov3-vitb16).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_net_config(config: TurboVLAConfig, action_dim: int, state_dim: int, num_views: int) -> NetConfig:
    # Mirrors the route-A run: robotwin-style profile (learned patch positions,
    # zero-fill text padding, pre-norm, sdpa, bf16 autocast).
    return NetConfig(
        compatibility_profile="robotwin",
        text=TextEncoderConfig(
            model_name_or_path=config.bert_path,
            frozen=config.freeze_text_encoder,
            local_files_only=True,
            attention_implementation="sdpa",
        ),
        vision=VisionEncoderConfig(
            model_name_or_path=config.dinov3_path,
            image_size=config.image_size,
            num_views=num_views,
            position_embedding="learned_patch",
            encode_views_separately=False,
            frozen=config.freeze_vision_encoder,
            local_files_only=True,
            attention_implementation="sdpa",
            compute_precision="bf16_autocast",
            dropout=config.vision_dropout,
        ),
        interaction=InteractionConfig(
            hidden_dim=config.hidden_dim,
            nheads=config.nheads,
            num_layers=config.num_interaction_layers,
            dim_feedforward=config.dim_feedforward,
            enhancer_inner_dim=config.enhancer_inner_dim,
            padding_strategy="zero_fill",
            residual_style="pre_norm",
            attention_backend="sdpa",
            compute_precision="bf16_autocast",
        ),
        action=ActionHeadConfig(
            action_dim=action_dim,
            state_dim=state_dim,
            horizon=config.chunk_size,
            num_state_tokens=config.num_state_tokens,
            num_layers=config.act_num_layers,
            mlp_hidden_dim=config.act_mlp_hidden_dim,
            state_hidden_dim=config.act_state_hidden_dim,
            dropout=config.act_dropout,
        ),
    )


class TurboVLAPolicy(PreTrainedPolicy):
    """TurboVLA: Real-Time Vision-Language-Action Model (arXiv:2607.27205)."""

    config_class = TurboVLAConfig
    name = "turbovla"

    def __init__(self, config: TurboVLAConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        config.validate_features()

        action_dim = config.output_features[ACTION].shape[0]
        state_dim = config.input_features[OBS_STATE].shape[0]
        num_views = len(config.image_features)

        self.model = TurboVLA(
            _build_net_config(config, action_dim=action_dim, state_dim=state_dim, num_views=num_views)
        )

        # The network consumes the pre-final-norm hidden state, so DINOv3's final
        # LayerNorm never participates in the loss; freeze it to keep it out of
        # the optimizer (the upstream RoboTwin trainer does the equivalent).
        self.model.vision_encoder.backbone.norm.requires_grad_(False)

        if config.init_checkpoint:
            self._load_initialization(config.init_checkpoint)

        self.reset()

    def _load_initialization(self, path: str) -> None:
        source = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(source, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if isinstance(source.get(key), dict):
                    source = source[key]
                    break
        source = {k[7:] if k.startswith("module.") else k: v for k, v in source.items()}

        if any(k.startswith("model.") for k in source):
            # Trained TurboVLA checkpoint (route-A style, framework keys "model.*"):
            # load every tensor whose name and shape match the local network.
            source = {k[len("model."):]: v for k, v in source.items()}
            target = self.model.state_dict()
            loaded = {
                k: v for k, v in source.items() if k in target and tuple(target[k].shape) == tuple(v.shape)
            }
            target.update(loaded)
            self.model.load_state_dict(target, strict=True)
            print(f"[turbovla] loaded {len(loaded)}/{len(target)} tensors from trained checkpoint {path}")
            return

        # GroundingDINO release: transfer BERT, text projection and interaction
        # layers only (vision projection and action head stay freshly initialized).
        mappings = [
            ("bert.", "text_encoder.bert."),
            ("feat_map.", "text_encoder.text_projection."),
            (
                "transformer.encoder.text_layers.",
                "vision_language_interaction.text_layers.",
            ),
            (
                "transformer.encoder.fusion_layers.",
                "vision_language_interaction.fusion_layers.",
            ),
        ]
        target = self.model.state_dict()
        loaded = {}
        for source_key, value in source.items():
            for source_prefix, target_prefix in mappings:
                if not source_key.startswith(source_prefix):
                    continue
                target_key = target_prefix + source_key[len(source_prefix):]
                if target_key in target and tuple(target[target_key].shape) == tuple(value.shape):
                    loaded[target_key] = value
                break
        if not loaded:
            raise RuntimeError(f"no compatible initialization tensors found in {path}")
        target.update(loaded)
        self.model.load_state_dict(target, strict=True)
        print(f"[turbovla] loaded {len(loaded)} initialization tensors from {path}")

    def get_optim_params(self) -> dict:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _prepare_images(self, batch: dict[str, Tensor]) -> Tensor:
        """[B, V, 3, H, W] float in [0, 1] -> resized + ImageNet-normalized."""
        views = batch[OBS_IMAGES] if OBS_IMAGES in batch else [batch[k] for k in self.config.image_features]
        size = self.config.image_size
        mean = torch.tensor(IMAGENET_MEAN, device=views[0].device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=views[0].device).view(1, 3, 1, 1)
        processed = []
        for img in views:  # each [B, 3, H, W]
            if img.shape[-2:] != (size, size):
                img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            processed.append((img - mean) / std)
        return torch.stack(processed, dim=1)

    def _prepare_instructions(self, batch: dict[str, Tensor]) -> list[str]:
        task = batch.get("task", None)
        if task is None:
            task = getattr(self.config, "default_task", "") or ""
        if isinstance(task, str):
            return [task]
        return [str(t) for t in task]

    def _model_inputs(self, batch: dict[str, Tensor]) -> tuple[list[str], dict[str, Tensor], Tensor]:
        return self._prepare_instructions(batch), {"dinov3": self._prepare_images(batch)}, batch[OBS_STATE]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # `predict_action_chunk` returns a (batch_size, n_action_steps, action_dim)
            # tensor, but the queue effectively has shape (n_action_steps, batch, *).
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        instructions, samples, state = self._model_inputs(batch)
        return self.model(instructions, samples, state)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the L1 loss for training."""
        instructions, samples, state = self._model_inputs(batch)
        actions_hat = self.model(instructions, samples, state)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        if "action_is_pad" in batch:
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = valid_mask.sum() * abs_err.shape[-1]
            l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        else:
            l1_loss = abs_err.mean()
        return l1_loss, {"l1_loss": l1_loss.item()}
