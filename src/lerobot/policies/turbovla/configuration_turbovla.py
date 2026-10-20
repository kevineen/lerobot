#!/usr/bin/env python

# TurboVLA policy configuration for the lerobot fork.
# Paper: https://arxiv.org/abs/2607.27205 — LLM-free VLA (DINOv3 + BERT + bi-attention + ACT head).
# Local runbook: so101-lab/docs/20_turbovla.md (route B).

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

BERT_PATH = "/home/kevin/models/bert-base-uncased"
DINOV3_PATH = "/home/kevin/models/dinov3-vitb16"
GROUNDINGDINO_INIT = "/home/kevin/models/groundingdino/groundingdino_swint_ogc.pth"


@PreTrainedConfig.register_subclass("turbovla")
@dataclass
class TurboVLAConfig(PreTrainedConfig):
    """Configuration for the TurboVLA policy.

    Notes on the inputs and outputs:
        - At least one "observation.images.*" input is required (multiple keys are
          treated as multiple camera views, encoded in `image_features` order).
        - "observation.state" is required.
        - "action" is required as an output key. Actions are absolute joint
          positions and are min-max normalized to [-1, 1] by the processor
          (identical convention to the upstream TurboVLA RoboTwin pipeline).
        - The batch must carry the instruction string under "task".
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 12
    n_action_steps: int = 12

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # imagenet norm happens inside the model
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Vision (DINOv3) — matches the route-A run (robotwin-style profile, 224²).
    image_size: int = 224
    freeze_vision_encoder: bool = False
    vision_dropout: float = 0.1

    # Text (BERT base uncased, frozen).
    freeze_text_encoder: bool = True

    # Interaction (bi-directional vision-language attention).
    hidden_dim: int = 256
    nheads: int = 8
    num_interaction_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024

    # Action head (ACT-style decoder).
    act_num_layers: int = 3
    num_state_tokens: int = 2
    act_state_hidden_dim: int = 256
    act_mlp_hidden_dim: int = 512
    act_dropout: float = 0.1

    # Initialization. Either the GroundingDINO release (bert/text-proj/interaction
    # transfer only) or a trained TurboVLA checkpoint from route A (keys starting
    # with "model.", e.g. the EMA file) whose matching tensors are loaded.
    init_checkpoint: str | None = GROUNDINGDINO_INIT
    bert_path: str = BERT_PATH
    dinov3_path: str = DINOV3_PATH

    # Training preset (route-A recipe).
    optimizer_lr: float = 5.0e-5
    optimizer_weight_decay: float = 1.0e-10

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be <= `chunk_size`. Got {self.n_action_steps} > {self.chunk_size}."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("TurboVLA requires at least one camera among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
