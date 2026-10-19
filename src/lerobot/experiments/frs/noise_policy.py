#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DSBC noise policy (Diffusion/flow Steering via Behavioral Cloning).

Tang et al., arXiv:2606.13675, Section "DSBC": the noise recovered by flow
reversal from demonstrations is treated as an "expert noise action", and a
small auxiliary policy pi_phi(a0 | o) is trained on (o, a0) pairs with
behavioral cloning. At inference time a0 ~ pi_phi(.|o) is sampled and denoised
by the frozen VLA, steering it without per-step references.

The paper's Appendix C.4 shows the recovered noise can be averaged along the
chunk axis, so this policy predicts a single (action_dim,) noise vector per
observation; expand it to a chunk with
`lerobot.policies.common.frs.expand_noise`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

NOISE_POLICY_FILENAME = "noise_policy.pt"


@dataclass
class NoisePolicyConfig:
    """Configuration of the DSBC noise policy MLP."""

    state_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    # Std of the Gaussian exploration noise added on top of the predicted mean
    # at inference (paper Appendix C.3 found sigma in [1, 2] to help).
    sigma: float = 1.0


class NoisePolicyMLP(nn.Module):
    """Small MLP mapping a normalized state to a noise-action mean."""

    def __init__(self, config: NoisePolicyConfig):
        """Build the MLP from `config`'s layer sizes."""
        super().__init__()
        self.config = config
        dims = [config.state_dim, *config.hidden_dims, config.action_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers += [nn.Linear(in_dim, out_dim), nn.SiLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict the noise-action mean for a normalized state (B, state_dim)."""
        return self.net(state.to(torch.float32))

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a noise action: predicted mean + N(0, sigma^2) exploration."""
        mean = self.forward(state)
        sigma = self.config.sigma if sigma is None else sigma
        if sigma <= 0:
            return mean
        eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + sigma * eps


def save_noise_policy(model: NoisePolicyMLP, output_dir: str | Path) -> Path:
    """Save the noise policy weights + config into `output_dir/noise_policy.pt`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / NOISE_POLICY_FILENAME
    payload = {
        "config": asdict(model.config),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def load_noise_policy(path: str | Path, device: str = "cpu") -> NoisePolicyMLP:
    """Load a noise policy saved by `save_noise_policy`."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = NoisePolicyConfig(**payload["config"])
    model = NoisePolicyMLP(config)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
