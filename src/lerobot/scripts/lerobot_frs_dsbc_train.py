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

"""Offline DSBC training: BC a small noise policy on flow-reversal noise.

Implements the offline variant of "Diffusion Steering via Behavioral Cloning"
from arXiv:2606.13675: existing demonstrations are pushed through the frozen
flow-matching policy *backwards* (actions -> noise with `reverse_actions`),
and the recovered noise is imitated by `pi_phi_noise(a0 | state)`.

Usage:
    lerobot-frs-dsbc-train \
        --policy_path outputs/train/pi0_so101/pretrained_model \
        --dataset_repo_id <user>/so101_task \
        --output_dir outputs/frs_dsbc
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.experiments.frs.configs import FrsDsbcTrainConfig
from lerobot.experiments.frs.noise_policy import (
    NoisePolicyConfig,
    NoisePolicyMLP,
    save_noise_policy,
)
from lerobot.experiments.frs.policy_bridge import (
    action_dim_of,
    get_dataset_stats,
    load_policy_with_processors,
    pad_action_chunk,
    reverse_actions_for_policy,
)
from lerobot.utils.random_utils import set_seed


def _valid_frame_indices(dataset: LeRobotDataset, policy_config) -> list[int]:
    """Frames whose forward action window stays inside their episode.

    `dataset_to_index` is exclusive (see `LeRobotDataset._get_query_indices`);
    the trailing `chunk_size - 1` frames of each episode would have padded
    (repeated-end) action chunks, which we skip.
    """
    chunk_size = policy_config.chunk_size
    indices: list[int] = []
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        last_valid = end - chunk_size
        indices.extend(range(start, last_valid + 1))
    return indices


def _collate_preprocessed(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack per-item pre-processed (batch-1) dicts into one batch."""
    batch = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.cat(values, dim=0)
        else:
            batch[key] = values
    return batch


def _recover_noise_targets(cfg: FrsDsbcTrainConfig, policy, preprocessor, dataset, indices):
    """Phase 1: run flow reversal over the dataset -> (state, noise_mean) pairs."""
    stats_cpu = []
    model_device = policy.config.device
    selected = indices if cfg.max_samples is None else indices[: cfg.max_samples]

    for start in range(0, len(selected), cfg.batch_size):
        chunk_ids = selected[start : start + cfg.batch_size]
        samples = [preprocessor(dataset[i]) for i in chunk_ids]
        batch = _collate_preprocessed(samples)
        padded_actions = pad_action_chunk(batch["action"], policy)
        noise = reverse_actions_for_policy(
            policy, batch, padded_actions, num_steps=cfg.num_reverse_steps
        )
        noise = noise.mean(dim=1) if cfg.chunk_noise_mode == "mean" else noise[:, 0, :]
        noise = noise[:, : action_dim_of(policy)].cpu()
        states = batch["observation.state"].cpu()
        if states.ndim == 3:  # policies with observation history (e.g. pi05 MEM)
            states = states[:, -1, :]
        stats_cpu.append((states, noise))
        logging.info(
            "flow reversal: %d / %d frames", min(start + cfg.batch_size, len(selected)), len(selected)
        )

    all_states = torch.cat([s for s, _ in stats_cpu])
    all_noise = torch.cat([n for _, n in stats_cpu])
    return all_states.to(model_device), all_noise.to(model_device)


def _train_noise_policy(cfg: FrsDsbcTrainConfig, states: torch.Tensor, noise: torch.Tensor):
    """Phase 2: behavioral cloning of the noise policy (fast, <1 GB VRAM)."""
    noise_cfg = NoisePolicyConfig(
        state_dim=states.shape[-1],
        action_dim=noise.shape[-1],
        hidden_dims=tuple(cfg.hidden_dims),
        sigma=cfg.noise_sigma,
    )
    model = NoisePolicyMLP(noise_cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = torch.utils.data.TensorDataset(states, noise)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_states, batch_noise in loader:
            pred = model(batch_states)
            loss = torch.nn.functional.mse_loss(pred, batch_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, cfg.epochs // 10) == 0 or epoch == 0:
            logging.info("epoch %d/%d | mse %.5f", epoch + 1, cfg.epochs, epoch_loss / n_batches)

    model.eval()
    return model


@parser.wrap()
def run(cfg: FrsDsbcTrainConfig) -> None:
    set_seed(cfg.seed)
    if not cfg.dataset_repo_id:
        raise ValueError("--dataset-repo-id is required")

    logging.info("Loading policy from %s", cfg.policy_path)
    policy, preprocessor, _, policy_config = load_policy_with_processors(
        cfg.policy_path, device=cfg.device
    )
    # Fail fast if the checkpoint has no stats (needed by the steering script too).
    get_dataset_stats(preprocessor)

    logging.info("Loading dataset %s", cfg.dataset_repo_id)
    dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_config, dataset.meta)
    if delta_timestamps is None or "action" not in delta_timestamps:
        raise ValueError(
            "Could not resolve action delta timestamps from the policy config; "
            "the policy must define a future action window (chunk_size > 1)."
        )
    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        delta_timestamps=delta_timestamps,
    )

    all_indices = _valid_frame_indices(dataset, policy_config)
    if cfg.dataset_episodes is not None:
        allowed: set[int] = set()
        for i in cfg.dataset_episodes:
            ep = dataset.meta.episodes[i]
            allowed.update(range(ep["dataset_from_index"], ep["dataset_to_index"]))
        selected = [i for i in all_indices if i in allowed]
    else:
        selected = all_indices

    logging.info("Frames with complete action windows: %d", len(selected))
    if not selected:
        raise ValueError("No valid frames found (episodes shorter than chunk_size?)")

    states, noise = _recover_noise_targets(cfg, policy, preprocessor, dataset, selected)
    logging.info("Recovered noise targets: %s -> %s", tuple(states.shape), tuple(noise.shape))

    model = _train_noise_policy(cfg, states, noise)

    output_path = save_noise_policy(model, Path(cfg.output_dir) / cfg.dataset_repo_id.replace("/", "_"))
    logging.info("Saved noise policy to %s", output_path)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
