#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Configs for the Flow Reversal Steering (FRS) experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrsDsbcTrainConfig:
    """Offline DSBC: train the noise policy by BC on flow-reversal noise.

    The frozen flow-matching policy maps each demonstration action chunk to its
    noise sample (actions -> noise), the noise is averaged along the chunk axis
    (paper Appendix C.4), and a small MLP is fit on (state, noise) pairs.
    """

    # Path to a trained flow-matching policy checkpoint (pi0 / pi05 / smolvla).
    policy_path: str = "outputs/train/frs_policy/pretrained_model"
    device: str = "cuda"

    dataset_repo_id: str = ""
    dataset_root: str | None = None
    # Restrict training to given episode indices (default: all episodes).
    dataset_episodes: list[int] | None = None

    # Number of reverse Euler steps (paper: 10, i.e. h=0.1).
    num_reverse_steps: int = 10
    # "mean" averages the recovered noise over the chunk axis before BC.
    chunk_noise_mode: str = "mean"

    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256)
    # Exploration sigma used at inference (paper Appendix C.3: sigma 1-2 helps).
    noise_sigma: float = 1.0
    max_samples: int | None = None
    seed: int = 42

    output_dir: str = "outputs/frs_dsbc"


@dataclass
class KinematicsConfig:
    """SO-101 arm kinematics used to turn keyboard directions into joint targets."""

    urdf_path: str = "experiments/frs/assets/so101.urdf"
    # End-effector frame name in the URDF.
    ee_frame: str = "gripper_frame_link"
    # Arm joint names in the robot's motor order (arm joints only, no gripper).
    # None = use the URDF's joint order (logged at startup for alignment).
    joint_names: list[str] | None = None
    # Cartesian step size per key press, in meters.
    step_m: float = 0.01
    # IK weights (position dominates; orientation kept loose like the paper,
    # which steers translation only).
    position_weight: float = 1.0
    orientation_weight: float = 0.01
    # Safety clamp on per-step joint motion of the reference target.
    max_joint_delta_deg: float = 10.0


@dataclass
class FrsSteeringConfig:
    """Zero-shot FRS keyboard steering on a real robot (paper: W/A/S/D/Q/E + IK).

    Modes:
    - "steer": keyboard reference chunks -> flow reversal -> re-denoise (FRS);
    - "policy": plain policy execution (defer behavior);
    - "dsbc": execute the DSBC noise policy instead of sampled noise.
    """

    policy_path: str = "outputs/train/frs_policy/pretrained_model"
    device: str = "cuda"
    task: str = "Pick up the object"

    mode: str = "steer"  # steer | policy | dsbc
    # Path to noise_policy.pt (required for mode=dsbc).
    noise_policy_path: str | None = None

    num_reverse_steps: int = 10
    # How many actions of each chunk to execute before re-planning
    # (the paper's `action_repeats`=5).
    execute_steps: int = 5
    chunk_noise_mode: str = "tile"
    noise_sigma: float = 1.0

    # Reference chunk length fed to flow reversal before tiling to the policy
    # chunk size (paper: IK target repeated `action_repeats` times).
    action_repeats: int = 5
    gripper_open_pos: float = 60.0
    gripper_close_pos: float = 0.0

    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)

    robot_port: str = "/dev/ttyACM1"
    camera_type: str = "intelrealsense"  # intelrealsense | opencv
    camera_serial: str = ""
    camera_index_or_path: int | str = 0  # used when camera_type=opencv
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    fps: int = 30
    seed: int = 42
