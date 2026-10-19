#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Reference chunk construction for zero-shot FRS steering.

Turns a keyboard Cartesian direction into a short joint-space reference chunk,
mirroring the paper's pipeline: direction -> EE target (current FK + delta) ->
IK joint targets -> repeat `action_repeats` times. The chunk is later tiled to
the policy chunk size by `lerobot.policies.common.frs.build_reference_chunk`.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.experiments.frs.configs import KinematicsConfig
from lerobot.model.kinematics import RobotKinematics


class ReferenceBuilder:
    """Builds raw (robot-space) joint reference chunks from steering deltas."""

    def __init__(
        self,
        config: KinematicsConfig,
        arm_joint_names: list[str] | None = None,
        num_arm_joints: int = 5,
    ):
        """Load placo kinematics; `arm_joint_names` pins the joint order."""
        self.config = config
        self.kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.ee_frame,
            joint_names=arm_joint_names,
        )
        urdf_names = list(self.kinematics.joint_names)
        logging.info(
            "FRS kinematics loaded (%s): joints=%s, ee_frame=%s",
            config.urdf_path,
            urdf_names,
            config.ee_frame,
        )
        if arm_joint_names is not None and arm_joint_names != urdf_names:
            raise ValueError(
                f"Requested joint names {arm_joint_names} not found in the URDF "
                f"(available: {urdf_names}). Align --kinematics.joint-names or the URDF."
            )
        self.arm_joint_names = urdf_names
        self.num_arm_joints = len(urdf_names)
        if self.num_arm_joints != num_arm_joints:
            logging.warning(
                "URDF has %d arm joints; expected %d for this robot.", self.num_arm_joints, num_arm_joints
            )

    def build(
        self,
        current_joints_deg: np.ndarray,
        delta_xyz: np.ndarray,
        gripper_pos: float,
        action_repeats: int,
    ) -> torch.Tensor:
        """Map a steering delta to a raw reference chunk.

        Args:
            current_joints_deg: Full motor positions (arm joints + gripper), degrees.
            delta_xyz: Cartesian steering delta in the EE frame, meters.
            gripper_pos: Gripper target for the reference chunk.
            action_repeats: Reference chunk length (paper: 5).

        Returns:
            `(1, action_repeats, action_dim)` float32 tensor in robot space
            (arm joint targets + gripper), ready for normalization.
        """
        arm_current = current_joints_deg[: self.num_arm_joints]

        ee_pose = self.kinematics.forward_kinematics(arm_current)
        ee_target = ee_pose.copy()
        ee_target[:3, 3] += np.asarray(delta_xyz, dtype=np.float64)

        arm_target = self.kinematics.inverse_kinematics(
            arm_current,
            ee_target,
            position_weight=self.config.position_weight,
            orientation_weight=self.config.orientation_weight,
        )

        # Safety: clamp the per-step joint motion of the reference.
        max_delta = self.config.max_joint_delta_deg
        arm_target = np.clip(arm_target, arm_current - max_delta, arm_current + max_delta)

        step = np.concatenate([arm_target, [gripper_pos]]).astype(np.float32)
        chunk = np.tile(step[None, :], (action_repeats, 1))
        return torch.from_numpy(chunk[None, :, :])
