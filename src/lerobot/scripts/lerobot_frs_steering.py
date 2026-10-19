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

"""Zero-shot Flow Reversal Steering on the SO-101 (arXiv:2606.13675).

Runs a trained flow-matching policy (pi0 / pi05 / smolvla) on the arm while
letting the operator steer it every planning step: keyboard directions are
turned into a coarse joint reference chunk (IK), the chunk is integrated
backwards through the flow (actions -> noise) and re-denoised, producing an
in-distribution action chunk that moves toward the reference.

Keys: W/A/S/D/Q/E steer, G gripper toggle, Space defer to the base policy,
Esc/x quit.

Usage:
    lerobot-frs-steering \
        --policy_path outputs/train/pi0_so101/pretrained_model \
        --task "pick up the object" \
        --mode steer \
        --kinematics.urdf_path experiments/frs/assets/so101.urdf \
        --camera_serial 740112072058
"""

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.experiments.frs.configs import FrsSteeringConfig
from lerobot.experiments.frs.keyboard_listener import SteeringKeyboardState
from lerobot.experiments.frs.noise_policy import load_noise_policy
from lerobot.experiments.frs.policy_bridge import (
    build_normalized_reference,
    get_dataset_stats,
    load_policy_with_processors,
)
from lerobot.experiments.frs.reference_builder import ReferenceBuilder
from lerobot.policies.common.frs import expand_noise
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.random_utils import set_seed

FRS_POLICY_TYPES = {"pi0", "pi05", "smolvla"}


def _make_camera_config(cfg: FrsSteeringConfig):
    if cfg.camera_type == "intelrealsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        if not cfg.camera_serial:
            raise ValueError("--camera-serial is required for camera_type=intelrealsense")
        return RealSenseCameraConfig(
            serial_number_or_name=cfg.camera_serial,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
        )
    if cfg.camera_type == "opencv":
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        return OpenCVCameraConfig(
            index_or_path=cfg.camera_index_or_path,
            width=cfg.camera_width,
            height=cfg.camera_height,
            fps=cfg.camera_fps,
        )
    raise ValueError(f"Unknown camera_type {cfg.camera_type!r} (intelrealsense | opencv)")


def _make_robot(cfg: FrsSteeringConfig):
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.utils import make_robot_from_config

    robot_config = SOFollowerRobotConfig(
        port=cfg.robot_port,
        id="frs_follower",
        cameras={"front": _make_camera_config(cfg)},
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    return robot


def _motor_names(robot) -> list[str]:
    return list(robot.bus.motors.keys())


def _observe(robot, motor_names, device, task: str):
    """RobotObservation -> pre-processed policy batch."""
    obs = robot.get_observation()
    frame = {
        "observation.state": np.array([obs[f"{m}.pos"] for m in motor_names], dtype=np.float32),
        "observation.images.front": obs["front"],
    }
    batch = prepare_observation_for_inference(frame, device, task=task)
    return batch, obs


def _plan_chunk(
    cfg: FrsSteeringConfig,
    policy,
    preprocessor,
    stats,
    batch,
    steering: SteeringKeyboardState,
    reference_builder: ReferenceBuilder | None,
    noise_policy,
    state_raw: np.ndarray,
):
    """One planning step: steer (FRS), defer (plain policy), or DSBC noise."""
    batch = preprocessor(batch)

    delta = steering.consume_delta() * cfg.kinematics.step_m
    defer = steering.consume_defer()
    has_direction = bool(np.any(delta != 0.0))

    mode = cfg.mode
    if mode == "policy" or defer:
        return policy.predict_action_chunk(batch), "policy"
    if mode == "dsbc":
        state_norm = batch[OBS_STATE]
        noise_vec = noise_policy.sample(state_norm, sigma=cfg.noise_sigma)
        noise_chunk = expand_noise(
            noise_vec,
            chunk_size=policy.config.chunk_size,
            max_action_dim=policy.config.max_action_dim,
            sigma=cfg.noise_sigma,
        ).to(policy.config.device)
        return policy.predict_action_chunk(batch, noise=noise_chunk), "dsbc"

    # mode == "steer"
    if not has_direction or reference_builder is None:
        return policy.predict_action_chunk(batch), "policy"
    gripper_pos = cfg.gripper_open_pos if steering.gripper_open else cfg.gripper_close_pos
    raw_reference = reference_builder.build(
        state_raw,
        delta,
        gripper_pos,
        action_repeats=cfg.action_repeats,
    )
    reference = build_normalized_reference(
        raw_reference, policy, stats, mode=cfg.chunk_noise_mode
    )
    chunk = policy.predict_action_chunk(
        batch,
        reference_actions=reference,
        num_reverse_steps=cfg.num_reverse_steps,
    )
    return chunk, "frs"


@parser.wrap()
def run(cfg: FrsSteeringConfig) -> None:
    if cfg.mode not in {"steer", "policy", "dsbc"}:
        raise ValueError(f"mode must be steer|policy|dsbc, got {cfg.mode!r}")
    if cfg.mode == "dsbc" and not cfg.noise_policy_path:
        raise ValueError("mode=dsbc requires --noise-policy-path")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    logging.info("Loading policy from %s", cfg.policy_path)
    policy, preprocessor, postprocessor, policy_config = load_policy_with_processors(
        cfg.policy_path, device=cfg.device
    )
    if policy_config.type not in FRS_POLICY_TYPES:
        raise ValueError(
            f"FRS steering supports flow-matching policies {sorted(FRS_POLICY_TYPES)}, "
            f"got {policy_config.type!r}"
        )
    stats = get_dataset_stats(preprocessor)

    reference_builder = None
    if cfg.mode == "steer":
        logging.info("Loading arm kinematics")
        reference_builder = ReferenceBuilder(cfg.kinematics)

    noise_policy = None
    if cfg.mode == "dsbc":
        noise_policy = load_noise_policy(Path(cfg.noise_policy_path), device=cfg.device)
        state_dim = policy_config.input_features["observation.state"].shape[0]
        if noise_policy.config.state_dim != state_dim:
            raise ValueError(
                f"Noise policy was trained on state_dim={noise_policy.config.state_dim}, "
                f"but the policy expects {state_dim}."
            )
        logging.info("DSBC noise policy loaded from %s", cfg.noise_policy_path)

    logging.info("Connecting robot on %s", cfg.robot_port)
    robot = _make_robot(cfg)
    motor_names = _motor_names(robot)
    logging.info("Motors: %s", motor_names)

    steering = SteeringKeyboardState()
    steering.start()
    print(
        "\n=== FRS steering ===\n"
        "W/S: +/-x  A/D: -/+y  Q/E: -/+z  (base frame of the URDF)\n"
        f"G: gripper toggle   Space: defer   Esc/x: quit   mode={cfg.mode}\n"
    )

    action_queue: deque[torch.Tensor] = deque()
    tick_duration = 1.0 / cfg.fps

    try:
        while not steering.quit:
            tick_start = time.perf_counter()

            batch, obs = _observe(robot, motor_names, device, cfg.task)
            state_raw = np.array([obs[f"{m}.pos"] for m in motor_names], dtype=np.float64)

            if len(action_queue) == 0:
                with torch.inference_mode():
                    chunk, source = _plan_chunk(
                        cfg,
                        policy,
                        preprocessor,
                        stats,
                        batch,
                        steering,
                        reference_builder,
                        noise_policy,
                        state_raw,
                    )
                for t in range(min(cfg.execute_steps, chunk.shape[1])):
                    action = postprocessor(chunk[:, t, :]).squeeze(0).cpu()
                    action_queue.append(action)
                logging.debug("planned via %s", source)

            action = action_queue.popleft()
            robot.send_action(
                {f"{m}.pos": float(v) for m, v in zip(motor_names, action.tolist(), strict=True)}
            )

            elapsed = time.perf_counter() - tick_start
            if elapsed < tick_duration:
                time.sleep(tick_duration - elapsed)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        steering.stop()
        robot.disconnect()
        logging.info("Robot disconnected. Bye!")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
