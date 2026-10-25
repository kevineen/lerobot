#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under this file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AmazingHand (pollen-robotics): 4指×2サーボ (SCS0009×8) の多指ハンド。

HopeJrHand と同じ構成（SCS0009 on protocol-1 bus）だが指の本数/命名が異なる。
プロトコル1（SCS系列）は sync_read 非対応のため、観測はモーターごとの逐次
読み出し（HopeJrHand と同じ手法）。Goal_Position の書き込みは sync_write。

サーボ空間 (finger{i}_motor{1,2}) と sim2rl モデル空間 (spread/flex) の対応は
``~/robot/sim2rl/assets/robots/amazinghand/calibration.json`` の線形写像:
spread = 0.26*(m1+m2), flex = 0.35*(m1-m2) [rad/deg単位に注意]。
"""

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_amazing_hand import AmazingHandConfig

logger = logging.getLogger(__name__)

#: 手指ごとのサーボID（公式 PythonExample/AmazingHand_Demo.py の順序に準拠:
#: Index=1,2 / Middle=3,4 / Ring=5,6 / Thumb=7,8。sim2rl の finger1..4 との対応は
#: 実機での点灯テストで確定させるまで仮置き）。
def _motors() -> dict[str, Motor]:
    out = {}
    for i in range(1, 5):
        out[f"finger{i}_motor1"] = Motor(2 * i - 1, "scs0009", MotorNormMode.RANGE_0_100)
        out[f"finger{i}_motor2"] = Motor(2 * i, "scs0009", MotorNormMode.RANGE_0_100)
    return out


class AmazingHand(Robot):
    config_class = AmazingHandConfig
    name = "amazing_hand"

    def __init__(self, config: AmazingHandConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=_motors(),
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        feats: dict[str, type | tuple] = dict(self._motors_ft)
        if self.config.read_present_load:
            feats.update({f"{motor}.load": float for motor in self.bus.motors})
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                feats[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                feats[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        fingers = {f"finger{i}": [f"finger{i}_motor1", f"finger{i}_motor2"] for i in range(1, 5)}
        self.calibration = RangeFinderGUI(self.bus, fingers).run()
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}
        start = time.perf_counter()
        for motor in self.bus.motors:
            obs_dict[f"{motor}.pos"] = self.bus.read(
                "Present_Position", motor, num_retry=self.config.num_read_retries
            )
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # モーター負荷（把持・接触力のプロキシ。力覚IL/VFA-lite student用）
        if self.config.read_present_load:
            start = time.perf_counter()
            for motor in self.bus.motors:
                obs_dict[f"{motor}.load"] = self.bus.read(
                    "Present_Load", motor, num_retry=self.config.num_read_retries
                )
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read load: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs_dict[cam_key] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)
        return action

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
