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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("amazing_hand")
@dataclass
class AmazingHandConfig(RobotConfig):
    """AmazingHand (pollen-robotics, 8x Feetech SCS0009, protocol 1).

    Motor naming follows the sim2rl servo space (``finger{i}_motor{1,2}``,
    i=1..4; see ~/robot/sim2rl assets/robots/amazinghand/calibration.json).
    The official demos name them index/middle/ring/thumb — the finger index
    mapping must be verified on hardware once (blink test) before use.
    """

    port: str  # Waveshare USB serial adapter (dedicated bus; never mix with the STS3215 arm bus)
    side: str  # "left" / "right"

    disable_torque_on_disconnect: bool = True

    # Read `Present_Load` (SCS0009 load feedback, read-only register) from every
    # motor in each observation and expose it as `{motor}.load` scalars — the
    # contact/grasp force proxy for force-aware IL (same pattern as the fork's
    # so101_follower read_present_load; sim2rl tweezers_pick student policies
    # consume this channel). Sequential per-motor reads (protocol 1 has no
    # sync_read), so it costs 8 extra single reads per frame.
    read_present_load: bool = False

    num_read_retries: int = 2

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)
