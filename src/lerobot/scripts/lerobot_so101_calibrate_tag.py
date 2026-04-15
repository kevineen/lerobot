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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture ArUco reference pose + optional joint snapshot for trainswap calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.experiments.trainswap.geometry import T_from_rvec_tvec
from lerobot.experiments.trainswap.realsense_aruco import RealSenseArucoCapture


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")


def _load_poses(path: Path, pose_name: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if pose_name not in data:
        raise KeyError(f"Pose {pose_name!r} not found in {path}")
    pose = data[pose_name]
    if not isinstance(pose, dict):
        raise TypeError(f"Invalid pose entry for {pose_name!r}")
    return {str(k): float(v) for k, v in pose.items()}


def _optional_T(path: str | None) -> list[list[float]] | None:
    if not path:
        return None
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"T_robot_from_cam must be 4x4 list-of-lists JSON at {p}")
    m = [[float(x) for x in row] for row in raw]
    if any(len(row) != 4 for row in m):
        raise ValueError("T_robot_from_cam must be 4x4")
    return m


@dataclass
class CalibrateTagConfig:
    """Grab one stable ArUco detection and write a calibration JSON."""

    serial_number_or_name: str
    width: int = 640
    height: int = 480
    fps: int = 30
    tag_id: int = 0
    marker_length_m: float = 0.04
    aruco_dictionary: str = "DICT_4X4_50"
    out_path: str = "so101_tag_calibration.json"
    warmup_frames: int = 30
    # Optional: embed joint snapshot from `lerobot-so101-save-pose` output file.
    poses_path: str | None = None
    reference_pose_name: str | None = None
    # Optional: 4x4 ``T_robot_from_cam`` (robot <- camera) as JSON file for logging ``T_robot_from_tag``.
    T_robot_from_cam_path: str | None = None


@parser.wrap()
def main(cfg: CalibrateTagConfig) -> None:
    _init_logging()
    logging.info("Calibrate tag config:\n%s", pformat(cfg))

    cap = RealSenseArucoCapture(
        serial_number=str(cfg.serial_number_or_name),
        width=int(cfg.width),
        height=int(cfg.height),
        fps=int(cfg.fps),
        marker_length_m=float(cfg.marker_length_m),
        aruco_dictionary=str(cfg.aruco_dictionary),
    )
    cap.connect()
    try:
        det = None
        for _ in range(max(1, int(cfg.warmup_frames))):
            d = cap.detect_tag(int(cfg.tag_id))
            if d is not None:
                det = d
        if det is None:
            raise RuntimeError(f"Could not detect ArUco id={cfg.tag_id} after warmup.")

        T_ct = T_from_rvec_tvec(det.rvec, det.tvec)
        joints: dict[str, float] = {}
        if cfg.poses_path and cfg.reference_pose_name:
            joints = _load_poses(Path(cfg.poses_path), str(cfg.reference_pose_name))

        T_robot_from_cam = _optional_T(cfg.T_robot_from_cam_path)
        T_robot_from_tag = None
        if T_robot_from_cam is not None:
            T_rc = np.asarray(T_robot_from_cam, dtype=np.float64).reshape(4, 4)
            T_robot_from_tag = (T_rc @ T_ct).tolist()

        out = {
            "schema_version": 1,
            "tag_id": int(cfg.tag_id),
            "aruco_dictionary": str(cfg.aruco_dictionary),
            "marker_length_m": float(cfg.marker_length_m),
            "reference_tag_tvec_cam_m": det.tvec.reshape(3).tolist(),
            "reference_tag_rvec_cam_rad": det.rvec.reshape(3).tolist(),
            "T_cam_from_tag": T_ct.tolist(),
            "reference_joints": joints,
            # User-tuned: degrees (or normalized gripper units) per meter of tag translation in camera frame.
            "joint_gains_per_tag_tvec_m": {
                "shoulder_pan.pos": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
                "shoulder_lift.pos": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
            },
            "T_robot_from_cam": T_robot_from_cam,
            "T_robot_from_tag_at_reference": T_robot_from_tag,
            "notes": (
                "Edit joint_gains_per_tag_tvec_m after small experiments: "
                "when the tag moves +0.01m in +tx in camera frame, increase shoulder_pan.pos by gain*0.01."
            ),
        }

        out_p = Path(cfg.out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        logging.info("Wrote calibration to %s", out_p)
    finally:
        cap.disconnect()


if __name__ == "__main__":
    main()
