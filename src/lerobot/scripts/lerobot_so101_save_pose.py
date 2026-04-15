#!/usr/bin/env python

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config


@dataclass
class SavePoseConfig:
    """Save current SO-101 joint positions as a named pose.

    This is a small utility to build a library of task poses (home / pre_pick / pick / pre_place / place / etc.)
    that can be consumed by higher-level scripts (e.g. SayCan-driven trainswap).
    """

    robot: RobotConfig
    pose_name: str
    out_path: str = "so101_poses.json"
    overwrite: bool = False


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")


def _load_existing(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ValueError(f"Invalid pose file schema at {path}")
    out: dict[str, dict[str, float]] = {}
    for name, pose in data.items():
        if not isinstance(pose, dict) or not all(isinstance(m, str) and isinstance(v, (int, float)) for m, v in pose.items()):
            raise ValueError(f"Invalid pose '{name}' in {path}")
        out[name] = {m: float(v) for m, v in pose.items()}
    return out


def _get_joint_pose(obs: dict[str, object]) -> dict[str, float]:
    pose: dict[str, float] = {}
    for k, v in obs.items():
        if k.endswith(".pos") and isinstance(v, (int, float)):
            pose[k] = float(v)
    if not pose:
        raise RuntimeError("No joint positions found in robot observation.")
    return pose


@parser.wrap()
def save_pose(cfg: SavePoseConfig) -> None:
    _init_logging()
    logging.info("Save pose config:\n%s", pformat(asdict(cfg)))

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        obs = robot.get_observation()
        pose = _get_joint_pose(obs)
    finally:
        robot.disconnect()

    poses = _load_existing(out_path)
    if (cfg.pose_name in poses) and not cfg.overwrite:
        raise FileExistsError(
            f"Pose '{cfg.pose_name}' already exists in {out_path}. Use --overwrite=true to replace it."
        )

    poses[cfg.pose_name] = pose
    out_path.write_text(json.dumps(poses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Saved pose '%s' with %d joints to %s", cfg.pose_name, len(pose), out_path)


def main() -> None:
    save_pose()


if __name__ == "__main__":
    main()

