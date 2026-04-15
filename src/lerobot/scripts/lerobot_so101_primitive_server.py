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

"""HTTP JSON server on Raspberry Pi: execute SO-101 motion primitives from a pose library."""

from __future__ import annotations

import json
import logging
import numbers
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.configs import parser
from lerobot.processor import RobotAction
from lerobot.robots import RobotConfig, make_robot_from_config


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")


def _load_poses(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid poses file (expected dict): {path}")
    out: dict[str, dict[str, float]] = {}
    for name, pose in data.items():
        if not isinstance(pose, dict):
            continue
        out[str(name)] = {str(k): float(v) for k, v in pose.items()}
    return out


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _interpolate_joints(
    start: dict[str, float], end: dict[str, float], *, seconds: float, hz: float, send_action
) -> dict[str, float]:
    """Linearly interpolate all keys present in either dict."""
    keys = sorted(set(start) | set(end))
    steps = max(1, int(round(seconds * hz)))
    last_sent: dict[str, float] = dict(start)
    for s in range(1, steps + 1):
        t = s / steps
        action: dict[str, float] = {}
        for k in keys:
            a = float(start.get(k, end.get(k, 0.0)))
            b = float(end.get(k, start.get(k, 0.0)))
            action[k] = _lerp(a, b, t)
        send_action(action)  # type: ignore[arg-type]
        last_sent = action
        time.sleep(1.0 / hz)
    # Ensure we end exactly on `end` for keys that exist in end
    final = {k: float(end[k]) for k in end}
    merged = {**last_sent, **final}
    send_action(merged)  # type: ignore[arg-type]
    return merged


@dataclass
class PrimitiveServerConfig:
    robot: RobotConfig
    poses_path: str = "so101_poses.json"
    host: str = "0.0.0.0"
    port: int = 8765
    default_hz: float = 30.0


class _Ctx:
    robot = None
    poses: dict[str, dict[str, float]] = {}
    poses_path: Path | None = None
    hz: float = 30.0


_CTX = _Ctx()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _handle_primitive(cmd: dict[str, Any]) -> dict[str, Any]:
    if _CTX.robot is None:
        raise RuntimeError("Server not initialized.")

    name = str(cmd.get("name", ""))
    if name == "ping":
        return {"ok": True, "name": "ping"}

    if name == "reload_poses":
        if _CTX.poses_path is None:
            raise RuntimeError("poses_path not set")
        _CTX.poses = _load_poses(_CTX.poses_path)
        return {"ok": True, "name": "reload_poses", "num_poses": len(_CTX.poses)}

    if name == "get_observation":
        obs = _CTX.robot.get_observation()
        # Convert to JSON-serializable (images are heavy; omit large arrays)
        slim: dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, str):
                slim[k] = v
            elif isinstance(v, bool):
                slim[k] = v
            elif isinstance(v, numbers.Real):
                slim[k] = float(v)
            elif hasattr(v, "tolist"):
                continue
        return {"ok": True, "name": "get_observation", "observation": slim}

    if name == "move_to_named_pose":
        pose_name = str(cmd["pose_name"])
        if pose_name not in _CTX.poses:
            raise KeyError(f"Unknown pose_name={pose_name!r}. Known={sorted(_CTX.poses.keys())}")
        action: RobotAction = {k: float(v) for k, v in _CTX.poses[pose_name].items()}
        sent = _CTX.robot.send_action(action)
        return {"ok": True, "name": name, "sent": sent}

    if name == "set_joints":
        joints = cmd.get("joints", {})
        if not isinstance(joints, dict):
            raise TypeError("joints must be a dict")
        action = {str(k): float(v) for k, v in joints.items()}
        sent = _CTX.robot.send_action(action)
        return {"ok": True, "name": name, "sent": sent}

    if name == "interpolate_poses":
        a = str(cmd["from_pose"])
        b = str(cmd["to_pose"])
        seconds = float(cmd.get("seconds", 2.0))
        hz = float(cmd.get("hz", _CTX.hz))
        if a not in _CTX.poses or b not in _CTX.poses:
            raise KeyError(f"Unknown poses: from={a!r} to={b!r}")
        last = _interpolate_joints(
            _CTX.poses[a],
            _CTX.poses[b],
            seconds=seconds,
            hz=hz,
            send_action=_CTX.robot.send_action,
        )
        return {"ok": True, "name": name, "sent": last}

    if name == "interpolate_joints":
        start = cmd.get("from_joints", {})
        end = cmd.get("to_joints", {})
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise TypeError("from_joints and to_joints must be dicts")
        seconds = float(cmd.get("seconds", 2.0))
        hz = float(cmd.get("hz", _CTX.hz))
        s = {str(k): float(v) for k, v in start.items()}
        e = {str(k): float(v) for k, v in end.items()}
        last = _interpolate_joints(
            s,
            e,
            seconds=seconds,
            hz=hz,
            send_action=_CTX.robot.send_action,
        )
        return {"ok": True, "name": name, "sent": last}

    if name == "open_gripper":
        value = float(cmd.get("value", 100.0))
        obs = _CTX.robot.get_observation()
        action = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        action["gripper.pos"] = value
        sent = _CTX.robot.send_action(action)
        return {"ok": True, "name": name, "sent": sent}

    if name == "close_gripper":
        value = float(cmd.get("value", 35.0))
        obs = _CTX.robot.get_observation()
        action = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        action["gripper.pos"] = value
        sent = _CTX.robot.send_action(action)
        return {"ok": True, "name": name, "sent": sent}

    if name == "sleep":
        time.sleep(float(cmd.get("seconds", 0.2)))
        return {"ok": True, "name": name}

    raise ValueError(f"Unknown primitive name={name!r}")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/primitive":
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            cmd = json.loads(raw.decode("utf-8"))
            if not isinstance(cmd, dict):
                raise TypeError("JSON body must be an object")
            out = _handle_primitive(cmd)
            _json_response(self, 200, out)
        except Exception as e:
            logging.exception("primitive failed")
            _json_response(self, 400, {"ok": False, "error": str(e)})


@parser.wrap()
def main(cfg: PrimitiveServerConfig) -> None:
    _init_logging()
    logging.info("Primitive server config:\n%s", pformat(cfg))

    poses_path = Path(cfg.poses_path)
    _CTX.poses_path = poses_path
    _CTX.poses = _load_poses(poses_path)
    _CTX.hz = float(cfg.default_hz)

    robot = make_robot_from_config(cfg.robot)
    robot.connect(calibrate=False)
    _CTX.robot = robot

    httpd = ThreadingHTTPServer((cfg.host, int(cfg.port)), _Handler)
    logging.info("Listening on http://%s:%s/primitive", cfg.host, int(cfg.port))

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        httpd.shutdown()
        httpd.server_close()
        robot.disconnect()
        _CTX.robot = None


if __name__ == "__main__":
    main()
