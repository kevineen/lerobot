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

"""SayCan + CLIP + ArUco trainswap loop (PC): plan with LM Studio, score with CLIP, act via Pi HTTP primitives."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from lerobot.configs import parser
from lerobot.experiments.saycan_clip.clip_scorer import ClipScorer
from lerobot.experiments.saycan_clip.configs import CLIPConfig, LLMBackendConfig, SayCanConfig
from lerobot.experiments.saycan_clip.llm_client import ChatMessage, OpenAICompatibleClient
from lerobot.experiments.trainswap.joint_correction import (
    apply_tvec_joint_gains,
    load_gains_from_calibration,
    ref_tvec_from_calibration,
)
from lerobot.experiments.trainswap.pi_json_client import post_primitive
from lerobot.experiments.trainswap.realsense_aruco import RealSenseArucoCapture
from lerobot.experiments.trainswap.track_vision import estimate_track_offset_from_bgr


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")


def _parse_candidates_from_llm(text: str) -> list[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    raise ValueError(f"Could not parse JSON array from LLM output: {text[:200]!r}")


def _get_candidates(cfg: "SayCanTrainswapConfig") -> list[str]:
    if not cfg.saycan.llm_generate_candidates:
        return list(cfg.saycan.candidates)
    client = OpenAICompatibleClient(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        timeout_s=cfg.llm.timeout_s,
    )
    prompt = cfg.saycan.prompt_template.format(instruction=cfg.saycan.instruction, n=cfg.saycan.llm_num_candidates)
    out = client.chat_completions(
        model=cfg.llm.model,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    return _parse_candidates_from_llm(out)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_poses(path: Path) -> dict[str, dict[str, float]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"poses must be a dict: {path}")
    out: dict[str, dict[str, float]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = {str(kk): float(vv) for kk, vv in v.items()}
    return out


def _map_text_to_skill(text: str) -> str:
    t = text.lower()
    if "done" in t or "finished" in t or "complete" in t:
        return "done"
    if "power" in t and "button" in t:
        return "power"
    if "place" in t or ("put" in t and "track" in t):
        return "place"
    if "pick" in t or "grasp" in t or "grab" in t:
        return "pick"
    if "open" in t and "gripper" in t:
        return "open"
    if "close" in t and "gripper" in t:
        return "close"
    if "home" in t or "retreat" in t or "reset" in t:
        return "home"
    return "noop"


@dataclass
class TrainswapPoseNames:
    """Named poses expected in ``poses_path`` (same file on PC and Pi)."""

    home: str = "home"
    pre_pick: str = "pre_pick"
    pick: str = "pick"
    lift: str = "lift"
    pre_place: str = "pre_place"
    place: str = "place"
    post_place: str = "post_place"
    power_pre: str = "power_pre"
    power_press: str = "power_press"


def _default_saycan() -> SayCanConfig:
    return SayCanConfig(
        instruction="Pick up the stopped toy train off the table and place it on the track.",
        candidates=[
            "pick up the toy train from the table",
            "move the gripper to the train and close the gripper",
            "lift the train up safely",
            "move above the track area",
            "place the train onto the track",
            "press the power on button",
            "open the gripper",
            "close the gripper",
            "go to home pose",
            "done",
        ],
        top_k=5,
        llm_generate_candidates=True,
        llm_num_candidates=12,
    )


@dataclass
class SayCanTrainswapConfig:
    """Configuration for the PC-side SayCan trainswap driver."""

    pi_base_url: str = "http://127.0.0.1:8765"
    poses_path: str = "so101_poses.json"
    calibration_path: str = "so101_tag_calibration.json"

    realsense_serial: str = ""
    rs_width: int = 640
    rs_height: int = 480
    rs_fps: int = 30
    tag_id: int = 0
    marker_length_m: float = 0.04
    aruco_dictionary: str = "DICT_4X4_50"

    saycan: SayCanConfig = field(default_factory=_default_saycan)
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)

    pose_names: TrainswapPoseNames = field(default_factory=TrainswapPoseNames)
    motion_seconds: float = 2.0
    motion_hz: float = 30.0
    gripper_open: float = 100.0
    gripper_close: float = 35.0

    # Phase 4: optional USB camera index (OpenCV) for rail line heuristic.
    usb_track_camera_index: int | None = None
    use_track_vision: bool = False
    track_shoulder_pan_deg_per_px: float = 0.0

    num_steps: int = 8


class TrainswapExecutor:
    """Builds corrected joint targets and sends them to the Pi primitive server."""

    def __init__(self, cfg: SayCanTrainswapConfig) -> None:
        self.cfg = cfg
        self.poses = _load_poses(Path(cfg.poses_path))
        self.cal = _load_json(Path(cfg.calibration_path))
        self.ref_tvec = ref_tvec_from_calibration(self.cal)
        self.gains = load_gains_from_calibration(self.cal)
        self._usb_cap = None
        if cfg.use_track_vision and cfg.usb_track_camera_index is not None:
            import cv2  # type: ignore

            self._usb_cap = cv2.VideoCapture(int(cfg.usb_track_camera_index))

    def close(self) -> None:
        if self._usb_cap is not None:
            self._usb_cap.release()
            self._usb_cap = None

    def _correct(self, joints: dict[str, float], *, tvec: np.ndarray | None) -> dict[str, float]:
        if tvec is None or not self.gains:
            return dict(joints)
        return apply_tvec_joint_gains(joints, tvec_m=tvec, ref_tvec_m=self.ref_tvec, gains=self.gains)

    def _track_extra_pan(self) -> float:
        if self._usb_cap is None:
            return 0.0
        ok, bgr = self._usb_cap.read()
        if not ok or bgr is None:
            return 0.0
        est = estimate_track_offset_from_bgr(bgr)
        if est is None:
            return 0.0
        return float(self.cfg.track_shoulder_pan_deg_per_px) * float(est.offset_x_px)

    def _interpolate(self, a: dict[str, float], b: dict[str, float], *, seconds: float | None = None) -> None:
        sec = float(seconds if seconds is not None else self.cfg.motion_seconds)
        post_primitive(
            self.cfg.pi_base_url,
            {
                "name": "interpolate_joints",
                "from_joints": a,
                "to_joints": b,
                "seconds": sec,
                "hz": float(self.cfg.motion_hz),
            },
        )

    def run_home(self, *, tvec: np.ndarray | None) -> None:
        h = self._correct(self.poses[self.cfg.pose_names.home], tvec=tvec)
        post_primitive(self.cfg.pi_base_url, {"name": "set_joints", "joints": h})

    def run_open(self) -> None:
        post_primitive(self.cfg.pi_base_url, {"name": "open_gripper", "value": float(self.cfg.gripper_open)})

    def run_close(self) -> None:
        post_primitive(self.cfg.pi_base_url, {"name": "close_gripper", "value": float(self.cfg.gripper_close)})

    def run_pick(self, *, tvec: np.ndarray | None) -> None:
        pn = self.cfg.pose_names
        pre_p = self._correct(self.poses[pn.pre_pick], tvec=tvec)
        pick_p = self._correct(self.poses[pn.pick], tvec=tvec)
        lift_p = self._correct(self.poses[pn.lift], tvec=tvec)
        home_p = self._correct(self.poses[pn.home], tvec=tvec)

        cur = post_primitive(self.cfg.pi_base_url, {"name": "get_observation"})["observation"]
        cur_j = {k: float(v) for k, v in cur.items() if k.endswith(".pos")}
        self._interpolate(cur_j, pre_p)
        self.run_open()
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.3})
        self._interpolate(pre_p, pick_p)
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.2})
        self.run_close()
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.3})
        self._interpolate(pick_p, lift_p)
        self._interpolate(lift_p, home_p)

    def run_place(self, *, tvec: np.ndarray | None) -> None:
        pn = self.cfg.pose_names
        pre_pl = self._correct(self.poses[pn.pre_place], tvec=tvec)
        place_p = self._correct(self.poses[pn.place], tvec=tvec)
        post_p = self._correct(self.poses[pn.post_place], tvec=tvec)
        home_p = self._correct(self.poses[pn.home], tvec=tvec)

        extra_pan = self._track_extra_pan()
        if extra_pan != 0.0:
            k = "shoulder_pan.pos"
            place_p = dict(place_p)
            pre_pl = dict(pre_pl)
            post_p = dict(post_p)
            place_p[k] = float(place_p.get(k, 0.0)) + extra_pan
            pre_pl[k] = float(pre_pl.get(k, 0.0)) + extra_pan
            post_p[k] = float(post_p.get(k, 0.0)) + extra_pan

        cur = post_primitive(self.cfg.pi_base_url, {"name": "get_observation"})["observation"]
        cur_j = {k: float(v) for k, v in cur.items() if k.endswith(".pos")}
        self._interpolate(cur_j, pre_pl)
        self._interpolate(pre_pl, place_p)
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.2})
        self.run_open()
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.25})
        self._interpolate(place_p, post_p)
        self._interpolate(post_p, home_p)

    def run_power(self, *, tvec: np.ndarray | None) -> None:
        pn = self.cfg.pose_names
        a = self._correct(self.poses[pn.power_pre], tvec=tvec)
        b = self._correct(self.poses[pn.power_press], tvec=tvec)
        home_p = self._correct(self.poses[pn.home], tvec=tvec)
        cur = post_primitive(self.cfg.pi_base_url, {"name": "get_observation"})["observation"]
        cur_j = {k: float(v) for k, v in cur.items() if k.endswith(".pos")}
        self._interpolate(cur_j, a)
        self._interpolate(a, b)
        post_primitive(self.cfg.pi_base_url, {"name": "sleep", "seconds": 0.15})
        self._interpolate(b, home_p)


@parser.wrap()
def main(cfg: SayCanTrainswapConfig) -> None:
    _init_logging()
    logging.info("SayCan trainswap config:\n%s", pformat(asdict(cfg)))

    if not cfg.realsense_serial:
        raise ValueError("realsense_serial must be set (use `lerobot-find-cameras realsense`).")

    scorer = ClipScorer(
        pretrained_name=cfg.clip.pretrained_name,
        device=cfg.clip.device,
        batch_size=cfg.clip.batch_size,
        normalize_embeddings=cfg.clip.normalize_embeddings,
    )

    cap = RealSenseArucoCapture(
        serial_number=str(cfg.realsense_serial),
        width=int(cfg.rs_width),
        height=int(cfg.rs_height),
        fps=int(cfg.rs_fps),
        marker_length_m=float(cfg.marker_length_m),
        aruco_dictionary=str(cfg.aruco_dictionary),
    )
    cap.connect()

    ex = TrainswapExecutor(cfg)
    try:
        for step in range(int(cfg.num_steps)):
            rgb, _ = cap.grab()
            det = cap.detect_tag(int(cfg.tag_id))
            tvec = det.tvec if det is not None else None

            candidates = _get_candidates(cfg)
            scored = scorer.score(rgb, candidates)
            top = scored[0] if scored else None
            logging.info("Step %s top=%s score=%s", step + 1, getattr(top, "text", None), getattr(top, "score", None))

            if top is None:
                continue
            skill = _map_text_to_skill(top.text)
            logging.info("Mapped skill=%s", skill)

            if skill == "done":
                break
            if skill == "noop":
                continue
            if skill == "home":
                ex.run_home(tvec=tvec)
            elif skill == "open":
                ex.run_open()
            elif skill == "close":
                ex.run_close()
            elif skill == "pick":
                ex.run_pick(tvec=tvec)
            elif skill == "place":
                ex.run_place(tvec=tvec)
            elif skill == "power":
                ex.run_power(tvec=tvec)
    finally:
        ex.close()
        cap.disconnect()


if __name__ == "__main__":
    main()
