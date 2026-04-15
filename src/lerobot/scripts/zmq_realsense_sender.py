#!/usr/bin/env python
"""
ZMQ RealSense sender (Jetson/TX2/Nano side).

Publishes color frames (and optional inference payload) for consumption by `ZMQCamera`.

Message protocol (JSON string):
    {
      "images": {"<camera_name>": "<base64-jpeg>"},
      "timestamps": {"<camera_name>": <float perf_counter seconds>},
      "frame_ids": {"<camera_name>": <int>},
      "inference": {"<camera_name>": "<json-string>"}
    }

Notes:
    - `timestamps` SHOULD be monotonic seconds (`time.perf_counter()`), not wall clock.
    - `inference` is an opaque JSON string. Keep it schema-free unless you also
      update your downstream feature schema.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SenderConfig:
    bind: str
    camera_name: str
    width: int
    height: int
    fps: int
    jpeg_quality: int


def parse_args() -> SenderConfig:
    p = argparse.ArgumentParser(description="Publish RealSense frames over ZMQ as JSON/base64-JPEG.")
    p.add_argument("--bind", default="tcp://0.0.0.0:5555", help="ZMQ bind address (PUB).")
    p.add_argument("--camera-name", default="front", help="Camera name key used by ZMQCamera.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=85, help="OpenCV JPEG quality (0-100).")
    args = p.parse_args()
    return SenderConfig(
        bind=args.bind,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
    )


def encode_jpeg_b64(bgr: np.ndarray, *, quality: int) -> str:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def main() -> None:
    cfg = parse_args()

    try:
        import zmq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyzmq is required on the sender side") from e

    try:
        import pyrealsense2 as rs
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyrealsense2 is required on the sender side") from e

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(cfg.bind)

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
    pipeline.start(rs_cfg)

    frame_id = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())
            ts = time.perf_counter()

            # Placeholder: replace with your inference result.
            # Keep it compact; it's sent every frame.
            inference_json = ""

            payload = {
                "images": {cfg.camera_name: encode_jpeg_b64(img, quality=cfg.jpeg_quality)},
                "timestamps": {cfg.camera_name: ts},
                "frame_ids": {cfg.camera_name: frame_id},
                "inference": {cfg.camera_name: inference_json},
            }
            sock.send_string(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            frame_id += 1
    finally:
        pipeline.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()

