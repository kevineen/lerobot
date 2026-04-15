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

"""Intel RealSense capture + ArUco pose estimation (aligned depth to color)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.experiments.trainswap.geometry import T_from_rvec_tvec


def _cv2():
    import cv2  # type: ignore

    return cv2


def _rs():
    import pyrealsense2 as rs  # type: ignore

    return rs


def aruco_dictionary_by_name(name: str):
    """Resolve OpenCV ArUco dictionary from a string like ``DICT_4X4_50``."""
    cv2 = _cv2()
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was built without aruco module; install opencv-contrib-python-headless.")
    dict_id = getattr(cv2.aruco, name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary name: {name!r}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


@dataclass(frozen=True)
class TagDetection:
    """One detected marker."""

    tag_id: int
    rvec: np.ndarray  # (3,) float64
    tvec: np.ndarray  # (3,) float64, meters in camera frame
    corners_px: np.ndarray  # (1, 4, 2) float32
    depth_center_m: float | None  # median depth at marker center (aligned to color), if available


@dataclass
class RealSenseArucoCapture:
    """Minimal RealSense pipeline with depth aligned to color for ArUco + depth sampling."""

    serial_number: str
    width: int
    height: int
    fps: int
    marker_length_m: float
    aruco_dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        self._pipeline = None
        self._profile = None
        self._align = None
        self._aruco_dict = aruco_dictionary_by_name(self.aruco_dictionary)
        cv2 = _cv2()
        self._aruco_params = cv2.aruco.DetectorParameters()

    def connect(self) -> None:
        rs = _rs()
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial_number)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._profile = None
        self._align = None

    def _camera_matrix_and_dist(self) -> tuple[np.ndarray, np.ndarray]:
        if self._profile is None:
            raise RuntimeError("RealSenseArucoCapture is not connected.")
        rs = _rs()
        stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        # Build K from intrinsics (pixel coordinates, z in meters for projected points)
        k = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.array(intr.coeffs, dtype=np.float64).reshape(5, 1)
        return k, dist

    def grab(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return RGB uint8 (H,W,3) and depth uint16 (H,W) aligned to color, or (rgb, None) if no depth."""
        if self._pipeline is None or self._align is None:
            raise RuntimeError("RealSenseArucoCapture is not connected.")
        rs = _rs()
        frames = self._pipeline.wait_for_frames(timeout_ms=10000)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            raise RuntimeError("No color frame from RealSense.")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return color, depth

    def detect_tag(self, tag_id: int) -> TagDetection | None:
        """Detect a single ArUco marker id; returns None if not found."""
        rgb, depth_u16 = self.grab()
        cv2 = _cv2()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # OpenCV 4.7+ prefers ``ArucoDetector``; older versions use ``detectMarkers`` directly.
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)
        if ids is None or len(ids) == 0:
            return None
        flat = ids.flatten().tolist()
        if tag_id not in flat:
            return None
        idx = flat.index(tag_id)
        k, dist = self._camera_matrix_and_dist()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx], float(self.marker_length_m), k, dist
        )
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3)
        c = corners[idx][0]  # (4,2)
        center = np.mean(c, axis=0)
        depth_m: float | None = None
        if depth_u16 is not None:
            u = int(round(float(center[0])))
            v = int(round(float(center[1])))
            h, w = depth_u16.shape[:2]
            r = 3
            patch = depth_u16[max(0, v - r) : min(h, v + r + 1), max(0, u - r) : min(w, u + r + 1)]
            zs = patch.astype(np.float64).reshape(-1)
            zs = zs[zs > 0]
            if zs.size:
                z_mm = float(np.median(zs))
                depth_m = z_mm * 1e-3
        return TagDetection(
            tag_id=int(tag_id),
            rvec=rvec,
            tvec=tvec,
            corners_px=corners[idx],
            depth_center_m=depth_m,
        )

    def T_cam_from_tag(self, tag_id: int) -> np.ndarray | None:
        """Return 4x4 ``T_cam_from_tag`` if marker is visible."""
        det = self.detect_tag(tag_id)
        if det is None:
            return None
        return T_from_rvec_tvec(det.rvec, det.tvec)
