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

"""Small rigid-body helpers for tag poses (OpenCV rvec/tvec convention)."""

from __future__ import annotations

import numpy as np


def _cv2():
    import cv2  # type: ignore

    return cv2


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector (3,) to 3x3 rotation matrix."""
    cv2 = _cv2()
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rmat, _ = cv2.Rodrigues(rvec)
    return np.asarray(rmat, dtype=np.float64)


def T_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build 4x4 transform mapping points from marker frame into camera frame.

    Args:
        rvec: (3,) rotation vector (marker -> camera) in OpenCV `solvePnP` convention.
        tvec: (3,) translation of marker origin expressed in camera frame (meters).

    Returns:
        T_cam_from_marker: (4, 4) homogeneous matrix.
    """
    rmat = rodrigues_to_matrix(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    t_out = np.eye(4, dtype=np.float64)
    t_out[:3, :3] = rmat
    t_out[:3, 3] = t
    return t_out


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return homogeneous product ``a @ b`` (4x4)."""
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform represented as 4x4."""
    t = np.asarray(T, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    rt = r.T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ p
    return out
