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

"""Apply small joint corrections from tag translation deltas (camera frame, meters)."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_tvec_joint_gains(
    base_joints: dict[str, float],
    *,
    tvec_m: np.ndarray,
    ref_tvec_m: np.ndarray,
    gains: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Add joint offsets proportional to (tvec - ref_tvec).

    Args:
        base_joints: Joint positions keyed like ``\"shoulder_pan.pos\"``.
        tvec_m: Current tag translation in camera frame (meters), shape (3,).
        ref_tvec_m: Reference tag translation at calibration time, shape (3,).
        gains: Mapping ``motor_key -> {\"tx\": deg_per_m, \"ty\": ..., \"tz\": ...}``.
            Motor keys should include the ``.pos`` suffix used by LeRobot SO follower.

    Returns:
        New joint dictionary (copied, with corrections applied).
    """
    d = np.asarray(tvec_m, dtype=np.float64).reshape(3) - np.asarray(ref_tvec_m, dtype=np.float64).reshape(3)
    tx, ty, tz = float(d[0]), float(d[1]), float(d[2])
    out = dict(base_joints)
    for motor_key, axis_gains in gains.items():
        if not motor_key.endswith(".pos"):
            continue
        corr = 0.0
        corr += float(axis_gains.get("tx", 0.0)) * tx
        corr += float(axis_gains.get("ty", 0.0)) * ty
        corr += float(axis_gains.get("tz", 0.0)) * tz
        out[motor_key] = float(out.get(motor_key, 0.0)) + corr
    return out


def load_gains_from_calibration(cal: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract gain map from a calibration JSON dict."""
    raw = cal.get("joint_gains_per_tag_tvec_m", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for motor_key, gains in raw.items():
        if not isinstance(gains, dict):
            continue
        out[str(motor_key)] = {str(k): float(v) for k, v in gains.items() if k in ("tx", "ty", "tz")}
    return out


def ref_tvec_from_calibration(cal: dict[str, Any]) -> np.ndarray:
    """Read reference tag tvec (meters) from calibration JSON."""
    v = cal.get("reference_tag_tvec_cam_m", None)
    if v is None:
        raise KeyError("calibration missing reference_tag_tvec_cam_m")
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    return arr
