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

"""Heuristic rail / line detection from a USB (OpenCV) camera (Phase 4 fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cv2():
    import cv2  # type: ignore

    return cv2


@dataclass(frozen=True)
class TrackOffsetEstimate:
    """Rough lateral offset of a dominant line near the image center (pixels)."""

    offset_x_px: float
    angle_deg: float
    confidence: float


def estimate_track_offset_from_bgr(bgr: np.ndarray, *, roi_half_height: int = 120) -> TrackOffsetEstimate | None:
    """Detect a near-vertical dominant line in a horizontal band around the image center.

    This is intentionally lightweight (Canny + probabilistic Hough) for CPU-only PCs.

    Args:
        bgr: uint8 image (H,W,3) in BGR order (OpenCV default).
        roi_half_height: Half-height of the central ROI band used for line search.

    Returns:
        TrackOffsetEstimate or None if no line is found.
    """
    cv2 = _cv2()
    h, w = bgr.shape[:2]
    cy = h // 2
    y0 = max(0, cy - roi_half_height)
    y1 = min(h, cy + roi_half_height)
    roi = bgr[y0:y1, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=40, maxLineGap=12)
    if lines is None or len(lines) == 0:
        return None

    cx = w * 0.5
    best_score = 0.0
    best_x_at_cy = cx
    best_angle = 0.0

    for ln in lines.reshape(-1, 4):
        x1, y1l, x2, y2l = ln.astype(np.float64)
        dx = x2 - x1
        dy = y2 - y1l
        if abs(dx) < 1e-3:
            angle = 90.0
            x_at_cy = x1
        else:
            angle = float(np.degrees(np.arctan2(dy, dx)))
            # Intersect line with global y = cy (in full image coords: y in roi is y0 + y_roi)
            # Map roi y to global: y_global = y0 + y_roi
            # We want x at y_global = cy -> y_roi = cy - y0
            y_target_roi = cy - y0
            t = (y_target_roi - y1l) / (y2l - y1l + 1e-9)
            x_at_cy = x1 + t * (x2 - x1)

        # Prefer near-vertical lines
        score = abs(np.sin(np.radians(angle)))
        if score > best_score:
            best_score = float(score)
            best_x_at_cy = float(x_at_cy)
            best_angle = float(angle)

    if best_score < 0.25:
        return None

    return TrackOffsetEstimate(offset_x_px=float(best_x_at_cy - cx), angle_deg=best_angle, confidence=best_score)
