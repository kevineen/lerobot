from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@dataclass(frozen=True)
class ImageWithMeta:
    image: np.ndarray  # HWC uint8 RGB
    source: str


def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    return np.array(rgb, dtype=np.uint8)


def load_images_from_paths(patterns: list[str]) -> list[ImageWithMeta]:
    paths: list[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    unique = sorted(set(paths))
    out: list[ImageWithMeta] = []
    for p in unique:
        img = Image.open(p)
        out.append(ImageWithMeta(image=_pil_to_rgb_array(img), source=str(Path(p))))
    return out


class FrontRealSenseSource:
    """One-frame capture source for an Intel RealSense device."""

    def __init__(self, cfg: RealSenseCameraConfig) -> None:
        self.cfg = cfg
        self._cam: RealSenseCamera | None = None

    def connect(self) -> None:
        if self._cam is not None:
            return
        self._cam = RealSenseCamera(self.cfg)
        self._cam.connect()

    def read(self) -> ImageWithMeta:
        if self._cam is None or not self._cam.is_connected:
            raise RuntimeError("FrontRealSenseSource is not connected.")
        frame = self._cam.read_latest()
        # RealSenseCamera returns CHW or HWC depending on internal pipeline; normalize to HWC uint8.
        arr = frame
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (
            1,
            3,
            4,
        ):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if float(arr.max()) <= 1.0 else arr.astype(np.uint8)
        return ImageWithMeta(image=arr, source=f"realsense:{self.cfg.serial_number_or_name}")

    def disconnect(self) -> None:
        if self._cam is None:
            return
        self._cam.disconnect()
        self._cam = None

