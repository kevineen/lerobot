#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch device helpers.

ROCm/HIP builds still expose ``torch.cuda`` and use device type ``"cuda"``.
This module makes that mapping explicit so callers can log, alias, and skip
NVIDIA-only extras (FlashAttention2, CUDA-ABI torchcodec) on AMD GPUs.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# User-facing aliases that PyTorch does not accept as device types.
_ROCM_ALIASES = ("rocm", "hip")


def is_rocm() -> bool:
    """Return True when this PyTorch build is ROCm/HIP rather than NVIDIA CUDA.

    Detection uses ``torch.version.hip``, which is set only on ROCm wheels.
    ``torch.cuda.is_available()`` can still be True on those wheels.
    """
    hip = getattr(torch.version, "hip", None)
    return bool(hip)


def is_nvidia_cuda() -> bool:
    """Return True when a GPU is available and this is an NVIDIA CUDA build."""
    return torch.cuda.is_available() and not is_rocm()


def canonical_torch_device(try_device: str) -> str:
    """Map user aliases onto names ``torch.device`` understands.

    ``rocm`` / ``hip`` (optional index) become ``cuda`` / ``cuda:N`` because
    ROCm PyTorch reuses the CUDA device type.
    """
    try_device = str(try_device).strip()
    lower = try_device.lower()
    for alias in _ROCM_ALIASES:
        if lower == alias:
            return "cuda"
        prefix = f"{alias}:"
        if lower.startswith(prefix):
            return f"cuda:{try_device.split(':', 1)[1]}"
    return try_device


def select_attn_implementation(*, allow_flash: bool = True) -> str:
    """Pick a Hugging Face Transformers attention backend.

    FlashAttention2 is NVIDIA CUDA only (and often compiled against a CUDA
    PyTorch ABI). ROCm and missing ``flash_attn`` fall back to SDPA.
    """
    if not allow_flash or is_rocm():
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return "flash_attention_2"


def apply_attn_implementation(config: Any, implementation: str | None = None) -> str:
    """Set ``_attn_implementation`` on a transformers config and nested configs.

    GR00T's Eagle backbone stores attention choice on the parent config plus
    ``text_config`` / ``vision_config``. Overriding all three keeps SDPA in
    effect even when a checkpoint JSON still says ``flash_attention_2``.
    """
    impl = implementation or select_attn_implementation()
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = impl
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "_attn_implementation"):
        text_config._attn_implementation = impl
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "_attn_implementation"):
        vision_config._attn_implementation = impl
    return impl


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        if is_rocm():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "ROCm GPU"
            logging.info(
                "ROCm/HIP backend detected (%s, hip=%s). Using torch device 'cuda' "
                "(ROCm maps the CUDA API to HIP).",
                name,
                torch.version.hip,
            )
        else:
            logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = canonical_torch_device(str(try_device))
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        device = torch.device(try_device)
        if log and is_rocm():
            logging.info("Using ROCm via torch device %s.", device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    elif try_device == "xpu":
        assert torch.xpu.is_available()
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    device = canonical_torch_device(str(device))
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # NOTE: Some Intel XPU devices do not support double precision (FP64).
            # The `has_fp64` flag is returned by `torch.xpu.get_device_capability()`
            # when available; if False, we fall back to float32 for compatibility.
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = canonical_torch_device(str(try_device))
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, rocm, hip, mps, xpu or cpu.")


def is_amp_available(device: str):
    device = canonical_torch_device(str(device))
    if device.startswith("cuda") or device.startswith("xpu") or device == "cpu":
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")
