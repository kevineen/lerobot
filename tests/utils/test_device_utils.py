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

"""ROCm-aware device helpers used by this fork's training path."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.utils.device_utils import (
    apply_attn_implementation,
    auto_select_torch_device,
    canonical_torch_device,
    get_safe_torch_device,
    is_amp_available,
    is_rocm,
    is_torch_device_available,
    select_attn_implementation,
)
from lerobot.utils.import_utils import get_safe_default_codec, torchcodec_is_usable


def test_canonical_torch_device_aliases():
    """rocm/hip are user aliases; PyTorch still wants the cuda device type."""
    assert canonical_torch_device("rocm") == "cuda"
    assert canonical_torch_device("hip") == "cuda"
    assert canonical_torch_device("rocm:0") == "cuda:0"
    assert canonical_torch_device("HIP:1") == "cuda:1"
    assert canonical_torch_device("cuda:0") == "cuda:0"
    assert canonical_torch_device("cpu") == "cpu"


def test_select_attn_implementation_skips_flash_on_rocm(monkeypatch):
    import lerobot.utils.device_utils as device_utils

    monkeypatch.setattr(device_utils, "is_rocm", lambda: True)
    assert device_utils.select_attn_implementation() == "sdpa"
    assert device_utils.select_attn_implementation(allow_flash=True) == "sdpa"


def test_select_attn_implementation_without_flash_attn(monkeypatch):
    import lerobot.utils.device_utils as device_utils

    monkeypatch.setattr(device_utils, "is_rocm", lambda: False)
    original = device_utils.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "flash_attn":
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(device_utils.importlib.util, "find_spec", fake_find_spec)
    assert device_utils.select_attn_implementation(allow_flash=True) == "sdpa"
    assert device_utils.select_attn_implementation(allow_flash=False) == "sdpa"


def test_apply_attn_implementation_nested_configs():
    text = SimpleNamespace(_attn_implementation="flash_attention_2")
    vision = SimpleNamespace(_attn_implementation="flash_attention_2")
    config = SimpleNamespace(_attn_implementation="flash_attention_2", text_config=text, vision_config=vision)
    impl = apply_attn_implementation(config, "sdpa")
    assert impl == "sdpa"
    assert config._attn_implementation == "sdpa"
    assert text._attn_implementation == "sdpa"
    assert vision._attn_implementation == "sdpa"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm not available")
def test_rocm_alias_resolves_to_cuda_device():
    device = get_safe_torch_device("rocm")
    assert device.type == "cuda"
    assert is_torch_device_available("rocm")
    assert is_amp_available("rocm")
    assert is_amp_available("cuda:0")


def test_auto_select_logs_rocm(monkeypatch, caplog):
    import lerobot.utils.device_utils as device_utils

    if not torch.cuda.is_available():
        pytest.skip("needs torch.cuda")
    monkeypatch.setattr(device_utils, "is_rocm", lambda: True)
    with caplog.at_level("INFO"):
        selected = auto_select_torch_device()
    assert selected.type == "cuda"
    assert any("ROCm" in record.message for record in caplog.records)


def test_torchcodec_unusable_on_rocm(monkeypatch):
    monkeypatch.setattr("lerobot.utils.device_utils.is_rocm", lambda: True)
    assert torchcodec_is_usable() is False
    assert get_safe_default_codec() == "pyav"


def test_live_rocm_codec_policy():
    """On this fork's AMD training host, default codec must be pyav."""
    if not is_rocm():
        pytest.skip("not a ROCm PyTorch build")
    assert torchcodec_is_usable() is False
    assert get_safe_default_codec() == "pyav"
