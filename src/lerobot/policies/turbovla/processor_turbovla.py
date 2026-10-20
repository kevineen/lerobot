#!/usr/bin/env python

# TurboVLA pre/post-processing: standard lerobot normalization pipelines
# (STATE/ACTION min-max to [-1, 1], VISUAL identity — the model applies
# ImageNet normalization internally).

from typing import Any

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_turbovla import TurboVLAConfig


def make_turbovla_pre_post_processors(
    config: TurboVLAConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
