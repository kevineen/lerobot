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

"""YAML-driven experiment report rendering (Markdown via Jinja2)."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment

EXPERIMENT_REPORT_TEMPLATE = "experiment_report.md.jinja"


def load_experiment_yaml(path: Path) -> dict[str, Any]:
    """Load experiment definition from a YAML file."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Experiment YAML must be a mapping at the root, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def validate_experiment_data(data: dict[str, Any]) -> None:
    """Ensure required keys exist for report generation."""
    meta = data.get("meta")
    if not isinstance(meta, dict):
        msg = "Experiment YAML must contain a 'meta' mapping."
        raise ValueError(msg)
    title = meta.get("title")
    if not title or not isinstance(title, str) or not title.strip():
        msg = "Experiment YAML must set 'meta.title' to a non-empty string."
        raise ValueError(msg)


def default_template_text() -> str:
    """Return the bundled Markdown/Jinja template for experiment reports."""
    path = files("lerobot.templates").joinpath(EXPERIMENT_REPORT_TEMPLATE)
    return path.read_text(encoding="utf-8")


def render_experiment_report(data: dict[str, Any], template_str: str) -> str:
    """Render Markdown from structured experiment data and a Jinja2 template string."""
    validate_experiment_data(data)
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=False,
    )
    tmpl = env.from_string(template_str)
    return tmpl.render(report=data)


def render_experiment_report_from_files(
    config_path: Path,
    template_path: Path | None = None,
) -> str:
    """Load YAML from disk and render using the bundled or a custom template file."""
    data = load_experiment_yaml(config_path)
    if template_path is not None:
        template_str = template_path.read_text(encoding="utf-8")
    else:
        template_str = default_template_text()
    return render_experiment_report(data, template_str)
