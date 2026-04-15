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

"""
Render a paper-style Markdown experiment report from a YAML config (Jinja2 template).

Example:

```shell
lerobot-experiment-report \\
  --config examples/experiment_reports/so101_network_realsense_saycan.yaml \\
  --output reports/experiment.md
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot.utils.experiment_report import render_experiment_report_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML (must include meta.title).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown here. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Custom Jinja template path (default: bundled experiment_report.md.jinja).",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.template is not None and not args.template.is_file():
        print(f"Template file not found: {args.template}", file=sys.stderr)
        sys.exit(1)

    try:
        markdown = render_experiment_report_from_files(args.config, template_path=args.template)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
