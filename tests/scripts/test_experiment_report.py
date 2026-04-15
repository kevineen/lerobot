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

import pytest

from lerobot.utils.experiment_report import (
    default_template_text,
    render_experiment_report,
    validate_experiment_data,
)


def test_render_experiment_report_minimal():
    data = {
        "meta": {"title": "Test experiment"},
        "abstract": "Short abstract.",
        "results": {
            "tables": [
                {
                    "caption": "Metrics",
                    "rows": [
                        {"metric": "latency", "value": "42", "unit": "ms", "notes": "ok"},
                    ],
                }
            ]
        },
    }
    out = render_experiment_report(data, default_template_text())
    assert "# Test experiment" in out
    assert "## 要旨 (Abstract)" in out
    assert "Short abstract." in out
    assert "| latency | 42 | ms | ok |" in out


@pytest.mark.parametrize(
    "bad_data,match",
    [
        ({}, "meta"),
        ({"meta": {}}, "title"),
        ({"meta": {"title": ""}}, "title"),
        ({"meta": {"title": "   "}}, "title"),
    ],
)
def test_validate_experiment_data_rejects(bad_data, match):
    with pytest.raises(ValueError, match=match):
        validate_experiment_data(bad_data)
