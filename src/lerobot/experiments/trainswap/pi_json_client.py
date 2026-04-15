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

"""Minimal JSON-over-HTTP client (stdlib) for the SO-101 primitive server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def post_primitive(base_url: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    """POST ``/primitive`` and parse JSON response.

    Args:
        base_url: Example ``http://192.168.1.50:8765`` (no trailing slash).
        payload: JSON-serializable command dict.
        timeout_s: Socket timeout seconds.

    Returns:
        Parsed JSON object from the server.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
    """
    url = base_url.rstrip("/") + "/primitive"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e

    return json.loads(body)
