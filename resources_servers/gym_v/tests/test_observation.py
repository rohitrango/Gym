# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
from dataclasses import dataclass, field
from typing import Any

import pytest
from PIL import Image

from resources_servers.gym_v._observation import (
    _attach_env_info,
    _format_visible_metadata,
    image_to_data_url,
    observation_to_user_message,
)


@dataclass
class _StubObservation:
    """Lightweight stand-in for `gym_v.Observation` used in helper-level tests."""

    text: str | None = None
    image: Image.Image | list[Image.Image] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _decode_data_url(data_url: str) -> Image.Image:
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_image_to_data_url_round_trips_png() -> None:
    image = Image.new("RGB", (4, 4), (12, 34, 56))
    data_url = image_to_data_url(image)

    decoded = _decode_data_url(data_url)

    assert decoded.size == (4, 4)
    assert decoded.getpixel((0, 0)) == (12, 34, 56)


def test_observation_to_user_message_includes_text_image_and_visible_metadata() -> None:
    obs = _StubObservation(
        text="board caption",
        image=Image.new("RGB", (4, 4), (0, 0, 0)),
        metadata={
            "state_text": "[[0,1],[1,0]]",
            "text_prompt": "Predict the next board.",
            "oracle_answer": "do not leak",
        },
    )

    message = observation_to_user_message(
        obs,
        env_id="Algorithmic/GameOfLife-v0",
        prefix_text="Rules text",
    )

    content = message.content
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert "Rules text" in content[0]["text"]
    assert "board caption" in content[0]["text"]
    assert "state_text: [[0,1],[1,0]]" in content[0]["text"]
    assert "text_prompt" not in content[0]["text"]
    assert "oracle_answer" not in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert message.env_info is None


def test_format_visible_metadata_unlisted_env_defaults_to_empty_allowlist() -> None:
    """Unlisted envs return None (no metadata exposed), matching the explicit
    `[]` entries in `_VISIBLE_METADATA_KEYS` (e.g. Games/FrozenLake-v0). This
    keeps the allow-list a true allow-list — adding a new env still requires
    an explicit entry to surface its metadata — while not crashing the
    viewer / endpoint code path on exploratory envs that haven't been
    audited yet.
    """
    assert _format_visible_metadata("Unknown/Env-v0", {"anything": "hidden"}) is None


def test_format_visible_metadata_empty_allowlist_returns_none() -> None:
    assert _format_visible_metadata("Games/FrozenLake-v0", {"anything": "hidden"}) is None


def test_attach_env_info_sanitizes_metadata() -> None:
    message = observation_to_user_message(
        _StubObservation(text="state", image=None, metadata={}),
        env_id="Games/FrozenLake-v0",
    )
    updated = _attach_env_info(message, {"invalid_action": False, "coords": (1, 2)})

    assert updated.env_info == {"invalid_action": False, "coords": [1, 2]}
