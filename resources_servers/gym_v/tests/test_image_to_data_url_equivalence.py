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
"""Pinned byte-equivalence regression guard for the PNG image-to-data-URL path.

The resources server and NeMo-RL each implement `image_to_data_url`. They MUST
emit byte-identical strings for the same input, or training-data drift goes
silently uncaught. This test pins the encoder to the contract NeMo-RL ships in
`nemo_rl/environments/nemo_gym.py::image_to_data_url`. The reference encoder is
inlined here so the test runs in the resources-server's lightweight venv (no
need to import torch / ray / transformers via `nemo_rl.environments.nemo_gym`).

If NeMo-RL's helper changes, update the inline reference here AND the resources
server's `_observation.image_to_data_url` in the same PR. PR review is not a
strong enough safeguard for this contract, hence the test.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

from resources_servers.gym_v._observation import (
    image_to_data_url as resources_image_to_data_url,
)


def _nemo_rl_reference_image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Pinned copy of `nemo_rl.environments.nemo_gym.image_to_data_url`."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def _deterministic_image() -> Image.Image:
    image = Image.new("RGB", (32, 32))
    pixels = image.load()
    for x in range(image.width):
        for y in range(image.height):
            pixels[x, y] = ((x * 17) % 256, (y * 29) % 256, ((x + y) * 13) % 256)
    return image


def test_image_to_data_url_matches_nemo_rl_png_encoder() -> None:
    image = _deterministic_image()

    assert resources_image_to_data_url(image) == _nemo_rl_reference_image_to_data_url(image)
