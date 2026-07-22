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
"""Generate a polygon-naming JSONL dataset.

Each row bundles three pre-rendered 128×128 canvases (base64 data URLs)
into `turn_1_images` (2) and `turn_2_images` (1), plus the multiset of
ground-truth `(sides, colour)` tuples across all three.

Usage:

    python generate_data.py --num-rows 5 --output data/example.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


CANVAS_SIZE = 128
CENTER = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
RADIUS = 50
BACKGROUND = (0, 0, 0)
PALETTE = {
    "red":     (255,   0,   0),
    "green":   (  0, 200,   0),
    "blue":    (  0,   0, 255),
    "yellow":  (255, 230,   0),
    "magenta": (255,   0, 255),
}
MIN_SIDES = 3
MAX_SIDES = 6

SYSTEM_PROMPT = (
    "You are a visual perception assistant. Each turn you will be shown one "
    "or more small 128×128 canvases. Each canvas contains a single regular "
    "polygon drawn in a bright colour on a black background. Your job is "
    "to identify each polygon's `(number_of_sides, colour_name)` and submit "
    "them via the `submit_turn` tool.\n\n"
    "Colour vocabulary (use these lower-case names EXACTLY): "
    "red, green, blue, yellow, magenta.\n"
    "Number of sides is an integer in [3, 6]. A 3-sided polygon is a "
    "triangle, 4-sided is a square, 5-sided is a pentagon, 6-sided is a "
    "hexagon.\n\n"
    "On each turn, call `submit_turn` exactly once with `answers` set to "
    "the list of `[sides, colour]` pairs for the images shown that turn "
    "(preserve image order). After the last turn, reply with a short "
    "natural-language message such as `done` to end the episode."
)


TOOL_SCHEMA = {
    "type": "function",
    "name": "submit_turn",
    "description": (
        "Submit the (sides, colour) pairs for the images shown in the "
        "current turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "description": (
                    "List of `[sides, colour]` pairs, one per image shown "
                    "in the current turn, in image order."
                ),
                "items": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "integer", "minimum": MIN_SIDES, "maximum": MAX_SIDES},
                        {"type": "string", "enum": list(PALETTE.keys())},
                    ],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "required": ["answers"],
        "additionalProperties": False,
    },
    "strict": False,
}


def _regular_polygon(n: int, radius: int, center: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Vertices of a regular n-gon inscribed in a circle, apex up."""
    cx, cy = center
    # -pi/2 rotates so the first vertex points up.
    return [
        (
            cx + radius * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
            cy + radius * math.sin(-math.pi / 2 + 2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def render_polygon(sides: int, colour_name: str) -> str:
    """Render a filled regular n-gon and return a base64 PNG data URL."""
    fill = PALETTE[colour_name]
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    vertices = _regular_polygon(sides, RADIUS, CENTER)
    draw.polygon(vertices, fill=fill)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _sample_polygon(rng: random.Random) -> Tuple[int, str, str]:
    sides = rng.randint(MIN_SIDES, MAX_SIDES)
    colour = rng.choice(list(PALETTE.keys()))
    return sides, colour, render_polygon(sides, colour)


def _make_row(row_id: int, rng: random.Random) -> dict:
    n1, c1, url1 = _sample_polygon(rng)
    n2, c2, url2 = _sample_polygon(rng)
    n3, c3, url3 = _sample_polygon(rng)

    return {
        "id": row_id,
        # NeMo-Gym's RolloutCollectionHelper routes each row to the agent
        # server named here (see 3rdparty/Gym-workspace/Gym/nemo_gym/
        # rollout_collection.py:689 — `row["agent_ref"]["name"]`). Must
        # match the outer instance key in polygon_naming.yaml.
        "agent_ref": {"name": "polygon_naming_multimodal_simple_agent"},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # Kickoff user turn is empty text; the actual first-turn
                # user message (with images) is injected by /seed_session
                # via multimodal_simple_agent.
                {"role": "user", "content": "Awaiting the first turn."},
            ],
            "tools": [TOOL_SCHEMA],
            "parallel_tool_calls": False,
        },
        "truth_tuples": [[n1, c1], [n2, c2], [n3, c3]],
        "turn_1_images": [url1, url2],
        "turn_2_images": [url3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for i in range(args.num_rows):
            f.write(json.dumps(_make_row(i, rng)) + "\n")
    print(f"Wrote {args.num_rows} rows to {args.output}")


if __name__ == "__main__":
    main()
