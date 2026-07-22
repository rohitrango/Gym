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
"""Polygon-naming multi-turn multimodal benchmark.

Each task ships three pre-rendered 128×128 canvases, each showing a
regular n-gon (n in [3, 6]) drawn in one of five primary colours.

Rollout, driven by `multimodal_simple_agent`:

  1. `/seed_session` → injects a user turn containing the first two
     images and instructs the model to call `submit_turn` with the list
     of `(sides, colour)` tuples matching those two images.
  2. Model calls `submit_turn(answers=[[3, "red"], [4, "green"]])`.
  3. `/submit_turn` records the answer, injects a second user turn
     containing the third image, and instructs the model to call
     `submit_turn` again with the tuple for that image.
  4. Model calls `submit_turn(answers=[[6, "yellow"]])`.
  5. `/submit_turn` records the answer and returns a plain-text "done".
  6. Model emits an assistant message; agent loop terminates.
  7. `/verify` compares the accumulated tuples against ground truth as
     a multiset (order-independent) and returns reward 0.0 or 1.0.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


PALETTE = ("red", "green", "blue", "yellow", "magenta")
MIN_SIDES = 3
MAX_SIDES = 6


class PolygonNamingConfig(BaseResourcesServerConfig):
    pass


class PolygonPair(BaseModel):
    """A single `(sides, colour)` tuple. Wire format matches what the model
    writes in the tool-call arguments: `[3, "red"]`."""

    sides: int
    colour: str


def _pairs_from_wire(raw: List[Any]) -> List[Tuple[int, str]]:
    """Coerce the tool-call payload into `(int, str)` tuples with the
    colour normalized to lower-case. Skips anything malformed rather than
    crashing; the multiset compare in `/verify` decides pass/fail."""
    pairs: List[Tuple[int, str]] = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        sides_raw, colour_raw = item
        try:
            sides = int(sides_raw)
        except (TypeError, ValueError):
            continue
        colour = str(colour_raw).strip().lower()
        pairs.append((sides, colour))
    return pairs


class PolygonSeedRequest(BaseSeedSessionRequest):
    """Extended seed_session request. Forwarded from `SimpleAgentRunRequest`
    (which allows extra fields), so the top-level JSONL row is available
    here field-by-field."""

    model_config = ConfigDict(extra="allow")

    truth_tuples: List[Tuple[int, str]] = Field(default_factory=list)
    turn_1_images: List[str] = Field(default_factory=list)
    turn_2_images: List[str] = Field(default_factory=list)


class SubmitTurnRequest(BaseModel):
    """Tool-call payload. The model provides a list of two-element
    lists — matched to `\\boxed{[(3, red), (4, green)]}` intent."""

    answers: List[List[Any]] = Field(default_factory=list)


class PolygonRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    truth_tuples: List[Tuple[int, str]] = Field(default_factory=list)
    turn_1_images: List[str] = Field(default_factory=list)
    turn_2_images: List[str] = Field(default_factory=list)


class PolygonVerifyRequest(PolygonRunRequest, BaseVerifyRequest):
    pass


class PolygonVerifyResponse(BaseVerifyResponse):
    submitted_tuples: List[Tuple[int, str]]
    turns_completed: int
    exact_match: bool


class _SessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn: int = 0
    answers: List[Tuple[int, str]] = Field(default_factory=list)
    truth: List[Tuple[int, str]] = Field(default_factory=list)
    turn_2_image_url: str | None = None


def _image_content_part(image_url: str) -> Dict[str, Any]:
    # `detail` is required by openai-python's ResponseInputImageParam schema
    # (validated in multimodal_simple_agent's seed_session handler via
    # NeMoGymEasyInputMessage.model_validate). "auto" defers to the model's
    # default handling; explicit "low"/"high" would force a specific token
    # budget for the image tile.
    return {"type": "input_image", "image_url": image_url, "detail": "auto"}


def _text_content_part(text: str) -> Dict[str, Any]:
    return {"type": "input_text", "text": text}


class PolygonNamingResourcesServer(SimpleResourcesServer):
    config: PolygonNamingConfig

    # In-process per-session store. Fine for eval workloads; the
    # session id lives in the signed session cookie, so parallel
    # rollouts do not collide.
    sessions: Dict[str, _SessionState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/submit_turn")(self.submit_turn)
        return app

    def _get_or_create_state(self, request: Request) -> Tuple[str, _SessionState]:
        sid = request.session[SESSION_ID_KEY]
        state = self.sessions.setdefault(sid, _SessionState())
        return sid, state

    async def seed_session(
        self,
        request: Request,
        body: PolygonSeedRequest,
    ) -> Dict[str, Any]:
        sid, _ = self._get_or_create_state(request)

        # Ensure the second-turn image is available for injection later.
        turn_2_url = body.turn_2_images[0] if body.turn_2_images else None

        self.sessions[sid] = _SessionState(
            turn=0,
            answers=[],
            truth=[(int(n), str(c).lower()) for n, c in body.truth_tuples],
            turn_2_image_url=turn_2_url,
        )

        content: List[Dict[str, Any]] = [
            _text_content_part(
                "Turn 1: two polygons are shown below. For EACH image, "
                "identify (number_of_sides, colour) and call the "
                "`submit_turn` tool with `answers` set to the two-element "
                "list of `[sides, colour]` pairs. Use lower-case colour "
                "names. Do not include reasoning in the tool arguments."
            ),
        ]
        for image_url in body.turn_1_images:
            content.append(_image_content_part(image_url))

        return {
            "as_user_messages": [
                {"role": "user", "content": content},
            ],
        }

    async def submit_turn(
        self,
        request: Request,
        body: SubmitTurnRequest,
    ) -> Dict[str, Any]:
        sid, state = self._get_or_create_state(request)

        parsed = _pairs_from_wire(body.answers)
        state.answers.extend(parsed)
        state.turn += 1
        self.sessions[sid] = state

        if state.turn == 1:
            # Reveal image 3 and ask for its tuple.
            if state.turn_2_image_url is None:
                # Malformed seed — no image to reveal. End the trajectory
                # cleanly with a plain-text acknowledgement so /verify can
                # still score whatever answers came in on turn 1.
                return {
                    "function_call_output": (
                        "No second-turn image was provided by the environment. "
                        "Reply with `done` to finish."
                    ),
                }
            return {
                "function_call_output": (
                    f"Turn 1 recorded ({len(parsed)} pair(s))."
                ),
                "as_user_messages": [
                    {
                        "role": "user",
                        "content": [
                            _text_content_part(
                                "Turn 2: one more polygon is shown below. "
                                "Call `submit_turn` again with `answers` set "
                                "to a single-element list containing the "
                                "`[sides, colour]` tuple for THIS image only."
                            ),
                            _image_content_part(state.turn_2_image_url),
                        ],
                    }
                ],
            }

        if state.turn == 2:
            return {
                "function_call_output": (
                    f"Turn 2 recorded ({len(parsed)} pair(s)). "
                    "Reply with a short natural-language message (e.g. `done`) "
                    "to finish the episode."
                ),
            }

        # Extra tool calls beyond the two scripted turns are a mild
        # protocol violation. Record the pairs (they still count toward
        # the multiset) but tell the model to stop.
        return {
            "function_call_output": (
                "All turns are complete. Do not call `submit_turn` again — "
                "reply with a short natural-language message to finish."
            ),
        }

    async def verify(self, request: Request, body: PolygonVerifyRequest) -> PolygonVerifyResponse:
        sid = request.session[SESSION_ID_KEY]
        state = self.sessions.pop(sid, None)

        if state is None:
            # No session state ⇒ seed_session never ran or was reset.
            # Fall back to using the row's ground truth against no
            # submissions so /verify still returns a well-formed response.
            truth = [(int(n), str(c).lower()) for n, c in body.truth_tuples]
            submitted: List[Tuple[int, str]] = []
            turns = 0
        else:
            truth = state.truth
            submitted = state.answers
            turns = state.turn

        exact_match = Counter(truth) == Counter(submitted)
        reward = 1.0 if exact_match else 0.0

        return PolygonVerifyResponse(
            **body.model_dump(),
            reward=reward,
            submitted_tuples=submitted,
            turns_completed=turns,
            exact_match=exact_match,
        )


if __name__ == "__main__":
    PolygonNamingResourcesServer.run_webserver()
