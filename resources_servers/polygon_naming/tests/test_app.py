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
"""Tests for the polygon_naming resources server.

Exercises the full `/seed_session → /submit_turn (×2) → /verify` flow,
plus the multiset semantics of the reward and error paths.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.polygon_naming.app import (
    PolygonNamingConfig,
    PolygonNamingResourcesServer,
    _pairs_from_wire,
)


_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
)


def _make_client() -> TestClient:
    config = PolygonNamingConfig(host="0.0.0.0", port=8080, entrypoint="", name="polygon_naming")
    server = PolygonNamingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver())


def _seed_payload(
    truth_tuples: list[list],
    turn_1_images: list[str] | None = None,
    turn_2_images: list[str] | None = None,
) -> dict:
    return {
        "truth_tuples": truth_tuples,
        "turn_1_images": turn_1_images or [_TINY_PNG_DATA_URL, _TINY_PNG_DATA_URL],
        "turn_2_images": turn_2_images or [_TINY_PNG_DATA_URL],
    }


def _verify_payload(truth_tuples: list[list]) -> dict:
    return {
        "responses_create_params": {"input": []},
        "response": {
            "id": "resp",
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
        "truth_tuples": truth_tuples,
        "turn_1_images": [],
        "turn_2_images": [],
    }


class TestPairsFromWire:
    def test_normalizes_colour_case(self) -> None:
        assert _pairs_from_wire([[3, "Red"], [4, "GREEN"]]) == [(3, "red"), (4, "green")]

    def test_coerces_string_ints(self) -> None:
        assert _pairs_from_wire([["3", "red"]]) == [(3, "red")]

    def test_skips_malformed(self) -> None:
        assert _pairs_from_wire([[3], "junk", [None, "red"], [4, "green"]]) == [(4, "green")]

    def test_empty_input(self) -> None:
        assert _pairs_from_wire([]) == []
        assert _pairs_from_wire(None) == []


class TestFullFlow:
    def test_exact_match_yields_reward_1(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        seed_resp = client.post("/seed_session", json=_seed_payload(truth))
        assert seed_resp.status_code == 200
        body = seed_resp.json()
        # The seed envelope must inject a user message with 2 images.
        msgs = body["as_user_messages"]
        assert len(msgs) == 1
        content = msgs[0]["content"]
        types = [p["type"] for p in content]
        assert types == ["input_text", "input_image", "input_image"]

        turn1 = client.post("/submit_turn", json={"answers": [[3, "red"], [4, "green"]]})
        assert turn1.status_code == 200
        turn1_body = turn1.json()
        # Turn 1 response must reveal image 3 via as_user_messages.
        assert "as_user_messages" in turn1_body
        t1_content = turn1_body["as_user_messages"][0]["content"]
        t1_types = [p["type"] for p in t1_content]
        assert t1_types == ["input_text", "input_image"]

        turn2 = client.post("/submit_turn", json={"answers": [[6, "yellow"]]})
        assert turn2.status_code == 200
        turn2_body = turn2.json()
        # Turn 2 response is plain text — no image reveal.
        assert "as_user_messages" not in turn2_body
        assert "function_call_output" in turn2_body

        verify = client.post("/verify", json=_verify_payload(truth))
        assert verify.status_code == 200
        vbody = verify.json()
        assert vbody["reward"] == 1.0
        assert vbody["exact_match"] is True
        assert vbody["turns_completed"] == 2
        assert sorted(map(tuple, vbody["submitted_tuples"])) == sorted(
            [(3, "red"), (4, "green"), (6, "yellow")]
        )

    def test_permuted_answers_still_pass(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        # Order swapped inside a turn, and the two turns swapped as well.
        client.post("/submit_turn", json={"answers": [[4, "green"], [3, "red"]]})
        client.post("/submit_turn", json={"answers": [[6, "yellow"]]})

        verify = client.post("/verify", json=_verify_payload(truth))
        assert verify.json()["reward"] == 1.0

    def test_wrong_colour_yields_reward_0(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        client.post("/submit_turn", json={"answers": [[3, "red"], [4, "blue"]]})
        client.post("/submit_turn", json={"answers": [[6, "yellow"]]})

        vbody = client.post("/verify", json=_verify_payload(truth)).json()
        assert vbody["reward"] == 0.0
        assert vbody["exact_match"] is False

    def test_case_insensitive_colour(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        client.post("/submit_turn", json={"answers": [[3, "RED"], [4, "Green"]]})
        client.post("/submit_turn", json={"answers": [[6, "YELLOW"]]})

        assert client.post("/verify", json=_verify_payload(truth)).json()["reward"] == 1.0

    def test_missing_third_answer_yields_reward_0(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        client.post("/submit_turn", json={"answers": [[3, "red"], [4, "green"]]})
        client.post("/submit_turn", json={"answers": []})

        assert client.post("/verify", json=_verify_payload(truth)).json()["reward"] == 0.0

    def test_extra_answers_yield_reward_0(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        client.post("/submit_turn", json={"answers": [[3, "red"], [4, "green"], [5, "blue"]]})
        client.post("/submit_turn", json={"answers": [[6, "yellow"]]})

        assert client.post("/verify", json=_verify_payload(truth)).json()["reward"] == 0.0

    def test_extra_submit_turn_call_does_not_crash(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]

        client.post("/seed_session", json=_seed_payload(truth))
        client.post("/submit_turn", json={"answers": [[3, "red"], [4, "green"]]})
        client.post("/submit_turn", json={"answers": [[6, "yellow"]]})
        extra = client.post("/submit_turn", json={"answers": [[3, "red"]]})
        assert extra.status_code == 200
        # Still just a plain-text nudge — no image.
        assert "as_user_messages" not in extra.json()

    def test_verify_without_seed_returns_zero(self) -> None:
        client = _make_client()
        truth = [[3, "red"], [4, "green"], [6, "yellow"]]
        vbody = client.post("/verify", json=_verify_payload(truth)).json()
        assert vbody["reward"] == 0.0
        assert vbody["turns_completed"] == 0
        assert vbody["submitted_tuples"] == []


class TestSessionIsolation:
    def test_two_clients_do_not_collide(self) -> None:
        client_a = _make_client()
        client_b = _make_client()
        truth_a = [[3, "red"], [4, "green"], [5, "blue"]]
        truth_b = [[6, "yellow"], [3, "magenta"], [4, "red"]]

        client_a.post("/seed_session", json=_seed_payload(truth_a))
        client_b.post("/seed_session", json=_seed_payload(truth_b))
        client_a.post("/submit_turn", json={"answers": [[3, "red"], [4, "green"]]})
        client_b.post("/submit_turn", json={"answers": [[6, "yellow"], [3, "magenta"]]})
        client_a.post("/submit_turn", json={"answers": [[5, "blue"]]})
        client_b.post("/submit_turn", json={"answers": [[4, "red"]]})

        assert client_a.post("/verify", json=_verify_payload(truth_a)).json()["reward"] == 1.0
        assert client_b.post("/verify", json=_verify_payload(truth_b)).json()["reward"] == 1.0
