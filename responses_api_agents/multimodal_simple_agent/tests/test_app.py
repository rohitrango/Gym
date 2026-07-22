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
"""Tests for MultimodalSimpleAgent.

Focus on the *new* behaviour compared to SimpleAgent: turning tool /
seed_session responses into user messages with `input_image` content
parts. Baseline SimpleAgent behaviour is already covered by the parent
package's tests.
"""
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.multimodal_simple_agent.app import (
    MultimodalSimpleAgent,
    MultimodalSimpleAgentConfig,
    _envelope_messages,
)
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
)


_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
)


def _make_server() -> MultimodalSimpleAgent:
    config = MultimodalSimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(type="responses_api_models", name="model_server"),
        resources_server=ResourcesServerRef(type="resources_servers", name="res_server"),
        max_steps=4,
    )
    return MultimodalSimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _tool_call_output(name: str, call_id: str, args: dict) -> dict:
    return {
        "id": f"resp_{call_id}",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(args),
                "type": "function_call",
                "status": "completed",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _assistant_message_output(text: str) -> dict:
    return {
        "id": "resp_final",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": "msg_final",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class _FakeToolReply:
    """Mimics the aiohttp response object the agent reads from."""

    def __init__(self, body: bytes | str, content_type: str) -> None:
        self._body = body.encode() if isinstance(body, str) else body
        self.headers = {"Content-Type": content_type}
        self.cookies = MagicMock()
        self.content = AsyncMock()
        self.content.read = AsyncMock(return_value=self._body)


class TestEnvelopeHelper:
    def test_none_for_non_dict(self) -> None:
        assert _envelope_messages("hi") is None
        assert _envelope_messages(None) is None
        assert _envelope_messages([{"role": "user"}]) is None

    def test_none_when_key_missing(self) -> None:
        assert _envelope_messages({"function_call_output": "ok"}) is None

    def test_singular_dict(self) -> None:
        msg = {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        assert _envelope_messages({"as_user_message": msg}) == [msg]

    def test_plural_list(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "input_text", "text": "a"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "b"}]},
        ]
        assert _envelope_messages({"as_user_messages": msgs}) == msgs

    def test_plural_dict_is_wrapped(self) -> None:
        msg = {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        assert _envelope_messages({"as_user_messages": msg}) == [msg]


class TestToolResponseInjection:
    async def test_legacy_string_tool_response(self) -> None:
        """Non-JSON tool response ⇒ single function_call_output, no user msg."""
        server = _make_server()
        tool_call_resp = _tool_call_output("do_thing", "c1", {"x": 1})
        final_resp = _assistant_message_output("done")

        tool_reply = _FakeToolReply("plain-text-body", content_type="text/plain")

        model_reply = AsyncMock()
        model_reply.read = AsyncMock(side_effect=[
            json.dumps(tool_call_resp),
            json.dumps(final_resp),
        ])
        model_reply.cookies = MagicMock()

        async def post_side_effect(**kwargs):
            if kwargs["server_name"] == server.config.resources_server.name:
                return tool_reply
            return _model_shim(model_reply)

        server.server_client.post = AsyncMock(side_effect=post_side_effect)
        app = server.setup_webserver()
        client = TestClient(app)

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "go"}]})
        assert res.status_code == 200

        # The second model call must have exactly the original function_call +
        # a plain-text function_call_output — no NeMoGymEasyInputMessage.
        second_input = server.server_client.post.call_args_list[1].kwargs["json"].input
        tool_outputs = [
            item for item in second_input if isinstance(item, NeMoGymFunctionCallOutput)
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0].output == "plain-text-body"
        injected_msgs = [
            item
            for item in second_input
            if isinstance(item, NeMoGymEasyInputMessage) and item.role == "user"
            # skip the seed user turn
            and item.content != "go"
        ]
        assert injected_msgs == []

    async def test_envelope_injects_user_message_with_image(self) -> None:
        """JSON envelope ⇒ ack + user message with input_image content part."""
        server = _make_server()
        tool_call_resp = _tool_call_output("submit_turn", "c1", {"answers": [[3, "red"]]})
        final_resp = _assistant_message_output("done")

        envelope = {
            "function_call_output": "recorded",
            "as_user_messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Now this one:"},
                        {"type": "input_image", "image_url": _TINY_PNG_DATA_URL},
                    ],
                }
            ],
        }
        tool_reply = _FakeToolReply(json.dumps(envelope), content_type="application/json")

        model_reply = AsyncMock()
        model_reply.read = AsyncMock(side_effect=[
            json.dumps(tool_call_resp),
            json.dumps(final_resp),
        ])
        model_reply.cookies = MagicMock()

        async def post_side_effect(**kwargs):
            if kwargs["server_name"] == server.config.resources_server.name:
                return tool_reply
            return _model_shim(model_reply)

        server.server_client.post = AsyncMock(side_effect=post_side_effect)
        app = server.setup_webserver()
        client = TestClient(app)

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "start"}]})
        assert res.status_code == 200

        second_input = server.server_client.post.call_args_list[1].kwargs["json"].input
        # Exactly one function_call_output, and it carries the envelope's ack.
        tool_outputs = [
            item for item in second_input if isinstance(item, NeMoGymFunctionCallOutput)
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0].output == "recorded"

        injected_user_msgs = [
            item
            for item in second_input
            if isinstance(item, NeMoGymEasyInputMessage)
            and item.role == "user"
            and not isinstance(item.content, str)
        ]
        assert len(injected_user_msgs) == 1
        content = injected_user_msgs[0].content
        types = [part["type"] for part in content]
        assert types == ["input_text", "input_image"]
        assert content[1]["image_url"] == _TINY_PNG_DATA_URL

    async def test_envelope_default_ack(self) -> None:
        server = _make_server()
        tool_call_resp = _tool_call_output("submit_turn", "c1", {"answers": []})
        final_resp = _assistant_message_output("done")
        envelope = {"as_user_message": {"role": "user", "content": "next!"}}
        tool_reply = _FakeToolReply(json.dumps(envelope), content_type="application/json")

        model_reply = AsyncMock()
        model_reply.read = AsyncMock(side_effect=[
            json.dumps(tool_call_resp),
            json.dumps(final_resp),
        ])
        model_reply.cookies = MagicMock()

        async def post_side_effect(**kwargs):
            if kwargs["server_name"] == server.config.resources_server.name:
                return tool_reply
            return _model_shim(model_reply)

        server.server_client.post = AsyncMock(side_effect=post_side_effect)
        app = server.setup_webserver()
        client = TestClient(app)
        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "start"}]})
        assert res.status_code == 200

        second_input = server.server_client.post.call_args_list[1].kwargs["json"].input
        tool_outputs = [
            item for item in second_input if isinstance(item, NeMoGymFunctionCallOutput)
        ]
        assert tool_outputs[0].output == "OK"

    async def test_json_body_without_envelope_key_is_forwarded_verbatim(self) -> None:
        server = _make_server()
        tool_call_resp = _tool_call_output("do_thing", "c1", {})
        final_resp = _assistant_message_output("done")
        body_json = {"unrelated": "value"}
        tool_reply = _FakeToolReply(json.dumps(body_json), content_type="application/json")

        model_reply = AsyncMock()
        model_reply.read = AsyncMock(side_effect=[
            json.dumps(tool_call_resp),
            json.dumps(final_resp),
        ])
        model_reply.cookies = MagicMock()

        async def post_side_effect(**kwargs):
            if kwargs["server_name"] == server.config.resources_server.name:
                return tool_reply
            return _model_shim(model_reply)

        server.server_client.post = AsyncMock(side_effect=post_side_effect)
        app = server.setup_webserver()
        client = TestClient(app)
        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "go"}]})
        assert res.status_code == 200

        second_input = server.server_client.post.call_args_list[1].kwargs["json"].input
        tool_outputs = [
            item for item in second_input if isinstance(item, NeMoGymFunctionCallOutput)
        ]
        assert tool_outputs[0].output == json.dumps(body_json)


def _model_shim(reply: AsyncMock):
    """Wrap a model-server reply so successive `server_client.post(...)`
    calls share the same `side_effect` sequence on `.read()` (matches how
    `get_response_json` unpacks model responses)."""
    shim = MagicMock()
    shim.headers = {"Content-Type": "application/json"}
    shim.cookies = reply.cookies
    shim.read = reply.read
    shim.ok = True
    return shim
