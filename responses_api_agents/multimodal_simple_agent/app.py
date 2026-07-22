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
"""Multimodal-aware SimpleAgent.

Extends `simple_agent` so any resources-server endpoint (`/seed_session`,
tool calls, ...) may return a JSON envelope that injects one or more
`NeMoGymEasyInputMessage` items — with `input_image` content parts —
into the model's trajectory, in addition to (or instead of) the standard
text-only `function_call_output`.

Envelope shape:

    {
      "function_call_output": "ok",          # optional; ack for tool calls
      "as_user_messages": [                    # or singular "as_user_message"
        {"role": "user", "content": [
          {"type": "input_text",  "text": "..."},
          {"type": "input_image", "image_url": "data:image/png;base64,..."}
        ]},
        ...
      ]
    }

Backward-compatible: if the response is not JSON, or the JSON does not
carry `as_user_message(s)`, the body is forwarded as a plain
`function_call_output`, matching legacy `simple_agent` behaviour.
"""
import json
from typing import Any, List, Union

from fastapi import Request, Response
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
    SimpleAgentVerifyResponse,
)


class MultimodalSimpleAgentConfig(SimpleAgentConfig):
    pass


InjectedOutput = Union[NeMoGymFunctionCallOutput, NeMoGymEasyInputMessage]


def _envelope_messages(body_json: Any) -> list[dict] | None:
    """Return the list of user-message dicts carried by an envelope body, or
    None if the body is not an envelope."""
    if not isinstance(body_json, dict):
        return None
    if "as_user_messages" in body_json:
        raw = body_json["as_user_messages"]
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return raw
        return None
    if "as_user_message" in body_json:
        raw = body_json["as_user_message"]
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return raw
        return None
    return None


def _decode_body(raw_bytes: bytes, content_type: str) -> tuple[str, Any]:
    """Return (raw_text, parsed_json_or_None). parsed_json is only non-None
    when the content-type advertises JSON and the body decodes cleanly."""
    text = raw_bytes.decode(errors="replace")
    if "application/json" not in (content_type or "").lower():
        return text, None
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        return text, None


class MultimodalSimpleAgent(SimpleAgent):
    config: MultimodalSimpleAgentConfig

    async def _tool_response_items(
        self,
        api_response,
        call_id: str,
    ) -> list[InjectedOutput]:
        """Turn a resources-server tool response into trajectory items."""
        raw_bytes = await api_response.content.read()
        content_type = api_response.headers.get("Content-Type", "")
        text, parsed = _decode_body(raw_bytes, content_type)

        msgs = _envelope_messages(parsed) if parsed is not None else None
        if msgs is None:
            # Legacy path: forward the body verbatim as function_call_output.
            return [
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=text,
                )
            ]

        ack = parsed.get("function_call_output", "OK") if isinstance(parsed, dict) else "OK"
        emitted: list[InjectedOutput] = [
            NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=str(ack),
            )
        ]
        for m in msgs:
            emitted.append(NeMoGymEasyInputMessage.model_validate(m))
        return emitted

    def _seed_session_messages(
        self, raw_bytes: bytes, content_type: str
    ) -> list[NeMoGymEasyInputMessage]:
        """Extract user messages from a /seed_session response envelope.

        Returns [] when the response body is not an envelope, keeping
        legacy `simple_agent` behaviour for existing benches.
        """
        _, parsed = _decode_body(raw_bytes, content_type)
        msgs = _envelope_messages(parsed) if parsed is not None else None
        if not msgs:
            return []
        return [NeMoGymEasyInputMessage.model_validate(m) for m in msgs]

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs: list = []
        usage = None
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if o.type == "function_call"
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                    )
                    new_outputs.append(tool_response)
                    continue

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=parsed_arguments,
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                try:
                    injected = await self._tool_response_items(
                        api_response, output_function_call.call_id
                    )
                except ValidationError as e:
                    # The resources server sent a malformed envelope. Surface
                    # the error to the model as a function_call_output so the
                    # loop can continue rather than crash the batch.
                    injected = [
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=json.dumps(
                                {"error": f"Invalid tool envelope: {e!r}"}
                            ),
                        )
                    ]
                new_outputs.extend(injected)

            if self.config.max_steps and step >= self.config.max_steps:
                break

        for k, v in (
            *resources_server_cookies.items(),
            *model_server_cookies.items(),
        ):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        # Envelope from /seed_session: prepend messages to the first
        # /v1/responses call. This is how a benchmark can hand the model
        # its opening (multimodal) user turn without pre-baking it into
        # the JSONL row.
        seed_bytes = await seed_session_response.content.read()
        seed_ctype = seed_session_response.headers.get("Content-Type", "")
        seed_msgs = self._seed_session_messages(seed_bytes, seed_ctype)

        responses_body = body.responses_create_params
        if seed_msgs:
            responses_body = responses_body.model_copy(deep=True)
            if isinstance(responses_body.input, str):
                responses_body.input = [
                    NeMoGymEasyInputMessage(role="user", content=responses_body.input)
                ]
            responses_body.input = list(responses_body.input) + list(seed_msgs)

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=responses_body,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        response_json = await get_response_json(response)

        # Prepend seed_msgs to `response.output` so any downstream consumer
        # that walks the output list (training-framework trajectory
        # reconstruction, verifiers, per-turn image binning) sees the full
        # trajectory the model actually saw. Without this, the seed user
        # turn is only present in the tokenized prompt (via
        # `responses_body.input`) but absent from the returned `output`,
        # leading to a mismatch between the model-input token stream (which
        # contains seed image placeholder tokens) and any per-turn tensor
        # extraction (which walks `output` and misses those images).
        #
        # `_tool_response_items` already emits `NeMoGymEasyInputMessage`
        # instances into `output` for mid-rollout env injections, so
        # user-role message items in `output` are a supported shape —
        # seed_msgs are the same shape from `_seed_session_messages`.
        if seed_msgs:
            response_json.setdefault("output", [])
            response_json["output"] = [
                m.model_dump() for m in seed_msgs
            ] + response_json["output"]

        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response_json}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))


__all__ = [
    "MultimodalSimpleAgent",
    "MultimodalSimpleAgentConfig",
    "InjectedOutput",
]


if __name__ == "__main__":
    MultimodalSimpleAgent.run_webserver()
