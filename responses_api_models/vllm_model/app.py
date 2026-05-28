# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import BaseModel, Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionDeveloperMessageParam,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageForTraining,
    NeMoGymChatCompletionMessageParam,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionToolParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionDefinition,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    TokenIDLogProbMixin,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)


def _debug_jsonl_path(component: str) -> Optional[Path]:
    debug_dir = os.environ.get("NEMO_RL_DEBUG_RESPONSES_PIPELINE_DIR")
    if not debug_dir:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{component}.jsonl"


def _preview_text(value: Any, limit: int = 500) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _seq_summary(values: Any, *, limit: int = 12) -> Dict[str, Any]:
    if values is None:
        return {"present": False, "len": 0}
    if not isinstance(values, list):
        return {"present": True, "type": type(values).__name__}
    return {
        "present": True,
        "len": len(values),
        "head": values[:limit],
        "tail": values[-limit:] if len(values) > limit else [],
    }


def _image_url_summary(image_url: Any) -> Dict[str, Any]:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str):
        return {"present": False, "type": type(image_url).__name__}
    return {
        "present": True,
        "len": len(image_url),
        "sha256_16": hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16],
        "prefix": image_url[:32],
    }


def _content_summary(content: Any) -> Any:
    if isinstance(content, str):
        return {"kind": "str", "len": len(content), "preview": _preview_text(content)}
    if not isinstance(content, list):
        return {"kind": type(content).__name__, "repr": _preview_text(content)}
    parts = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"kind": type(part).__name__, "repr": _preview_text(part)})
            continue
        part_type = part.get("type")
        summary = {"type": part_type, "keys": sorted(part.keys())}
        if "text" in part:
            text = part.get("text")
            summary["text_len"] = len(text) if isinstance(text, str) else None
            summary["text_preview"] = _preview_text(text)
        if "image_url" in part:
            summary["image_url"] = _image_url_summary(part.get("image_url"))
        parts.append(summary)
    return {"kind": "list", "len": len(content), "parts": parts}


def _message_summary(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    if not isinstance(message, dict):
        return {"type": type(message).__name__, "repr": _preview_text(message)}
    return {
        "keys": sorted(message.keys()),
        "role": message.get("role"),
        "type": message.get("type"),
        "content": _content_summary(message.get("content")),
        "reasoning_len": len(message.get("reasoning") or "")
        if isinstance(message.get("reasoning"), str)
        else None,
        "reasoning_content_len": len(message.get("reasoning_content") or "")
        if isinstance(message.get("reasoning_content"), str)
        else None,
        "prompt_token_ids": _seq_summary(message.get("prompt_token_ids")),
        "generation_token_ids": _seq_summary(message.get("generation_token_ids")),
        "generation_log_probs": _seq_summary(message.get("generation_log_probs")),
    }


def _output_item_summary(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"type": type(item).__name__, "repr": _preview_text(item)}
    summary = _message_summary(item)
    summary["id"] = item.get("id")
    if "summary" in item:
        summary["summary"] = _content_summary(
            [
                {"type": s.get("type"), "text": s.get("text")}
                for s in item.get("summary", [])
                if isinstance(s, dict)
            ]
        )
    return summary


def _logprobs_summary(logprobs_content: Any) -> Dict[str, Any]:
    if not isinstance(logprobs_content, list):
        return {"present": logprobs_content is not None, "type": type(logprobs_content).__name__}
    selected = logprobs_content[:5] + (logprobs_content[-5:] if len(logprobs_content) > 5 else [])
    return {
        "present": True,
        "len": len(logprobs_content),
        "sample": [
            {
                "token": entry.get("token") if isinstance(entry, dict) else None,
                "logprob": entry.get("logprob") if isinstance(entry, dict) else None,
            }
            for entry in selected
        ],
    }


def _debug_dump(component: str, event: str, payload: Dict[str, Any]) -> None:
    path = _debug_jsonl_path(component)
    if path is None:
        return
    row = {"event": event, "created_at": time(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool
    max_input_tokens: Optional[int] = None

    uses_reasoning_parser: bool
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class VLLMModel(SimpleResponsesAPIModel):
    config: VLLMModelConfig

    def get_converter(self) -> "VLLMConverter":
        """Return the converter used for Responses API <-> Chat Completions mapping.

        Override in subclasses (e.g. GenRMModel) to use a specialized converter.
        """
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
        )

    def model_post_init(self, context):
        self._post_init()
        return super().model_post_init(context)

    def _post_init(self) -> None:
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()

        self._converter = self.get_converter()

    def _create_context_length_exceeded_chat_completion(
        self, prompt_token_ids: Optional[List[int]] = None
    ) -> NeMoGymChatCompletion:
        message_kwargs = dict(
            role="assistant",
            content=None,
            tool_calls=None,
        )
        if self.config.return_token_id_information and prompt_token_ids is not None:
            message = NeMoGymChatCompletionMessageForTraining(
                **message_kwargs,
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=[],
                generation_log_probs=[],
            )
        else:
            message = NeMoGymChatCompletionMessage(**message_kwargs)

        return NeMoGymChatCompletion(
            id="chtcmpl-context-length-exceeded",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="context_length_exceeded",
                    message=message,
                )
            ],
            context_length_exceeded=True,
        )

    @staticmethod
    def _get_tokenize_body_dict(body_dict: Dict[str, Any]) -> Dict[str, Any]:
        tokenize_body_dict = {}
        for key in ("model", "messages", "tools", "chat_template_kwargs", "mm_processor_kwargs"):
            if key in body_dict:
                tokenize_body_dict[key] = body_dict[key]
        return tokenize_body_dict

    async def _get_prompt_token_ids(
        self, client: NeMoGymAsyncOpenAI, body_dict: Dict[str, Any]
    ) -> List[int]:
        tokenize_response = await self._get_tokenize_response(client, body_dict)
        return tokenize_response["tokens"]

    async def _get_tokenize_response(
        self, client: NeMoGymAsyncOpenAI, body_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        tokenize_response = await client.create_tokenize(
            **self._get_tokenize_body_dict(body_dict)
        )
        return tokenize_response

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.is_responses_native:
            return await self._responses_native(request, body)

        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        choice = chat_completion_response.choices[0]

        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]
        _debug_dump(
            "gym_vllm_model",
            "responses_postprocess_output",
            {
                "response_id": chat_completion_response.id,
                "finish_reason": choice.finish_reason,
                "output": [_output_item_summary(item) for item in response_output_dicts],
                "usage": chat_completion_response.usage.model_dump()
                if chat_completion_response.usage and hasattr(chat_completion_response.usage, "model_dump")
                else chat_completion_response.usage,
            },
        )

        usage = None
        if chat_completion_response.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion_response.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion_response.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion_response.usage.prompt_tokens
                + chat_completion_response.usage.completion_tokens,
            )

        response_metadata = deepcopy(body.metadata) if body.metadata else None
        if chat_completion_response.context_length_exceeded:
            response_metadata = response_metadata or {}
            response_metadata["context_length_exceeded"] = "true"

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if "tool_choice" in body else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=response_metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """
        The following config parameters are effectively no-ops with Responses native models:
        - uses_reasoning_parser: bool (Not applicable)
        """
        # The following parameters could be supported, but have not been supported yet for Responses-native models:
        if self.config.return_token_id_information:
            raise NotImplementedError
        if self.config.replace_developer_role_with_system:
            raise NotImplementedError
        if not self.config.sequential_reasoning_allowed:
            raise NotImplementedError

        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model
        if self.config.chat_template_kwargs:
            body_dict["chat_template_kwargs"] = deepcopy(self.config.chat_template_kwargs)
        if self.config.extra_body:
            body_dict = self.config.extra_body | body_dict

        client = self._resolve_client(request)
        response_dict = await client.create_response(**body_dict)

        return NeMoGymResponse.model_validate(response_dict)

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the body dict before issuing a chat completion request.

        Subclasses can override this to apply model-specific transformations
        (e.g. role remapping, extra sampling params).  The base implementation
        handles the features driven by ``VLLMModelConfig``.

        Args:
            request: The originating FastAPI request (available for session /
                client resolution if needed by subclasses).
            body_dict: Mutable dict produced by ``body.model_dump(exclude_unset=True)``.

        Returns:
            The (possibly mutated) ``body_dict`` that will be forwarded to
            ``client.create_chat_completion``.
        """
        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata") or dict()

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs", "{}")
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        metadata_extra_body_str = metadata.get("extra_body", "{}")
        extra_body.update(json.loads(metadata_extra_body_str))

        if self.config.return_token_id_information:
            body_dict |= dict(
                logprobs=True,
                # Typically passed via OpenAI client extra_body.
                return_tokens_as_token_ids=True,
                # For prompt and generation token IDs.
                return_token_ids=True,
            )

        if self.config.uses_reasoning_parser:
            # Keep historical assistant turns as literal text, including any
            # <think>...</think> tags. Splitting history back into reasoning
            # fields changes chat-template whitespace and breaks prefix
            # comparisons for generated turns.
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue

                content = message_dict["content"]
                if isinstance(content, str) or not content:
                    continue
                elif isinstance(content, list):
                    message_dict["content"] = "".join(part.get("text", "") for part in content)
                else:
                    raise NotImplementedError

        if extra_body:
            body_dict = extra_body | body_dict

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        client = self._resolve_client(request)
        prompt_token_ids: Optional[List[int]] = None
        vllm_max_model_len: Optional[int] = None

        should_tokenize_prompt = self.config.max_input_tokens is not None
        if should_tokenize_prompt:
            tokenize_response = await self._get_tokenize_response(client, body_dict)
            prompt_token_ids = tokenize_response["tokens"]
            if tokenize_response.get("max_model_len") is not None:
                vllm_max_model_len = int(tokenize_response["max_model_len"])

        max_input_tokens = self.config.max_input_tokens
        if vllm_max_model_len is not None:
            max_input_tokens = (
                vllm_max_model_len
                if max_input_tokens is None
                else min(max_input_tokens, vllm_max_model_len)
            )

        _debug_dump(
            "gym_vllm_model",
            "chat_request_pre_vllm",
            {
                "server_name": self.config.name,
                "body_keys": sorted(body_dict.keys()),
                "message_count": len(body_dict.get("messages", [])),
                "messages": [_message_summary(message) for message in body_dict.get("messages", [])],
                "chat_template_kwargs": body_dict.get("chat_template_kwargs"),
                "return_token_id_information": self.config.return_token_id_information,
                "prompt_token_ids": _seq_summary(prompt_token_ids),
                "max_input_tokens": max_input_tokens,
                "max_tokens": body_dict.get("max_tokens"),
            },
        )

        if max_input_tokens is not None and prompt_token_ids is not None:
            prompt_len = len(prompt_token_ids)
            if prompt_len >= max_input_tokens:
                return self._create_context_length_exceeded_chat_completion(
                    prompt_token_ids
                )

            remaining_budget = max_input_tokens - prompt_len
            requested_max_tokens = body_dict.get("max_tokens")
            body_dict["max_tokens"] = (
                remaining_budget
                if requested_max_tokens is None
                else min(requested_max_tokens, remaining_budget)
            )
            if requested_max_tokens != body_dict["max_tokens"]:
                LOGGER.info(
                    "Clamped vLLM max_tokens from %s to %s for prompt_len=%s max_input_tokens=%s",
                    requested_max_tokens,
                    body_dict["max_tokens"],
                    prompt_len,
                    max_input_tokens,
                )

        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (
                self._has_visible_assistant_content(last_message) or last_message.get("tool_calls")
            ):
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "content_filter"
                return res

        try:
            chat_completion_dict = await client.create_chat_completion(**body_dict)
        except ClientResponseError as e:
            """
            Example messages for out of context length:

            1. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L914
            ```json
            {"object":"error","message":"This model\'s maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages, Please reduce the length of the messages. None","type":"BadRequestError","param":null,"code":400}
            ```
            2. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L940
            3. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L948
            4. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/sampling_params.py#L463
            """
            result_content_str = e.response_content.decode()

            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                res = self._create_empty_chat_completion()
                res.choices[0].finish_reason = "length"
                return res
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
        _debug_dump(
            "gym_vllm_model",
            "raw_chat_completion",
            {
                "server_name": self.config.name,
                "chat_completion_keys": sorted(chat_completion_dict.keys()),
                "choice_keys": sorted(choice_dict.keys()),
                "finish_reason": choice_dict.get("finish_reason"),
                "message": _message_summary(choice_dict.get("message", {})),
                "logprobs": _logprobs_summary(
                    (choice_dict.get("logprobs") or {}).get("content")
                    if isinstance(choice_dict.get("logprobs"), dict)
                    else None
                ),
                "usage": chat_completion_dict.get("usage"),
            },
        )
        if self.config.uses_reasoning_parser:
            # See the TODO wrt reasoning_content above
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                # See the TODO wrt reasoning_content above
                choice_dict["message"].pop("reasoning", None)

                # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"]["content"] or "")
        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self.config.return_token_id_information:
            log_probs = (choice_dict.get("logprobs") or {}).get("content") or []
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            generation_token_ids = choice_dict.get("token_ids")
            if generation_token_ids is None:
                # Fallback for older vLLM responses that only expose token IDs
                # as synthetic logprob token strings like `"token_id:151667"`.
                generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            prompt_token_ids_for_message = chat_completion_dict.get("prompt_token_ids")
            if prompt_token_ids_for_message is None:
                # The tokenize endpoint doesn't accept sampling parameters.
                # Preserve chat-template knobs so fallback prompt IDs match
                # the generation request as closely as possible.
                tokenize_body_dict = dict()
                for key in ("model", "messages", "tools", "chat_template_kwargs"):
                    if key in body_dict:
                        tokenize_body_dict[key] = body_dict[key]

                # The base URL has /v1 at the end but vLLM's tokenize endpoint
                # does not have v1, hence the client shim routes to ../tokenize.
                tokenize_response = await client.create_tokenize(**tokenize_body_dict)
                prompt_token_ids_for_message = tokenize_response["tokens"]

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    prompt_token_ids=prompt_token_ids_for_message,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

            # Clean the duplicated information
            choice_dict.pop("logprobs", None)
            chat_completion_dict.pop("prompt_token_ids", None)
            choice_dict.pop("token_ids", None)

        _debug_dump(
            "gym_vllm_model",
            "chat_completion_after_token_metadata",
            {
                "server_name": self.config.name,
                "finish_reason": choice_dict.get("finish_reason"),
                "message": _message_summary(choice_dict.get("message", {})),
            },
        )

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    def _create_empty_chat_completion(self) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion(
            id="chtcmpl-123",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client

    def _has_visible_assistant_content(self, message: Dict[str, Any]) -> bool:
        content = message.get("content")
        if not content:
            return False
        if isinstance(content, str):
            _, remaining_content = self._converter._extract_reasoning_from_content(content)
            return bool(remaining_content)
        return True


class VLLMConverterResponsesToChatCompletionsState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    # We are mapping from Response input items to chat completions messages, which is many to one.
    # Our state will accumulate the reasoning, chat, and tool calls for assistant messages.
    content_buffer: str = ""  # Buffer for reasoning and chat
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)

    # Will only be populated if return_token_id_information is True.
    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.content_buffer or self.tool_calls_buffer):
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
            tool_calls=self.tool_calls_buffer,
        )

        # We check here that self.token_information is non-empty since it's possible that some assistant messages are entirely inputs and are not generated by the model in this trajectory.
        if self.return_token_id_information and self.token_information:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.tool_calls_buffer = []


class VLLMConverter(BaseModel):
    return_token_id_information: bool

    # =======================================================
    # Reasoning handling. This may change across models and model families
    # =======================================================

    THINK_TAG_PATTERN: ClassVar = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>\n{t.lstrip()}</think>" for t in texts if t)

    @classmethod
    def _parse_think_tags(cls, content: str) -> Tuple[List[str], str]:
        # Extract reasoning content from between <think></think> tags.
        matches = cls.THINK_TAG_PATTERN.findall(content)
        # Remove reasoning from main content
        cleaned = cls.THINK_TAG_PATTERN.sub("", content)
        return matches, cleaned

    # =======================================================
    # Response create params to Chat Completion create params
    # =======================================================

    def responses_to_chat_completion_create_params(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        responses_create_params = responses_create_params.model_dump(exclude_unset=True)

        # Tracks messages including reasoning for each respective message type helper function
        state = VLLMConverterResponsesToChatCompletionsState(
            return_token_id_information=self.return_token_id_information
        )

        # Input can be a string. Wrap in a ResponseInput-like
        response_input = responses_create_params["input"]
        if isinstance(response_input, str):
            wrapped_input = {
                "content": [
                    {
                        "text": response_input,
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
            input_messages = [wrapped_input]
        else:
            input_messages = responses_create_params.pop("input", [])

        for m in input_messages:
            if not m.get("type") and m.get("role"):
                m["type"] = "message"

            match m["type"]:
                case "message":
                    self._format_message(m, state)
                case "reasoning":
                    self._format_reasoning(m, state)
                case "function_call":
                    self._format_function_call(m, state)
                case "function_call_output":
                    self._format_function_call_output(m, state)
                case _:  # pragma: no cover
                    raise NotImplementedError(f"Unsupported message type: {m}")

            if self.return_token_id_information and m.get("prompt_token_ids"):
                state.token_information = TokenIDLogProbMixin(
                    prompt_token_ids=m["prompt_token_ids"],
                    generation_token_ids=m["generation_token_ids"],
                    generation_log_probs=m["generation_log_probs"],
                )

        state.flush_assistant()

        model = responses_create_params.pop("model", None)
        if model is not None:
            responses_create_params["model"] = model

        # The corresponding parameter to `max_output_tokens`` is `max_tokens`
        max_output_tokens = responses_create_params.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            responses_create_params["max_tokens"] = max_output_tokens

        tools = responses_create_params.pop("tools", None)
        if tools is not None:
            chat_completion_tools = []
            for tool_dict in tools:
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)

                # As of vLLM 0.17.1, vLLM Chat Completions does not accept this `strict` parameter on tool definitions that OpenAI accepts.
                tool_dict.pop("strict", None)
                chat_completion_tools.append(
                    NeMoGymChatCompletionToolParam(type="function", function=NeMoGymFunctionDefinition(**tool_dict))
                )
            if chat_completion_tools:
                responses_create_params["tools"] = chat_completion_tools

        chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=state.messages,
            **responses_create_params,
        )

        return chat_completion_create_params

    def _format_function_call_output(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        state.flush_assistant()

        assert "call_id" in m
        converted = NeMoGymChatCompletionToolMessageParam(
            content=m["output"],
            role="tool",
            tool_call_id=m["call_id"],
        )
        state.messages.append(converted)

    def _format_message(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        content = m["content"]

        if isinstance(content, list) and m["role"] != "assistant":
            converted_parts = []
            for part_param in content:
                match part_param["type"]:
                    case "input_text":
                        converted_parts.append({"type": "text", "text": part_param["text"]})
                    case "input_image":
                        image_url = part_param.get("image_url", "")
                        detail = part_param.get("detail", "auto")
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                        )
                    case _:
                        raise NotImplementedError(f"Unsupported part param type: {part_param['type']}")
            content = converted_parts
            m["content"] = content

        match m["role"]:
            case "assistant":
                # Handle reasoning
                final_content = ""
                if isinstance(m["content"], list):
                    content_str = "".join([part.get("text", "") for part in m["content"]])
                    final_content += content_str
                elif isinstance(m["content"], str):
                    final_content += m["content"]
                else:
                    raise NotImplementedError(
                        f"Expected m['content'] to be str or list[dict], but got {type(m['content']).__name__!r}: {m['content']!r}"
                    )

                converted = []
                state.content_buffer += final_content
            case "user":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionUserMessageParam(
                        content=content,
                        role="user",
                    )
                ]
            # TODO: Revisit this in case we need separate handling. Not all chat templates may support the 'developer' role.
            case "system":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionSystemMessageParam(
                        content=content,
                        role="system",
                    )
                ]
            case "developer":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionDeveloperMessageParam(
                        content=content,
                        role="developer",
                    )
                ]
            case _:  # pragma: no cover
                raise NotImplementedError(f"Unrecognized role for message: `{m['role']}`")

        state.messages.extend(converted)

    def _format_reasoning(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        """
        Collects text from 'reasoning' messages in responses api and appends it to a buffer.

        This is done to group together one (or multiple) reasoning message(s) into a single,
        cohesive block, later prepending it to a subsequent assistant message.
        See: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/how-to-faq.md#faq-openai-responses-vs-chat-completions-api for an example of reasoning in responses api.
        """
        if "summary" in m and m["summary"]:
            texts = [s["text"] for s in m["summary"]]
            state.content_buffer += self._wrap_reasoning_in_think_tags(texts)

    def _format_function_call(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        assert "call_id" in m
        tool_call = NeMoGymChatCompletionMessageToolCallParam(
            id=m["call_id"],
            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                arguments=m["arguments"],
                name=m["name"],
            ),
            type="function",
        )
        state.tool_calls_buffer.append(tool_call)

    # =======================================================
    # Chat Completion to Response
    # =======================================================

    def postprocess_chat_response(self, choice: NeMoGymChoice) -> List[NeMoGymResponseOutputItem]:
        raw_message = choice.message.model_dump()
        response_output = self.postprocess_assistant_message_dict(raw_message)
        _debug_dump(
            "gym_vllm_model",
            "converter_postprocess_chat_response",
            {
                "choice_finish_reason": choice.finish_reason,
                "raw_message": _message_summary(raw_message),
                "output": [_output_item_summary(item) for item in response_output],
            },
        )
        return response_output

    def postprocess_assistant_message_dict(self, message_dict: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        response_output = []

        content = message_dict.get("content") or ""
        reasoning_matches, content = self._extract_reasoning_from_content(content)
        if reasoning_matches:
            reasoning_item = NeMoGymResponseReasoningItem(
                id=f"rs_{uuid4().hex}",
                type="reasoning",
                summary=[
                    NeMoGymSummary(text=reasoning_text, type="summary_text") for reasoning_text in reasoning_matches
                ],
                status="completed",
            )
            response_output.append(reasoning_item)

        tool_calls_raw = message_dict.get("tool_calls", []) or []
        # We need to return at least one output item. When the model decides to just stop with no chat or tool calls
        # We just add an output item with empty or null content here. This is prevalent e.g. in the case of base models that may not be the most reliable since they have not been instruction tuned.
        has_empty_output = not (response_output or tool_calls_raw)

        if content or has_empty_output:
            response_output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role=message_dict.get("role"),
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=content,
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                )
            )

        for tc in tool_calls_raw:
            assert "id" in tc
            response_output.append(
                NeMoGymResponseFunctionToolCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    call_id=tc["id"],
                    type="function_call",
                    status="completed",
                    id=tc["id"],
                )
            )

        # `"prompt_token_ids" in raw_message`: sometimes the model endpoint may go out of context length, in which case we return an empty response
        # In these cases, there are no token id information provided.
        if self.return_token_id_information and "prompt_token_ids" in message_dict:
            last_response_output_item = response_output[-1]
            train_cls = RESPONSES_TO_TRAIN[last_response_output_item.__class__]
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                prompt_token_ids=message_dict["prompt_token_ids"],
                generation_token_ids=message_dict["generation_token_ids"],
                generation_log_probs=message_dict["generation_log_probs"],
            )

        return response_output

    def _extract_reasoning_from_content(self, content: str) -> Tuple[List[str], str]:
        # TODO: Currently only parses reasoning wrapped in <think>...</think> tags.
        # Maybe parameterize to support other model formats in the future.
        return self._parse_think_tags(content)

    def chat_completions_messages_to_responses_items(
        self, messages: List[Dict[str, Any]]
    ) -> List[NeMoGymResponseOutputItem]:
        output_items = []

        for message in messages:
            role = message["role"]
            if role in ("user", "system", "developer"):
                # vLLM may return None content
                if message["content"] is None:
                    message["content"] = ""
                output_items.append(NeMoGymEasyInputMessage.model_validate(message))
            elif role == "assistant":
                output_items.extend(self.postprocess_assistant_message_dict(message))
            elif role == "tool":
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                        status="completed",
                    )
                )
            else:
                raise NotImplementedError(f"Unrecognized role: {role}!")

        return output_items


def split_responses_input_output_items(
    items: List[NeMoGymResponseOutputItem],
) -> Tuple[List[NeMoGymResponseOutputItem], List[NeMoGymResponseOutputItem]]:
    if not items:
        return [], []

    for i, item in enumerate(items):
        if (
            getattr(item, "role", None) == "assistant"
            or getattr(item, "type", None)
            in {
                "reasoning",
                "reasoning_item",
            }
            or getattr(item, "type", None) in ("function_call",)
        ):
            break

    return items[:i], items[i:]


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModel.run_webserver()  # noqa: F401
