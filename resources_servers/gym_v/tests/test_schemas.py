# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from pydantic import ValidationError

from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVAgentVerifyResponse,
    GymVNeMoGymResponse,
    GymVSeedSessionRequest,
    GymVStepRequest,
    GymVTaskRow,
)


def _row(**overrides):
    row = {
        "env_id": "Games/FrozenLake-v0",
        "env_kwargs": {"size": 4, "num_holes": 3, "tile_size": 32},
        "seed": 1234,
        "task_id": "frozenlake_4x4_h3_seed1234_stage1",
        "act_grammar_regex": r"^\[(up|down|left|right|w|a|s|d)\]$",
        "horizon_cap": 30,
        "task_metadata": {"difficulty_stage": 1, "category": "Games"},
        "responses_create_params": {
            "model": "policy_model",
            "input": [],
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "tools": [],
        },
    }
    row.update(overrides)
    return row


def test_task_row_round_trips_frozenlake_worked_example() -> None:
    parsed = GymVTaskRow.model_validate(_row())

    dumped = parsed.model_dump(mode="json")
    reparsed = GymVTaskRow.model_validate(dumped)

    assert reparsed.env_id == "Games/FrozenLake-v0"
    assert reparsed.responses_create_params.input == []
    assert reparsed.responses_create_params.tools == [], (
        "Path B contract: per-row tools list must be empty (no act schema sent)."
    )


def test_task_row_round_trips_maze_qa_worked_example() -> None:
    parsed = GymVTaskRow.model_validate(
        _row(
            env_id="Puzzles/Maze-QA-v0",
            env_kwargs={"size": "small", "cell_size": 40, "question_type": None},
            seed=1234,
            task_id="maze_qa_small_seed1234_stage1",
            act_grammar_regex=r"^([A-H]|-?\d+)$",
            horizon_cap=1,
            task_metadata={"difficulty_stage": 1, "category": "Puzzles"},
        )
    )

    assert parsed.env_id == "Puzzles/Maze-QA-v0"
    assert parsed.horizon_cap == 1


def test_step_request_requires_action_string() -> None:
    """Path A's `tool_calls` transport was removed; `action_string` is now
    a required field on every /step request."""
    with pytest.raises(ValidationError):
        GymVStepRequest(env_id="env")


def test_step_request_rejects_extra_action_transport_fields() -> None:
    """Schema has extra='forbid'; legacy Path A `tool_calls` / `action`
    fields must be rejected as unknown rather than silently accepted."""
    with pytest.raises(ValidationError):
        GymVStepRequest.model_validate(
            {
                "env_id": "env",
                "action_string": "[right]",
                "tool_calls": [],
            }
        )


def test_step_request_accepts_path_b_action_string() -> None:
    request = GymVStepRequest(env_id="env", action_string="[right]")

    assert request.action_string == "[right]"
    assert request.env_id == "env"


def test_seed_session_request_accepts_task_row_without_task_idx() -> None:
    request = GymVSeedSessionRequest(task_row=_row())

    assert request.task_idx is None
    assert request.task_row is not None
    assert request.task_row.seed == 1234


def test_seed_session_request_rejects_missing_selector() -> None:
    with pytest.raises(ValidationError, match="Either task_row or task_idx"):
        GymVSeedSessionRequest()


def _verify_response() -> GymVNeMoGymResponse:
    return GymVNeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        env_id="env_1",
    )


def test_verify_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GymVAgentVerifyRequest(response=_verify_response(), unexpected=True)


def test_verify_response_uses_direct_fields() -> None:
    response = GymVAgentVerifyResponse(response=_verify_response(), reward=1.0)

    assert response.response.env_id == "env_1"
    assert response.reward == 1.0


def test_gym_v_response_preserves_training_token_metadata() -> None:
    response = GymVNeMoGymResponse.model_validate(
        {
            "id": "resp_test",
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {
                            "annotations": [],
                            "text": "\\boxed{[right]}",
                            "type": "output_text",
                        }
                    ],
                    "status": "completed",
                    "type": "message",
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "env_id": "env_1",
        }
    )

    dumped = response.model_dump(mode="json")
    assert dumped["output"][0]["prompt_token_ids"] == [1, 2, 3]
    assert dumped["output"][0]["generation_token_ids"] == [4, 5]
    assert dumped["output"][0]["generation_log_probs"] == [-0.1, -0.2]
