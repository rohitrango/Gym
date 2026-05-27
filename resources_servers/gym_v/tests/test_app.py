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
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from omegaconf import OmegaConf
from PIL import Image

from resources_servers.gym_v import app as gym_v_app
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVCloseRequest,
    GymVNeMoGymResponse,
    GymVResourcesServerConfig,
    GymVSeedSessionRequest,
    GymVStepRequest,
)


class StubGymVEnv:
    def __init__(
        self,
        done_after: int = 3,
        raise_on_step: bool = False,
        multi_agent: bool = False,
    ) -> None:
        self.done_after = done_after
        self.raise_on_step = raise_on_step
        self.multi_agent = multi_agent
        self.step_count = 0
        self.closed = False
        self.actions: list[str] = []

    @property
    def description(self) -> str:
        return "STUB ENV: wrap your action in \\boxed{...}."

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.step_count = 0
        obs = gym_v_app.Observation(
            image=Image.new("RGB", (4, 4), (0, 0, 0)),
            text=f"seed={seed}; step=0",
            metadata={},
        )
        if self.multi_agent:
            return {"agent_0": obs, "agent_1": obs}, {"agent_0": {}, "agent_1": {}}
        return {"agent_0": obs}, {"agent_0": {}}

    def step(self, action: dict[str, str]):
        if self.raise_on_step:
            raise ValueError("boom")

        answer = action["agent_0"]
        self.actions.append(answer)
        self.step_count += 1
        done = self.step_count >= self.done_after
        reward = 1.0 if done else 0.0
        obs = gym_v_app.Observation(
            image=Image.new("RGB", (4, 4), (255 if done else 128, 0, 0)),
            text=f"step={self.step_count}",
            metadata={"action_received": answer},
        )
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": done, "__all__": done},
            {"agent_0": False, "__all__": False},
            {"agent_0": {"action_received": answer, "turn": self.step_count}},
        )

    def close(self) -> None:
        self.closed = True


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "env_id": "Games/FrozenLake-v0",
        "env_kwargs": {"size": 4, "num_holes": 3, "tile_size": 32},
        "seed": 1234,
        "task_id": "stub_seed1234",
        "act_grammar_regex": r"^.+$",
        "horizon_cap": None,
        "task_metadata": {"category": "stub"},
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


def _server(tmp_path: Path, rows: list[dict[str, Any]]) -> gym_v_app.GymVResourcesServer:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = GymVResourcesServerConfig(
        name="gym_v_test",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        task_jsonl_fpaths=[str(path)],
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="0.0.0.0", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    return gym_v_app.GymVResourcesServer(config=config, server_client=server_client)


def _request() -> Request:
    return MagicMock(spec=Request)


def _verify_request(env_id: str) -> GymVAgentVerifyRequest:
    response = GymVNeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        env_id=env_id,
    )
    return GymVAgentVerifyRequest(response=response)


@pytest.mark.asyncio
async def test_seed_session_returns_image_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: StubGymVEnv())
    server = _server(tmp_path, [_row()])

    response = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    assert response.env_id in server.env_id_to_env
    content = response.obs[0].content
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["input_text", "input_image"]
    assert "STUB ENV" in content[0]["text"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert response.obs[0].env_info is None


@pytest.mark.asyncio
async def test_seed_session_accepts_full_task_row_without_server_jsonl(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: StubGymVEnv())
    server = _server(tmp_path, [])

    response = await server.seed_session(
        _request(), GymVSeedSessionRequest(task_row=_row())
    )

    assert response.env_id in server.env_id_to_env
    assert server.env_id_to_task_row[response.env_id]["seed"] == 1234
    assert server.task_rows == []


@pytest.mark.asyncio
async def test_seed_session_out_of_range_returns_400(tmp_path: Path) -> None:
    server = _server(tmp_path, [_row()])

    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=99))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_seed_session_rejects_multi_agent_env(monkeypatch, tmp_path: Path) -> None:
    env = StubGymVEnv(multi_agent=True)
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: env)
    server = _server(tmp_path, [_row()])

    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    assert exc_info.value.status_code == 400
    assert env.closed is True
    assert not server.env_id_to_env


@pytest.mark.asyncio
async def test_step_advances_env_and_accumulates_reward(
    monkeypatch, tmp_path: Path
) -> None:
    """Path B: each /step call carries the model's `action_string` extracted
    from `\\boxed{...}` by text_action_agent. Multi-step rollouts accumulate
    reward on the session record."""
    env = StubGymVEnv(done_after=2)
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: env)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    first = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="a"),
    )
    second = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="b"),
    )

    assert first.reward == 0.0
    assert first.done is False
    assert first.obs[0].env_info == {"action_received": "a", "turn": 1}
    assert second.reward == 1.0
    assert second.done is True
    assert server.env_id_to_total_reward[seeded.env_id] == 1.0
    assert env.actions == ["a", "b"]


@pytest.mark.asyncio
async def test_step_env_exception_returns_recovery(monkeypatch, tmp_path: Path) -> None:
    env = StubGymVEnv(raise_on_step=True)
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: env)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    response = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="bad"),
    )

    assert response.reward == 0.0
    assert response.done is False
    assert response.obs[0].env_info == {"env_step_exception": "boom"}


@pytest.mark.asyncio
async def test_step_horizon_cap_terminates_episode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: StubGymVEnv(done_after=99))
    server = _server(tmp_path, [_row(horizon_cap=2)])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    first = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="a"),
    )
    second = await server.step(
        _request(),
        GymVStepRequest(env_id=seeded.env_id, action_string="b"),
    )

    assert first.done is False
    assert first.horizon_terminated is False
    assert second.done is True
    assert second.horizon_terminated is True


@pytest.mark.asyncio
async def test_unknown_env_id_step_returns_404(tmp_path: Path) -> None:
    server = _server(tmp_path, [_row()])

    with pytest.raises(HTTPException) as exc_info:
        await server.step(_request(), GymVStepRequest(env_id="missing", action_string="x"))

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_close_releases_session_state_idempotently(monkeypatch, tmp_path: Path) -> None:
    env = StubGymVEnv()
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: env)
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))

    first = await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))
    second = await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))

    assert first.success is True
    assert first.message == "ok"
    assert second.success is True
    assert second.message == "already closed"
    assert env.closed is True
    assert seeded.env_id not in server.env_id_to_env
    assert seeded.env_id in server.env_id_to_total_reward


@pytest.mark.asyncio
async def test_verify_drains_reward_even_after_close(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym_v_app.gym_v, "make", lambda *args, **kwargs: StubGymVEnv(done_after=1))
    server = _server(tmp_path, [_row()])
    seeded = await server.seed_session(_request(), GymVSeedSessionRequest(task_idx=0))
    await server.step(_request(), GymVStepRequest(env_id=seeded.env_id, action_string="x"))
    await server.close(_request(), GymVCloseRequest(env_id=seeded.env_id))

    verified = await server.verify(_request(), _verify_request(seeded.env_id))
    second = await server.verify(_request(), _verify_request(seeded.env_id))

    assert verified.reward == 1.0
    assert second.reward == 0.0
