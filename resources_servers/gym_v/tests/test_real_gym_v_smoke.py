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
"""Real-env smoke tests gated on the container shipping `gym_v` + extras.

These tests intentionally exercise the helper modules against actual Gym-V
envs without going through the FastAPI endpoint layer. They are the first
integration gate the new container image must satisfy.
"""
from __future__ import annotations

from conftest import requires_gym_v, requires_reasoning_gym


@requires_gym_v
@requires_reasoning_gym
def test_real_gameoflife_reset_produces_user_message_with_image() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make(
        "Algorithmic/GameOfLife-v0",
        dataset_kwargs={"size": 5},
        cell_px=8,
        padding=4,
    )
    obs_dict, _info_dict = env.reset(seed=0)
    assert set(obs_dict.keys()) == {"agent_0"}, "Stage A requires single-agent envs."

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Algorithmic/GameOfLife-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )

    assert message.role == "user"
    content = message.content
    assert isinstance(content, list)
    types = [part.get("type") for part in content]
    assert "input_text" in types
    assert "input_image" in types
    image_part = next(part for part in content if part.get("type") == "input_image")
    assert image_part["image_url"].startswith("data:image/png;base64,")
    assert image_part.get("detail") == "auto"


@requires_gym_v
def test_real_frozenlake_step_round_trip_with_env_info() -> None:
    import gym_v

    from resources_servers.gym_v._observation import (
        _attach_env_info,
        observation_to_user_message,
    )

    env = gym_v.make(
        "Games/FrozenLake-v0",
        size=4,
        num_holes=3,
        tile_size=16,
    )
    obs_dict, _info_dict = env.reset(seed=1234)
    assert set(obs_dict.keys()) == {"agent_0"}

    obs_dict, _reward, _terminated, _truncated, info_dict = env.step({"agent_0": "[right]"})

    message = _attach_env_info(
        observation_to_user_message(
            obs_dict["agent_0"],
            env_id="Games/FrozenLake-v0",
        ),
        info_dict["agent_0"],
    )

    assert message.env_info is not None
    assert "invalid_action" in message.env_info, "FrozenLake step info must expose invalid_action."
