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

from conftest import (
    requires_gym_v,
    requires_matplotlib,
    requires_networkx,
    requires_reasoning_gym,
)


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


@requires_gym_v
def test_real_gridcomponent_reset_and_step_with_oracle_answer() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make(
        "Graphs/GridComponent-v0",
        max_n_m=4,
        cell_px=56,
        padding=24,
    )
    obs_dict, info_dict = env.reset(seed=1729)
    assert set(obs_dict.keys()) == {"agent_0"}

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Graphs/GridComponent-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )
    types = [part.get("type") for part in message.content]
    assert "input_image" in types, "GridComponent must produce a rendered grid image."

    oracle = info_dict["agent_0"]["oracle_answer"]
    _obs, reward_dict, _term, _trunc, _info = env.step({"agent_0": oracle})
    assert reward_dict["agent_0"] == 1.0, "Stepping with the oracle answer must score 1.0."

    env2 = gym_v.make("Graphs/GridComponent-v0", max_n_m=4, cell_px=56, padding=24)
    env2.reset(seed=1729)
    _obs, reward_dict, _term, _trunc, _info = env2.step({"agent_0": "not-an-integer"})
    assert reward_dict["agent_0"] == 0.0, "Non-integer answer must score 0.0."


@requires_gym_v
@requires_networkx
def test_real_treeevenpartitioning_reset_and_step_with_oracle_answer() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make(
        "Graphs/TreeEvenPartitioning-v0",
        max_n=3,
        max_k=3,
        node_radius=20,
        image_size=800,
        padding=80,
    )
    obs_dict, info_dict = env.reset(seed=1729)
    assert set(obs_dict.keys()) == {"agent_0"}

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Graphs/TreeEvenPartitioning-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )
    types = [part.get("type") for part in message.content]
    assert "input_image" in types, "TreeEvenPartitioning must produce a rendered graph image."

    oracle = info_dict["agent_0"]["oracle_answer"]
    assert isinstance(oracle, str) and "\n" in oracle, (
        "oracle_answer is a newline-joined multi-line partition string."
    )

    _obs, reward_dict, _term, _trunc, _info = env.step({"agent_0": oracle})
    assert reward_dict["agent_0"] == 1.0, "Stepping with the oracle partition must score 1.0."

    env2 = gym_v.make(
        "Graphs/TreeEvenPartitioning-v0",
        max_n=3, max_k=3, node_radius=20, image_size=800, padding=80,
    )
    env2.reset(seed=1729)
    _obs, reward_dict, _term, _trunc, _info = env2.step({"agent_0": "garbage answer"})
    assert reward_dict["agent_0"] == 0.0, "Malformed partition must score 0.0."


@requires_gym_v
def test_real_misetree_reset_and_step_with_oracle_answer() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make(
        "Graphs/MaximumIndependentSetTree-v0",
        max_n=6,
        node_radius=22,
        image_size=700,
        padding=60,
    )
    obs_dict, info_dict = env.reset(seed=1729)
    assert set(obs_dict.keys()) == {"agent_0"}

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Graphs/MaximumIndependentSetTree-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )
    types = [part.get("type") for part in message.content]
    assert "input_image" in types, "MISETree must produce a rendered tree image."

    oracle = info_dict["agent_0"]["oracle_answer"]
    assert isinstance(oracle, str) and oracle.strip(), (
        "oracle_answer is a space-separated vertex list."
    )

    _obs, reward_dict, _term, _trunc, _info = env.step({"agent_0": oracle})
    # Graded reward (answer/gold)^beta with beta=3; oracle should score exactly 1.0.
    assert reward_dict["agent_0"] == 1.0, "Stepping with the oracle MIS must score 1.0."

    env2 = gym_v.make(
        "Graphs/MaximumIndependentSetTree-v0",
        max_n=6, node_radius=22, image_size=700, padding=60,
    )
    env2.reset(seed=1729)
    _obs, reward_dict, _term, _trunc, _info = env2.step({"agent_0": "not-an-integer"})
    assert reward_dict["agent_0"] == 0.0, "Non-integer answer must score 0.0."


@requires_gym_v
def test_real_tangram_reset_and_step_with_oracle_answer() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make(
        "Geometry/Tangram-QA-v0",
        grid_size=5,
        num_seeds=4,
        num_pieces_to_remove=1,
        question_type=0,  # piece_count
    )
    obs_dict, info_dict = env.reset(seed=1729)
    assert set(obs_dict.keys()) == {"agent_0"}

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Geometry/Tangram-QA-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )
    types = [part.get("type") for part in message.content]
    assert "input_image" in types, "Tangram-QA must produce a rendered puzzle image."

    oracle = info_dict["agent_0"]["oracle_answer"]
    assert isinstance(oracle, str) and oracle.strip(), (
        "oracle_answer is an MCQ letter or option string."
    )

    _obs, reward_dict, _term, _trunc, _info = env.step({"agent_0": oracle})
    assert reward_dict["agent_0"] == 1.0, "Stepping with the oracle MCQ must score 1.0."

    env2 = gym_v.make(
        "Geometry/Tangram-QA-v0",
        grid_size=5, num_seeds=4, num_pieces_to_remove=1, question_type=0,
    )
    env2.reset(seed=1729)
    _obs, reward_dict, _term, _trunc, _info = env2.step({"agent_0": "definitely_not_an_option"})
    assert reward_dict["agent_0"] == 0.0, "Garbage MCQ answer must score 0.0."


@requires_gym_v
@requires_matplotlib
def test_real_smallestcircle_reset_and_step_with_oracle_answer() -> None:
    import gym_v

    from resources_servers.gym_v._observation import observation_to_user_message

    env = gym_v.make("Geometry/SmallestCircle-v0", n_points=6)
    obs_dict, info_dict = env.reset(seed=1729)
    assert set(obs_dict.keys()) == {"agent_0"}

    message = observation_to_user_message(
        obs_dict["agent_0"],
        env_id="Geometry/SmallestCircle-v0",
        prefix_text=env.description if isinstance(env.description, str) else "",
    )
    types = [part.get("type") for part in message.content]
    assert "input_image" in types, "SmallestCircle must produce a rendered scatter plot."

    oracle = info_dict["agent_0"]["oracle_answer"]
    assert isinstance(oracle, str) and len(oracle.split()) == 3, (
        "oracle_answer is 'x y r' space-separated."
    )

    _obs, reward_dict, _term, _trunc, _info = env.step({"agent_0": oracle})
    # Reward is (gold/answer)^beta; oracle ↔ gold so reward == 1.0 exactly.
    assert reward_dict["agent_0"] == 1.0, "Stepping with the oracle circle must score 1.0."

    env2 = gym_v.make("Geometry/SmallestCircle-v0", n_points=6)
    env2.reset(seed=1729)
    # Tiny circle at origin: infeasible (won't cover the points) → 0.
    _obs, reward_dict, _term, _trunc, _info = env2.step({"agent_0": "0 0 0.001"})
    assert reward_dict["agent_0"] == 0.0, "Infeasible circle must score 0.0."
