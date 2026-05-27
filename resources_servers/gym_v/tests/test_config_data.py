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

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from omegaconf import OmegaConf

from resources_servers.gym_v.app import GymVResourcesServer
from resources_servers.gym_v.schemas import GymVResourcesServerConfig, GymVTaskRow

GYM_ROOT = Path(__file__).parents[3]
GYM_V_ROOT = GYM_ROOT / "resources_servers" / "gym_v"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_path_b_example_jsonl_rows_validate() -> None:
    """Path A example JSONLs were removed; only `_path_b.jsonl` files remain."""
    paths = [
        GYM_V_ROOT / "data" / "frozenlake_path_b.jsonl",
        GYM_V_ROOT / "data" / "gameoflife_path_b.jsonl",
        GYM_V_ROOT / "data" / "doorkey_path_b.jsonl",
        GYM_V_ROOT / "data" / "all_example_tasks_path_b.jsonl",
    ]

    for path in paths:
        rows = _load_jsonl(path)
        assert rows, f"{path} must contain at least one row"
        for row in rows:
            parsed = GymVTaskRow.model_validate(row)
            assert parsed.responses_create_params.input == []
            assert parsed.responses_create_params.tools == [], (
                f"{path} row has non-empty tools list; Path B contract requires []."
            )


def test_all_path_b_example_tasks_is_concatenation_of_single_row_files() -> None:
    expected = (
        _load_jsonl(GYM_V_ROOT / "data" / "frozenlake_path_b.jsonl")
        + _load_jsonl(GYM_V_ROOT / "data" / "gameoflife_path_b.jsonl")
        + _load_jsonl(GYM_V_ROOT / "data" / "doorkey_path_b.jsonl")
    )

    assert _load_jsonl(GYM_V_ROOT / "data" / "all_example_tasks_path_b.jsonl") == expected


def test_gym_v_yaml_points_at_existing_valid_jsonls() -> None:
    cfg = OmegaConf.to_container(
        OmegaConf.load(GYM_V_ROOT / "configs" / "gym_v.yaml"),
        resolve=True,
    )
    server_cfg = cfg["gym_v_resources_server"]["resources_servers"]["gym_v"]
    jsonl_paths = server_cfg["task_jsonl_fpaths"]

    # New clients pass full GymVTaskRow payloads to /seed_session. The
    # server-side JSONL list remains only for task_idx compatibility.
    assert jsonl_paths == []
    config = GymVResourcesServerConfig.model_validate(
        server_cfg | {"name": "gym_v_resources_server", "host": "0.0.0.0", "port": 8080}
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="0.0.0.0", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    server = GymVResourcesServer(config=config, server_client=server_client)

    assert server.task_rows == []
    # Default agent is `text_action_agent` (Path B); Path A's `aviary_agent`
    # was removed from the canonical config when the tool-call transport was
    # deprecated.
    agent_block = cfg["gym_v_agent"]["responses_api_agents"]["text_action_agent"]
    assert agent_block["done_if_no_boxed_answer"] is False
    assert agent_block["return_transitions"] is False
    assert agent_block["datasets"][0]["jsonl_fpath"].endswith("_path_b.jsonl")
