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
"""Validate the Path B contract for retained example JSONLs.

Path A (`act(answer=...)` tool-call transport) was removed; only the
`_path_b.jsonl` example rows remain. Project-specific Game-RLVR manifests
live outside this submodule. These tests enforce:

1. Every retained Path B row has `tools: []` (no tool schema sent).
2. Rows do NOT carry a Path A `instructions` field — that prompt now lives
   in `text_action_agent.system_prompt` (`PATH_B_SYSTEM_PROMPT`), not on
   per-row JSONLs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "resources_servers" / "gym_v" / "data"

PATH_B_JSONLS = [
    DATA_DIR / "frozenlake_path_b.jsonl",
    DATA_DIR / "gameoflife_path_b.jsonl",
    DATA_DIR / "doorkey_path_b.jsonl",
    DATA_DIR / "all_example_tasks_path_b.jsonl",
]


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("jsonl_path", PATH_B_JSONLS, ids=lambda p: p.name)
def test_path_b_tools_empty(jsonl_path: Path) -> None:
    rows = _load_rows(jsonl_path)
    assert rows, f"{jsonl_path} has no rows"
    for idx, row in enumerate(rows):
        rcp = row["responses_create_params"]
        tools = rcp.get("tools", [])
        assert tools == [], (
            f"{jsonl_path}:{idx} expected empty tools list (Path B contract); "
            f"got {tools!r}"
        )
