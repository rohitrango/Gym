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
"""Canonical Path B system-prompt template for Gym-V.

Single source of truth referenced by:
- The `text_action_agent` system prompt default (`responses_api_agents/
  text_action_agent/app.py` imports this).
- `tests/test_prompts.py`, which validates the example JSONL rows'
  `responses_create_params.instructions` match this template byte-for-byte.
  Drift between this module and the JSONL-shipped copy means the curriculum
  builder has gone out of sync with the prompt contract.

Path A (`act(answer=...)` tool-call transport) was removed; this module now
only exports the Path B prompt. The git history retains the prior
`PATH_A_SYSTEM_PROMPT` / `ACT_TOOL_SCHEMA` definitions if anyone needs to
replay a Path A rollout for archaeology.
"""
from __future__ import annotations

# Path B: model emits plain text containing the last \boxed{...} as the action.
#
# Design notes:
# - The active trajectory config keeps `reasoning_parser: deepseek_r1` and
#   enables the model's native chat-template thinking path. The assistant turn
#   starts in a <think> block, so the prompt must explicitly constrain that
#   block and require the boxed action immediately after </think>.
# - The goal is enough reasoning to use the env's image/text without letting
#   the model spend the whole generation budget inside <think>.
# - Prior history:
#   - Job 11729628: unbounded <think> → max_output_tokens hit.
#   - Jobs 11731129/11731385: enable_thinking=false → model skipped
#     reasoning entirely and pattern-matched reminder templates.
#   - Runs 1/3 (this project): <think> bounded to 1 sentence via system
#     prompt → still 36-64% truncation because vLLM prefills <think> and
#     the model's pretraining overrides the brevity instruction.
#   - Current approach: keep thinking enabled, but make the turn contract
#     explicit: brief <think>...</think>, then \boxed{}.
# - NO env-specific examples (no "[up] for FrozenLake, grid for GoL"). Each
#   env's per-turn user message already shows its own action format; baking
#   FrozenLake examples into the system prompt makes the model emit
#   `[direction]` even when the env is GoL (observed in 11731385's GoL rows).
#   The instruction is intentionally format-only and env-agnostic.
PATH_B_SYSTEM_PROMPT = """\
You are playing a game. Each turn, you receive an image of the current game
state and a short text description. Read the rules in the first user message
carefully — they specify the action grammar and the goal.

For every turn:
1. Think briefly inside the <think> block in 1-2 short sentences. Just state
   your observation and decision — do NOT enumerate grid rows, list options,
   or restate history.
2. Close the think block with </think>, then immediately write your final
   action wrapped in \\boxed{...} on the LAST line.
   The exact format inside \\boxed{...} is dictated by the env's rules.

Only the LAST \\boxed{...} in your response is sent to the game.
"""
