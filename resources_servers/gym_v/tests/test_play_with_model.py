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

from pathlib import Path
from typing import Any

from PIL import Image

from resources_servers.gym_v import play_with_model
from resources_servers.gym_v.view_rollouts import build_report


class StubEnv:
    @property
    def description(self) -> str:
        return "Stub env. Answer with \\boxed{...}."

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        obs = play_with_model.gym_v.Observation(
            image=Image.new("RGB", (4, 4), (0, 0, 0)),
            text=f"seed={seed}",
            metadata={},
        )
        return {"agent_0": obs}, {"agent_0": {}}

    def step(self, action: dict[str, str]):
        self.step_count += 1
        done = self.step_count >= 2
        obs = play_with_model.gym_v.Observation(
            image=Image.new("RGB", (4, 4), (128, 0, 0)),
            text=f"got {action['agent_0']}",
            metadata={},
        )
        return (
            {"agent_0": obs},
            {"agent_0": 1.0 if done else 0.0},
            {"agent_0": done, "__all__": done},
            {"agent_0": False, "__all__": False},
            {"agent_0": {"invalid_action": False}},
        )


def _path_b_model_call(
    input_items: list[dict[str, Any]],
    model_endpoint_url: str,
    model_name: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Stub model that emits a Path B response: plain assistant text
    containing `<think>...</think>` plus a final `\\boxed{...}`. No tool
    calls, no `tools` field sent in the request."""
    assert input_items
    assert model_name == "policy_model"
    assert "\\boxed" in system_prompt, (
        "Path B system prompt must instruct the model to emit \\boxed{...}"
    )
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<think>go right</think>\n\\boxed{[right]}",
                    }
                ],
            }
        ]
    }


def test_direct_mode_model_step_path_b_and_save_transcript(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(play_with_model.gym_v, "make", lambda *args, **kwargs: StubEnv())

    state, image, status, env_feedback, assistant_text = play_with_model.reset_direct(
        "Games/FrozenLake-v0",
        '{"size": 4}',
        1234,
    )

    assert image is not None
    assert "turn=0" in status
    assert "Stub env" in env_feedback
    assert assistant_text == "", "reset_direct must clear the assistant text textbox"

    state, image, status, action, env_feedback, assistant_text = play_with_model.step_model_direct(
        state,
        "http://model",
        "policy_model",
        play_with_model.DEFAULT_SYSTEM_PROMPT,
        model_caller=_path_b_model_call,
    )

    assert image is not None
    assert action == "[right]", "extract_boxed_action must return the inner of the LAST \\boxed{...}"
    assert "turn=1" in status
    # Env feedback (separate textbox) shows the env's response to the model's action.
    assert "got [right]" in env_feedback
    # Assistant text (separate textbox) shows the model's CoT + \boxed line verbatim.
    assert "<think>go right</think>" in assistant_text
    assert "\\boxed{[right]}" in assistant_text

    path = play_with_model.save_transcript(state, tmp_path)
    assert path.is_file()

    report = build_report(path)
    assert report["per_env"]["Games/FrozenLake-v0"]["n"] == 1
    assert report["per_env"]["Games/FrozenLake-v0"]["format_match"] == 1.0


def test_extract_boxed_action_handles_no_boxed_answer() -> None:
    """Missing \\boxed{...} -> format_error=True semantics in the viewer."""
    action, assistant, ok = play_with_model.extract_boxed_action(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I forgot to box my answer"}],
                }
            ]
        }
    )

    assert action is None
    assert assistant == "I forgot to box my answer"
    assert ok is False


def test_extract_boxed_action_returns_last_boxed() -> None:
    """Reasoning models often write a placeholder \\boxed{example} in their
    CoT and a final \\boxed{decision}; the LAST capture is the action."""
    action, assistant, ok = play_with_model.extract_boxed_action(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "<think>For example, write \\boxed{example} like this. "
                                "I'll go up.</think>\n\\boxed{[up]}"
                            ),
                        }
                    ],
                }
            ]
        }
    )

    assert action == "[up]"
    assert ok is True
    assert "example" in assistant


def test_extract_boxed_action_ignores_empty_boxed() -> None:
    """An empty \\boxed{} is treated as a recoverable failure (matches
    text_action_agent._extract_boxed semantics)."""
    action, _assistant, ok = play_with_model.extract_boxed_action(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\\boxed{}"}],
                }
            ]
        }
    )

    assert action is None
    assert ok is False
