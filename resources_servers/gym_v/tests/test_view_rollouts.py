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

from PIL import Image
import pytest

from resources_servers.gym_v._observation import image_to_data_url
from resources_servers.gym_v.view_rollouts import (
    RolloutInspectionError,
    build_report,
    write_report,
)


def _user_message(text: str, *, env_info=None, image=False) -> dict:
    content = [{"type": "input_text", "text": text}]
    if image:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(Image.new("RGB", (2, 2), (1, 2, 3))),
                "detail": "auto",
            }
        )
    return {"type": "message", "role": "user", "content": content, "env_info": env_info}


def _assistant(text: str = "<think>abc</think> answer") -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
        "generation_token_ids": [1, 2, 3],
    }


def _boxed_assistant(answer: str) -> dict:
    """Path B assistant turn: assistant text ending in `\\boxed{<answer>}`."""
    return _assistant(text=f"<think>thinking...</think> answer: \\boxed{{{answer}}}")


def _rollout(
    *,
    env_id: str = "Games/FrozenLake-v0",
    group_id: str = "g0",
    reward: float = 0.0,
    answer: str | None = "[up]",
    step_exception: bool = False,
) -> dict:
    """Single-turn Path B rollout. `answer=None` produces an assistant turn
    with NO `\\boxed{...}` (the format-error case)."""
    output: list[dict] = [_user_message("seed", env_info=None, image=True)]
    if answer is None:
        output.append(_assistant(text="<think>uh</think> I am stuck."))
    else:
        output.append(_boxed_assistant(answer))
    output.append(
        _user_message(
            "step",
            env_info={"env_step_exception": "boom"} if step_exception else {},
            image=True,
        )
    )
    return {
        "id": f"{env_id}-{group_id}-{reward}-{answer}",
        "env_id": env_id,
        "group_id": group_id,
        "reward": reward,
        "contains_transitions": False,
        "output": output,
    }


def _multi_turn_rollout(
    *,
    env_id: str = "Games/FrozenLake-v0",
    group_id: str = "g0",
    reward: float = 0.0,
    answers: list[str | None],
) -> dict:
    """Multi-turn Path B rollout: one assistant message per turn, each
    optionally producing a `\\boxed{<answer>}`. `None` entries mean a turn
    with no boxed answer (per-turn fmt_match=False)."""
    output: list[dict] = [_user_message("seed", image=True)]
    for ans in answers:
        if ans is None:
            output.append(_assistant(text="<think>uh</think> I am stuck."))
        else:
            output.append(_boxed_assistant(ans))
        output.append(_user_message("step", image=False))
    return {
        "id": f"{env_id}-{group_id}-{reward}",
        "env_id": env_id,
        "group_id": group_id,
        "reward": reward,
        "contains_transitions": False,
        "output": output,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_wrapped_rollouts(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({"full_result": json.dumps(row)}) + "\n")


def _row_source(path: Path) -> None:
    """Path B row source (the only kind retained): `responses_create_params.tools`
    is always the empty list."""
    _write_jsonl(
        path,
        [
            {
                "env_id": "Games/FrozenLake-v0",
                "seed": 1,
                "responses_create_params": {
                    "model": "policy_model",
                    "input": [],
                    "tools": [],
                },
                "act_grammar_regex": r"^\[(up|down|left|right|w|a|s|d)\]$",
            }
        ],
    )


def test_per_env_report_counts_are_correct(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    _write_wrapped_rollouts(
        rollouts_path,
        [
            _rollout(reward=1.0, answer="[up]"),
            _rollout(reward=0.0, answer=None),
            _rollout(reward=0.0, answer="[diagonal]"),
        ],
    )

    report = build_report(rollouts_path, row_source_path)
    frozenlake = report["per_env"]["Games/FrozenLake-v0"]

    assert frozenlake["n"] == 3
    assert frozenlake["any_success"] == pytest.approx(1 / 3)
    assert frozenlake["format_match"] == pytest.approx(2 / 3)
    assert frozenlake["vocab_in_set"] == pytest.approx(1 / 2)


def test_per_group_report_filter_rate(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    rows = [
        _rollout(group_id="g0", reward=0.0),
        _rollout(group_id="g0", reward=0.0),
        _rollout(group_id="g1", reward=0.0),
        _rollout(group_id="g1", reward=1.0),
        _rollout(group_id="g2", reward=0.0),
        _rollout(group_id="g2", reward=1.0),
        _rollout(group_id="g3", reward=0.0),
        _rollout(group_id="g3", reward=1.0),
    ]
    _write_wrapped_rollouts(rollouts_path, rows)

    report = build_report(rollouts_path, row_source_path)

    assert report["per_group"]["filter_rate"] == pytest.approx(0.25)
    by_env = report["per_group"]["effective_nonzero_groups_per_step_per_env"]
    assert by_env["Games/FrozenLake-v0"]["nonzero"] == 3
    assert by_env["Games/FrozenLake-v0"]["total"] == 4


def test_step_exception_column_counts_recoveries(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    _write_wrapped_rollouts(
        rollouts_path,
        [
            _rollout(answer="[up]", step_exception=True),
            _rollout(answer="[down]", step_exception=False),
        ],
    )

    report = build_report(rollouts_path, row_source_path)

    assert report["per_env"]["Games/FrozenLake-v0"]["step_exception_rate"] == pytest.approx(
        0.5
    )


def test_inspector_rejects_nested_transitions_format(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    row = _rollout()
    row["contains_transitions"] = True
    _write_wrapped_rollouts(rollouts_path, [row])

    with pytest.raises(RolloutInspectionError, match="return_transitions=false"):
        build_report(rollouts_path, row_source_path)


def test_per_turn_columns_count_each_assistant_turn(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    # 1 rollout, 4 assistant turns: 3 fmt-matched (2 in-vocab, 1 not), 1 not.
    _write_wrapped_rollouts(
        rollouts_path,
        [
            _multi_turn_rollout(answers=["[up]", "[down]", "[diagonal]", None]),
        ],
    )

    report = build_report(rollouts_path, row_source_path)
    frozenlake = report["per_env"]["Games/FrozenLake-v0"]

    # Per-turn fmt-match: 3 of 4 turns produced a boxed answer.
    assert frozenlake["per_turn_fmt_match"] == pytest.approx(3 / 4)
    # Per-turn vocab-in-set is conditioned on fmt_match: of the 3 boxed turns,
    # 2 ("[up]", "[down]") match the grammar regex.
    assert frozenlake["per_turn_vocab_in_set"] == pytest.approx(2 / 3)
    # Per-turn executable-action-rate = pt_fmt_match * pt_vocab_in_set.
    assert frozenlake["per_turn_executable_action_rate"] == pytest.approx(
        (3 / 4) * (2 / 3)
    )


def test_per_turn_columns_handle_no_assistant_turns(tmp_path: Path) -> None:
    """If a rollout has no assistant items at all (e.g., crashed before the
    first /v1/responses call), per-turn columns must report None rather than
    crashing."""
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    _row_source(row_source_path)
    empty_row = {
        "id": "empty",
        "env_id": "Games/FrozenLake-v0",
        "group_id": "g0",
        "reward": 0.0,
        "contains_transitions": False,
        "output": [_user_message("seed", image=True)],
    }
    _write_wrapped_rollouts(rollouts_path, [empty_row])

    report = build_report(rollouts_path, row_source_path)
    frozenlake = report["per_env"]["Games/FrozenLake-v0"]
    assert frozenlake["per_turn_fmt_match"] is None
    assert frozenlake["per_turn_vocab_in_set"] is None
    assert frozenlake["per_turn_executable_action_rate"] is None


def test_write_report_saves_json_and_sample_markdown(tmp_path: Path) -> None:
    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    out_dir = tmp_path / "report"
    _row_source(row_source_path)
    _write_wrapped_rollouts(rollouts_path, [_rollout(reward=1.0, answer="[up]")])

    report = build_report(rollouts_path, row_source_path)
    write_report(report, out_dir, top_k=1, bottom_k=0)

    assert (out_dir / "report.json").is_file()
    sample_dir = out_dir / "samples" / "Games__FrozenLake-v0"
    assert (sample_dir / "best_0.md").is_file()
    assert list(sample_dir.glob("best_0_turn_*.png"))


def test_sample_markdown_renders_seed_obs_before_output(tmp_path: Path) -> None:
    """Option B (seed-obs-persistence-problem.md): seed_obs is stored
    separately from output. The sample markdown must render seed_obs first
    (## Seed Observation) before any ## Assistant / ## User from output.
    """
    seed_user = _user_message("initial prompt with image", image=True)
    rollout = {
        "id": "r0",
        "env_id": "Games/FrozenLake-v0",
        "group_id": "g0",
        "reward": 1.0,
        "contains_transitions": False,
        "seed_obs": [seed_user],
        "output": [
            _boxed_assistant("[up]"),
            _user_message("step result", image=True),
        ],
    }

    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    out_dir = tmp_path / "report"
    _row_source(row_source_path)
    _write_wrapped_rollouts(rollouts_path, [rollout])

    report = build_report(rollouts_path, row_source_path)
    write_report(report, out_dir, top_k=1, bottom_k=0)

    sample_dir = out_dir / "samples" / "Games__FrozenLake-v0"
    md_path = sample_dir / "best_0.md"
    assert md_path.is_file()
    md_text = md_path.read_text()

    # seed_obs rendered before assistant output
    assert "## Seed Observation" in md_text
    seed_pos = md_text.index("## Seed Observation")
    assert "## Assistant" in md_text
    asst_pos = md_text.index("## Assistant")
    assert seed_pos < asst_pos, "Seed observation must appear before assistant output"

    # seed image saved as a PNG
    seed_pngs = list(sample_dir.glob("best_0_seed_*.png"))
    assert seed_pngs, "Seed observation image must be saved as a PNG"

    # initial prompt text visible in markdown
    assert "initial prompt with image" in md_text


def test_sample_markdown_works_without_seed_obs(tmp_path: Path) -> None:
    """Legacy trajectories without seed_obs must still render cleanly."""
    rollout = _rollout(reward=1.0, answer="[up]")
    assert "seed_obs" not in rollout  # confirm legacy shape

    rollouts_path = tmp_path / "trajectory_collection.jsonl"
    row_source_path = tmp_path / "rows.jsonl"
    out_dir = tmp_path / "report"
    _row_source(row_source_path)
    _write_wrapped_rollouts(rollouts_path, [rollout])

    report = build_report(rollouts_path, row_source_path)
    write_report(report, out_dir, top_k=1, bottom_k=0)

    md_path = out_dir / "samples" / "Games__FrozenLake-v0" / "best_0.md"
    assert md_path.is_file()
    md_text = md_path.read_text()
    assert "## Seed Observation" not in md_text
    assert "## Assistant" in md_text
