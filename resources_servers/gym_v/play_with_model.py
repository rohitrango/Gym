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
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable
import uuid

import gradio as gr
import gym_v
import httpx
from PIL import Image

from ._observation import _attach_env_info, observation_to_user_message
from ._prompts import PATH_B_SYSTEM_PROMPT
from ._viewer_common import first_input_image_url, input_text, write_jsonl
from .schemas import GymVEnvStateEasyInputMessage

DEFAULT_SYSTEM_PROMPT = PATH_B_SYSTEM_PROMPT

# Inline copy of the BOXED extraction regex from
# `string_match/app.py` / `text_action_agent/app.py`. Kept inline so the
# viewer doesn't have to import from another resources-server / agent
# package boundary — the regex hasn't changed in any of the three call
# sites and a one-line manual sync is cheaper than the cross-package
# dependency. Anyone updating it should grep for r"\\boxed\\{" across the
# tree.
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)


@dataclass
class ViewerState:
    mode: str
    env_id: str
    env_kwargs: dict[str, Any]
    seed: int
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    env: Any | None = None
    output: list[dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0
    turn: int = 0
    done: bool = False


ModelCaller = Callable[[list[dict[str, Any]], str, str, str], dict[str, Any]]


def _message_dump(message: GymVEnvStateEasyInputMessage) -> dict[str, Any]:
    return message.model_dump(mode="json")


def _description_for_agent_0(env: Any) -> str:
    description = env.description
    if isinstance(description, str):
        return description
    return description.get("agent_0", "")


def _first_image_from_obs(obs: Any) -> Image.Image | None:
    if obs.image is None:
        return None
    if isinstance(obs.image, list):
        return obs.image[0] if obs.image else None
    return obs.image


def _first_image_from_message(message: dict[str, Any]) -> Image.Image | None:
    data_url = first_input_image_url(message)
    if not data_url:
        return None
    from ._viewer_common import decode_image_data_url

    return decode_image_data_url(data_url)


def parse_env_kwargs(env_kwargs_json: str) -> dict[str, Any]:
    if not env_kwargs_json.strip():
        return {}
    parsed = json.loads(env_kwargs_json)
    if not isinstance(parsed, dict):
        raise ValueError("Env kwargs must be a JSON object.")
    return parsed


def extract_boxed_action(model_response: dict[str, Any]) -> tuple[str | None, str, bool]:
    """Path B action extraction: walk the assistant text in an OpenAI
    Responses-style body and return the LAST `\\boxed{...}` capture.

    Returns `(answer, full_assistant_text, ok)` where:
    - `answer` is the stripped contents of the last `\\boxed{...}`
      (or `None` if no well-formed boxed answer is present),
    - `full_assistant_text` is the concatenated assistant content for
      display in the viewer's "Assistant text" panel,
    - `ok` indicates whether `answer` is a non-empty string.

    Mirrors `text_action_agent._extract_boxed`'s semantics — reasoning
    models routinely write a placeholder `\\boxed{example}` earlier in
    their CoT and the FINAL `\\boxed{...}` is the decision that counts.
    """
    assistant_text_parts: list[str] = []
    for item in model_response.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str):
            assistant_text_parts.append(content)
        elif isinstance(content, list):
            assistant_text_parts.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and "text" in part
            )

    full_text = "\n".join(assistant_text_parts)
    matches = BOXED_PATTERN.findall(full_text)
    if not matches:
        return None, full_text, False
    answer = matches[-1].strip()
    if not answer:
        return None, full_text, False
    return answer, full_text, True


def call_model_server(
    input_items: list[dict[str, Any]],
    model_endpoint_url: str,
    model_name: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Post a Path B Responses-API request: plain text in, plain text out.

    No `tools` field is sent — the model is expected to emit its action as
    `\\boxed{...}` in plain text (Path B). vLLM's reasoning parser
    (`reasoning_parser: deepseek_r1` in the project config) handles
    `<think>...</think>` blocks.
    """
    payload = {
        "model": model_name,
        "instructions": system_prompt,
        "input": input_items,
    }
    response = httpx.post(
        f"{model_endpoint_url.rstrip('/')}/v1/responses",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def reset_direct(env_id: str, env_kwargs_json: str, seed: int) -> tuple[ViewerState, Image.Image | None, str, str, str]:
    """Reset the env. Returns (state, image, status, env_feedback, assistant_text).

    The 5th element is an empty string — there is no model assistant text yet
    at reset time. The Gradio wiring passes this through to the
    `assistant_text` textbox so it clears whatever was left from the prior
    session.
    """
    env_kwargs = parse_env_kwargs(env_kwargs_json)
    env = gym_v.make(env_id, **env_kwargs)
    obs_dict, _info_dict = env.reset(seed=seed)
    if set(obs_dict.keys()) != {"agent_0"}:
        env.close()
        raise ValueError(f"Stage A viewer supports only single-agent envs; got {list(obs_dict)}")

    obs = obs_dict["agent_0"]
    message = observation_to_user_message(
        obs,
        env_id=env_id,
        prefix_text=_description_for_agent_0(env),
    )
    state = ViewerState(
        mode="Direct",
        env_id=env_id,
        env_kwargs=env_kwargs,
        seed=seed,
        env=env,
        output=[_message_dump(message)],
    )
    return state, _first_image_from_obs(obs), "turn=0 reward=0.0 done=False", input_text(state.output[-1]), ""


def _append_model_output(state: ViewerState, model_response: dict[str, Any]) -> tuple[str | None, str, bool]:
    output = model_response.get("output", [])
    if isinstance(output, list):
        state.output.extend(output)
    return extract_boxed_action(model_response)


def step_manual_direct(
    state: ViewerState, action: str, append_action: bool = True
) -> tuple[ViewerState, Image.Image | None, str, str, str]:
    if state.env is None:
        raise ValueError("Reset direct mode before stepping.")
    if state.done:
        return state, _first_image_from_message(state.output[-1]), "episode already done", action, input_text(state.output[-1])

    obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = state.env.step({"agent_0": action})
    reward = float(reward_dict["agent_0"])
    state.total_reward += reward
    state.turn += 1
    state.done = bool(terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False))

    obs = obs_dict["agent_0"]
    message = _attach_env_info(
        observation_to_user_message(obs, env_id=state.env_id),
        info_dict.get("agent_0", {}),
    )
    if append_action:
        # Path B: record the manual action as a synthesized assistant
        # message containing `\boxed{<action>}` — matches the wire shape
        # text_action_agent produces, so saved transcripts can be
        # consumed by view_rollouts.py unchanged.
        state.output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": f"\\boxed{{{action}}}"}
                ],
            }
        )
    state.output.append(_message_dump(message))
    status = f"turn={state.turn} reward={state.total_reward} done={state.done}"
    return state, _first_image_from_obs(obs), status, action, input_text(state.output[-1])


def step_model_direct(
    state: ViewerState,
    model_endpoint_url: str,
    model_name: str,
    system_prompt: str,
    model_caller: ModelCaller = call_model_server,
) -> tuple[ViewerState, Image.Image | None, str, str, str, str]:
    """Call the model (Path B Responses API), extract its `\\boxed{...}`
    action, and step the env.

    Returns (state, image, status, action, env_feedback, assistant_text). The
    6th element is the concatenated assistant text the model emitted
    (`<think>...</think>` block plus the `\\boxed{...}` line), surfaced
    separately from `env_feedback` so a subsequent manual step doesn't
    clobber it.
    """
    response = model_caller(
        state.output,
        model_endpoint_url,
        model_name,
        system_prompt,
    )
    action, assistant_text, ok = _append_model_output(state, response)
    if not ok or action is None:
        status = (
            f"turn={state.turn} reward={state.total_reward} done={state.done} "
            f"format_error=True (no \\boxed{{...}} action found)"
        )
        # On format error there's no env step to take; leave env_feedback as
        # the prior turn's text (last message) so the user can still read it
        # alongside the model output that just failed to parse.
        return (
            state,
            _first_image_from_message(state.output[-1]),
            status,
            "<format error>",
            input_text(state.output[-1]),
            assistant_text,
        )
    new_state, image, status, action_str, env_feedback = step_manual_direct(
        state, action, append_action=False
    )
    return new_state, image, status, action_str, env_feedback, assistant_text


def save_transcript(state: ViewerState, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out / f"play_session_{stamp}.jsonl"
    row = {
        "full_result": json.dumps(
            {
                "id": f"viewer_{state.session_id}",
                "env_id": state.env_id,
                "group_id": f"viewer_session_{state.session_id}",
                "reward": state.total_reward,
                "contains_transitions": False,
                "output": state.output,
            }
        )
    }
    write_jsonl(path, [row])
    return path


DEFAULT_ENV_ID = "Games/FrozenLake-v0"
DEFAULT_ENV_KWARGS_JSON = '{"size": 4, "num_holes": 3, "tile_size": 32}'
DEFAULT_SEED = 1234

# Per-env kwargs presets used when --env-id is passed without --env-kwargs.
# Mirrors the curated catalog in `docs/guides/manual-game-play.md`. Envs
# absent from this map default to `{}` (registry defaults), which works for
# every registered env per gym_v.make's kwargs-override semantics (see
# `registration.py`'s env_spec_kwargs.update(make_kwargs)).
_ENV_KWARGS_PRESETS: dict[str, str] = {
    "Games/FrozenLake-v0": '{"size": 4, "num_holes": 3, "tile_size": 32}',
    "Games/Game2048-v0": '{"target_tile": 2048, "tile_size": 100}',
    "Games/LightsOut-v0": '{"size": 5, "cell_size": 80}',
    "Games/Minesweeper-v0": '{"rows": 8, "cols": 8, "num_mines": 10}',
    "Games/Sokoban-v0": '{"dim_room": [6, 6], "num_boxes": 3, "tile_size": 48}',
    "Games/Wordle-v0": '{"word_length": 5, "num_guesses": 6, "hardcore": false}',
    "Spatial/DoorKey-v0": '{"size": 6, "tile_size": 16}',
    "Spatial/FourRooms2D-v0": '{"tile_size": 32}',
    "Spatial/LavaGap-v0": '{"size": 7, "tile_size": 32}',
    "Spatial/MultiRoom-v0": '{"min_num_rooms": 6, "max_num_rooms": 6, "tile_size": 32}',
    "Puzzles/Maze-QA-v0": '{"size": "small", "cell_size": 40, "question_type": null}',
    "Algorithmic/GameOfLife-v0": '{"dataset_kwargs": {"size": 100}, "cell_px": 24, "padding": 12}',
    "Logic/MiniSudoku-v0": '{"dataset_kwargs": {"size": 500}, "cell_px": 80, "padding": 24}',
    "Puzzles/TowerOfHanoi-v0": '{"dataset_kwargs": {"size": 500, "min_disks": 3, "max_disks": 4, "min_pegs": 3, "max_pegs": 4}}',
    "Arc/ArcAgi-v0": '{"dataset_kwargs": {"size": 500}, "cell_px": 16, "padding": 16}',
}


def _resolve_default_kwargs(env_id_value: str) -> str:
    """Pick a sensible kwargs JSON default for a given env id."""
    return _ENV_KWARGS_PRESETS.get(env_id_value, "{}")


def build_app(
    default_env_id: str = DEFAULT_ENV_ID,
    default_env_kwargs: str | None = None,
    default_seed: int = DEFAULT_SEED,
) -> gr.Blocks:
    """Build the Gradio app, optionally pre-filling the env-id / kwargs / seed
    textboxes from CLI overrides. Backwards compatible: with no args, the
    viewer launches with the historical FrozenLake defaults.

    `default_env_kwargs=None` means "look up the preset for this env id";
    pass an explicit string (including `"{}"`) to bypass the lookup.
    """
    if default_env_kwargs is None:
        default_env_kwargs = _resolve_default_kwargs(default_env_id)

    with gr.Blocks() as app:
        state = gr.State()
        gr.Markdown("# Gym-V model-in-the-loop viewer")
        with gr.Row():
            env_id = gr.Textbox(value=default_env_id, label="Env ID")
            env_kwargs = gr.Textbox(value=default_env_kwargs, label="Env kwargs JSON")
            seed = gr.Number(value=default_seed, precision=0, label="Seed")
        with gr.Row():
            endpoint = gr.Textbox(value="http://localhost:8000", label="Model endpoint")
            model = gr.Textbox(value="policy_model", label="Model name")
        system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System prompt")
        image = gr.Image(type="pil", label="Observation")
        status = gr.Textbox(label="Status")
        action = gr.Textbox(label="Last action")
        # `lines` sets the initial visible height; `max_lines` caps the
        # expanded height before Gradio falls back to internal scrolling.
        # Env descriptions for envs like Maze-QA, MiniSudoku, TowerOfHanoi
        # routinely run 15-30 lines (rules + question + A-H options), so a
        # default single-line Textbox would truncate them with no UI signal.
        # Env feedback and assistant text are kept in separate textboxes so a
        # subsequent manual step doesn't overwrite the model's prior CoT.
        env_feedback = gr.Textbox(
            label="Env feedback",
            lines=20,
            max_lines=40,
            show_copy_button=True,
        )
        assistant_text = gr.Textbox(
            label="Assistant text (model CoT + action)",
            lines=10,
            max_lines=40,
            show_copy_button=True,
        )
        manual_action = gr.Textbox(value="[right]", label="Manual action")
        out_dir = gr.Textbox(value="viewer_transcripts", label="Transcript output dir")

        reset_btn = gr.Button("Reset")
        model_btn = gr.Button("Step (Model)")
        manual_btn = gr.Button("Step (Manual)")
        save_btn = gr.Button("Save Transcript")

        # Reset clears `assistant_text` (5th return value of `reset_direct`).
        reset_btn.click(
            reset_direct,
            [env_id, env_kwargs, seed],
            [state, image, status, env_feedback, assistant_text],
        )
        # Step (Model) updates BOTH env feedback and assistant text.
        model_btn.click(
            step_model_direct,
            [state, endpoint, model, system_prompt],
            [state, image, status, action, env_feedback, assistant_text],
        )
        # Step (Manual) intentionally omits `assistant_text` from the outputs
        # list — Gradio leaves components not listed in `outputs` untouched,
        # so the model's prior CoT stays visible while you single-step the
        # env manually.
        manual_btn.click(
            step_manual_direct,
            [state, manual_action],
            [state, image, status, action, env_feedback],
        )
        save_btn.click(lambda s, d: str(save_transcript(s, d)), [state, out_dir], [status])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gradio model-in-the-loop viewer for Gym-V envs. The --env-id / "
            "--env-kwargs / --seed flags pre-fill the textboxes at launch so "
            "you don't have to retype them across viewer restarts."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help=(
            "Pre-fill the Env ID textbox at launch. Examples: "
            "'Games/FrozenLake-v0', 'Puzzles/Maze-QA-v0', 'Spatial/DoorKey-v0'. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--env-kwargs",
        default=None,
        help=(
            "JSON-encoded kwargs dict to pre-fill the Env kwargs textbox. "
            "If omitted, looks up a per-env preset (see _ENV_KWARGS_PRESETS); "
            "envs without a preset default to '{}' (registry defaults). "
            "Quote carefully — typical shell example: "
            "--env-kwargs '{\"size\": \"small\", \"question_type\": 0}'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Pre-fill the Seed input. Default: %(default)s.",
    )
    args = parser.parse_args()
    build_app(
        default_env_id=args.env_id,
        default_env_kwargs=args.env_kwargs,
        default_seed=args.seed,
    ).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
