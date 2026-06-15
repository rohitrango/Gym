# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-visualization viewer for the Gym-V envs added in this worktree.

A lightweight sibling of ``play_with_model.py``. Where the model-in-the-loop
viewer is a full play surface (manual action, model step, transcript save),
this one is just: pick an env from a dropdown, change the seed, click Reset,
look at the rendered observation. No model, no step buttons, no transcripts.

Use it to sanity-check the per-env stage-1 JSONL rows committed under
``data/<env>_path_b.jsonl`` — every entry in the dropdown is pre-filled with
the same kwargs the stage-1 rows use, so resetting at seed=1729 reproduces
the exact image a trainee would see for that JSONL row.

Launch (in module mode, from ``3rdparty/Gym-workspace/Gym``):

    PYTHONPATH=. $GYM_V_VENV/bin/python \\
        -m resources_servers.gym_v.visualize_added_envs \\
        --host 0.0.0.0 --port 7861

To add another env, append a row to ``GAMES`` below — that's the only edit.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import gradio as gr
import gym_v


# Each entry: env_id → (default-kwargs JSON string, one-line blurb).
# Mirrors the stage-1 settings used by the per-env JSONLs under data/.
GAMES: dict[str, tuple[str, str]] = {
    "Graphs/GridComponent-v0": (
        '{"max_n_m": 16, "cell_px": 56, "padding": 24}',
        "Count the largest connected component of 1s in a binary grid. "
        "Action: single integer. Reward: 1/0.",
    ),
    "Graphs/TreeEvenPartitioning-v0": (
        '{"max_n": 3, "max_k": 3, "node_radius": 20, "image_size": 800, "padding": 80}',
        "Partition tree vertices into N equal-sized connected subtrees. "
        "Action: N lines of K space-separated ints. Reward: (connected/N)^5.",
    ),
    "Graphs/MaximumIndependentSetTree-v0": (
        '{"max_n": 6, "node_radius": 22, "image_size": 700, "padding": 60}',
        "Pick a max-weight independent set in a weighted tree. "
        "Action: space-separated vertex ids. Reward: (weight/gold)^3.",
    ),
    "Geometry/Tangram-QA-v0": (
        '{"grid_size": 5, "num_seeds": 4, "num_pieces_to_remove": 1, "question_type": 0}',
        "MCQ over a Voronoi-tiled grid puzzle. question_type 0=piece-count "
        "(easy), 1=area, 2=adjacency, 3=rotation, 4=placement. Reward: 1/0.",
    ),
    "Geometry/SmallestCircle-v0": (
        '{"n_points": 6}',
        "Find the smallest enclosing circle of N integer-coordinate points. "
        "Action: 'x y r' (three floats). Reward: (gold/answer)^10 when feasible, "
        "else 0. Needs matplotlib.",
    ),
}

DEFAULT_ENV_ID = next(iter(GAMES))
DEFAULT_SEED = 1729


def _description_for_agent_0(env: Any) -> str:
    """Mirror play_with_model.py's helper: env.description is a str or a dict."""
    description = env.description
    if isinstance(description, str):
        return description
    if isinstance(description, dict):
        return description.get("agent_0", "")
    return str(description)


def _coerce_seed(value: Any) -> int:
    """Gradio's Number returns float; coerce safely, fall back to DEFAULT_SEED."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_SEED


def reset_env(env_id: str, kwargs_json: str, seed: Any) -> tuple[Any, str, str]:
    """Build the env, reset, render the agent_0 observation.

    Returns (PIL image | None, status string, env.description text).
    """
    try:
        kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
    except json.JSONDecodeError as exc:
        return None, f"kwargs JSON parse error: {exc}", ""

    seed_int = _coerce_seed(seed)

    try:
        env = gym_v.make(env_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — surface anything to the UI
        return None, f"gym_v.make({env_id!r}) failed: {type(exc).__name__}: {exc}", ""

    try:
        obs_dict, info_dict = env.reset(seed=seed_int)
    except Exception as exc:  # noqa: BLE001
        env.close()
        return None, f"env.reset(seed={seed_int}) failed: {type(exc).__name__}: {exc}", ""

    try:
        if set(obs_dict.keys()) != {"agent_0"}:
            return (
                None,
                f"Multi-agent envs not supported here; got agents={list(obs_dict)}.",
                "",
            )

        obs = obs_dict["agent_0"]
        info = info_dict.get("agent_0", {})
        image = getattr(obs, "image", None)
        # Some Observation classes wrap a single image; others may wrap a list.
        if isinstance(image, list):
            image = image[0] if image else None

        status_lines = [
            f"env_id        : {env_id}",
            f"env_kwargs    : {json.dumps(kwargs)}",
            f"seed          : {seed_int}",
            f"image size    : {image.size if image is not None else '<no image>'}",
        ]
        oracle = info.get("oracle_answer")
        if oracle is not None:
            status_lines.append(f"oracle_answer : {oracle!r}")
        if "question_type" in info:
            status_lines.append(f"question_type : {info['question_type']}")
        # Surface any state_text the env exposes (gives the textual prompt
        # alongside the image — useful for envs whose prompt is the actual
        # task spec, like the Graphs/* envs).
        meta = getattr(obs, "metadata", None) or {}
        if "text_prompt" in meta:
            status_lines.append("")
            status_lines.append("text_prompt:")
            status_lines.append(str(meta["text_prompt"]))
        elif "state_text" in meta:
            status_lines.append("")
            status_lines.append("state_text:")
            status_lines.append(str(meta["state_text"]))

        description = _description_for_agent_0(env)
        return image, "\n".join(status_lines), description
    finally:
        env.close()


def on_env_change(env_id: str) -> tuple[str, str]:
    """Update kwargs and blurb when the dropdown selection changes."""
    kwargs, blurb = GAMES.get(env_id, ("{}", ""))
    return kwargs, blurb


def build_app() -> gr.Blocks:
    default_kwargs, default_blurb = GAMES[DEFAULT_ENV_ID]

    with gr.Blocks(title="Gym-V added-envs dataset visualizer") as app:
        gr.Markdown(
            "# Gym-V added-envs dataset visualizer\n"
            "Pick an env, optionally tweak kwargs JSON, change the seed, click "
            "**Reset** to render that dataset entry's observation. The dropdown "
            "lists every env added to this worktree under "
            "`resources_servers/gym_v/data/*_path_b.jsonl`."
        )

        with gr.Row():
            env_id = gr.Dropdown(
                choices=list(GAMES.keys()),
                value=DEFAULT_ENV_ID,
                label="Env ID",
                interactive=True,
            )
            seed = gr.Number(
                value=DEFAULT_SEED,
                label="Seed",
                precision=0,
                interactive=True,
            )

        blurb = gr.Markdown(f"**About this env:** {default_blurb}")
        kwargs_json = gr.Textbox(
            value=default_kwargs,
            label="Env kwargs JSON",
            lines=2,
            interactive=True,
        )

        reset_btn = gr.Button("Reset", variant="primary")

        with gr.Row():
            image_out = gr.Image(label="Observation (agent_0)", type="pil", height=600)
            status_out = gr.Textbox(label="Status / info", lines=20, max_lines=40)

        desc_out = gr.Textbox(label="env.description", lines=10, max_lines=20)

        # Auto-fill kwargs + blurb when the env selection changes.
        env_id.change(
            fn=lambda eid: (
                GAMES.get(eid, ("{}", ""))[0],
                f"**About this env:** {GAMES.get(eid, ('', ''))[1]}",
            ),
            inputs=env_id,
            outputs=[kwargs_json, blurb],
        )

        reset_btn.click(
            fn=reset_env,
            inputs=[env_id, kwargs_json, seed],
            outputs=[image_out, status_out, desc_out],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
