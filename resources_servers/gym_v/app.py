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

import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import gym_v
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from gym_v import Observation
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import SimpleResourcesServer

# NOTE: absolute imports (matching aviary_agent/app.py and other resources
# servers) — NeMo-Gym's `cli.py` launches this with `python app.py` (script
# mode), where relative imports raise `ImportError: attempted relative import
# with no known parent package`. Modules-style invocation (`python -m
# resources_servers.gym_v.app`) would work with `from .x import y`, but
# that's an upstream NeMo-Gym change with bigger blast radius. Absolute
# imports here are the local fix.
from resources_servers.gym_v._dataset_cache import install_dataset_cache
from resources_servers.gym_v._observation import (
    _attach_env_info,
    observation_to_user_message,
)
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVAgentVerifyResponse,
    GymVCloseRequest,
    GymVCloseResponse,
    GymVEnvStateEasyInputMessage,
    GymVResourcesServerConfig,
    GymVSeedSessionRequest,
    GymVSeedSessionResponse,
    GymVStepRequest,
    GymVStepResponse,
    GymVTaskRow,
)

logger = logging.getLogger(__name__)

# Pinned Gym-V commit. Mirrors the requirements.txt's
# `gym-v[...] @ git+https://github.com/ModalMinds/gym-v.git@d47a022`.
_GYM_V_GIT_PIN = "d47a022"
_GYM_V_GIT_URL = "https://github.com/ModalMinds/gym-v.git"


def _ensure_gym_v_assets_present() -> None:
    """Backfill `gym_v.envs.assets/` if the installed wheel omits it.

    Background: gym-v@d47a022's wheel metadata doesn't ship the
    `[games]` PNG assets (FrozenLake's `ice.png`, etc.), but the
    runtime imports them via `importlib.resources`. The
    `Dockerfile.gym_v_extras` overlay used to copy these in at build
    time. When the per-server gym_v venv is built at job-launch (no
    overlay), the asset copy never happens and FrozenLake `reset()`
    raises `FileNotFoundError: Asset not found: .../assets/frozenlake/ice.png`.

    This helper restores the asset directory at server startup if it's
    missing. The git clone happens at most once per per-server-venv
    instantiation; subsequent server starts reuse the populated
    directory.
    """
    import gym_v.envs

    assets_dir = Path(gym_v.envs.__file__).parent / "assets"
    sentinel = assets_dir / "frozenlake" / "ice.png"
    if sentinel.is_file():
        return  # already populated

    logger.info(
        "Backfilling gym_v assets at %s (gym_v wheel ships without them; "
        "fetching from %s @ %s).",
        assets_dir, _GYM_V_GIT_URL, _GYM_V_GIT_PIN,
    )
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--quiet", _GYM_V_GIT_URL, tmp],
            check=True,
        )
        subprocess.run(
            ["git", "-C", tmp, "checkout", "--detach", "--quiet", _GYM_V_GIT_PIN],
            check=True,
        )
        src_assets = Path(tmp) / "gym_v" / "envs" / "assets"
        if not src_assets.is_dir():
            raise FileNotFoundError(
                f"Upstream gym-v@{_GYM_V_GIT_PIN} doesn't have "
                f"gym_v/envs/assets/ at {src_assets}; cannot backfill."
            )
        assets_dir.mkdir(parents=True, exist_ok=True)
        for item in src_assets.iterdir():
            dst = assets_dir / item.name
            if dst.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
    logger.info("gym_v asset backfill complete.")


class GymVResourcesServer(SimpleResourcesServer):
    """Env-id-parametric server wrapping Gym-V single-agent envs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: GymVResourcesServerConfig

    task_rows: list[dict[str, Any]] = Field(default_factory=list)
    env_id_to_env: dict[str, gym_v.Env] = Field(default_factory=dict)
    env_id_to_total_reward: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    env_id_to_task_row: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_id_to_turn_count: dict[str, int] = Field(default_factory=lambda: defaultdict(int))

    def model_post_init(self, _ctx: Any) -> None:
        # Restore Gym-V's PNG assets if the installed wheel omits them
        # (per-server-venv builds skip the Dockerfile-level asset copy).
        _ensure_gym_v_assets_present()

        for jsonl_path in self.config.task_jsonl_fpaths:
            with Path(jsonl_path).open() as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        validated = GymVTaskRow.model_validate(row)
                    except Exception:
                        logger.exception("Invalid Gym-V task row in %s:%s", jsonl_path, line_no)
                        raise
                    self.task_rows.append(validated.model_dump(mode="json"))

        install_dataset_cache()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def seed_session(
        self, request: Request, body: GymVSeedSessionRequest
    ) -> GymVSeedSessionResponse:
        if body.task_row is not None:
            row = body.task_row
        else:
            if body.task_idx is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either task_row or task_idx must be provided.",
                )
            if body.task_idx >= len(self.task_rows):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"task_idx={body.task_idx} out of range; "
                        f"server has {len(self.task_rows)} rows. "
                        "Did the dataset reload between training restarts?"
                    ),
                )
            row = GymVTaskRow.model_validate(self.task_rows[body.task_idx])

        # Per-row env_kwargs override the server default.
        effective_env_kwargs = {
            "disable_text_feedback": self.config.disable_text_feedback,
            **row.env_kwargs,
        }
        try:
            env = await run_in_threadpool(gym_v.make, row.env_id, **effective_env_kwargs)
            obs_dict, _info_dict = await run_in_threadpool(env.reset, seed=row.seed)
        except Exception as exc:
            logger.exception("Failed to seed env %s with kwargs=%s", row.env_id, effective_env_kwargs)
            raise HTTPException(
                status_code=500,
                detail=f"env construction failed: {type(exc).__name__}: {exc}",
            ) from exc

        if set(obs_dict.keys()) != {"agent_0"}:
            try:
                await run_in_threadpool(env.close)
            except Exception:
                logger.debug("Failed to close rejected multi-agent env", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multi-agent envs not supported in Stage A. "
                    f"env_id={row.env_id} returned agents={list(obs_dict.keys())}. "
                    "Filed under parent-plan Phase III."
                ),
            )

        env_id = str(uuid.uuid4())
        self.env_id_to_env[env_id] = env
        self.env_id_to_task_row[env_id] = row.model_dump(mode="json")
        self.env_id_to_total_reward[env_id] = 0.0
        self.env_id_to_turn_count[env_id] = 0

        obs_msg = observation_to_user_message(
            obs_dict["agent_0"],
            env_id=row.env_id,
            prefix_text=self._description_for_agent_0(env),
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
            skip_images=self.config.skip_images,
        )

        return GymVSeedSessionResponse(
            env_id=env_id,
            obs=[obs_msg],
        )

    async def step(self, request: Request, body: GymVStepRequest) -> GymVStepResponse:
        if body.env_id not in self.env_id_to_env:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )

        env = self.env_id_to_env[body.env_id]
        row = self.env_id_to_task_row[body.env_id]
        env_id_str = row["env_id"]

        # Path B action transport: action_string is the canonical (and only)
        # transport since Path A was removed. The schema's Pydantic validator
        # already enforces presence.
        answer = body.action_string

        try:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                await run_in_threadpool(env.step, {"agent_0": answer})
            )
        except Exception as exc:
            logger.warning(
                "env.step raised on env_id=%s (%s) with answer=%r: %s: %s",
                body.env_id,
                env_id_str,
                answer,
                type(exc).__name__,
                exc,
            )
            recovery = self._recovery_message(
                env_id_str,
                f"Invalid action {answer!r}: {type(exc).__name__}: {exc}",
                {"env_step_exception": str(exc)},
            )
            return GymVStepResponse(
                obs=[recovery],
                reward=0.0,
                done=False,
                horizon_terminated=False,
            )

        if set(obs_dict.keys()) != {"agent_0"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multi-agent envs not supported in Stage A. "
                    f"env_id={env_id_str} returned agents={list(obs_dict.keys())}."
                ),
            )

        reward = float(reward_dict["agent_0"])
        self.env_id_to_total_reward[body.env_id] += reward

        done = bool(
            terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False)
        )
        self.env_id_to_turn_count[body.env_id] += 1

        horizon_terminated = False
        if (
            self.config.enforce_horizon_cap
            and row.get("horizon_cap") is not None
            and self.env_id_to_turn_count[body.env_id] >= row["horizon_cap"]
        ):
            done = True
            horizon_terminated = True

        obs_msg = observation_to_user_message(
            obs_dict["agent_0"],
            env_id=env_id_str,
            prefix_text=None,
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
            skip_images=self.config.skip_images,
        )
        obs_msg = _attach_env_info(obs_msg, self._agent_0_info(info_dict))

        return GymVStepResponse(
            obs=[obs_msg],
            reward=reward,
            done=done,
            horizon_terminated=horizon_terminated,
        )

    async def close(self, request: Request, body: GymVCloseRequest) -> GymVCloseResponse:
        env = self.env_id_to_env.pop(body.env_id, None)
        self.env_id_to_task_row.pop(body.env_id, None)
        self.env_id_to_turn_count.pop(body.env_id, None)

        if env is None:
            return GymVCloseResponse(success=True, message="already closed")

        try:
            await run_in_threadpool(env.close)
        except Exception as exc:
            logger.warning("env.close raised on env_id=%s: %s", body.env_id, exc)
            return GymVCloseResponse(success=False, message=repr(exc))

        return GymVCloseResponse(success=True, message="ok")

    async def verify(
        self, request: Request, body: GymVAgentVerifyRequest
    ) -> GymVAgentVerifyResponse:
        env_id = body.response.env_id
        known_env_id = env_id in self.env_id_to_total_reward
        reward = self.env_id_to_total_reward.pop(env_id, 0.0)
        if not known_env_id:
            logger.info("/verify drained unknown env_id=%s; returning 0.0", env_id)
        return GymVAgentVerifyResponse(response=body.response, reward=reward)

    @staticmethod
    def _description_for_agent_0(env: gym_v.Env) -> str:
        description = env.description
        if isinstance(description, str):
            return description
        return description.get("agent_0", "")

    @staticmethod
    def _agent_0_info(info_dict: dict[str, Any]) -> dict[str, Any]:
        info = info_dict.get("agent_0", info_dict)
        return info if isinstance(info, dict) else {"info": info}

    def _recovery_message(
        self,
        env_id_str: str,
        text: str,
        env_info: dict[str, Any],
    ) -> GymVEnvStateEasyInputMessage:
        recovery = observation_to_user_message(
            Observation(image=None, text=text, metadata=env_info),
            env_id=env_id_str,
            prefix_text=None,
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
        )
        return _attach_env_info(recovery, env_info)


if __name__ == "__main__":
    GymVResourcesServer.run_webserver()
