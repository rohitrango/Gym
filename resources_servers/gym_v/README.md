# Gym-V NeMo-Gym resources server

Env-id-parametric NeMo-Gym resources server that wraps any
[`gym_v.Env`](https://github.com/ModalMinds/gym-v) behind `/seed_session`,
`/step`, `/close`, and `/verify`. Design and acceptance gates live in
[`docs/design-docs/doc-1-gym-v-adapter-and-rollout-inspector.md`](../../../../docs/design-docs/doc-1-gym-v-adapter-and-rollout-inspector.md).

This directory contains the schemas + helper layer today (`schemas.py`,
`_metadata.py`, `_observation.py`, `_dataset_cache.py`) and a unit-test suite
under `tests/`. The endpoint layer (`app.py`) and integration tests against
real Gym-V envs land alongside the container image that pip-installs Gym-V.

## Layout

```
resources_servers/gym_v/
  __init__.py
  schemas.py              # Pydantic wire shapes
  _metadata.py            # JSON-safe Observation.metadata walker
  _observation.py         # observation_to_user_message + image_to_data_url
  _dataset_cache.py       # per-class LRU cache on reasoning-gym _make_dataset
  requirements.txt        # per-server runtime deps for the container build
  tests/
    conftest.py           # gym_v / reasoning_gym skipif markers
    test_schemas.py
    test_metadata.py
    test_observation.py
    test_dataset_cache.py
    test_image_to_data_url_equivalence.py
```

## Running the helper-level tests

Run from a venv with `pytest`, `openai`, `pydantic`, `pillow`, and
`nemo_gym.openai_utils` on `PYTHONPATH`:

```bash
export PYTHONPATH=<repo>:<repo>/3rdparty/Gym-workspace/Gym
cd <repo>/3rdparty/Gym-workspace/Gym
python -m pytest resources_servers/gym_v/tests -v --tb=short
```

The integration tests (`test_real_*`) auto-skip until the container image
includes the pip-installed `gym-v[games,spatial,reasoning-gym]` extras. Once
that lands, the same `pytest` invocation runs everything.

## Container image expectations

See the design doc's "Installing Gym-V" section and the operations doc
[`docs/design-docs/gym-v-container-build.md`](../../../../docs/design-docs/gym-v-container-build.md).
The container overlay installs `requirements.txt` plus
`requirements-viewer.txt` into `/opt/per_server_venvs/gym_v/` and smoke-checks
GameOfLife + FrozenLake + DoorKey reset paths before the image is allowed to
commit.
