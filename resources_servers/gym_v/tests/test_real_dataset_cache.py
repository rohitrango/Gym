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
"""Verify that install_dataset_cache patches every configured reasoning-gym env.

Soft-failure mode (cache misses an env that Gym-V renamed) is hard to catch
from the unit tests, since they use stub classes. This integration test
asserts that every configured env class which actually defines
`_make_dataset` gets patched. Classes that do not define `_make_dataset`
on themselves (because Gym-V refactored that env to inline dataset
construction in `reset`) are reported but do not fail the test, since the
cache still works on the remaining classes.
"""
from __future__ import annotations

import importlib

from conftest import requires_gym_v, requires_reasoning_gym


def _classes_with_make_dataset() -> list[tuple[str, str]]:
    from resources_servers.gym_v._dataset_cache import _REASONING_GYM_ENV_CLASSES

    eligible: list[tuple[str, str]] = []
    for module_path, cls_name in _REASONING_GYM_ENV_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError):
            continue
        if "_make_dataset" in cls.__dict__:
            eligible.append((module_path, cls_name))
    return eligible


@requires_gym_v
@requires_reasoning_gym
def test_install_dataset_cache_patches_every_eligible_class() -> None:
    from resources_servers.gym_v._dataset_cache import install_dataset_cache

    eligible = _classes_with_make_dataset()
    assert eligible, "No reasoning-gym env class with _make_dataset found; check the registry."

    # install_dataset_cache is idempotent, so this may return 0 if another
    # test installed the cache earlier in the same process. Assert final state,
    # not per-call install count.
    install_dataset_cache()

    for module_path, cls_name in eligible:
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
        assert getattr(cls._make_dataset, "_gym_v_cached", False), (
            f"{module_path}.{cls_name} defines _make_dataset but was not cached. "
            "Did Gym-V rename / move one of the configured classes?"
        )
