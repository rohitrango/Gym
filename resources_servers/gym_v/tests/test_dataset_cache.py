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
import sys
import types

import pytest

from resources_servers.gym_v import _dataset_cache


def _register_module(monkeypatch: pytest.MonkeyPatch, module_path: str, cls: type) -> None:
    module = types.ModuleType(module_path)
    setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, module_path, module)
    monkeypatch.setattr(
        _dataset_cache,
        "_REASONING_GYM_ENV_CLASSES",
        [(module_path, cls.__name__)],
    )


def test_install_dataset_cache_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubEnv:
        def _make_dataset(self, *, seed=None):
            return {"seed": seed}

    _register_module(monkeypatch, "tests.stub_cache_idempotent", StubEnv)

    assert _dataset_cache.install_dataset_cache() == 1
    assert _dataset_cache.install_dataset_cache() == 0
    assert getattr(StubEnv._make_dataset, "_gym_v_cached") is True


def test_cache_hit_after_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubEnv:
        calls = 0

        def __init__(self):
            self._dataset_kwargs = {"size": 5}

        def _make_dataset(self, *, seed=None):
            type(self).calls += 1
            return object()

    _register_module(monkeypatch, "tests.stub_cache_hit", StubEnv)
    _dataset_cache.install_dataset_cache()

    env_a = StubEnv()
    env_b = StubEnv()
    first = env_a._make_dataset(seed=1)
    second = env_b._make_dataset(seed=1)

    assert first is second
    assert StubEnv.calls == 1


def test_different_kwargs_or_seed_use_different_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubEnv:
        calls = 0

        def __init__(self, size: int):
            self._dataset_kwargs = {"size": size}

        def _make_dataset(self, *, seed=None):
            type(self).calls += 1
            return object()

    _register_module(monkeypatch, "tests.stub_cache_miss", StubEnv)
    _dataset_cache.install_dataset_cache()

    size_5 = StubEnv(5)._make_dataset(seed=1)
    size_6 = StubEnv(6)._make_dataset(seed=1)
    seed_2 = StubEnv(5)._make_dataset(seed=2)

    assert len({id(size_5), id(size_6), id(seed_2)}) == 3
    assert StubEnv.calls == 3


def test_missing_extra_skips_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _dataset_cache,
        "_REASONING_GYM_ENV_CLASSES",
        [("missing.module", "MissingClass")],
    )

    assert _dataset_cache.install_dataset_cache() == 0


def test_hashable_handles_numpy_arrays() -> None:
    np = pytest.importorskip("numpy")

    key = _dataset_cache._hashable({"array": np.array([1, 2, 3])})

    assert key == (("array", (1, 2, 3)),)
    hash(key)
