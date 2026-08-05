# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
from omnidreams.interactive_drive.backends.world_model import (
    _physx_elapsed_ms,
    _total_chunk_elapsed_ms,
)

pytestmark = pytest.mark.ci_cpu


def test_physx_elapsed_ms_prefers_detailed_timing() -> None:
    trajectory = SimpleNamespace(
        physx_timings=SimpleNamespace(total_ms=338.1),
        physx_elapsed_s=0.123,
    )

    assert _physx_elapsed_ms(trajectory) == pytest.approx(338.1)


def test_physx_elapsed_ms_falls_back_to_elapsed_seconds() -> None:
    trajectory = SimpleNamespace(physx_timings=None, physx_elapsed_s=0.3381)

    assert _physx_elapsed_ms(trajectory) == pytest.approx(338.1)


def test_total_chunk_elapsed_ms_includes_physx() -> None:
    trajectory = SimpleNamespace(
        physx_timings=SimpleNamespace(total_ms=338.1),
        physx_elapsed_s=0.3381,
    )

    total_ms = _total_chunk_elapsed_ms(trajectory, render_elapsed_s=0.2586)

    assert total_ms == pytest.approx(596.7)
