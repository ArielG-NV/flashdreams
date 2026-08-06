# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.utils.cpp_extension
from ludus_renderer._ops import _plugin

pytestmark = pytest.mark.ci_cpu


def test_prefer_cuda13_toolkit_updates_pytorch_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_nvcc = tmp_path / "cuda-12.0" / "bin" / "nvcc"
    new_nvcc = tmp_path / "cuda-13.3" / "bin" / "nvcc"
    for nvcc in (old_nvcc, new_nvcc):
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()

    monkeypatch.setattr(_plugin.shutil, "which", lambda _name: str(old_nvcc))
    monkeypatch.setattr(
        _plugin.glob,
        "glob",
        lambda pattern: [str(new_nvcc)] if pattern.startswith("/usr/local") else [],
    )

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        version = "13.3" if "13.3" in command[0] else "12.0"
        return SimpleNamespace(
            stdout=f"Cuda compilation tools, release {version}, V0",
            stderr="",
        )

    monkeypatch.setattr(_plugin.subprocess, "run", fake_run)
    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", "/old/cuda")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    toolkit = _plugin._prefer_cuda13_toolkit()

    assert toolkit == str(new_nvcc.parent.parent)
    assert torch.utils.cpp_extension.CUDA_HOME == toolkit
    assert _plugin.os.environ["CUDA_HOME"] == toolkit
    assert _plugin.os.environ["PATH"].startswith(f"{new_nvcc.parent}:")
