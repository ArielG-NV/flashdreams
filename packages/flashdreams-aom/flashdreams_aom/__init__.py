# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled AVIF encoder and decoder backed by libaom."""

import os
import sys
from pathlib import Path


def encoder_path() -> Path:
    """Return the installed encoder without searching the system PATH."""
    return _executable_path("avifenc")


def decoder_path() -> Path:
    """Return the bundled decoder used to expand a recording's canvas."""
    return _executable_path("avifdec")


def _executable_path(name: str) -> Path:
    path = Path(__file__).with_name(name + (".exe" if sys.platform == "win32" else ""))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"AVIF native dependency is missing or not executable: {path}. "
            "Build and install it automatically with: "
            "uv sync --reinstall-package flashdreams-aom. "
            "Build dependencies: Git, a C/C++ compiler and Windows SDK on Windows "
            "(Visual Studio Build Tools, Desktop development with C++; "
            "run from an x64 Developer PowerShell), "
            "or a C/C++ toolchain on Linux/macOS; YASM on x86 Linux/macOS. "
            "Windows downloads YASM automatically. The isolated build installs "
            "scikit-build-core, CMake and Ninja. See the build log for missing dependencies."
        )
    return path
