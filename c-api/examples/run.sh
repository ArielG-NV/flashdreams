#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
api_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="$(cat "$api_dir/build/python-executable.txt")"
export PATH="$(cd -- "$(dirname -- "${python_executable//\\//}")" && pwd -P):$PATH"
example="$api_dir/build/flashdreams_example"
if [[ -f "$api_dir/build/Release/flashdreams_example" ]]; then
    example="$api_dir/build/Release/flashdreams_example"
elif [[ -f "$api_dir/build/Release/flashdreams_example.exe" ]]; then
    example="$api_dir/build/Release/flashdreams_example.exe"
elif [[ -f "$example.exe" ]]; then
    example="$example.exe"
fi
cd "$api_dir/build"
exec "$example" "$python_executable" "$@"
