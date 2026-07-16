#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Capture an interactive-drive Nsight Systems report with Python and CUDA stacks.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
output_dir="${NSYS_OUTPUT_DIR:-${repo_root}/profiles}"

capture_main_loop=false
if [[ "${1:-}" == "--capture-main-loop" ]]; then
    capture_main_loop=true
    shift
fi

web_server_cmd="--stream-mjpeg :8080"
if [[ "${1:-}" == "--native-ui" ]]; then
    web_server_cmd=""
    shift
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: profile_interactive_drive.sh [--capture-main-loop] [interactive-drive options]

  --capture-main-loop  Start collection when the renderer enters run_main_loop.
EOF
    exit 0
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "Nsight Systems CLI (nsys) is required but was not found on PATH." >&2
    exit 127
fi

mkdir -p "${output_dir}"
output="${output_dir}/interactive-drive-$(date -u +%Y-%m-%d-at-%H-%M-%S-UTC)"

capture_args=()
if "${capture_main_loop}"; then
    capture_args=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    export INTERACTIVE_DRIVE_NSIGHT_CAPTURE_MAIN_LOOP=1
fi

uv run --package flashdreams-omnidreams omnidreams-prepare --perf
uv sync --package flashdreams-omnidreams --extra interactive-drive
NVTX_PROFILE_PYTHON=1 exec nsys profile "${capture_args[@]}" \
    --trace=cuda,nvtx,osrt,python-gil \
    --sample=process-tree \
    --samples-per-backtrace=1 \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    --cudabacktrace=all \
    --pytorch=autograd-shapes-nvtx \
    --python-backtrace=cuda \
    --backtrace=dwarf \
    --cuda-event-trace=true \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output "${output}" \
    -- \
    uv run --package flashdreams-omnidreams interactive-drive \
      ${web_server_cmd} \
       --manifest example_world_model_perf.yaml \
    --auto-start \
    "$@"
