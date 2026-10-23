#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cd -- "$root"
case ${1:-} in
    flashdreams)
        bench_env="$here/.venv-flashdreams"
        UV_PROJECT_ENVIRONMENT="$bench_env" uv sync --locked --python 3.12 --package flashdreams-cosmos3-nano-sim-bimanual
        ;;
    auto)
        bench_env=${COSMOS3_AUTO_ENV:-$here/.venv-auto}
        UV_PROJECT_ENVIRONMENT="$bench_env" uv sync --locked --python 3.12 --package flashdreams-cosmos3-nano-sim-auto --extra interactive-drive --package flashdreams --group cuda13
        ;;
    vllm)
        bench_env=${COSMOS3_OFFICIAL_ENV:-$here/.venv-vllm}
        "$here/.venv-flashdreams/bin/python" "$here/fetch-official.py"
        uv venv --python 3.12 --allow-existing "$bench_env"
        VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.0.dev0+cosmos3official \
            uv --no-config pip install --python "$bench_env/bin/python" \
            -e "$here/vendor/official-image/app/vllm-omni" 'vllm==0.31.0'
        ;;
    *) echo 'Usage: setup.sh [flashdreams|auto|vllm]' >&2; exit 2 ;;
esac
mkdir -p "$here/results"
uv pip freeze --python "$bench_env/bin/python" > "$here/results/${1}-packages.txt"
