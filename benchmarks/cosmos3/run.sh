#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cd -- "$root"
case ${1:-} in ''|--skip-setup) ;; *) echo 'Usage: run.sh [--skip-setup]' >&2; exit 2 ;; esac
export COSMOS3_AUTO_ENV=${COSMOS3_AUTO_ENV:-$here/.venv-auto}
export COSMOS3_OFFICIAL_ENV=${COSMOS3_OFFICIAL_ENV:-$here/.venv-vllm}
fd="$here/.venv-flashdreams/bin/python"
auto="$COSMOS3_AUTO_ENV/bin/python"
vo="$COSMOS3_OFFICIAL_ENV/bin/python"
out="$here/results/three-seconds"
mkdir -p "$out"
if [[ ${1:-} != --skip-setup ]]; then
    bash "$here/setup.sh" flashdreams > "$out/setup-fd.log" 2>&1
    bash "$here/setup.sh" auto > "$out/setup-auto.log" 2>&1
    bash "$here/setup.sh" vllm > "$out/setup-vllm.log" 2>&1
fi
python3 "$here/check-three-seconds.py" > "$out/check.log"
if [[ ! -s "$out/bimanual-inputs/actions.npy" ]]; then
    "$fd" "$here/prepare-bimanual-smoke.py" > "$out/prepare-bimanual.log" 2>&1
fi
if [[ ! -s "$out/auto-inputs/raw.npz" ]]; then
    "$auto" "$here/export-auto-inputs.py" > "$out/prepare-auto.log" 2>&1
fi
"$fd" "$here/matched-bimanual.py" > "$out/bimanual-fd.log" 2>&1
"$vo" "$here/benchmark-official.py" bimanual > "$out/bimanual-vllm.log" 2>&1
"$auto" "$here/benchmark-auto-fd.py" > "$out/auto-fd.log" 2>&1
"$vo" "$here/benchmark-official.py" auto > "$out/auto-vllm.log" 2>&1
python3 "$here/report-three-seconds.py"
