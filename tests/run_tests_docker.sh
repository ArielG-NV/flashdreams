#!/usr/bin/env bash
# Run the flashsim test suite inside a fresh NVIDIA PyTorch docker container.
#
# Use this on a local machine that has docker and at least one GPU.
# The script installs flashsim + integration packages on the fly, then
# invokes pytest. Caches for uv / huggingface / flashsim are bind-mounted
# from the host so subsequent runs are fast.
#
# Usage:
#   ./tests/run_tests_docker.sh [TEST_TARGET...]
#
# Environment overrides:
#   FLASHSIM_TEST_IMAGE         (default: nvcr.io/nvidia/pytorch:26.02-py3)
#   FLASHSIM_UV_CACHE_DIR       (default: ${HOME}/.cache/uv)
#   FLASHSIM_HF_CACHE_DIR       (default: ${HOME}/.cache/huggingface)
#   FLASHSIM_CACHE_DIR          (default: ${HOME}/.cache/flashsim)
#   FLASHSIM_TRITON_CACHE_DIR   (default: ${HOME}/.cache/triton)
#
# Examples:
#   # Run all tests
#   ./tests/run_tests_docker.sh
#
#   # Run a specific test file
#   ./tests/run_tests_docker.sh flashsim/tests/test_attention.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHSIM_TEST_IMAGE:-nvcr.io/nvidia/pytorch:26.02-py3}"

UV_CACHE_HOST="${FLASHSIM_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHSIM_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHSIM_CACHE_HOST="${FLASHSIM_CACHE_DIR:-${HOME}/.cache/flashsim}"
TRITON_CACHE_HOST="${FLASHSIM_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHSIM_CACHE_HOST}" "${TRITON_CACHE_HOST}"

docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/flashsim" \
    -v "${UV_CACHE_HOST}:/root/.cache/uv" \
    -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
    -v "${FLASHSIM_CACHE_HOST}:/root/.cache/flashsim" \
    -v "${TRITON_CACHE_HOST}:/root/.cache/triton" \
    -e HF_HOME=/root/.cache/huggingface \
    -e UV_LINK_MODE=copy \
    -e TRITON_CACHE_DIR=/root/.cache/triton \
    -w /workspace/flashsim \
    "${IMAGE}" \
    bash -s -- "$@" <<'EOF'
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --break-system-packages --no-cache-dir uv
fi

INSTALL_TARGETS=("flashsim[dev]")
for integration_dir in integrations/*; do
    if [[ -f "${integration_dir}/pyproject.toml" ]]; then
        INSTALL_TARGETS+=("${integration_dir}[dev]")
    fi
done

INSTALL_ARGS=()
for target in "${INSTALL_TARGETS[@]}"; do
    INSTALL_ARGS+=("-e" "${target}")
done

uv pip install --system --break-system-packages --no-build-isolation "${INSTALL_ARGS[@]}"

# mediapy.read_video shells out to ffmpeg; the base image doesn't ship it.
if ! command -v ffmpeg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ffmpeg
fi

exec bash /workspace/flashsim/tests/_run_tests.sh "$@"
EOF
