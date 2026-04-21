#!/usr/bin/env bash
# Run the flashsim test suite in the *current* environment.
#
# Use this when you are already inside a container / virtualenv that has all
# flashsim dependencies installed. It does NOT install anything; it just
# discovers test files and invokes pytest (via tests/_run_tests.sh).
#
# Usage:
#   ./tests/run_tests_local.sh [TEST_TARGET...]
#
# Examples:
#   # Run all discovered tests
#   ./tests/run_tests_local.sh
#
#   # Run a specific test file
#   ./tests/run_tests_local.sh flashsim/tests/test_attention.py
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

python -m pip install --break-system-packages --no-build-isolation "${INSTALL_ARGS[@]}"

# mediapy.read_video shells out to ffmpeg; the base image doesn't ship it.
if ! command -v ffmpeg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ffmpeg
fi

exec bash ./tests/_run_tests.sh "$@"