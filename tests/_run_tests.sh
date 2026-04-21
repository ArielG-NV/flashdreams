#!/usr/bin/env bash
# INTERNAL helper: discover flashsim test files and invoke pytest.
#
# It assumes flashsim and its integrations are already installed in the active
# Python environment. CWD is changed to the repo root so the discovery globs
# resolve correctly regardless of where the caller invoked us from.
#
# Usage:
#   ./tests/_run_tests.sh [TEST_TARGET...]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 ]]; then
    TEST_TARGETS=()
    shopt -s nullglob
    for test_file in flashsim/tests/test_*.py integrations/*/tests/test_*.py tests/test_*.py; do
        TEST_TARGETS+=("${test_file}")
    done
    shopt -u nullglob
else
    TEST_TARGETS=("$@")
fi

if [[ ${#TEST_TARGETS[@]} -eq 0 ]]; then
    echo "No test targets found (expected flashsim/tests, integrations/*/tests, or tests)." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-$(command -v python3)}"
echo "=== Running pytest for ${#TEST_TARGETS[@]} target(s) using ${PYTHON_BIN} ==="
exec "${PYTHON_BIN}" -m pytest -m "not manual" "${TEST_TARGETS[@]}"
