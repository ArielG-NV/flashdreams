# Test environment matching nvcr.io/nvidian/qiwu:fast-infer-v9
#
# Build:
#   docker build -t gitlab-master.nvidia.com:5005/sil/flashsim:base-v0.1dev0 -f docker/Dockerfile .
#
# Run tests (GPU required):
#   docker/run_tests.sh
#
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace/flashsim

# Copy only manifests for dependency installation; source is bind-mounted at
# runtime by docker/run_tests.sh.
COPY flashsim/pyproject.toml flashsim/README.md /workspace/flashsim/flashsim/

# Stub package directory so editable install can be created during image build.
RUN mkdir -p /workspace/flashsim/flashsim/flashsim && touch /workspace/flashsim/flashsim/flashsim/__init__.py

# Install dependencies from the real flashsim package manifest.
# torch and transformer-engine are pre-installed by the base image with NVIDIA
# custom builds; the >=2.11.0a0 / >=2.12 lower bounds accept them.
RUN uv pip install --system --break-system-packages --no-cache-dir -e "/workspace/flashsim/flashsim[dev,streaming,streaming_viser]"

# Source code is bind-mounted at runtime (see docker/run_tests.sh).
# Include monorepo package roots so integration packages are importable.
ENV PYTHONPATH=/workspace/flashsim/flashsim:/workspace/flashsim/integrations/alpadreams:/workspace/flashsim/integrations/causal_wan2_1:/workspace/flashsim/integrations/causal_wan2_2:/workspace/flashsim/integrations/lingbot_world:/workspace/flashsim/integrations/streaming_ws:/workspace/flashsim/integrations/wan2_1
