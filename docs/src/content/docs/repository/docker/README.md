---
title: '``docker/`` -- FlashDreams base container image'
---

<a id="docker-readme--docker-flashdreams-base-container-image"></a>

This folder contains the Dockerfile and build tooling for a development base
image. Build it locally or push it to your own registry. The image does not
copy or install the FlashDreams repository; mount a checkout and install its
dependencies inside the container when you use it.

The image is based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` and adds
Python 3.12, build tools (gcc, g++, ninja), ffmpeg, libnccl-dev, uv, and the
AWS CLI v2.

---

<a id="docker-readme--contents"></a>

## Contents

| File | Purpose |
| --- | --- |
| `Dockerfile` | Base image definition. Based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. |
| `build_with_docker.sh` | Build + push a multi-arch (`linux/arm64` + `linux/amd64`) image to a registry you specify. |

---

<a id="docker-readme--building-locally"></a>

## Building locally

For a quick single-arch local image (no registry push):

```bash

docker build -t flashdreams:local -f docker/Dockerfile .

```

Then start a shell with the repository mounted (run this from the repository
root):

```bash

docker run --rm --gpus all -it \
  -v "$PWD:/workspace/flashdreams" \
  -w /workspace/flashdreams \
  flashdreams:local bash

```

---

<a id="docker-readme--building-pushing-multi-arch"></a>

## Building + pushing (multi-arch)

Use `build_with_docker.sh` to produce a multi-arch manifest (linux/arm64 +
linux/amd64) and push it to your registry:

```bash

# Log in to your target registry first
docker login <your-registry>

# Build and push -- at least one fully-qualified tag is required
bash docker/build_with_docker.sh <your-registry>/flashdreams:<your-tag>

# Multiple tags are supported
bash docker/build_with_docker.sh reg1/flashdreams:v1.0 reg2/flashdreams:latest

```

---

<a id="docker-readme--multi-arch-builds"></a>

## Multi-arch builds

`build_with_docker.sh` expects a Buildx builder that can build both
`linux/arm64` and `linux/amd64`. A default local builder can use QEMU
emulation for the non-native architecture, or you can configure your own
remote Buildx nodes outside this repository.

---

<a id="docker-readme--troubleshooting"></a>

## Troubleshooting

**`ERROR: failed to solve: ... network ...` during build.**
`build_with_docker.sh` passes `--allow network.host --network host` so apt
and PyPI traffic can use the host's configured network path.

**Buildx can't find an arm64 node.**
Run `docker buildx ls` and confirm your selected builder supports
`linux/arm64`. If it does not, configure a multi-platform builder or use
QEMU emulation for the non-native architecture.

**`docker buildx build ... --load` complains about multi-platform.**
`--load` imports a single image into the local Docker daemon and is
incompatible with multi-arch output. Drop one of the `--platform` values
if you need a local-only build for testing.
