# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch verified source from the official Cosmos3 image, without a container daemon."""

import hashlib
import json
import platform
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "sha256:86fb713bf73e118e537a55fa013dcdbdf2468ed61b98f4f2e7934a9482181dc1"
BASE = "https://registry-1.docker.io/v2/vllm/vllm-omni/"


def main():
    out = HERE / "results/official-image"
    out.mkdir(parents=True, exist_ok=True)
    target = HERE / "vendor/official-image"
    token = json.load(
        urllib.request.urlopen(
            "https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-omni:pull"
        )
    )["token"]

    def fetch(kind, digest):
        req = urllib.request.Request(
            BASE + kind + "/" + digest,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest.split(":")[1]:
            raise ValueError("Image digest verification failed")
        return json.loads(data)

    arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "amd64"
    index = fetch("manifests", IMAGE)
    digest = next(
        m["digest"] for m in index["manifests"] if m["platform"]["architecture"] == arch
    )
    manifest = fetch("manifests", digest)
    config = fetch("blobs", manifest["config"]["digest"])
    layers = iter(manifest["layers"])
    source = None
    for history in config["history"]:
        if history.get("empty_layer"):
            continue
        layer = next(layers)
        if "COPY . /app/vllm-omni" in history.get("created_by", ""):
            source = layer
    if source is None:
        raise ValueError("Official source layer not found")
    archive = out / (arch + "-source.tar.gz")
    if not archive.exists():
        req = urllib.request.Request(
            BASE + "blobs/" + source["digest"],
            headers={"Authorization": "Bearer " + token},
        )
        temporary = archive.with_suffix(".partial")
        with (
            urllib.request.urlopen(req, timeout=120) as response,
            temporary.open("wb") as file,
        ):
            while block := response.read(1024 * 1024):
                file.write(block)
        temporary.replace(archive)
    if (
        hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        != source["digest"].split(":")[1]
    ):
        raise ValueError("Official source archive hash mismatch")
    provenance = {
        "image": IMAGE,
        "architecture": arch,
        "manifest": digest,
        "source_layer": source["digest"],
    }
    stamp = target / "provenance.json"
    if not stamp.exists() or json.loads(stamp.read_text()) != provenance:
        if stamp.exists():
            raise ValueError(
                "Existing official source is from a different image; use a fresh folder"
            )
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            members = [
                m for m in tar if m.name == "app" or m.name.startswith("app/vllm-omni")
            ]
            tar.extractall(target, members=members, filter="data")
        stamp.write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
