# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnidreams.native import omnidreams_singleview as native


def _fake_thirdparty_info(tmp_path: Path) -> dict[str, dict[str, object]]:
    cutlass = tmp_path / "cutlass"
    (cutlass / "include").mkdir(parents=True)
    return {
        "cutlass": {
            "name": "cutlass",
            "path": str(cutlass),
            "repo": "https://github.com/NVIDIA/cutlass.git",
            "commit": "cutlass-test-sha",
            "source_sha256": "cutlass-source-hash",
            "tree_sha256": "cutlass-tree-hash",
            "stamp_path": str(cutlass / ".flashdreams_source.json"),
        },
        "SageAttention": {
            "name": "SageAttention",
            "path": str(tmp_path / "SageAttention"),
            "repo": "https://github.com/thu-ml/SageAttention.git",
            "commit": "sage-test-sha",
            "source_sha256": "sage-source-hash",
            "tree_sha256": "sage-tree-hash",
            "stamp_path": str(tmp_path / "SageAttention" / ".flashdreams_source.json"),
        },
        "SpargeAttn": {
            "name": "SpargeAttn",
            "path": str(tmp_path / "SpargeAttn"),
            "repo": "https://github.com/thu-ml/SpargeAttn.git",
            "commit": "sparge-test-sha",
            "source_sha256": "sparge-source-hash",
            "tree_sha256": "sparge-tree-hash",
            "stamp_path": str(tmp_path / "SpargeAttn" / ".flashdreams_source.json"),
        },
    }


def _fake_source_infos(helper: object, tmp_path: Path) -> dict[str, object]:
    return {
        name: helper.SourceInfo(
            name=str(info["name"]),
            path=Path(str(info["path"])),
            repo=str(info["repo"]),
            commit=str(info["commit"]),
            source_sha256=str(info["source_sha256"]),
            tree_sha256=str(info["tree_sha256"]),
            stamp_path=Path(str(info["stamp_path"])),
        )
        for name, info in _fake_thirdparty_info(tmp_path).items()
    }


@pytest.mark.ci_cpu
def test_build_info_uses_script_managed_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = native._native_build()
    monkeypatch.setattr(
        helper,
        "validate_thirdparty",
        lambda: _fake_source_infos(helper, tmp_path),
    )

    build_root = tmp_path / "native-build"
    info = native.build_info(build_root=build_root)

    assert info["build_root"] == str(build_root.resolve())
    assert info["thirdparty"]["cutlass"]["commit"] == "cutlass-test-sha"
    assert info["thirdparty"]["SageAttention"]["commit"] == "sage-test-sha"
    assert info["thirdparty"]["SpargeAttn"]["commit"] == "sparge-test-sha"
    assert info["cutlass_include"].endswith("3rdparty/cutlass/include")


@pytest.mark.ci_cpu
def test_load_extension_uses_build_root_for_torch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    build_root = tmp_path / "native-build"
    thirdparty_info = _fake_thirdparty_info(tmp_path)
    captured: dict[str, object] = {}

    def fake_load_torch_extension(**kwargs: object) -> object:
        captured.update(kwargs)
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        captured["cuda_arch_list_env"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        return SimpleNamespace()

    monkeypatch.setattr(native, "_extension", None)
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_NATIVE_MAX_JOBS", raising=False)

    extension = native.load_extension(build_root=build_root)

    assert extension is not None
    extension_name = captured["name"]
    assert captured["build_directory"] == str(
        build_root / "torch_extensions" / str(extension_name)
    )
    assert captured["extra_include_paths"] == [
        str(Path(str(thirdparty_info["cutlass"]["path"])) / "include")
    ]
    sources = [Path(str(source)).name for source in captured["sources"]]
    assert sources == ["omnidreams_singleview_ext.cpp", "omnidreams_singleview_cuda.cu"]
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA=\\\"cutlass-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SOURCE_SHA=\\\"cutlass-source-hash\\\""
        in captured["extra_cflags"]
    )
    assert any(
        flag.startswith("-DOMNIDREAMS_SINGLEVIEW_CUDA_SOURCE_SHA=")
        for flag in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA=\\\"sage-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA=\\\"sparge-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert captured["extra_cuda_cflags"] == ["-O3"]
    assert captured["with_cuda"] is True
    assert captured["max_jobs_env"] == "1"
    assert captured["cuda_arch_list_env"] == "12.0a"
    assert "MAX_JOBS" not in os.environ
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ


@pytest.mark.ci_cpu
def test_load_extension_respects_existing_max_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    captured: dict[str, object] = {}

    def fake_load_torch_extension(**kwargs: object) -> object:
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        captured["cuda_arch_list_env"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        return SimpleNamespace()

    monkeypatch.setattr(native, "_extension", None)
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: _fake_thirdparty_info(tmp_path))
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setenv("MAX_JOBS", "3")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")

    extension = native.load_extension(build_root=tmp_path / "native-build")

    assert extension is not None
    assert captured["max_jobs_env"] == "3"
    assert captured["cuda_arch_list_env"] == "8.9"
    assert os.environ["MAX_JOBS"] == "3"
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


@pytest.mark.ci_cpu
def test_load_extension_retries_after_failed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_load_torch_extension(**_: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first build failed")
        return SimpleNamespace()

    thirdparty_info = _fake_thirdparty_info(tmp_path)
    monkeypatch.setattr(native, "_extension", None)
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: thirdparty_info)
    monkeypatch.setattr("torch.utils.cpp_extension.load", fake_load_torch_extension)

    assert native.load_extension(build_root=tmp_path / "native-build") is None
    assert attempts == 1
    assert isinstance(native.extension_load_error(), RuntimeError)

    extension = native.load_extension(build_root=tmp_path / "native-build")
    assert attempts == 2
    assert extension is not None, native.extension_load_error()
    assert native.extension_load_error() is None
    assert attempts == 2


@pytest.mark.ci_cpu
def test_native_build_wraps_sync_setup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = native._native_build()

    class BrokenSyncTool:
        class ThirdPartySyncError(RuntimeError):
            pass

        @staticmethod
        def load_manifest() -> tuple[object, ...]:
            raise FileNotFoundError("missing manifest")

    monkeypatch.setattr(helper, "_sync_thirdparty_module", BrokenSyncTool)

    with pytest.raises(helper.NativeBuildError, match="missing manifest"):
        helper.validate_thirdparty()


@pytest.mark.manual
@pytest.mark.skipif(
    os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_THIRDPARTY_VERIFY") != "1",
    reason="Set OMNIDREAMS_SINGLEVIEW_RUN_THIRDPARTY_VERIFY=1 to verify downloaded sources.",
)
def test_real_thirdparty_sources_verify() -> None:
    info = native.validate_thirdparty()

    assert set(info) == {"cutlass", "SageAttention", "SpargeAttn"}


@pytest.mark.manual
@pytest.mark.skipif(
    os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST") != "1",
    reason="Set OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 to build the native extension.",
)
def test_cuda_native_extension_builds(tmp_path: Path) -> None:
    extension = native.load_extension(build_root=tmp_path)

    assert extension is not None, native.extension_load_error()
    assert extension.is_available()
    assert extension.build_info()["with_cuda"] is True
