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

"""CPU contract tests for MIRA WebRTC serving."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer
from mira_integration.configs.manifest import load_demo_config, load_manifest
from mira_integration.configs.schema import preview_grid_dimensions
from mira_integration.transformer import MiraTransformerConfig
from mira_integration.webrtc.media import video_chunk_to_rgb_frames
from mira_integration.webrtc.room import (
    MiraBrowserSession,
    MiraMultiplayerSessionManager,
    tile_player_video,
)
from mira_integration.webrtc.server import build_runtime_config, create_app, parse_args
from mira_integration.webrtc.session import (
    MiraInferenceRuntime,
    MiraRuntimeConfig,
    checkpoint_keys,
)

from flashdreams.serving.webrtc.server import PACKAGE_RESOURCE_STACK_KEY

pytestmark = pytest.mark.ci_cpu

MANIFEST_PATH = (
    Path(__file__).parents[1] / "mira_integration" / "configs" / "mira_car_soccer.yaml"
)
DEMO_1P = load_demo_config(MANIFEST_PATH, "mira-mini-1p")
DEMO_4P = load_demo_config(MANIFEST_PATH, "mira-mini-4p")


class _FakePipeline:
    def __init__(self) -> None:
        self.generated_keys: list[list[list[str] | None]] = []
        self.closed = False
        self.initialize_cache_calls = 0
        self.restore_cache_calls = 0

    def to(self, **_kwargs: Any) -> _FakePipeline:
        return self

    def eval(self) -> _FakePipeline:
        return self

    def initialize_cache(self, *, n_diffusion_steps: int) -> dict[str, int]:
        self.initialize_cache_calls += 1
        return {"n_diffusion_steps": n_diffusion_steps}

    def restore_cache(self, cache: dict[str, int]) -> None:
        del cache
        self.restore_cache_calls += 1

    def generate(
        self,
        autoregressive_index: int,
        cache: object,
        input: list[list[str] | None],
    ) -> torch.Tensor:
        del autoregressive_index, cache
        self.generated_keys.append(input)
        values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        return values.view(1, 1, 3, 1, 1).expand(4, 2, 3, 2, 2).clone()

    def finalize(self, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"total_ms": 4.0}

    def close(self) -> None:
        self.closed = True


class _FakeSessionManager:
    def __init__(self) -> None:
        self.preload_calls = 0
        self.shutdown_calls = 0

    def has_active_session(self) -> bool:
        return False

    def is_runtime_ready(self) -> bool:
        return self.preload_calls > 0

    async def preload_runtime(self) -> None:
        self.preload_calls += 1

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        del offer_sdp, offer_type
        return {"sdp": "answer-sdp", "type": "answer"}

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def public_config(self) -> dict[str, Any]:
        return DEMO_4P.metadata.to_public_dict()


def test_checkpoint_keys_follow_mira_vocabulary_order() -> None:
    assert checkpoint_keys(
        frozenset({"control", "w", "space", "q", "shift"}),
        DEMO_4P.metadata,
    ) == [
        "W",
        "Q",
        "Space",
        "LShiftKey",
        "LControlKey",
    ]


def test_runtime_config_derives_media_shape_from_manifest() -> None:
    config = MiraRuntimeConfig(model_config=DEMO_4P)
    assert config.video_width == DEMO_4P.metadata.video_width
    assert config.video_height == DEMO_4P.metadata.video_height
    assert config.frames_per_chunk == DEMO_4P.metadata.frames_per_chunk


@pytest.mark.parametrize(
    "argv,missing",
    [
        (["--demo", "mira-mini-1p"], "--manifest"),
        (["--manifest", str(MANIFEST_PATH)], "--demo"),
    ],
)
def test_server_requires_manifest_and_demo(
    argv: list[str], missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    "removed_option",
    ("--bundle-path", "--checkpoint-path", "--context-path"),
)
def test_server_rejects_asset_path_overrides(
    removed_option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--manifest",
        str(MANIFEST_PATH),
        "--demo",
        "mira-mini-1p",
        removed_option,
        "unused",
    ]
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert f"unrecognized arguments: {removed_option}" in capsys.readouterr().err


def test_webrtc_runtime_config_enables_fast_transformer_path() -> None:
    args = parse_args(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "--demo",
            "mira-mini-1p",
        ]
    )
    config = build_runtime_config(args)
    transformer = config.model_config.pipeline.diffusion_model.transformer
    assert transformer.compile_network is True
    assert transformer.use_cuda_graph is True
    assert transformer.cuda_graph_warmup_iters == 2


def test_webrtc_runtime_config_can_disable_cuda_graphs() -> None:
    args = parse_args(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "--demo",
            "mira-mini-1p",
            "--no-compile-network",
            "--no-cuda-graph",
            "--cuda-graph-warmup-iters",
            "0",
        ]
    )
    config = build_runtime_config(args)
    transformer = config.model_config.pipeline.diffusion_model.transformer
    assert transformer.compile_network is False
    assert transformer.use_cuda_graph is False
    assert transformer.cuda_graph_warmup_iters == 0


@pytest.mark.asyncio
async def test_runtime_translates_latest_keys_and_keeps_chunk_in_model_range() -> None:
    pipeline = _FakePipeline()
    runtime = MiraInferenceRuntime(
        config=MiraRuntimeConfig(
            model_config=DEMO_4P,
            device="cpu",
            n_diffusion_steps=3,
        ),
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    try:
        await runtime.initialize()
        await runtime.reset_for_new_session()
        result = await runtime.generate_chunk(
            player_keys=(
                frozenset({"w", "d", "space"}),
                frozenset(),
                frozenset({"q"}),
                frozenset({"shift"}),
            ),
        )

        assert pipeline.generated_keys == [
            [["W", "D", "Space"], [], ["Q"], ["LShiftKey"]]
        ]
        assert result.chunk_index == 0
        assert result.num_frames == 2
        assert result.video_chunk.dtype == torch.float32
        assert result.video_chunk.shape == (4, 2, 3, 2, 2)
        assert result.video_chunk[:, :, 0].unique().item() == 0.0
        assert result.video_chunk[:, :, 1].unique().item() == 0.5
        assert result.video_chunk[:, :, 2].unique().item() == 1.0
        assert result.stats == {"total_ms": 4.0}

        frames = video_chunk_to_rgb_frames(result.video_chunk[0])
        assert len(frames) == 2
        assert frames[0].dtype == "uint8"
        assert frames[0].shape == (2, 2, 3)
        assert frames[0][:, :, 0].max().item() == 0
        assert frames[0][:, :, 1].max().item() == 128
        assert frames[0][:, :, 2].max().item() == 255
    finally:
        await runtime.close()
    assert pipeline.closed


@pytest.mark.asyncio
async def test_runtime_render_uses_latest_published_input_state() -> None:
    pipeline = _FakePipeline()
    runtime = MiraInferenceRuntime(
        config=MiraRuntimeConfig(model_config=DEMO_4P, device="cpu"),
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    try:
        await runtime.initialize()
        await runtime.reset_for_new_session()

        runtime.publish_player_keys(
            (
                frozenset({"w"}),
                None,
                None,
                None,
            )
        )
        runtime.publish_player_keys(
            (
                frozenset({"d"}),
                frozenset({"space"}),
                None,
                frozenset(),
            )
        )
        result = await runtime.render_next_chunk()

        assert pipeline.generated_keys == [[["D"], ["Space"], None, []]]
        assert result.chunk_index == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_reset_restores_preloaded_cache_in_place() -> None:
    pipeline = _FakePipeline()
    runtime = MiraInferenceRuntime(
        config=MiraRuntimeConfig(model_config=DEMO_4P, device="cpu"),
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    try:
        await runtime.initialize()
        await runtime.reset_for_new_session()
        initial_cache = runtime._cache
        await runtime.reset_for_new_session()

        assert runtime._cache is initial_cache
        assert pipeline.initialize_cache_calls == 1
        assert pipeline.restore_cache_calls == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_multiplayer_preload_warms_manual_branch_before_in_place_reset() -> None:
    pipeline = _FakePipeline()
    config = MiraRuntimeConfig(
        model_config=DEMO_4P,
        device="cpu",
        warmup_chunks=3,
    )
    runtime = MiraInferenceRuntime(
        config=config,
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    manager = MiraMultiplayerSessionManager(runtime=runtime)
    try:
        await manager.preload_runtime()

        assert pipeline.initialize_cache_calls == 1
        assert pipeline.restore_cache_calls == 1
        assert pipeline.generated_keys == [
            [[], None, None, None],
            [[], None, None, None],
            [[], None, None, None],
        ]
        assert manager.is_runtime_ready()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_packaged_app_serves_mira_control_ui() -> None:
    manager = _FakeSessionManager()
    app = create_app(
        request_session_url="http://127.0.0.1:8083/request_session",
        session_manager=manager,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/request_session")
        html = await response.text()
        assert response.status == 200
        assert manager.preload_calls == 1
        assert "MIRA Mini · FlashDreams" in html
        assert '<div class="playerGrid" id="playerGrid"></div>' in html
        assert '<div class="controlGrid" id="controlGrid"></div>' in html
        assert "request_session.js?v=preview-phases" in html
        assert html.index('class="previewControls"') > html.index("</section>")
        assert html.index('id="modelValue"') > html.index('id="playerGrid"')

        response = await client.get("/api/mira/config")
        model_config = await response.json()
        assert response.status == 200
        assert model_config["checkpoint"] == "alakazamworld/mira-mini-4p"
        assert model_config["playerCount"] == 4
        assert {item["key"] for item in model_config["inputs"]} == {
            "w",
            "a",
            "s",
            "d",
            "q",
            "e",
            "space",
            "shift",
            "control",
        }

        response = await client.get("/static/request_session.js")
        javascript = await response.text()
        assert response.status == 200
        assert 'addTransceiver("video", { direction: "recvonly" })' in javascript
        assert 'createDataChannel("mira-controls"' in javascript
        assert 'fetch("/api/mira/offer"' in javascript
        assert 'fetch("/api/mira/config"' in javascript
        assert 'sendMessage({ type: "claim", seat })' in javascript
        assert 'sendMessage({ type: "release" })' in javascript
        assert 'label.textContent = "Disconnect Preview"' in javascript
        assert 'label.textContent = "End Session"' in javascript
        assert 'payload.type === "seat_released"' in javascript
        assert 'document.execCommand("copy")' in javascript
        assert "window.isSecureContext" in javascript
        assert 'report.type === "inbound-rtp"' in javascript
        assert "framesPerSecond" in javascript
        assert "config.playerCount" in javascript

        response = await client.get("/static/request_session.css")
        stylesheet = await response.text()
        assert response.status == 200
        assert ".controlDeck" in stylesheet
        assert "body.has-video #remoteVideo" in stylesheet
    finally:
        await client.close()


def test_packaged_app_keeps_web_resource_alive() -> None:
    app = create_app(
        request_session_url="http://127.0.0.1:8083/request_session",
        session_manager=_FakeSessionManager(),
    )
    try:
        assert isinstance(app[PACKAGE_RESOURCE_STACK_KEY], ExitStack)
    finally:
        app[PACKAGE_RESOURCE_STACK_KEY].close()


def test_runtime_config_reports_native_output_shape() -> None:
    config = MiraRuntimeConfig(model_config=DEMO_4P)
    assert (config.video_height, config.video_width) == (
        DEMO_4P.metadata.video_height,
        DEMO_4P.metadata.video_width,
    )
    assert config.frames_per_chunk == DEMO_4P.metadata.frames_per_chunk


def test_model_metadata_drives_player_count_and_checkpoint_keys() -> None:
    assert DEMO_4P.metadata.player_count == 4
    assert DEMO_4P.metadata.checkpoint_keys(frozenset({"control", "w", "space"})) == [
        "W",
        "Space",
        "LControlKey",
    ]


def test_manifest_generates_mira_test_pipeline() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    metadata = manifest.demos["mira-mini-1p"]
    assert metadata.checkpoint == "alakazamworld/mira-mini"
    assert metadata.input_key_map is manifest.input_maps["car-soccer"]
    assert DEMO_1P.pipeline.name == "mira-mini-1p"
    assert DEMO_1P.pipeline.model_repo == metadata.checkpoint
    assert DEMO_1P.pipeline.n_players == metadata.player_count == 1
    assert DEMO_1P.pipeline.n_context_frames == metadata.n_context_frames == 39
    assert DEMO_4P.pipeline.n_context_frames == DEMO_4P.metadata.n_context_frames == 78
    one_player_transformer = DEMO_1P.pipeline.diffusion_model.transformer
    four_player_transformer = DEMO_4P.pipeline.diffusion_model.transformer
    assert isinstance(one_player_transformer, MiraTransformerConfig)
    assert isinstance(four_player_transformer, MiraTransformerConfig)
    assert one_player_transformer.action_guidance_scale == 1.0
    assert four_player_transformer.action_guidance_scale == 4.0


def test_manifest_and_demo_selection_are_required() -> None:
    with pytest.raises(ValueError, match="manifest path is required"):
        load_manifest(None)
    with pytest.raises(ValueError, match="demo name is required"):
        load_demo_config(MANIFEST_PATH, None)
    with pytest.raises(ValueError, match="Unknown MIRA demo 'missing'"):
        load_demo_config(MANIFEST_PATH, "missing")


def test_manifest_rejects_unknown_input_map(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.yaml"
    manifest_path.write_text(
        """
input-map:
  controls:
    forward:
      browser_key: [w]
      checkpoint_key: W
      label: W
      action: Forward
      group: Movement
demos:
  mira-mini-1p:
    checkpoint-hugging-face: example/model
    input-map: missing
    player_count: 1
    display_name: Test
""".strip()
    )
    with pytest.raises(ValueError, match="unknown map 'missing'"):
        load_manifest(manifest_path)


def test_manifest_requires_context_frame_count(tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing-context.yaml"
    manifest_path.write_text(
        MANIFEST_PATH.read_text(encoding="utf-8").replace(
            "    n_context_frames: 39\n",
            "",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"demos\.mira-mini-1p\.n_context_frames must be a positive integer",
    ):
        load_manifest(manifest_path)


def test_preview_grid_and_tiling_support_eight_players() -> None:
    assert preview_grid_dimensions(8) == (3, 3)
    video = torch.arange(8, dtype=torch.uint8).view(8, 1, 1, 1, 1)
    preview = tile_player_video(video)
    assert preview.shape == (1, 1, 3, 3)
    assert preview.flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 0]


@pytest.mark.asyncio
async def test_release_returns_player_session_to_preview() -> None:
    pipeline = _FakePipeline()
    runtime = MiraInferenceRuntime(
        config=MiraRuntimeConfig(model_config=DEMO_4P, device="cpu"),
        pipeline_factory=lambda _config: pipeline,  # ty: ignore[invalid-argument-type]
    )
    manager = MiraMultiplayerSessionManager(runtime=runtime)
    sent: list[str] = []
    session = MiraBrowserSession(
        session_id="browser-1",
        peer_connection=object(),
        video_track=object(),  # ty: ignore[invalid-argument-type]
        seat=2,
        held_keys={"w", "space"},
        control_channel=SimpleNamespace(readyState="open", send=sent.append),
    )
    manager._sessions[session.session_id] = session
    manager._seat_owners[2] = session.session_id

    try:
        await manager._handle_message(session, json.dumps({"type": "release"}))

        assert session.seat is None
        assert session.held_keys == set()
        assert 2 not in manager._seat_owners
        assert json.loads(sent[-1]) == {"type": "seat_released", "seat": 2}
    finally:
        await runtime.close()
