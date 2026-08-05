# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the full local OmniDreams WebRTC game-mode serving path."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from omnidreams.webrtc.server import create_app
from omnidreams.webrtc.session import (
    OmnidreamsRuntimeConfig,
    OmnidreamsWebRTCSessionManager,
)


async def _drive_client(url: str, target_chunks: int) -> None:
    peer = RTCPeerConnection()
    channel = peer.createDataChannel("controls")
    peer.addTransceiver("video", direction="recvonly")
    opened = asyncio.Event()
    finished = asyncio.Event()
    chunk_count = 0
    frame_count = 0

    @channel.on("open")
    def on_open() -> None:
        opened.set()

    @channel.on("message")
    def on_message(raw: object) -> None:
        nonlocal chunk_count
        if not isinstance(raw, str):
            return
        payload = json.loads(raw)
        if payload.get("type") == "chunk_done":
            chunk_count += 1
            if chunk_count >= target_chunks:
                finished.set()

    @peer.on("track")
    def on_track(track: object) -> None:
        async def consume() -> None:
            nonlocal frame_count
            while True:
                try:
                    await track.recv()
                except Exception:
                    return
                frame_count += 1

        asyncio.create_task(consume())

    offer = await peer.createOffer()
    await peer.setLocalDescription(offer)
    while peer.iceGatheringState != "complete":
        await asyncio.sleep(0.05)
    local = peer.localDescription
    assert local is not None
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/api/webrtc/offer",
            json={"sdp": local.sdp, "type": local.type},
        ) as response:
            response.raise_for_status()
            answer = await response.json()
    await peer.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    await asyncio.wait_for(opened.wait(), timeout=30.0)
    channel.send(
        json.dumps(
            {"type": "action", "action": {"event": "keydown", "key": "w"}}
        )
    )

    async def heartbeat() -> None:
        while not finished.is_set():
            await asyncio.sleep(5.0)
            if channel.readyState == "open":
                channel.send(json.dumps({"type": "heartbeat", "t": time.time_ns()}))

    heartbeat_task = asyncio.create_task(heartbeat())
    started = time.monotonic()
    try:
        await asyncio.wait_for(finished.wait(), timeout=180.0)
        elapsed = time.monotonic() - started
        print(
            f"WebRTC client chunks={chunk_count} decoded_frames={frame_count} "
            f"elapsed_s={elapsed:.3f} decoded_fps={frame_count / elapsed:.3f}",
            flush=True,
        )
        channel.send(
            json.dumps(
                {"type": "action", "action": {"event": "keyup", "key": "w"}}
            )
        )
        channel.send(json.dumps({"type": "disconnect"}))
    finally:
        heartbeat_task.cancel()
        await peer.close()


async def _run(args: argparse.Namespace) -> None:
    manager = OmnidreamsWebRTCSessionManager(
        runtime_config=OmnidreamsRuntimeConfig(
            pipeline_config_name=args.pipeline_config_name,
            scene_uuid=args.scene_uuid,
            device=args.device,
            fps=args.fps,
            game_mode=True,
            warmup_chunks=args.warmup_chunks,
        )
    )
    url = f"http://127.0.0.1:{args.port}"
    app = create_app(
        session_manager=manager,
        request_session_url=f"{url}/request_session",
        auto_start=False,
    )
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, host="127.0.0.1", port=args.port)
        await site.start()
        await _drive_client(url, args.chunks)
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-config-name",
        default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
    )
    parser.add_argument(
        "--scene-uuid", default="0d404ff7-2b66-498c-b047-1ed8cded60d4"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-chunks", type=int, default=6)
    parser.add_argument("--chunks", type=int, default=55)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
