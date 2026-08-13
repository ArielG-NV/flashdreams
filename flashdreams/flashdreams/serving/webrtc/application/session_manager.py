# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session manager for WebRTC-hosted FlashDreams applications."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from flashdreams.demo.application import (
    IFlashDreamsApplicationSession,
    create_application,
)
from flashdreams.demo.factories import NullInputSink
from flashdreams.demo.io import (
    InputSink,
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.infra.results import StepResult
from flashdreams.runtime.output import OutputArtifact
from flashdreams.serving.webrtc.application.factory import WebRTCIOFactory
from flashdreams.serving.webrtc.media import BufferedVideoTrack
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.services import (
    WebRTCOutputBridge,
    WebRTCOutputBridgeDecision,
)
from flashdreams.serving.webrtc.warmup import wait_for_ice_gathering_complete


class BufferedTrackOutputBridge(WebRTCOutputBridge):
    """Deliver canonical video chunks to one bounded WebRTC media track."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        track: BufferedVideoTrack,
    ) -> None:
        self._loop = loop
        self._track = track
        self._closed = False
        self._generation = 0

    def begin_generation(self, generation: int) -> None:
        """Flush queued frames when a newer generation begins."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        if self._closed or generation <= self._generation:
            return
        self._generation = generation
        asyncio.run_coroutine_threadsafe(self._track.flush(), self._loop).result()

    def submit_chunk(
        self,
        result: StepResult,
        *,
        generation: int,
        force_keyframe: bool = False,
    ) -> WebRTCOutputBridgeDecision:
        """Convert and enqueue a complete chunk with bounded backpressure."""
        del force_keyframe
        if self._closed:
            return WebRTCOutputBridgeDecision(
                accepted=False,
                should_stop=True,
                dropped=True,
                drop_policy="drop_newest",
                metadata={"reason": "closed"},
            )
        if generation < self._generation:
            return WebRTCOutputBridgeDecision(
                accepted=False,
                dropped=True,
                drop_policy="drop_newest",
                metadata={"reason": "stale generation"},
            )
        started = time.monotonic()
        frames = self._track.prepare_result_frames(result)
        accepted = asyncio.run_coroutine_threadsafe(
            self._track.enqueue_frames(frames),
            self._loop,
        ).result()
        elapsed = time.monotonic() - started
        stopped = accepted != len(frames)
        return WebRTCOutputBridgeDecision(
            accepted=not stopped,
            should_stop=stopped,
            dropped=stopped,
            drop_policy="drop_newest" if stopped else "none",
            metadata={
                "accepted_frames": accepted,
                "queue_depth": self._track.qsize(),
                "write_blocked_s": elapsed,
            },
        )

    def close(self) -> None:
        """Stop accepting new chunks without truncating queued playback."""
        self._closed = True


class _DeferredOutputSink(OutputSink):
    """Preserve one sink identity while model metadata determines the track."""

    produces_artifacts = False

    def __init__(self) -> None:
        self._sink: OutputSink | None = None

    def bind(self, sink: OutputSink) -> None:
        """Bind the peer-owned output sink exactly once."""
        if self._sink is not None:
            raise RuntimeError("WebRTC output sink is already bound.")
        self._sink = sink

    def open(self, session_info: SessionInfo) -> None:
        self._required().open(session_info)

    def begin_generation(self, generation: int) -> None:
        self._required().begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        return self._required().write(result)

    def close(self) -> Sequence[OutputArtifact]:
        return self._required().close()

    def _required(self) -> OutputSink:
        if self._sink is None:
            raise RuntimeError("WebRTC output sink has not been bound.")
        return self._sink


class ApplicationWebRTCSessionManager:
    """Adapt any FlashDreams application to one browser WebRTC session."""

    def __init__(
        self,
        *,
        application_slug: str,
        commandline_args: Sequence[str],
    ) -> None:
        self._application_slug = application_slug
        self._commandline_args = tuple(commandline_args)
        self._runtime_ready = False
        self._peer: RTCPeerConnection | None = None
        self._track: BufferedVideoTrack | None = None
        self._generation_task: asyncio.Task[None] | None = None
        self._session_lock = asyncio.Lock()

    def has_active_session(self) -> bool:
        """Return whether a peer currently owns the single session."""
        return self._peer is not None and self._peer.connectionState != "closed"

    def is_runtime_ready(self) -> bool:
        """Return whether the application host is accepting offers."""
        return self._runtime_ready

    async def preload_runtime(self) -> None:
        """Mark the lightweight host ready; model setup starts for the first offer."""
        self._runtime_ready = True

    async def create_answer(
        self,
        *,
        offer_sdp: str,
        offer_type: str,
    ) -> dict[str, str]:
        """Initialize a model session, negotiate video, and start generation."""
        async with self._session_lock:
            if self.has_active_session():
                raise SessionBusyError(
                    "A WebRTC application session is already active."
                )

            input_sink = NullInputSink()
            deferred_output = _DeferredOutputSink()
            session, session_info = await asyncio.to_thread(
                self._create_initialized_session,
                input_sink,
                deferred_output,
            )
            fps = int(round(session_info.frames_per_second or 16.0))
            maxsize = max(1, session_info.steady_output_frame_count or fps)
            loop = asyncio.get_running_loop()
            track = BufferedVideoTrack(fps=fps, maxsize=maxsize)
            bridge = BufferedTrackOutputBridge(loop=loop, track=track)
            factory = WebRTCIOFactory(lambda: bridge)
            output_sink = factory.create_output_sink()
            deferred_output.bind(output_sink)

            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            peer.addTransceiver(track, direction="sendonly")
            self._peer = peer
            self._track = track

            @peer.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if peer.connectionState in {"failed", "disconnected", "closed"}:
                    await self._close_peer(peer)

            try:
                await peer.setRemoteDescription(
                    RTCSessionDescription(sdp=offer_sdp, type=offer_type)
                )
                answer = await peer.createAnswer()
                await peer.setLocalDescription(answer)
                await wait_for_ice_gathering_complete(peer)
                local = peer.localDescription
                if local is None:
                    raise RuntimeError("Peer connection did not produce an SDP answer.")
                self._generation_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._run_session,
                        session,
                        session_info,
                        input_sink,
                        deferred_output,
                    )
                )
                return {"sdp": local.sdp, "type": local.type}
            except Exception:
                await self._close_peer(peer)
                session.close()
                raise

    async def shutdown(self) -> None:
        """Close the active peer and release its media track."""
        if self._peer is not None:
            await self._close_peer(self._peer)
        self._runtime_ready = False

    def _create_initialized_session(
        self,
        input_sink: InputSink,
        output_sink: OutputSink,
    ) -> tuple[IFlashDreamsApplicationSession, SessionInfo]:
        application, slug_args = create_application(self._application_slug)
        application.init(
            [*slug_args, *self._commandline_args],
            input_sink,
            output_sink,
        )
        session = application.create_session(input_sink, output_sink)
        session.init()
        return session, session.session_info()

    @staticmethod
    def _run_session(
        session: IFlashDreamsApplicationSession,
        session_info: SessionInfo,
        input_sink: InputSink,
        output_sink: OutputSink,
    ) -> None:
        try:
            input_sink.open(session_info)
            output_sink.open(session_info)
            output_sink.begin_generation(0)
            session.generate(input_sink, output_sink)
        except Exception:
            logger.exception("WebRTC application generation failed.")
        finally:
            session.close()
            output_sink.close()
            input_sink.close()

    async def _close_peer(self, peer: RTCPeerConnection) -> None:
        if peer is not self._peer:
            return
        track = self._track
        self._peer = None
        self._track = None
        if track is not None:
            await track.close()
        if peer.connectionState != "closed":
            await peer.close()


__all__ = [
    "ApplicationWebRTCSessionManager",
    "BufferedTrackOutputBridge",
]
