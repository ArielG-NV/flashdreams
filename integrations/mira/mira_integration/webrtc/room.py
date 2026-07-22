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

"""Configurable multiplayer WebRTC room for synchronized MIRA inference."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import nvtx
import torch
from aiortc import RTCPeerConnection, RTCSessionDescription
from loguru import logger
from torch import Tensor

from flashdreams.serving.webrtc.media import BufferedVideoTrack
from flashdreams.serving.webrtc.messages import (
    make_chunk_done_payload,
    make_error_payload,
)
from flashdreams.serving.webrtc.server import SessionBusyError
from flashdreams.serving.webrtc.warmup import wait_for_ice_gathering_complete
from mira_integration.configs.schema import preview_grid_dimensions
from mira_integration.webrtc.session import (
    MiraInferenceRuntime,
    MiraRuntimeConfig,
)


@nvtx.annotate()
def tile_player_video(video: Tensor) -> Tensor:
    """Tile per-player video into one near-square preview stream.

    Args:
        video: Player video with shape ``[P, T, C, H, W]``.

    Returns:
        Tiled video with shape ``[T, C, rows*H, columns*W]``.

    Raises:
        ValueError: ``video`` does not contain a positive player dimension.
    """
    if video.ndim != 5 or video.shape[0] <= 0:
        raise ValueError(f"Expected [P,T,C,H,W] player video, got {tuple(video.shape)}")
    players, frames, channels, height, width = video.shape
    rows, columns = preview_grid_dimensions(players)
    preview = torch.zeros(
        frames,
        channels,
        rows * height,
        columns * width,
        dtype=video.dtype,
        device=video.device,
    )
    for player in range(players):
        row, column = divmod(player, columns)
        preview[
            :,
            :,
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = video[player]
    return preview


@dataclass(slots=True)
class MiraBrowserSession:
    """One browser peer observing the room or controlling one player seat."""

    session_id: str
    """Opaque identifier used to own and release a seat."""

    peer_connection: Any
    """WebRTC peer connection owned by this browser."""

    video_track: BufferedVideoTrack
    """Track receiving either the preview grid or one player view."""

    seat: int | None = None
    """Claimed player index; ``None`` keeps the browser in preview mode."""

    held_keys: set[str] = field(default_factory=set)
    """Normalized browser keys currently held for ``seat``."""

    control_channel: Any | None = None
    """Browser data channel once negotiation completes."""

    last_message_at: float = 0.0
    """Monotonic time of the latest heartbeat or control message."""

    liveness_task: asyncio.Task[Any] | None = None
    """Watchdog task that closes abandoned browser sessions."""

    closed: bool = False
    """Whether media and peer resources have been released."""

    @nvtx.annotate()
    async def close(self) -> None:
        """Close this browser's media and peer connection."""
        if self.closed:
            return
        self.closed = True
        task = self.liveness_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.video_track.close()
        await self.peer_connection.close()


class MiraMultiplayerSessionManager:
    """Own one model-configured rollout shared by observers and controllers."""

    def __init__(
        self,
        *,
        runtime_config: MiraRuntimeConfig | None = None,
        runtime: MiraInferenceRuntime | None = None,
        client_liveness_timeout_s: float = 10.0,
    ) -> None:
        if runtime_config is None:
            if runtime is None:
                raise ValueError(
                    "MIRA requires an explicit manifest and demo runtime config."
                )
            runtime_config = runtime.config
        self.runtime_config = runtime_config
        self._runtime = runtime or MiraInferenceRuntime(config=runtime_config)
        self.model_config = self._runtime.model_config
        self.metadata = self.model_config.metadata
        self._client_liveness_timeout_s = client_liveness_timeout_s
        self._sessions: dict[str, MiraBrowserSession] = {}
        self._seat_owners: dict[int, str] = {}
        self._runtime_ready = False
        self._preload_lock = asyncio.Lock()
        self._room_lock = asyncio.Lock()
        self._generation_task: asyncio.Task[Any] | None = None

    @nvtx.annotate()
    def has_active_session(self) -> bool:
        """Return whether at least one browser is observing the room."""
        return any(not session.closed for session in self._sessions.values())

    @nvtx.annotate()
    def is_runtime_ready(self) -> bool:
        """Return whether model initialization and warmup completed."""
        return self._runtime_ready

    @nvtx.annotate()
    def public_config(self) -> dict[str, Any]:
        """Return configuration used to construct the browser UI."""
        return self.metadata.to_public_dict()

    @nvtx.annotate()
    def room_state(self) -> dict[str, Any]:
        """Return public occupancy state for the browser seat picker."""
        return {
            "players": [
                {"seat": seat, "occupied": seat in self._seat_owners}
                for seat in range(self.metadata.player_count)
            ],
            "active_players": len(self._seat_owners),
            "observers": sum(
                session.seat is None for session in self._sessions.values()
            ),
            "capacity": self.metadata.player_count,
            "runtime_ready": self._runtime_ready,
        }

    @nvtx.annotate()
    async def preload_runtime(self) -> None:
        """Initialize and warm up the configured joint runtime once."""
        async with self._preload_lock:
            if self._runtime_ready:
                return
            await self._runtime.initialize()
            await self._runtime.reset_for_new_session()
            idle = tuple(None for _ in range(self.metadata.player_count))
            for _ in range(self.runtime_config.warmup_chunks):
                await self._runtime.generate_chunk(player_keys=idle)
            await self._runtime.reset_for_new_session()
            self._runtime_ready = True

    @nvtx.annotate()
    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        """Negotiate an observer connection that receives the preview grid."""
        return await self._create_answer(
            offer_sdp=offer_sdp,
            offer_type=offer_type,
            initial_seat=None,
        )

    @nvtx.annotate()
    async def create_player_answer(
        self, *, seat: int, offer_sdp: str, offer_type: str
    ) -> dict[str, str]:
        """Negotiate a connection that atomically claims ``seat``."""
        return await self._create_answer(
            offer_sdp=offer_sdp,
            offer_type=offer_type,
            initial_seat=seat,
        )

    @nvtx.annotate()
    async def _create_answer(
        self,
        *,
        offer_sdp: str,
        offer_type: str,
        initial_seat: int | None,
    ) -> dict[str, str]:
        await self.preload_runtime()
        async with self._room_lock:
            if initial_seat is not None:
                self._validate_seat(initial_seat)
                if initial_seat in self._seat_owners:
                    raise SessionBusyError(
                        f"Player {initial_seat + 1} is already controlled."
                    )
            if not self._sessions:
                await self._runtime.reset_for_new_session()

            peer = RTCPeerConnection()
            track = BufferedVideoTrack(
                fps=self.runtime_config.fps,
                maxsize=self.runtime_config.frames_per_chunk,
            )
            peer.addTrack(track)
            session_id = uuid4().hex
            session = MiraBrowserSession(
                session_id=session_id,
                seat=initial_seat,
                peer_connection=peer,
                video_track=track,
                last_message_at=asyncio.get_running_loop().time(),
            )
            self._sessions[session_id] = session
            if initial_seat is not None:
                self._seat_owners[initial_seat] = session_id
            session.liveness_task = asyncio.create_task(self._watch_liveness(session))
            self._wire_peer(session)
            try:
                await peer.setRemoteDescription(
                    RTCSessionDescription(sdp=offer_sdp, type=offer_type)
                )
                answer = await peer.createAnswer()
                await peer.setLocalDescription(answer)
                await wait_for_ice_gathering_complete(peer)
                description = peer.localDescription
                if description is None:
                    raise RuntimeError("Peer connection did not produce an answer.")
                return {"sdp": description.sdp, "type": description.type}
            except Exception:
                self._remove_session_locked(session)
                await session.close()
                raise

    @nvtx.annotate()
    def _wire_peer(self, session: MiraBrowserSession) -> None:
        peer = session.peer_connection

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            session.control_channel = channel

            @channel.on("message")
            def on_message(message: Any) -> None:
                asyncio.create_task(self._handle_message(session, message))

            @channel.on("close")
            def on_close() -> None:
                asyncio.create_task(self.close_session(session))

            if self._generation_task is None or self._generation_task.done():
                self._generation_task = asyncio.create_task(self._generation_worker())

        @peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if peer.connectionState in {"failed", "disconnected", "closed"}:
                await self.close_session(session)

    @nvtx.annotate()
    async def _handle_message(self, session: MiraBrowserSession, raw: Any) -> None:
        if session.closed or not isinstance(raw, str):
            return
        session.last_message_at = asyncio.get_running_loop().time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send(session, make_error_payload("Invalid JSON payload."))
            return
        message_type = str(payload.get("type", "")).lower()
        if message_type == "heartbeat":
            return
        if message_type == "disconnect":
            await self.close_session(session)
            return
        if message_type == "claim":
            await self._claim_seat(session, payload.get("seat"))
            return
        if message_type == "release":
            await self._release_seat(session)
            return
        action = payload.get("action", payload)
        if message_type != "action" or not isinstance(action, dict):
            self._send(
                session,
                make_error_payload("Expected a claim, release, or action message."),
            )
            return
        if session.seat is None:
            self._send(
                session, make_error_payload("Choose a player before sending input.")
            )
            return
        event = str(action.get("event", "")).lower()
        key = str(action.get("key", "")).lower()
        if event == "step":
            return
        if event not in {"keydown", "keyup"} or key not in self.metadata.browser_keys:
            self._send(session, make_error_payload("Unsupported keyboard action."))
            return
        if event == "keydown":
            session.held_keys.add(key)
        else:
            session.held_keys.discard(key)

    @nvtx.annotate()
    async def _claim_seat(self, session: MiraBrowserSession, raw_seat: Any) -> None:
        try:
            seat = int(raw_seat)
            self._validate_seat(seat)
        except (TypeError, ValueError) as exc:
            self._send(
                session,
                {"type": "seat_claim_failed", "message": str(exc)},
            )
            return
        async with self._room_lock:
            owner = self._seat_owners.get(seat)
            if owner is not None and owner != session.session_id:
                self._send(
                    session,
                    {
                        "type": "seat_claim_failed",
                        "seat": seat,
                        "message": f"Player {seat + 1} is already controlled.",
                    },
                )
                return
            if session.seat is not None:
                self._seat_owners.pop(session.seat, None)
            session.seat = seat
            session.held_keys.clear()
            self._seat_owners[seat] = session.session_id
        self._send(session, {"type": "seat_claimed", "seat": seat})

    @nvtx.annotate()
    async def _release_seat(self, session: MiraBrowserSession) -> None:
        """Release the browser's seat while keeping its preview session alive."""
        async with self._room_lock:
            seat = session.seat
            if seat is not None:
                owner = self._seat_owners.get(seat)
                if owner == session.session_id:
                    self._seat_owners.pop(seat, None)
            session.seat = None
            session.held_keys.clear()
        self._send(session, {"type": "seat_released", "seat": seat})

    @nvtx.annotate()
    def _validate_seat(self, seat: int) -> None:
        if seat not in range(self.metadata.player_count):
            raise ValueError(
                f"Player seat must be between 0 and {self.metadata.player_count - 1}."
            )

    @nvtx.annotate()
    async def _generation_worker(self) -> None:
        logger.info("MIRA room ready; starting continuous generation.")
        loop = asyncio.get_running_loop()
        try:
            while self._sessions:
                started = loop.time()
                with nvtx.annotate("MiraMultiplayerSessionManager.collect_keys"):
                    player_keys: list[frozenset[str] | None] = []
                    for seat in range(self.metadata.player_count):
                        controller = self._session_for_seat(seat)
                        player_keys.append(
                            frozenset(controller.held_keys)
                            if controller is not None
                            else None
                        )
                    keys = tuple(player_keys)
                try:
                    with nvtx.annotate(
                        "MiraMultiplayerSessionManager.generate_chunk"
                    ):
                        result = await self._runtime.generate_chunk(player_keys=keys)
                except Exception as exc:
                    logger.exception("MIRA multiplayer generation failed.")
                    for session in tuple(self._sessions.values()):
                        self._send(session, make_error_payload(str(exc)))
                    return
                generated = loop.time()
                sessions = tuple(self._sessions.values())
                with nvtx.annotate("MiraMultiplayerSessionManager.prepare_chunks"):
                    preview = (
                        tile_player_video(result.video_chunk)
                        if any(session.seat is None for session in sessions)
                        else None
                    )
                    chunks = [
                        preview
                        if session.seat is None
                        else result.video_chunk[session.seat]
                        for session in sessions
                    ]
                assert all(chunk is not None for chunk in chunks)
                with nvtx.annotate("MiraMultiplayerSessionManager.enqueue_chunks"):
                    enqueued = await asyncio.gather(
                        *(
                            session.video_track.enqueue_chunk(chunk)
                            for session, chunk in zip(sessions, chunks, strict=True)
                            if chunk is not None
                        )
                    )
                finished = loop.time()
                rows, columns = preview_grid_dimensions(self.metadata.player_count)
                for session, count in zip(sessions, enqueued, strict=True):
                    preview_mode = session.seat is None
                    width = self.metadata.video_width * (columns if preview_mode else 1)
                    height = self.metadata.video_height * (rows if preview_mode else 1)
                    self._send(
                        session,
                        make_chunk_done_payload(
                            chunk_index=result.chunk_index,
                            num_frames=result.num_frames,
                            enqueued_frames=count,
                            fps=self.runtime_config.fps,
                            width=width,
                            height=height,
                            model=self.metadata.display_name,
                            gen_ms=(generated - started) * 1000,
                            enqueue_ms=(finished - generated) * 1000,
                            play_ms=result.num_frames * 1000 / self.runtime_config.fps,
                            queue_depth=session.video_track.qsize(),
                            lag_ms=max(
                                0.0,
                                (finished - started) * 1000
                                - result.num_frames * 1000 / self.runtime_config.fps,
                            ),
                            extra={
                                "seat": session.seat,
                                "view": "preview" if preview_mode else "player",
                            },
                        ),
                    )
        except asyncio.CancelledError:
            raise
        finally:
            self._generation_task = None

    @nvtx.annotate()
    def _session_for_seat(self, seat: int) -> MiraBrowserSession | None:
        owner = self._seat_owners.get(seat)
        return self._sessions.get(owner) if owner is not None else None

    @nvtx.annotate()
    async def _watch_liveness(self, session: MiraBrowserSession) -> None:
        try:
            while not session.closed:
                await asyncio.sleep(1)
                if (
                    asyncio.get_running_loop().time() - session.last_message_at
                    >= self._client_liveness_timeout_s
                ):
                    await self.close_session(session)
                    return
        except asyncio.CancelledError:
            raise

    @nvtx.annotate()
    async def close_session(self, session: MiraBrowserSession) -> None:
        """Release one browser and its seat without disturbing the room."""
        async with self._room_lock:
            current = self._sessions.get(session.session_id)
            if current is not session:
                return
            self._remove_session_locked(session)
        await session.close()
        if not self._sessions and self._generation_task is not None:
            task = self._generation_task
            self._generation_task = None
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @nvtx.annotate()
    def _remove_session_locked(self, session: MiraBrowserSession) -> None:
        self._sessions.pop(session.session_id, None)
        if session.seat is not None:
            owner = self._seat_owners.get(session.seat)
            if owner == session.session_id:
                self._seat_owners.pop(session.seat, None)

    @nvtx.annotate()
    async def shutdown(self) -> None:
        """Close every browser and release the MIRA runtime."""
        for session in tuple(self._sessions.values()):
            await self.close_session(session)
        await self._runtime.close()
        self._runtime_ready = False

    @nvtx.annotate()
    def wait_for_termination(self) -> None:
        """Block the serving worker until shutdown is requested."""
        self._runtime.wait_for_termination()

    @nvtx.annotate()
    def send_exit_signal(self) -> None:
        """Release a worker waiting for server shutdown."""
        self._runtime.send_exit_signal()

    @staticmethod
    @nvtx.annotate()
    def _send(session: MiraBrowserSession, payload: dict[str, Any]) -> None:
        channel = session.control_channel
        if channel is None or getattr(channel, "readyState", "closed") != "open":
            return
        with contextlib.suppress(Exception):
            channel.send(json.dumps(payload))


__all__ = ["MiraBrowserSession", "MiraMultiplayerSessionManager", "tile_player_video"]
