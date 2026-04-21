"""Clock-driven playout: decouple display from network arrival.

The socket may deliver FRME batches **bursty** (e.g. 8 frames at once). This module
emits **one displayed frame per wall-clock tick** at ``target_fps``, draining a
deque of batches so the UI thread does not stutter when network jitter occurs.

If no frame is ready at a tick, we increment ``underruns`` and optionally re-invoke
``on_frame`` with ``held=True`` (same bytes as last frame) so logs stay periodic.
"""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


class OnFrame(Protocol):
    """Optional async hook each time a frame would be shown (real or held repeat)."""

    async def __call__(
        self, *, batch_id: int, frame_index: int, jpeg: bytes, held: bool
    ) -> None: ...


@dataclass
class PlayoutStats:
    frames_emitted: int = 0
    underruns: int = 0
    batches_started: int = 0
    # Time from batch arrival at client (recv stamp) to first displayed frame of that batch (ms).
    batch_first_display_latency_ms: list[float] = field(default_factory=list)
    # Wall-clock gap between successive displayed ticks (ms); should cluster near 1000/target_fps.
    frame_interval_ms: list[float] = field(default_factory=list)

    def profile_summary(self, *, target_fps: float) -> dict[str, float | int]:
        """Aggregate latency / smoothness metrics for logging."""
        target_iv = 1000.0 / max(target_fps, 1e-6)
        iv = self.frame_interval_ms
        lat = self.batch_first_display_latency_ms
        out: dict[str, float | int] = {
            "target_fps": target_fps,
            "target_frame_interval_ms": target_iv,
            "frames_emitted": self.frames_emitted,
            "underruns": self.underruns,
            "batches_started": self.batches_started,
        }
        if iv:
            out["interval_mean_ms"] = statistics.mean(iv)
            out["interval_p50_ms"] = _percentile(iv, 50)
            out["interval_p95_ms"] = _percentile(iv, 95)
            out["interval_max_ms"] = max(iv)
        else:
            out["interval_mean_ms"] = float("nan")
            out["interval_p50_ms"] = float("nan")
            out["interval_p95_ms"] = float("nan")
            out["interval_max_ms"] = float("nan")
        if lat:
            out["batch_to_first_frame_mean_ms"] = statistics.mean(lat)
            out["batch_to_first_frame_p95_ms"] = _percentile(lat, 95)
            out["batch_to_first_frame_max_ms"] = max(lat)
        else:
            out["batch_to_first_frame_mean_ms"] = float("nan")
            out["batch_to_first_frame_p95_ms"] = float("nan")
            out["batch_to_first_frame_max_ms"] = float("nan")
        return out


def _percentile(xs: list[float], pct: float) -> float:
    """Linear interpolation between sorted order-statistics (``pct`` in 0..100)."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] + (k - f) * (ys[c] - ys[f])


@dataclass
class _Current:
    """The batch currently being shown, frame-by-frame (``pos`` indexes ``frames``)."""

    batch_id: int
    frames: tuple[bytes, ...]
    recv_mono: float
    pos: int = 0


@dataclass
class PlayoutConfig:
    target_fps: float = 60.0
    min_batches_before_start: int = 2
    duration_s: float | None = None
    drop_stale_batches: bool = False
    # When set, ``is_set()`` ends playout after the current tick (e.g. SIGINT).
    stop_event: asyncio.Event | None = None


async def _await_incoming_or_stop(
    incoming: asyncio.Queue[tuple[int, tuple[bytes, ...], float]],
    stop_event: asyncio.Event | None,
) -> tuple[int, tuple[bytes, ...], float] | None:
    """Wait for the next queued batch, or abort priming if ``stop_event`` wins the race."""
    if stop_event is None:
        return await incoming.get()
    get_task = asyncio.create_task(incoming.get())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
    if stop_task in done:
        get_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await get_task
        return None
    return await get_task


def _extend_pending(
    pending: deque[tuple[int, tuple[bytes, ...], float]],
    new_batches: list[tuple[int, tuple[bytes, ...], float]],
    *,
    drop_stale: bool,
) -> None:
    """Merge recv-drained batches into ``pending`` (FIFO) or keep only the newest."""
    if not new_batches:
        return
    if drop_stale:
        pending.append(new_batches[-1])
    else:
        pending.extend(new_batches)


async def run_playout(
    incoming: asyncio.Queue[tuple[int, tuple[bytes, ...], float]],
    *,
    cfg: PlayoutConfig,
    on_frame: OnFrame | None = None,
) -> PlayoutStats:
    """Drain ``incoming`` (``batch_id``, frames, recv_mono) and emit one frame per tick.

    ``recv_mono`` is :func:`time.perf_counter` when the FRME was received and unpacked.

    Underrun: re-emit the last frame with ``held=True`` when no new bytes exist.
    """
    stats = PlayoutStats()
    interval = 1.0 / max(cfg.target_fps, 1e-6)
    # Batches already received but not yet started on the playout clock.
    pending: deque[tuple[int, tuple[bytes, ...], float]] = deque()
    current: _Current | None = None
    last_jpeg: bytes | None = None
    t_end = time.monotonic() + cfg.duration_s if cfg.duration_s else None
    last_shown_mono: float | None = None

    def mark_shown_tick(now: float) -> None:
        """Record inter-frame interval in ms (successive ``now`` = display tick times)."""
        nonlocal last_shown_mono
        if last_shown_mono is not None:
            stats.frame_interval_ms.append((now - last_shown_mono) * 1000.0)
        last_shown_mono = now

    async def drain_incoming() -> None:
        """Move everything currently in ``incoming`` into ``pending`` (non-blocking)."""
        new_batches: list[tuple[int, tuple[bytes, ...], float]] = []
        while True:
            try:
                new_batches.append(incoming.get_nowait())
            except asyncio.QueueEmpty:
                break
        _extend_pending(pending, new_batches, drop_stale=cfg.drop_stale_batches)

    # Prime: do not start the wall clock until we have some pipeline depth (smoother UX).
    while len(pending) < cfg.min_batches_before_start:
        row = await _await_incoming_or_stop(incoming, cfg.stop_event)
        if row is None:
            return stats
        bid, frames, recv_mono = row
        new = [(bid, frames, recv_mono)]
        while True:
            try:
                new.append(incoming.get_nowait())
            except asyncio.QueueEmpty:
                break
        _extend_pending(pending, new, drop_stale=cfg.drop_stale_batches)

    # next_tick: ideal absolute time for the *next* display tick (reduces drift vs sleep-only).
    next_tick = time.monotonic()
    while True:
        if cfg.stop_event is not None and cfg.stop_event.is_set():
            break
        if t_end is not None and time.monotonic() >= t_end:
            break
        await drain_incoming()
        if current is None or current.pos >= len(current.frames):
            if pending:
                bid, frames, recv_mono = pending.popleft()
                current = _Current(
                    batch_id=bid, frames=frames, recv_mono=recv_mono, pos=0
                )
                stats.batches_started += 1
            else:
                # Nothing to show yet: keep cadence; repeat last texture if available.
                stats.underruns += 1
                now = time.perf_counter()
                mark_shown_tick(now)
                if last_jpeg is not None and on_frame is not None:
                    res = on_frame(
                        batch_id=-1,
                        frame_index=-1,
                        jpeg=last_jpeg,
                        held=True,
                    )
                    if asyncio.iscoroutine(res) or asyncio.isfuture(res):
                        await res  # type: ignore[func-returns-value]
                stats.frames_emitted += 1
                next_tick += interval
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    await asyncio.sleep(sleep)
                else:
                    # We fell behind: reset phase so the next tick aligns to "now".
                    next_tick = time.monotonic()
                continue

        now = time.perf_counter()
        # First pixel of this batch: measures network+queue wait for this FRME.
        if current.pos == 0:
            stats.batch_first_display_latency_ms.append(
                (now - current.recv_mono) * 1000.0
            )

        jpeg = current.frames[current.pos]
        last_jpeg = jpeg
        mark_shown_tick(now)
        if on_frame is not None:
            res = on_frame(
                batch_id=current.batch_id,
                frame_index=current.pos,
                jpeg=jpeg,
                held=False,
            )
            if asyncio.iscoroutine(res) or asyncio.isfuture(res):
                await res  # type: ignore[func-returns-value]
        stats.frames_emitted += 1
        current.pos += 1

        next_tick += interval
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            await asyncio.sleep(sleep)
        else:
            next_tick = time.monotonic()

    return stats
