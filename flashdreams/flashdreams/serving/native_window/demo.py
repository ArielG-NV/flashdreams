# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a demo in a native window."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flashdreams.runtime.demo import (
    DemoSpec,
    ModelWarmupPlan,
    NoopTransportService,
    PreparedScenario,
    RunResult,
    RuntimeHost,
    StepPipeline,
    run_demo_session_async,
)
from flashdreams.runtime.demo.spec import DemoAdapter, NativeWindowOutputSpec

from .presenter import SlangPyNativePresenter
from .services import (
    NativeFrameQueue,
    NativeWindowInputSource,
    NativeWindowOutputSink,
    NativeWindowRunMode,
)


@dataclass(slots=True)
class _State:
    ready: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    input: NativeWindowInputSource | None = None
    transport: NoopTransportService | None = None
    result: RunResult | None = None
    error: BaseException | None = None


def run_native_window_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    presenter_factory: Callable[..., Any] = SlangPyNativePresenter,
) -> RunResult:
    """Run one keyboard-driven demo and present every emitted video frame."""
    output = spec.output
    if not isinstance(output, NativeWindowOutputSpec):
        raise ValueError("Native output requires NativeWindowOutputSpec.")
    if spec.input_mode != "realtime":
        raise ValueError("Native output requires realtime input.")
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    if spec.input_mode not in adapter.supported_input_modes():
        raise ValueError(f"Adapter does not support {spec.input_mode!r} input.")

    scenario = adapter.prepare_scenario(spec)
    queue = NativeFrameQueue(max_chunks=output.max_queued_chunks)
    state = _State()

    def on_key(event: str, key: str) -> None:
        if state.input is not None:
            state.input.record_key(event=event, key=key, timestamp_s=time.monotonic())

    presenter = presenter_factory(
        width=output.video_width,
        height=output.video_height,
        title=output.title,
        on_key=on_key,
    )
    try:
        host = RuntimeHost(adapter.create_runtime(spec.config))
    except BaseException:
        presenter.close()
        raise

    def worker() -> None:
        try:
            state.result = asyncio.run(
                _run_session(
                    state=state,
                    host=host,
                    spec=spec,
                    scenario=scenario,
                    adapter=adapter,
                    queue=queue,
                    output=output,
                )
            )
        except BaseException as exc:
            state.error = exc
        finally:
            state.finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    interval = 1.0 / output.fps
    next_frame = time.monotonic()
    try:
        while not state.finished.is_set() and not presenter.should_close:
            presenter.process_events()
            frame = queue.pop()
            if frame is None:
                time.sleep(0.001)
                continue
            time.sleep(max(0.0, next_frame - time.monotonic()))
            presenter.present_frame(frame)
            next_frame = max(next_frame + interval, time.monotonic())
    finally:
        while not state.ready.is_set() and not state.finished.is_set():
            state.ready.wait(0.05)
        if state.transport is not None:
            state.transport.close()
        state.finished.wait()
        presenter.close()
        queue.close()
        host.close()
        thread.join(1.0)

    if state.error is not None:
        raise RuntimeError("Native-window session failed.") from state.error
    if state.result is None:
        raise RuntimeError("Native-window session returned no result.")
    return state.result


async def _run_session(
    *,
    state: _State,
    host: RuntimeHost,
    spec: DemoSpec,
    scenario: PreparedScenario,
    adapter: DemoAdapter,
    queue: NativeFrameQueue,
    output: NativeWindowOutputSpec,
) -> RunResult:
    loop = asyncio.get_running_loop()
    source = NativeWindowInputSource(fps=output.fps)
    source.reset(start_v=loop.time())
    transport = NoopTransportService()
    mode = NativeWindowRunMode(
        input_source=source,
        output_sink=NativeWindowOutputSink(queue=queue, fps=output.fps),
        transport=transport,
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=host,
        model_warmup_plan=ModelWarmupPlan(),
    )
    state.input = source
    state.transport = transport
    state.ready.set()
    try:
        await asyncio.to_thread(host.preload)
        return await run_demo_session_async(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        await context.close_async()


__all__ = ["run_native_window_demo"]
