# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spawn-safe model-loop worker and its process protocol."""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from io import BytesIO
from multiprocessing.connection import Connection
from typing import Any

import cloudpickle
import torch.multiprocessing  # noqa: F401 - registers zero-copy tensor reducers

from flashdreams.api_v2.loop import (
    IModelLoop,
    _install_remote_operation_sender,
    _model_results,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_POLL_INTERVAL_SECONDS = 0.05


class ModelProcessError(RuntimeError):
    """Failure reported when the model process exits without an exception."""


def serialize_model_loop(model_loop: IModelLoop[object]) -> bytes:
    """Serialize a CUDA-free loop specification for a spawned interpreter."""
    return _process_dumps(model_loop)


def serialize_operation(operation: Any) -> bytes:
    """Serialize a cross-process state operation, including local callables."""
    return _process_dumps(operation)


def deserialize_exception(payload: bytes, traceback_text: str) -> BaseException:
    """Recover a worker exception, falling back to a process-boundary error."""
    try:
        error = cloudpickle.loads(payload)
    except BaseException:
        error = ModelProcessError("The model process failed.\n" + traceback_text)
    if isinstance(error, BaseException):
        return error
    return ModelProcessError("The model process failed.\n" + traceback_text)


def model_process_main(
    connection: Connection,
    model_loop_payload: bytes,
    max_steps: int | None,
    max_inflight_results: int,
) -> None:
    """Materialize and drive the model loop in a spawned process.

    Tensor-bearing result messages are sent directly through the multiprocessing
    connection. PyTorch's registered reducers transmit CPU shared-memory or CUDA
    IPC handles rather than copying tensor storage through the pipe.
    """
    shutdown = threading.Event()
    failures: queue.Queue[BaseException] = queue.Queue()
    model_loop: IModelLoop[object] | None = None
    quiesced = False
    try:
        loaded = cloudpickle.loads(model_loop_payload)
        if not isinstance(loaded, IModelLoop):
            raise TypeError("The model-process payload is not an IModelLoop.")
        model_loop = loaded
        model_loop._bind_process_runtime(
            shutdown_event=shutdown,
            failure_queue=failures,
        )

        def send_ui_operation(operation: Any) -> None:
            connection.send(("invoke_ui", serialize_operation(operation)))

        _install_remote_operation_sender("ui", send_ui_operation)
        connection.send(("ready", os.getpid()))

        pending_events: list[Any] = []
        generation = 0
        started = False
        terminate = False
        inflight_results = 0

        def receive_commands(timeout: float = 0.0) -> None:
            nonlocal generation, inflight_results, quiesced, started, terminate
            first = True
            while connection.poll(timeout if first else 0.0):
                first = False
                message = connection.recv()
                kind = message[0]
                if kind == "input":
                    events = message[1]
                    generation = message[2]
                    pending_events.extend(events.get_events())
                elif kind == "start":
                    started = True
                elif kind == "invoke_model":
                    model_loop._invoke_local(cloudpickle.loads(message[1]))
                elif kind == "result_received":
                    inflight_results -= 1
                    if inflight_results < 0:
                        raise RuntimeError(
                            "Received an unexpected result acknowledgement."
                        )
                elif kind == "quiesce":
                    quiesced = True
                    shutdown.set()
                elif kind == "terminate":
                    terminate = True
                    quiesced = True
                    shutdown.set()
                else:
                    raise RuntimeError(f"Unknown model-process command: {kind!r}.")

        while not started and not quiesced:
            receive_commands(_POLL_INTERVAL_SECONDS)

        steps_run = 0
        last_run_started: float | None = None
        while not quiesced and (max_steps is None or steps_run < max_steps):
            receive_commands()
            while not quiesced and inflight_results >= max_inflight_results:
                receive_commands(_POLL_INTERVAL_SECONDS)
            if quiesced:
                break
            events = UserInputEvents(list(pending_events))
            pending_events.clear()
            step_generation = generation
            step_index = model_loop._begin_run(events, step_generation)
            if step_index is None:
                break

            if model_loop.frequency and last_run_started is not None:
                due_at = last_run_started + 1.0 / model_loop.frequency
                while not quiesced:
                    remaining = due_at - time.monotonic()
                    if remaining <= 0.0:
                        break
                    receive_commands(min(remaining, _POLL_INTERVAL_SECONDS))
            if quiesced:
                break
            last_run_started = time.monotonic()

            results = _model_results(model_loop.step(step_index, events))
            model_loop._finish_run(results)
            steps_run += 1
            receive_commands()
            if quiesced:
                break
            connection.send(("result", step_generation, results))
            inflight_results += 1
            if model_loop.is_finished():
                break

        if not quiesced:
            quiesced = True
            shutdown.set()
        _shutdown_model_loop(model_loop)
        connection.send(("quiesced",))

        # CUDA IPC requires the producer to outlive all consumer references.
        # Stay resident until the original process has cleared presentation and
        # closed every output sink, then explicitly permits termination.
        while not terminate:
            receive_commands(_POLL_INTERVAL_SECONDS)
    except EOFError:
        pass
    except BaseException as error:
        shutdown.set()
        try:
            payload = cloudpickle.dumps(error)
        except BaseException:
            payload = cloudpickle.dumps(ModelProcessError(str(error)))
        try:
            connection.send(("failure", payload, traceback.format_exc()))
            if model_loop is not None:
                _shutdown_model_loop(model_loop)
            connection.send(("quiesced",))
            while True:
                message = connection.recv()
                if message[0] == "terminate":
                    break
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        _install_remote_operation_sender("ui", None)
        connection.close()


def _shutdown_model_loop(model_loop: IModelLoop[object]) -> None:
    try:
        model_loop._shutdown()
    except BaseException:
        # Let the outer worker handler preserve this as the primary error when
        # shutdown itself is what failed.
        raise


__all__ = [
    "ModelProcessError",
    "deserialize_exception",
    "model_process_main",
    "serialize_model_loop",
    "serialize_operation",
]


_LOCK_TYPE = type(threading.Lock())
_RLOCK_TYPE = type(threading.RLock())


def _new_event(is_set: bool) -> threading.Event:
    event = threading.Event()
    if is_set:
        event.set()
    return event


class _ProcessPickler(cloudpickle.CloudPickler):
    """Recreate process-local synchronization embedded in model state."""

    def reducer_override(self, obj: Any) -> Any:
        if type(obj) is _LOCK_TYPE:
            return threading.Lock, ()
        if type(obj) is _RLOCK_TYPE:
            return threading.RLock, ()
        if isinstance(obj, threading.Event):
            return _new_event, (obj.is_set(),)
        return super().reducer_override(obj)


def _process_dumps(value: Any) -> bytes:
    buffer = BytesIO()
    _ProcessPickler(buffer).dump(value)
    return buffer.getvalue()
