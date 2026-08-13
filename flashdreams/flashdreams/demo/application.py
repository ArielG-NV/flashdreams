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

"""Transport-neutral contracts and host loop for FlashDreams applications."""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from flashdreams.demo.factories import (
    CallableIOFactory,
    Mp4IOFactory,
    LocalWindowIOFactory,
    NullInputSink,
    ProvidedIOFactory,
)
from flashdreams.demo.io import InputSink, IOFactory, OutputSink, SessionInfo
from flashdreams.demo.outputs import LocalWindowOutputSink, NullOutputSink
from flashdreams.runtime.output import OutputArtifact

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
"""Entry-point group whose values expose a zero-argument ``createApp`` factory."""


class IFlashDreamsApplicationSession(ABC):
    """One isolated application session with sequential model state."""

    @abstractmethod
    def init(self) -> None:
        """Initialize model and per-session resources."""

    def session_info(self) -> SessionInfo:
        """Return sink-facing metadata after session initialization."""
        return SessionInfo()

    def generate(self, input_src: InputSink, output_sink: OutputSink) -> None:
        """Run sequential steps until the session reports completion."""
        while self.step(input_src, output_sink):
            pass

    @abstractmethod
    def step(self, input_src: InputSink, output_sink: OutputSink) -> bool:
        """Run one model step and return whether another step remains."""

    def close(self) -> None:
        """Release optional per-session resources."""


class IFlashDreamsApplication(ABC):
    """Application factory boundary independent of model and presentation backend."""

    @abstractmethod
    def init(
        self,
        commandline_args: Sequence[str],
        input_src: InputSink,
        output_sink: OutputSink,
    ) -> None:
        """Parse application arguments and validate startup state.

        Args:
            commandline_args: Arguments following the selected application slug.
            input_src: Host-provided input sink.
            output_sink: Host-provided output sink.
        """

    @abstractmethod
    def create_session(
        self,
        input_src: InputSink,
        output_sink: OutputSink,
    ) -> IFlashDreamsApplicationSession:
        """Create one isolated application session."""

    def createSession(
        self,
        input_src: InputSink,
        output_sink: OutputSink,
    ) -> IFlashDreamsApplicationSession:
        """Create a session through the package-facing compatibility spelling."""
        return self.create_session(input_src, output_sink)


ApplicationFactory = Callable[[], IFlashDreamsApplication]
"""Zero-argument factory exported from an application package as ``createApp``."""


def create_application(
    application_slug: str,
) -> tuple[IFlashDreamsApplication, list[str]]:
    """Load the application package registered for an exact slug.

    Args:
        application_slug: User-facing application slug.

    Returns:
        The created application and package-derived arguments, currently empty.

    Raises:
        LookupError: No installed application package matches the slug.
        TypeError: The package factory does not return the application contract.
    """
    if not application_slug.strip():
        raise ValueError("application_slug must be non-empty.")

    registered = sorted(
        entry_points(group=APPLICATION_ENTRY_POINT_GROUP),
        key=lambda item: item.name,
    )
    for entry_point in registered:
        if entry_point.name == application_slug:
            return _create_from_entry_point(entry_point), []

    module = _import_application_module(application_slug)
    factory = getattr(module, "createApp", None)
    if not callable(factory):
        raise TypeError(
            f"Application module {module.__name__!r} does not expose createApp()."
        )
    return _validate_application(factory(), origin=module.__name__), []


def run_application(
    application_slug: str,
    commandline_args: Sequence[str] = (),
    *,
    io_factory: IOFactory | None = None,
    input_src: InputSink | None = None,
    output_sink: OutputSink | None = None,
) -> tuple[OutputArtifact, ...]:
    """Load and run an application through one host-owned I/O factory.

    Args:
        application_slug: Installed application or concrete demo slug.
        commandline_args: Arguments forwarded to the application.
        io_factory: Factory for per-run input and output sinks.
        input_src: Compatibility injection for a caller-owned input sink.
        output_sink: Compatibility injection for a caller-owned output sink.

    Returns:
        Persistent artifacts returned by the output sink.

    Raises:
        ValueError: ``io_factory`` is combined with direct sink injection.
        TypeError: A factory returns an object outside the sink contracts.
    """
    resolved_factory = _resolve_io_factory(
        io_factory=io_factory,
        input_src=input_src,
        output_sink=output_sink,
    )
    resolved_input = resolved_factory.create_input_sink()
    resolved_output = resolved_factory.create_output_sink()
    if not isinstance(resolved_input, InputSink):
        raise TypeError("IOFactory.create_input_sink() must return an InputSink.")
    if not isinstance(resolved_output, OutputSink):
        raise TypeError("IOFactory.create_output_sink() must return an OutputSink.")

    session: IFlashDreamsApplicationSession | None = None
    artifacts: tuple[OutputArtifact, ...] = ()
    try:
        application, slug_args = create_application(application_slug)
        application.init(
            [*slug_args, *commandline_args],
            resolved_input,
            resolved_output,
        )
        session = application.create_session(resolved_input, resolved_output)
        session.init()
        session_info = session.session_info()
        resolved_input.open(session_info)
        resolved_output.open(session_info)
        resolved_output.begin_generation(0)
        session.generate(resolved_input, resolved_output)
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                artifacts = tuple(resolved_output.close())
            finally:
                resolved_input.close()
    return artifacts


def _resolve_io_factory(
    *,
    io_factory: IOFactory | None,
    input_src: InputSink | None,
    output_sink: OutputSink | None,
) -> IOFactory:
    if io_factory is not None:
        if input_src is not None or output_sink is not None:
            raise ValueError("Pass io_factory or direct input/output sinks, not both.")
        return io_factory
    if input_src is None and output_sink is None:
        return LocalWindowIOFactory()
    return ProvidedIOFactory(
        input_sink=input_src if input_src is not None else NullInputSink(),
        output_sink=(
            output_sink if output_sink is not None else LocalWindowOutputSink()
        ),
    )


def _create_from_entry_point(entry_point: EntryPoint) -> IFlashDreamsApplication:
    value = entry_point.load()
    application = value() if callable(value) else value
    return _validate_application(application, origin=entry_point.value)


def _validate_application(value: Any, *, origin: str) -> IFlashDreamsApplication:
    if not isinstance(value, IFlashDreamsApplication):
        raise TypeError(
            f"Application factory {origin!r} returned {type(value).__name__}; "
            "expected IFlashDreamsApplication."
        )
    return value


def _import_application_module(slug: str) -> Any:
    module_name = slug.replace("-", "_")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    available = ", ".join(
        sorted(ep.name for ep in entry_points(group=APPLICATION_ENTRY_POINT_GROUP))
    )
    raise LookupError(
        f"No FlashDreams application package matches {slug!r}. "
        f"Installed applications: {available or '(none)'}."
    )


def _parse_host_io(
    application_slug: str,
    args: Sequence[str],
) -> tuple[IOFactory, list[str]]:
    output_kind = "local-window"
    output_path: Path | None = None
    output_fps: float | None = None
    application_args: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in {"--output", "--output-path", "--output-fps"}:
            if index + 1 >= len(args):
                raise ValueError(f"{argument} requires a value.")
            value = args[index + 1]
            if argument == "--output":
                output_kind = value
            elif argument == "--output-path":
                output_path = Path(value)
            else:
                output_fps = float(value)
            index += 2
            continue
        application_args.append(argument)
        index += 1

    if output_kind == "local-window":
        return LocalWindowIOFactory(fps=output_fps), application_args
    if output_kind == "null":
        return CallableIOFactory(NullInputSink, NullOutputSink), application_args
    if output_kind == "mp4":
        path = output_path or Path("outputs") / f"{application_slug}.mp4"
        return Mp4IOFactory(output_path=path, fps=output_fps), application_args
    raise ValueError(
        f"Unsupported output {output_kind!r}; expected local-window, null, mp4, or webrtc."
    )


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the ``flashdreams`` console entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        args.pop(0)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: flashdreams [run] APPLICATION "
            "[--output local-window|null|mp4|webrtc] [--host HOST] [--port PORT] "
            "[APPLICATION_ARGS ...]"
        )
        return
    application_slug = args.pop(0)
    if _selected_output(args) == "webrtc":
        host, port, application_args = _parse_webrtc_host_args(args)
        from flashdreams.serving.webrtc.application import serve_application_webrtc

        serve_application_webrtc(
            application_slug, application_args, host=host, port=port
        )
        return
    io_factory, application_args = _parse_host_io(application_slug, args)
    artifacts = run_application(
        application_slug,
        application_args,
        io_factory=io_factory,
    )
    for artifact in artifacts:
        print(artifact.uri)


def _selected_output(args: Sequence[str]) -> str:
    try:
        index = args.index("--output")
    except ValueError:
        return "local-window"
    if index + 1 >= len(args):
        raise ValueError("--output requires a value.")
    return args[index + 1]


def _parse_webrtc_host_args(
    args: Sequence[str],
) -> tuple[str, int, list[str]]:
    host = "127.0.0.1"
    port = 8080
    application_args: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in {"--output", "--host", "--port"}:
            if index + 1 >= len(args):
                raise ValueError(f"{argument} requires a value.")
            value = args[index + 1]
            if argument == "--host":
                host = value
            elif argument == "--port":
                port = int(value)
            index += 2
            continue
        application_args.append(argument)
        index += 1
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    return host, port, application_args


__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationFactory",
    "IFlashDreamsApplication",
    "IFlashDreamsApplicationSession",
    "create_application",
    "entrypoint",
    "run_application",
]
