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
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol, runtime_checkable

from flashdreams.demo.outputs import VideoWindowOutputSink

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
"""Entry-point group whose values expose a zero-argument ``createApp`` factory."""


@runtime_checkable
class InputSink(Protocol):
    """Supplies opaque inputs to an application session."""

    def read(self) -> object | None:
        """Return the next available input, or ``None`` when none is available."""
        ...


@runtime_checkable
class OutputSink(Protocol):
    """Consumes opaque outputs produced by an application session."""

    def write(self, output: object) -> None:
        """Consume one output without imposing presentation semantics."""
        ...


@dataclass(slots=True)
class NullInputSink:
    """Input sink that never supplies runtime input."""

    def read(self) -> None:
        """Return ``None`` because no input is available."""
        return None


@dataclass(slots=True)
class NullOutputSink:
    """Output sink that optionally records values without presenting them."""

    store_outputs: bool = False
    """Whether to retain outputs for inspection."""

    output_count: int = field(default=0, init=False)
    """Number of outputs consumed since construction."""

    outputs: list[object] = field(default_factory=list, init=False)
    """Retained outputs when ``store_outputs`` is enabled."""

    def write(self, output: object) -> None:
        """Consume one output."""
        self.output_count += 1
        if self.store_outputs:
            self.outputs.append(output)


class IFlashDreamsApplicationSession(ABC):
    """One isolated application session with sequential model state."""

    @abstractmethod
    def init(self) -> None:
        """Initialize model and per-session resources."""

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
    input_src: InputSink | None = None,
    output_sink: OutputSink | None = None,
) -> None:
    """Load and run an application with host-provided input and output sinks.

    Args:
        application_slug: Installed application or concrete demo slug.
        commandline_args: Arguments forwarded to the application.
        input_src: Input sink; ``None`` supplies no dynamic inputs.
        output_sink: Output sink; ``None`` opens a local video window.
    """
    resolved_input = input_src if input_src is not None else NullInputSink()
    resolved_output = (
        output_sink if output_sink is not None else VideoWindowOutputSink()
    )
    application, slug_args = create_application(application_slug)
    application.init(
        [*slug_args, *commandline_args],
        resolved_input,
        resolved_output,
    )
    session = application.create_session(resolved_input, resolved_output)
    try:
        session.init()
        session.generate(resolved_input, resolved_output)
    finally:
        try:
            session.close()
        finally:
            close_output = getattr(resolved_output, "close", None)
            if callable(close_output):
                close_output()
            close_input = getattr(resolved_input, "close", None)
            if callable(close_input):
                close_input()


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


def entrypoint(argv: Sequence[str] | None = None) -> None:
    """Run the ``flashdreams`` console entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        args.pop(0)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: flashdreams [run] APPLICATION [APPLICATION_ARGS ...]")
        return
    application_slug = args.pop(0)
    run_application(application_slug, args)


__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "ApplicationFactory",
    "IFlashDreamsApplication",
    "IFlashDreamsApplicationSession",
    "InputSink",
    "NullInputSink",
    "NullOutputSink",
    "OutputSink",
    "VideoWindowOutputSink",
    "create_application",
    "entrypoint",
    "run_application",
]
