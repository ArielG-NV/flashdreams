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

"""Model-neutral interactive driving application and runner contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from flashdreams.demo import (
    CanonicalInputSchema,
    CanonicalInputWindow,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.results import StepResult
from flashdreams.runtime import DRIVER_COMMAND, StepRequirements


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractiveDriveCommand:
    """Normalized vehicle controls consumed by an integration runner."""

    throttle: float
    """Normalized acceleration input."""

    brake: float
    """Normalized braking input."""

    steer: float
    """Normalized steering input."""

    stop: bool
    """Whether the current session should stop."""

    reverse: bool
    """Whether reverse gear is requested."""


class InteractiveDriveRunnerSession(ABC):
    """Integration-owned state for one interactive driving session."""

    @abstractmethod
    def init(self) -> None:
        """Initialize model and scene resources."""

    @abstractmethod
    def session_info(self) -> SessionInfo:
        """Return output geometry and presentation timing."""

    @abstractmethod
    def next_step_requirements(self) -> StepRequirements | None:
        """Return requirements for the next driving chunk."""

    @abstractmethod
    def step(self, command: InteractiveDriveCommand) -> StepResult:
        """Simulate and render one driving chunk."""

    def close(self) -> None:
        """Release optional integration resources."""


class InteractiveDriveRunner(ABC):
    """Integration boundary used by the reusable driving application."""

    @abstractmethod
    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse integration arguments and validate startup state."""

    @abstractmethod
    def create_session(self) -> InteractiveDriveRunnerSession:
        """Create one isolated integration runner session."""


class InteractiveDriveApplication(IFlashDreamsApplication):
    """Transport-neutral interactive driving application."""

    def __init__(self, *, runner: InteractiveDriveRunner) -> None:
        self.runner = runner

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare the canonical driving command consumed every step."""
        return CanonicalInputSchema(
            modalities=(DRIVER_COMMAND,),
            description="interactive vehicle throttle, brake, and steering",
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Initialize the integration runner from application arguments."""
        self.runner.init(commandline_args)

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Wrap one integration runner session for the shared host."""
        return InteractiveDriveApplicationSession(
            runner_session=self.runner.create_session()
        )


class InteractiveDriveApplicationSession(IFlashDreamsApplicationSession):
    """Canonical-input adapter for one integration runner session."""

    def __init__(
        self,
        *,
        runner_session: InteractiveDriveRunnerSession,
    ) -> None:
        self.runner_session = runner_session

    def init(self) -> None:
        """Initialize the integration runner session."""
        self.runner_session.init()

    def session_info(self) -> SessionInfo:
        """Return integration-provided output metadata."""
        info = self.runner_session.session_info()
        if not isinstance(info, SessionInfo):
            raise TypeError(
                "InteractiveDriveRunnerSession.session_info() must return SessionInfo."
            )
        return info

    def next_step_requirements(self) -> StepRequirements | None:
        """Return integration-provided requirements for the next chunk."""
        return self.runner_session.next_step_requirements()

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Normalize canonical controls and run one integration step."""
        command = _driver_command(inputs.values.get(DRIVER_COMMAND.name))
        result = self.runner_session.step(command)
        if not isinstance(result, StepResult):
            raise TypeError(
                "InteractiveDriveRunnerSession.step() must return StepResult."
            )
        return replace(result, output_window=inputs.window)

    def close(self) -> None:
        """Release the integration runner session."""
        self.runner_session.close()


def _driver_command(value: object) -> InteractiveDriveCommand:
    if not isinstance(value, Mapping):
        raise TypeError("Interactive drive requires canonical driver_command input.")
    command = cast(Mapping[str, Any], value)
    return InteractiveDriveCommand(
        throttle=float(command["throttle"]),
        brake=float(command["brake"]),
        steer=float(command["steer"]),
        stop=bool(command["stop"]),
        reverse=bool(command["reverse"]),
    )


__all__ = [
    "InteractiveDriveApplication",
    "InteractiveDriveApplicationSession",
    "InteractiveDriveCommand",
    "InteractiveDriveRunner",
    "InteractiveDriveRunnerSession",
]
