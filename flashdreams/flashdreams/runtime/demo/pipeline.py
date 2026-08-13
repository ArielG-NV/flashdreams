# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared per-step pipeline for demo session drivers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from flashdreams.demo.io import OutputDecision, OutputSink
from flashdreams.runtime.inputs import CanonicalInputWindow
from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepRequirements, StepResult

from .run_modes import SessionMetricsRecorder
from .session_inputs import ControlDecision, ModelInputProvider, UserInputWindow


@dataclass(frozen=True, kw_only=True, slots=True)
class StepOutcome:
    """Combined output and control result from one shared model step."""

    output: OutputDecision = field(default_factory=OutputDecision)
    control: ControlDecision = field(default_factory=ControlDecision)


class StepPipeline:
    """Shared invariant for provider conversion, model step, output, and metrics."""

    def execute_application_step(
        self,
        *,
        inputs: CanonicalInputWindow,
        step: Callable[[CanonicalInputWindow], StepResult | None],
        output: OutputSink,
    ) -> StepOutcome | None:
        """Execute one application-authored step through shared output handling.

        Args:
            inputs: Canonical live inputs for the current application step.
            step: Application callback that produces one result; ``None``
                reports normal session completion.
            output: Sink that consumes the generated result.

        Returns:
            The output decision for a generated result, or ``None`` when the
            application session is complete.
        """
        result = step(inputs)
        if result is None:
            return None
        return StepOutcome(output=self._write_result(result=result, output=output))

    def execute_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
        provider: ModelInputProvider,
        session: InferenceSession,
        output: OutputSink,
        metrics: SessionMetricsRecorder,
    ) -> StepOutcome:
        prepared = provider.prepare_step(
            request=request,
            user_window=user_window,
        )
        if prepared.control.reset or prepared.control.close_session:
            metrics.record_control(
                request=request,
                user_window=user_window,
                control=prepared.control,
            )
            return StepOutcome(control=prepared.control)
        if prepared.inference_input is None:
            raise RuntimeError("ModelInputProvider returned no inference input.")

        result = session.step(prepared.inference_input)
        if not isinstance(result, StepResult):
            raise TypeError(
                "InferenceSession.step must return StepResult, "
                f"got {type(result).__name__}."
            )
        decision = self._write_result(result=result, output=output)
        metrics.record_step(
            request=request,
            user_window=user_window,
            inference_input=prepared.inference_input,
            result=result,
            decision=decision,
        )
        return StepOutcome(output=decision)

    @staticmethod
    def _write_result(*, result: StepResult, output: OutputSink) -> OutputDecision:
        if not isinstance(result, StepResult):
            raise TypeError(
                "Application session step must return StepResult or None, "
                f"got {type(result).__name__}."
            )
        decision = output.write(result)
        if not isinstance(decision, OutputDecision):
            raise TypeError(
                "OutputSink.write must return OutputDecision, "
                f"got {type(decision).__name__}."
            )
        return decision


__all__ = ["StepOutcome", "StepPipeline"]
