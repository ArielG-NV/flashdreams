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

"""Null, MP4, benchmark, composite, and local-window application output sinks."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from flashdreams.demo.io import OutputDecision, OutputSink, SessionInfo
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.results import StepResult
from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    write_video_tensor,
)
from flashdreams.infra.video_output import VideoResultCollector, prepare_video_for_mp4
from flashdreams.runtime.metrics import RuntimeMetricSample
from flashdreams.runtime.output import NullOutputTarget, OutputArtifact, OutputTarget
from flashdreams.runtime.video_output import Mp4VideoOutputTarget, VideoWriter

if TYPE_CHECKING:
    from flashdreams.runtime.demo.spec import OutputSpec


class CompositeOutputSinkError(RuntimeError):
    """Failure from one or more sinks during a composite lifecycle operation."""

    def __init__(self, operation: str, errors: Sequence[BaseException]) -> None:
        self.operation = operation
        """Composite lifecycle operation that failed."""

        self.errors = tuple(errors)
        """Errors raised by child sinks."""

        details = "; ".join(f"{type(error).__name__}: {error}" for error in self.errors)
        super().__init__(
            f"CompositeOutputSink.{operation} failed for {len(self.errors)} "
            f"sink(s): {details}"
        )


@dataclass(slots=True)
class NullOutputSink(OutputSink):
    """Consume results without presentation or persistence."""

    store_results: bool = False
    """Whether to retain serializable result records."""

    store_outputs: bool = False
    """Whether to retain raw result payloads for application tests."""

    produces_artifacts: bool = field(default=False, init=False)
    """Whether this sink produces artifacts."""

    output_count: int = field(default=0, init=False)
    """Number of results consumed since opening."""

    results: list[Mapping[str, object]] = field(default_factory=list, init=False)
    """Serializable result records retained when requested."""

    outputs: list[Any] = field(default_factory=list, init=False)
    """Raw result payloads retained when requested."""

    opened: bool = field(default=False, init=False)
    """Whether the sink is open."""

    closed: bool = field(default=True, init=False)
    """Whether the sink is closed."""

    session_info: SessionInfo | None = field(default=None, init=False)
    """Metadata supplied when opening the sink."""

    generation: int | None = field(default=None, init=False)
    """Most recently started generation."""

    def open(self, session_info: SessionInfo) -> None:
        """Prepare the sink for a session."""
        self.session_info = session_info
        self.output_count = 0
        self.results.clear()
        self.outputs.clear()
        self.opened = True
        self.closed = False

    def begin_generation(self, generation: int) -> None:
        """Record the active generation."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        self.generation = generation

    def write(self, result: StepResult) -> OutputDecision:
        """Consume one generated result."""
        if not self.opened or self.closed:
            raise RuntimeError("Cannot write to a closed output sink.")
        self.output_count += 1
        if self.store_results:
            self.results.append(_result_record(result))
        if self.store_outputs:
            self.outputs.append(result.output)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        """Close the sink without producing artifacts."""
        self.opened = False
        self.closed = True
        return ()


@dataclass(slots=True)
class Mp4OutputSink(OutputSink):
    """Collect generated video results into one MP4 artifact."""

    output_path: Path
    """Destination MP4 path."""

    fps: int | float | None = None
    """Output frame rate; ``None`` uses session metadata."""

    output_layout: VideoTensorLayout | None = None
    """Required video layout; ``None`` uses session metadata."""

    writer: VideoWriter = field(default=write_video_tensor, repr=False)
    """Video writer implementation."""

    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT
    """Dependency hint included in video-writer errors."""

    move_to_cpu: bool = True
    """Whether to move collected chunks to CPU memory immediately."""

    enabled: bool = True
    """Whether to retain and write submitted video chunks."""

    produces_artifacts: bool = field(default=True, init=False)
    """Whether this sink produces artifacts."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether the sink is open."""

    _collector: VideoResultCollector | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Per-session video result collector."""

    _artifacts: tuple[OutputArtifact, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Cached idempotent close result."""

    session_info: SessionInfo | None = field(default=None, init=False)
    """Metadata supplied when opening the sink."""

    def __post_init__(self) -> None:
        if self.fps is not None and float(self.fps) <= 0:
            raise ValueError("Mp4OutputSink.fps must be > 0 when set.")
        self.output_path = Path(self.output_path)

    def open(self, session_info: SessionInfo) -> None:
        """Prepare the MP4 collector for a session."""
        if self.fps is None:
            if session_info.frames_per_second is None:
                raise ValueError(
                    "Mp4OutputSink requires fps or SessionInfo.frames_per_second."
                )
            self.fps = session_info.frames_per_second
        if self.output_layout is None:
            if session_info.output_layout is None:
                raise ValueError(
                    "Mp4OutputSink requires output_layout or SessionInfo.output_layout."
                )
            self.output_layout = cast(
                VideoTensorLayout,
                session_info.output_layout,
            )
        elif (
            session_info.output_layout is not None
            and session_info.output_layout != self.output_layout
        ):
            raise ValueError(
                "Mp4OutputSink output_layout does not match SessionInfo: "
                f"{self.output_layout!r} != {session_info.output_layout!r}."
            )
        self.session_info = session_info
        self._collector = VideoResultCollector(
            output_layout=self.output_layout,
            enabled=self.enabled,
            move_to_cpu=self.move_to_cpu,
        )
        self._artifacts = None
        self._opened = True

    def begin_generation(self, generation: int) -> None:
        """Continue recording across generation resets."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")

    def write(self, result: StepResult) -> OutputDecision:
        """Collect one layout-aware video result."""
        if not self._opened or self._collector is None:
            raise RuntimeError("Cannot write to a closed output sink.")
        if result.layout is None:
            raise TypeError("Mp4OutputSink requires a video StepResult with layout.")
        if result.layout != self.output_layout:
            raise ValueError(
                "Mp4OutputSink received layout "
                f"{result.layout!r}; expected {self.output_layout!r}."
            )
        self._collector.add(result)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        """Write the collected video and return its artifact."""
        if self._artifacts is not None:
            return self._artifacts
        if self._collector is None:
            self._opened = False
            self._artifacts = ()
            return self._artifacts

        collector = self._collector
        self._collector = None
        self._opened = False
        video = collector.finish()
        if video is None:
            self._artifacts = ()
            return self._artifacts
        assert self.output_layout is not None
        writable_video, writable_layout = prepare_video_for_mp4(
            video,
            layout=self.output_layout,
        )
        assert self.fps is not None
        path = self.writer(
            writable_video,
            self.output_path,
            fps=self.fps,
            layout=writable_layout,
            install_hint=self.install_hint,
        )
        self._artifacts = (
            OutputArtifact(
                kind="video/mp4",
                uri=str(path),
                metadata={
                    "fps": self.fps,
                    "source_layout": self.output_layout,
                    "shape": tuple(int(dim) for dim in video.shape),
                    "stats_history": tuple(collector.stats_history),
                },
            ),
        )
        return self._artifacts



@dataclass(slots=True)
class BenchmarkStatsOutputSink(OutputSink):
    """Persist structured runtime metrics for a benchmark session."""

    output_path: Path
    """Destination JSON path."""

    schema_version: int = 1
    """Benchmark artifact schema version."""

    produces_artifacts: bool = field(default=True, init=False)
    """Whether this sink produces artifacts."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether the sink is open."""

    _closed: bool = field(default=True, init=False, repr=False)
    """Whether the sink is closed."""

    _session_info: SessionInfo | None = field(default=None, init=False, repr=False)
    """Metadata supplied when opening the sink."""

    _steps: list[Mapping[str, Any]] = field(default_factory=list, init=False)
    """Serializable records for consumed results."""

    _samples: list[RuntimeMetricSample] = field(default_factory=list, init=False)
    """Normalized runtime metric samples from consumed results."""

    _artifacts: tuple[OutputArtifact, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Cached idempotent close result."""

    generation: int | None = field(default=None, init=False)
    """Most recently started generation."""

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        if self.schema_version <= 0:
            raise ValueError("BenchmarkStatsOutputSink.schema_version must be > 0.")

    def open(self, session_info: SessionInfo) -> None:
        """Prepare metric collection for a benchmark session."""
        self._session_info = session_info
        self._steps.clear()
        self._samples.clear()
        self._artifacts = None
        self._opened = True
        self._closed = False

    def begin_generation(self, generation: int) -> None:
        """Record the active benchmark generation."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        self.generation = generation

    def write(self, result: StepResult) -> OutputDecision:
        """Collect one result and its normalized runtime metric samples."""
        if not self._opened or self._closed:
            raise RuntimeError("Cannot write to a closed output sink.")
        samples = tuple(_runtime_metric_samples_from_result(result))
        self._steps.append(_benchmark_step_record(result, sample_count=len(samples)))
        self._samples.extend(samples)
        return OutputDecision()

    def close(self) -> Sequence[OutputArtifact]:
        """Write the benchmark JSON and return its artifact."""
        if self._artifacts is not None:
            return self._artifacts

        payload = {
            "schema_version": self.schema_version,
            "artifact_type": "flashdreams.runtime.demo.benchmark_stats",
            "session": _session_info_record(self._session_info),
            "steps": [_json_value(step) for step in self._steps],
            "samples": [
                _runtime_metric_sample_record(sample) for sample in self._samples
            ],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._opened = False
        self._closed = True
        self._artifacts = (
            OutputArtifact(
                kind="application/json",
                uri=str(self.output_path),
                metadata={
                    "artifact_type": "benchmark_stats",
                    "schema_version": self.schema_version,
                    "step_count": len(self._steps),
                    "sample_count": len(self._samples),
                },
            ),
        )
        return self._artifacts


@dataclass(slots=True)
class CompositeOutputSink(OutputSink):
    """Fan out generated results to multiple output sinks."""

    sinks: Sequence[OutputSink]
    """Child sinks that receive every lifecycle operation."""

    produces_artifacts: bool = field(default=False, init=False)
    """Whether any child sink produces artifacts."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether every child sink opened successfully."""

    _closed: bool = field(default=True, init=False, repr=False)
    """Whether the composite is closed."""

    _artifacts: tuple[OutputArtifact, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Cached child artifacts from the most recent close."""

    def __post_init__(self) -> None:
        sinks = tuple(self.sinks)
        if not sinks:
            raise ValueError("CompositeOutputSink requires at least one sink.")
        self.sinks = sinks
        self.produces_artifacts = any(sink.produces_artifacts for sink in sinks)

    def open(self, session_info: SessionInfo) -> None:
        """Open every child sink and clean up opened siblings after failure."""
        opened_sinks: list[OutputSink] = []
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                sink.open(session_info)
            except Exception as exc:
                errors.append(exc)
            else:
                opened_sinks.append(sink)

        if errors:
            for sink in reversed(opened_sinks):
                try:
                    sink.close()
                except Exception as exc:
                    errors.append(exc)
            self._artifacts = None
            self._opened = False
            self._closed = True
            raise CompositeOutputSinkError("open", errors)

        self._artifacts = None
        self._opened = True
        self._closed = False

    def begin_generation(self, generation: int) -> None:
        """Start the same generation on every child sink."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        for sink in self.sinks:
            sink.begin_generation(generation)

    def write(self, result: StepResult) -> OutputDecision:
        """Write one result to every child and combine their decisions."""
        if not self._opened or self._closed:
            raise RuntimeError("Cannot write to a closed output sink.")
        return _combine_output_decisions(sink.write(result) for sink in self.sinks)

    def close(self) -> Sequence[OutputArtifact]:
        """Close every child and return all successfully produced artifacts."""
        if self._artifacts is not None:
            return self._artifacts

        artifacts: list[OutputArtifact] = []
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                artifacts.extend(sink.close())
            except Exception as exc:
                errors.append(exc)
        self._opened = False
        self._closed = True
        self._artifacts = tuple(artifacts)
        if errors:
            raise CompositeOutputSinkError("close", errors)
        return self._artifacts


@dataclass(slots=True)
class LocalWindowOutputSink(OutputSink):
    """Present ordered video frames through a SlangPy Vulkan window."""

    title: str = "FlashDreams"
    """Local window title."""

    fps: float | None = None
    """Playback rate; ``None`` uses session metadata."""

    presenter_factory: Callable[..., Any] | None = field(
        default=None,
        repr=False,
    )
    """Optional presenter factory used by tests and alternate native hosts."""

    produces_artifacts: bool = field(default=False, init=False)
    """Whether this sink produces artifacts."""

    _presenter: Any | None = field(default=None, init=False, repr=False)
    """Active local-window presenter."""

    _frame_interval_s: float = field(default=0.0, init=False, repr=False)
    """Target interval between presented frames."""

    _next_deadline_s: float | None = field(default=None, init=False, repr=False)
    """Absolute deadline for the next presentation tick."""

    _opened: bool = field(default=False, init=False, repr=False)
    """Whether the sink is open."""

    def __post_init__(self) -> None:
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be greater than zero when set.")

    def open(self, session_info: SessionInfo) -> None:
        """Create the SlangPy presenter for one application session."""
        if session_info.video_width is None or session_info.video_height is None:
            raise ValueError(
                "LocalWindowOutputSink requires video width and height in SessionInfo."
            )
        if self.fps is None:
            self.fps = session_info.frames_per_second or 16.0
        presenter_factory = self.presenter_factory
        if presenter_factory is None:
            from flashdreams.demo.local_window import (
                SlangPyLocalWindowPresenter,
            )

            presenter_factory = SlangPyLocalWindowPresenter
        self._presenter = presenter_factory(
            width=session_info.video_width,
            height=session_info.video_height,
            title=self.title,
        )
        self._frame_interval_s = 1.0 / self.fps
        self._next_deadline_s = None
        self._opened = True

    def begin_generation(self, generation: int) -> None:
        """Reset presentation pacing for a new generation."""
        if generation < 0:
            raise ValueError("generation must be >= 0.")
        self._next_deadline_s = None

    def write(self, result: StepResult) -> OutputDecision:
        """Present one result without materializing CUDA frames on the host."""
        if not self._opened or self._presenter is None:
            raise RuntimeError("Cannot write to a closed output sink.")

        for frame in result.lazy_rgb_frames():
            if not self._presenter.present(frame):
                return OutputDecision(should_stop=True)
            now_s = time.monotonic()
            if self._next_deadline_s is None:
                self._next_deadline_s = now_s + self._frame_interval_s
            else:
                self._next_deadline_s = max(
                    now_s,
                    self._next_deadline_s + self._frame_interval_s,
                )
            if not self._presenter.wait_until(self._next_deadline_s):
                return OutputDecision(should_stop=True)
        return OutputDecision(
            metadata={
                "presentation_backend": "slangpy",
                "cuda_resident": result.video_chunk.is_cuda,
            }
        )

    def close(self) -> Sequence[OutputArtifact]:
        """Close the local window without producing artifacts."""
        if self._presenter is not None:
            self._presenter.close()
        self._presenter = None
        self._opened = False
        self._next_deadline_s = None
        return ()


def build_output_sink(
    output: OutputSpec,
    *,
    mp4_writer: VideoWriter | None = None,
) -> OutputSink:
    """Build an output sink from a demo output specification."""
    from flashdreams.runtime.demo.spec import (
        Mp4OutputSpec,
        NullOutputSpec,
        WebRTCOutputSpec,
    )

    if isinstance(output, NullOutputSpec):
        return NullOutputSink(store_results=output.store_results)
    if isinstance(output, Mp4OutputSpec):
        writer = mp4_writer or write_video_tensor
        return Mp4OutputSink(
            output_path=Path(output.path),
            fps=output.fps,
            output_layout=output.output_layout,
            writer=writer,
            move_to_cpu=output.move_to_cpu,
        )
    if isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC output requires a realtime transport sink.")
    raise TypeError(f"Unsupported demo output spec: {type(output).__name__}.")



def build_benchmark_output_sink(
    output: OutputSpec | None,
    *,
    stats_path: Path,
    mp4_writer: VideoWriter | None = None,
) -> OutputSink:
    """Build a benchmark stats sink with optional output capture."""
    stats_sink = BenchmarkStatsOutputSink(output_path=stats_path)
    if output is None:
        return stats_sink
    return CompositeOutputSink(
        (
            build_output_sink(output, mp4_writer=mp4_writer),
            stats_sink,
        )
    )


def build_output_target(
    output: OutputSpec,
    *,
    mp4_writer: VideoWriter | None = None,
) -> OutputTarget:
    """Build a replay output target from a demo output specification."""
    from flashdreams.runtime.demo.spec import (
        Mp4OutputSpec,
        NullOutputSpec,
        WebRTCOutputSpec,
    )

    if isinstance(output, NullOutputSpec):
        return NullOutputTarget(store_results=output.store_results)
    if isinstance(output, Mp4OutputSpec):
        output_path = Path(output.path)
        if mp4_writer is not None:
            return Mp4VideoOutputTarget(
                output_path=output_path,
                fps=output.fps,
                output_layout=output.output_layout,
                writer=mp4_writer,
                move_to_cpu=output.move_to_cpu,
            )
        return Mp4VideoOutputTarget(
            output_path=output_path,
            fps=output.fps,
            output_layout=output.output_layout,
            move_to_cpu=output.move_to_cpu,
        )
    if isinstance(output, WebRTCOutputSpec):
        raise ValueError("WebRTC output does not create a replay OutputTarget.")
    raise TypeError(f"Unsupported demo output spec: {type(output).__name__}.")


def _result_record(result: StepResult) -> Mapping[str, object]:
    record: dict[str, object] = {
        "step_index": result.step_index,
        "frame_count": result.frame_count,
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
    }
    if result.layout is not None:
        record["layout"] = result.layout
    if result.output_window is not None:
        record["output_window"] = (
            result.output_window.start_s,
            result.output_window.end_s,
        )
    return record



def _benchmark_step_record(
    result: StepResult,
    *,
    sample_count: int,
) -> Mapping[str, Any]:
    record = dict(_result_record(result))
    record["sample_count"] = sample_count
    return record


def _runtime_metric_samples_from_result(
    result: StepResult,
) -> Sequence[RuntimeMetricSample]:
    samples: list[RuntimeMetricSample] = []
    metadata: dict[str, Any] = {"frame_count": result.frame_count}
    if result.layout is not None:
        metadata["layout"] = result.layout
    if result.output_window is not None:
        metadata["output_window"] = {
            "start_s": result.output_window.start_s,
            "end_s": result.output_window.end_s,
        }
    if result.metadata:
        metadata["result_metadata"] = dict(result.metadata)

    for name, value in result.metrics.items():
        if not name.strip():
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(float(value)):
            continue
        normalized_name, normalized_value, unit, category = _normalize_metric_sample(
            name,
            value,
        )
        samples.append(
            RuntimeMetricSample(
                name=normalized_name,
                value=normalized_value,
                unit=unit,
                category=category,
                step_index=result.step_index,
                metadata=metadata,
            )
        )
    return tuple(samples)


def _normalize_metric_sample(
    name: str,
    value: int | float,
) -> tuple[str, int | float, str, str]:
    if name.endswith("_ms"):
        return f"{name[:-3]}_s", float(value) / 1000.0, "s", "timing"
    if name.endswith("_s"):
        return name, value, "s", "timing"
    if name.endswith("_fps"):
        return name, value, "fps", "throughput"
    if name.endswith("_bytes"):
        return name, value, "bytes", "runtime"
    if name.endswith("_gib"):
        return name, value, "gib", "memory"
    if name.endswith("_count") or name in {"frames", "frame_count"}:
        return name, value, "count", "runtime"
    return name, value, "value", "runtime"


def _runtime_metric_sample_record(sample: RuntimeMetricSample) -> Mapping[str, Any]:
    return {
        "name": sample.name,
        "value": sample.value,
        "unit": sample.unit,
        "category": sample.category,
        "step_index": sample.step_index,
        "metadata": _json_value(sample.metadata),
    }


def _session_info_record(session_info: SessionInfo | None) -> Mapping[str, Any]:
    if session_info is None:
        return {}
    return {
        "output_layout": session_info.output_layout,
        "steady_output_frame_count": session_info.steady_output_frame_count,
        "metadata": _json_value(session_info.metadata),
    }


def _combine_output_decisions(decisions: Iterable[OutputDecision]) -> OutputDecision:
    decisions = tuple(decisions)
    if not decisions:
        return OutputDecision()

    metadata = tuple(
        _json_value(decision.metadata) for decision in decisions if decision.metadata
    )
    return OutputDecision(
        should_stop=any(decision.should_stop for decision in decisions),
        dropped=any(decision.dropped for decision in decisions),
        drop_policy=_combine_drop_policy(decisions),
        backpressure_s=max(decision.backpressure_s for decision in decisions),
        metadata={"decisions": metadata} if metadata else {},
    )


def _combine_drop_policy(
    decisions: Sequence[OutputDecision],
) -> Literal["none", "drop_newest", "drop_oldest"]:
    policies = tuple(
        decision.drop_policy for decision in decisions if decision.drop_policy != "none"
    )
    if not policies:
        return "none"
    if "drop_oldest" in policies:
        return "drop_oldest"
    return "drop_newest"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Sequence):
        return [_json_value(item) for item in value]
    return repr(value)


__all__ = [
    "BenchmarkStatsOutputSink",
    "CompositeOutputSink",
    "CompositeOutputSinkError",
    "LocalWindowOutputSink",
    "Mp4OutputSink",
    "NullOutputSink",
    "build_benchmark_output_sink",
    "build_output_sink",
    "build_output_target",
]
