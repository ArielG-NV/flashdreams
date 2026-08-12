# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-target selection for ``flashdreams-run`` demos."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast, runtime_checkable

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoSpec,
    Mp4OutputSpec,
    NativeWindowOutputSpec,
    NullOutputSpec,
    OutputSpec,
    WebRTCOutputSpec,
)

LaunchMode: TypeAlias = Literal["run", "mp4", "null", "webrtc", "native-window"]
DemoInputMode: TypeAlias = Literal["replay", "realtime"]


class LaunchModeUnavailableError(ValueError):
    """Raised when the selected output cannot consume a demo's input mode."""


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Model-neutral settings passed from the central CLI to a demo."""

    host: str | None = None
    port: int | None = None
    scenario: Mapping[str, object] = field(default_factory=dict)
    output: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DemoDefinition:
    """Target-neutral demo inputs and emitted video-track defaults."""

    model_id: str
    input_mode: DemoInputMode
    config: InferenceConfig
    preset_id: str | None = None
    scenario: object | None = None
    fps: int = 30
    video_width: int = 1280
    video_height: int = 720
    output_layout: VideoTensorLayout = "bvtchw"

    def build_spec(self, output: OutputSpec) -> DemoSpec:
        """Attach a shared output target after integration construction."""
        return DemoSpec(
            model_id=self.model_id,
            input_mode=self.input_mode,
            output=output,
            preset_id=self.preset_id,
            scenario=self.scenario,
            config=self.config,
        )


@dataclass(frozen=True, slots=True)
class ResolvedLaunch:
    """Validated launch ready to execute without invoking another CLI."""

    mode: LaunchMode
    label: str
    launch: Callable[[], object] = field(repr=False)
    summary: Mapping[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@runtime_checkable
class LaunchCapability(Protocol):
    """Target-neutral construction supplied by one demo integration."""

    def adapter(self, config: RunnerConfig) -> DemoAdapter: ...

    def demo(
        self,
        config: RunnerConfig,
        *,
        input_mode: DemoInputMode,
        scenario: Mapping[str, object],
    ) -> DemoDefinition: ...


@dataclass(frozen=True, slots=True)
class _OutputTarget:
    input_mode: DemoInputMode
    build: Callable[[RunnerConfig, LaunchOptions, DemoDefinition], OutputSpec]
    run: Callable[[DemoSpec, DemoAdapter], object]


def available_launch_modes(
    config: RunnerConfig,
    options: LaunchOptions | None = None,
) -> tuple[LaunchMode, ...]:
    """Infer compatible targets from the adapter's input capabilities."""
    del options
    capability = _resolve_capability(config)
    if capability is None:
        return ("run",)
    supported_inputs = frozenset(capability.adapter(config).supported_input_modes())
    modes = tuple(
        mode
        for mode, target in _OUTPUT_TARGETS.items()
        if target.input_mode in supported_inputs
    )
    return ("run", *modes)


def resolve_launch(
    config: RunnerConfig,
    *,
    mode: LaunchMode,
    options: LaunchOptions | None = None,
) -> ResolvedLaunch:
    """Validate a target against the demo API and construct its launch."""
    if mode == "run":
        raise ValueError("Mode 'run' is executed directly by the selected Runner.")
    options = options or LaunchOptions()
    capability = _resolve_capability(config)
    target = _OUTPUT_TARGETS.get(mode)
    if capability is None or target is None:
        _raise_unavailable(config, mode, options)
        raise AssertionError("unreachable")
    adapter = capability.adapter(config)
    if target.input_mode not in adapter.supported_input_modes():
        _raise_unavailable(config, mode, options)
    demo = capability.demo(
        config, input_mode=target.input_mode, scenario=options.scenario
    )
    if demo.input_mode != target.input_mode:
        raise ValueError(
            "Demo launch capability returned input_mode="
            f"{demo.input_mode!r}; expected {target.input_mode!r}."
        )
    output = target.build(config, options, demo)
    spec = demo.build_spec(output)
    return ResolvedLaunch(
        mode=mode,
        label=f"{mode} output",
        summary=_summary(config, mode, output),
        launch=partial(target.run, spec, adapter),
    )


def _raise_unavailable(
    config: RunnerConfig,
    mode: LaunchMode,
    options: LaunchOptions,
) -> None:
    supported = ", ".join(available_launch_modes(config, options))
    raise LaunchModeUnavailableError(
        f"Output target {mode!r} is not compatible with runner "
        f"{config.runner_name!r}. Available targets: {supported}."
    )


def _mp4_output(
    config: RunnerConfig,
    options: LaunchOptions,
    demo: DemoDefinition,
) -> OutputSpec:
    _validate_output_fields(options.output, {"path", "output", "fps"})
    path = options.output.get("path") or options.output.get("output")
    if path is None:
        path = getattr(config, "output", Path("outputs") / f"{config.runner_name}.mp4")
    return Mp4OutputSpec(
        path=Path(cast(Any, path)),
        fps=int(cast(Any, options.output.get("fps", demo.fps))),
        output_layout=demo.output_layout,
    )


def _null_output(
    config: RunnerConfig,
    options: LaunchOptions,
    demo: DemoDefinition,
) -> OutputSpec:
    del config, demo
    _validate_output_fields(options.output, set())
    return NullOutputSpec()


def _webrtc_output(
    config: RunnerConfig,
    options: LaunchOptions,
    demo: DemoDefinition,
) -> OutputSpec:
    _validate_output_fields(
        options.output,
        {
            "host",
            "port",
            "fps",
            "video_height",
            "video_width",
            "warmup_chunks",
            "warmup_timeout_s",
            "client_liveness_timeout_s",
        },
    )
    return WebRTCOutputSpec(
        host=str(options.host or options.output.get("host", "0.0.0.0")),
        port=int(cast(Any, options.port or options.output.get("port", 8080))),
        fps=int(cast(Any, options.output.get("fps", demo.fps))),
        video_height=int(
            cast(Any, options.output.get("video_height", demo.video_height))
        ),
        video_width=int(cast(Any, options.output.get("video_width", demo.video_width))),
        warmup_chunks=int(cast(Any, options.output.get("warmup_chunks", 0))),
        warmup_timeout_s=float(
            cast(Any, options.output.get("warmup_timeout_s", 600.0))
        ),
        client_liveness_timeout_s=float(
            cast(Any, options.output.get("client_liveness_timeout_s", 30.0))
        ),
    )


def _native_window_output(
    config: RunnerConfig,
    options: LaunchOptions,
    demo: DemoDefinition,
) -> OutputSpec:
    _validate_output_fields(
        options.output,
        {"fps", "video_height", "video_width", "title", "max_queued_chunks"},
    )
    return NativeWindowOutputSpec(
        fps=int(cast(Any, options.output.get("fps", demo.fps))),
        video_height=int(
            cast(Any, options.output.get("video_height", demo.video_height))
        ),
        video_width=int(cast(Any, options.output.get("video_width", demo.video_width))),
        title=str(options.output.get("title", config.runner_name)),
        max_queued_chunks=int(cast(Any, options.output.get("max_queued_chunks", 2))),
    )


def _validate_output_fields(values: Mapping[str, object], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported output fields: {', '.join(unknown)}.")


def _summary(
    config: RunnerConfig,
    mode: LaunchMode,
    output: OutputSpec,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "runner": config.runner_name,
        "output_target": mode,
        "device": config.device,
    }
    if isinstance(output, Mp4OutputSpec):
        summary["output_path"] = output.path
    if isinstance(output, WebRTCOutputSpec):
        summary.update(host=output.host, port=output.port)
    return summary


def _run_replay(spec: DemoSpec, adapter: DemoAdapter) -> object:
    from flashdreams.runtime.demo.replay import run_replay_demo

    result = run_replay_demo(spec=spec, adapter=adapter)
    if result.status != "completed":
        reason = result.reason or str(result.error) or "Demo replay failed."
        raise RuntimeError(reason)
    return result


def _run_native_window(spec: DemoSpec, adapter: DemoAdapter) -> object:
    from flashdreams.runtime.demo.bootstrap import initialize_cuda_distributed
    from flashdreams.serving.native_window import run_native_window_demo

    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    context = initialize_cuda_distributed(default_device=spec.config.device or "cuda")
    if context.world_size != 1:
        raise RuntimeError("Native-window output requires one process.")
    return run_native_window_demo(spec=spec, adapter=adapter)


@dataclass(frozen=True, slots=True)
class _WebRTCRuntimeConfig:
    video_width: int
    video_height: int
    warmup_chunks: int
    warmup_timeout_s: float
    device: str
    encoder_backend: Literal["auto", "default", "nvenc"] = "auto"


def _run_webrtc(spec: DemoSpec, adapter: DemoAdapter) -> object:
    from flashdreams.runtime.demo import RuntimeHost, WebRTCAppResources
    from flashdreams.runtime.demo.bootstrap import initialize_cuda_distributed
    from flashdreams.serving.webrtc.demo import serve_webrtc_demo
    from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

    output = spec.output
    if not isinstance(output, WebRTCOutputSpec):
        raise TypeError("WebRTC target requires WebRTCOutputSpec.")
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    context = initialize_cuda_distributed(default_device=spec.config.device or "cuda")
    scenario = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(spec.config)
    host = RuntimeHost(runtime)
    runtime_config = _WebRTCRuntimeConfig(
        video_width=output.video_width,
        video_height=output.video_height,
        warmup_chunks=output.warmup_chunks,
        warmup_timeout_s=output.warmup_timeout_s,
        device=str(context.device),
    )
    manager = BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=output.fps,
        identity=spec.model_id,
        client_liveness_timeout_s=output.client_liveness_timeout_s,
        shared_host=host,
        shared_adapter=adapter,
        shared_spec=spec,
        shared_scenario=scenario,
    )
    return serve_webrtc_demo(
        output=output,
        model_id=spec.model_id,
        session_manager=manager,
        app_resources=WebRTCAppResources(),
        world_rank=context.world_rank,
    )


_OUTPUT_TARGETS: dict[LaunchMode, _OutputTarget] = {
    "mp4": _OutputTarget("replay", _mp4_output, _run_replay),
    "null": _OutputTarget("replay", _null_output, _run_replay),
    "webrtc": _OutputTarget("realtime", _webrtc_output, _run_webrtc),
    "native-window": _OutputTarget(
        "realtime", _native_window_output, _run_native_window
    ),
}


def _resolve_capability(config: RunnerConfig) -> LaunchCapability | None:
    path = config.launch_capability
    if not path:
        return None
    return _load_launch_capability(path)


@cache
def _load_launch_capability(path: str) -> LaunchCapability:
    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "RunnerConfig.launch_capability must use 'module:attribute' syntax; "
            f"got {path!r}."
        ) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    if callable(value) and not isinstance(value, LaunchCapability):
        value = value()
    if not isinstance(value, LaunchCapability):
        raise TypeError(
            f"Launch capability {path!r} does not implement LaunchCapability."
        )
    return value


__all__ = [
    "DemoDefinition",
    "DemoInputMode",
    "LaunchCapability",
    "LaunchMode",
    "LaunchModeUnavailableError",
    "LaunchOptions",
    "ResolvedLaunch",
    "available_launch_modes",
    "resolve_launch",
]
