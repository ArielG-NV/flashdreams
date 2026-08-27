# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One text-to-video rollout: a prompt in, a chunk of frames per step out."""

from dataclasses import dataclass
from typing import Any

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass(slots=True)
class T2VModelState:
    """Mutable rollout state owned by the model loop."""

    pipeline_source: Any
    device: str
    prompt: str
    session_desc: SessionDesc
    total_blocks: int
    blocks_generated: int = 0
    cache: Any = None
    pipeline: Any = None


class T2VModelLoop(IModelLoop[T2VModelState]):
    """Run one autoregressive block per model-loop iteration."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate and finalize one autoregressive block."""
        del events
        state = self.state
        _ensure_pipeline(state)
        if state.cache is None:
            state.cache = _new_cache(state)
        state.blocks_generated += 1
        frames = state.pipeline.generate(
            autoregressive_index=step_index, cache=state.cache
        )
        metrics = state.pipeline.finalize(
            autoregressive_index=step_index, cache=state.cache
        )
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=int(frames.shape[0]),
                output_layout=state.session_desc.output_layout,
                metrics=dict(metrics or {}),
            )
        ]

    def is_finished(self) -> bool:
        return self.state.blocks_generated >= self.state.total_blocks

    def reset(self) -> None:
        self.state.blocks_generated = 0
        _ensure_pipeline(self.state)
        self.state.cache = _new_cache(self.state)

    def close(self) -> None:
        self.state.cache = None
        pipeline = self.state.pipeline
        self.state.pipeline = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()


class T2VSession(ISession):
    """One rollout of a streaming text-to-video model.

    A step is one autoregressive block, continuing the block before it rather
    than starting again. A causal decoder's first block usually decodes fewer
    frames than the rest.

    A rollout is ``total_blocks`` long and then reports itself finished, so
    nothing above it counts steps. What belongs to a run is the cache this
    initializes, holding the encoded prompt and the attention state; the
    pipeline is constructed and owned by the model process.
    """

    def __init__(
        self,
        pipeline: Any,
        prompt: str,
        session_desc: SessionDesc,
        total_blocks: int,
        *,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            pipeline: Serializable pipeline config, or a lightweight stand-in.
            prompt: Text to generate from.
            session_desc: Session the application accepted, already checked
                against what the model can produce.
            total_blocks: Blocks this rollout generates before it is finished.
        """
        self._pipeline = pipeline
        self._device = device
        self._prompt = prompt
        self._session_desc = session_desc
        self._total_blocks = total_blocks

    def init(self) -> None:
        """Register a CUDA-free model-loop specification for the worker."""
        state = T2VModelState(
            pipeline_source=self._pipeline,
            device=self._device,
            prompt=self._prompt,
            session_desc=self._session_desc,
            total_blocks=self._total_blocks,
        )
        self.register_model_loop(T2VModelLoop, state=state)

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc


def _new_cache(state: T2VModelState) -> Any:
    """Encode the prompt into a cache for one rollout."""
    ratio = state.pipeline.decoder.spatial_compression_ratio
    return state.pipeline.initialize_cache(
        text=[state.prompt],
        image=None,
        height=state.session_desc.video_height // ratio,
        width=state.session_desc.video_width // ratio,
    )


def _ensure_pipeline(state: T2VModelState) -> None:
    """Build and validate the pipeline inside the model process."""
    if state.pipeline is not None:
        return
    source = state.pipeline_source
    setup = getattr(source, "setup", None)
    pipeline = setup() if callable(setup) else source
    to = getattr(pipeline, "to", None)
    if callable(to):
        pipeline = to(state.device)
    evaluate = getattr(pipeline, "eval", None)
    if callable(evaluate):
        pipeline = evaluate()
    ratio = pipeline.decoder.spatial_compression_ratio
    if (
        state.session_desc.video_width % ratio
        or state.session_desc.video_height % ratio
    ):
        raise ValueError(
            f"Frame dimensions must be multiples of {ratio}, got "
            f"{state.session_desc.video_width}x{state.session_desc.video_height}."
        )
    state.pipeline = pipeline
