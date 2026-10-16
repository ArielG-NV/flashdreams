---
title: 'Create a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo is a model-agnostic application built using the `flashdreams.api_v2`
library. It handles user input, sessions, and display.

Check the [demo gallery](../../../demos/index.md) first. If an existing app
supports the interaction you need, [add a model adapter](integrate_model_adapter_for_a_demo.md)
instead.

## Overview

This example builds an app that sends a text prompt from an ImGui UI to a
text-to-video model (`StreamInferencePipeline`).

## Package layout

For a demo named `minimal_t2v`, add the app files and update the website:

```text
apps/minimal_t2v/
  pyproject.toml
  minimal_t2v/
    __init__.py       # export MinimalT2VApplication
    application.py   # MinimalT2VApplication(IApplication)
    session.py       # session, model loop, and model state
    ui.py            # ImGui text input and async message

docs/source/
  demos/
    minimal_t2v.md    # add demo page
    index.md          # add demo card
    .nav.yml          # add sidebar entry
  models/
    <model that demo supports>.md        # add quick start and link to demo page
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flashdreams-minimal-t2v"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["flashdreams[local-window]"]

[tool.uv.sources]
flashdreams = { workspace = true }

[tool.setuptools.packages.find]
include = ["minimal_t2v*"]
```

### `application.py`

The adapter supplies a pipeline and its default output description. No custom
command-line arguments are needed.

```python
from flashdreams.api_v2.application import IApplication
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.runtime_v2.session_desc import SessionDesc

from .session import MinimalT2VSession


class MinimalT2VApplication(IApplication):
    def __init__(self, pipeline: StreamInferencePipeline, desc: SessionDesc):
        self.pipeline = pipeline
        self.desc = desc

    def init(self, commandline_args):
        parser = argparse.ArgumentParser(
            prog=f"flashdreams-run-v2 {self.defaults.slug} --",
            description="Generate video from an image and keyboard/mouse actions.",
        )
        parser.add_argument("--default-prompt", type=str, required=True)
        args = parser.parse_args(commandline_args)
        self.prompt = args.prompt

    def session_desc(self):
        return self.desc

    def create_session(self, session_desc):
        return MinimalT2VSession(self.pipeline, session_desc, self.prompt)
```

### `__init__.py`

```python
from .application import MinimalT2VApplication
```

### `session.py`

Keep the pipeline, prompt, and cache in the model-loop state:

```python
from dataclasses import dataclass

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult

from .ui import MinimalT2VUILoop, MinimalT2VUIState


@dataclass
class TextToVideoState:
    pipeline: StreamInferencePipeline
    desc: SessionDesc
    prompt: str = ""
    cache: StreamInferencePipelineCache | None = None
    index: int = 0

    def set_prompt(self, prompt):
        self.prompt = prompt
        self.cache = None
        self.index = 0


class TextToVideoLoop(IModelLoop[TextToVideoState]):
    def step(self, step_index, events):
        state = self.state
        if not state.prompt:
            return []
        if state.cache is None:
            state.cache = state.pipeline.initialize_cache(text=[state.prompt])
        frames = state.pipeline.generate(state.index, state.cache)
        state.pipeline.finalize(state.index, state.cache)
        state.index += 1
        return [StepResult(
            step_index=step_index,
            output=frames,
            frame_count=frames.shape[0],
            output_layout=state.desc.output_layout,
        )]


class MinimalT2VSession(ISession):
    def __init__(self, pipeline: StreamInferencePipeline, desc: SessionDesc, prompt: str):
        self.pipeline = pipeline
        self.desc = desc
        self.prompt = prompt

    @property
    def session_desc(self):
        return self.desc

    def init(self):
        model_loop = self.register_model_loop(
            TextToVideoLoop,
            state=TextToVideoState(self.pipeline, self.desc),
        )
        self.register_ui_loop(
            MinimalT2VUILoop,
            state=MinimalT2VUIState(model_loop),
            width=self.desc.video_width,
            height=self.desc.video_height,
        )
        model_loop.state.set_prompt(self.prompt)
```

Use a `SessionDesc` matching the pipeline's output size, frame rate, and layout.
This example expects frames in `tchw` layout.

### `ui.py`

```python
from dataclasses import dataclass

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.runtime_v2.imgui_ui_loop import ImGuiUILoop


@dataclass
class MinimalT2VUIState:
    model_loop: IModelLoop
    text: str = ""


class MinimalT2VUILoop(ImGuiUILoop[MinimalT2VUIState]):
    def step_ui(self, imgui, step_index, events):
        imgui.begin("Minimal T2V")
        try:
            _, self.state.text = imgui.input_text("Prompt", self.state.text)
            if imgui.button("Send"):
                invoke_async(
                    self.state.model_loop,
                    lambda state, text=self.state.text: state.set_prompt(text),
                )
        finally:
            imgui.end()
        return self.presented_model_frame()
```

`invoke_async` runs the callback on the model thread before its next step.
The UI keeps its own editable text; it never writes model state directly.

The model loop waits for a prompt. Each `set_prompt` sets our pipeline cache to `None` to force our generation of `StreamInferencePipeline` to effectively reset (fill the cahce with the new prompt).

The [API reference](../api_reference/application.md) here describes our interfaces in more detail.

## Connect a model

[Add a model adapter](integrate_model_adapter_for_a_demo.md) to supply the pipeline and register
an application slug. The adapter would live in
`integrations_v2/<model>/apps/minimal_t2v/adapter.py`.
Run its registered slug with the standard format of:

```bash
uv run flashdreams-run-v2 minimal-t2v-<model> --mode native-window
```

Use the [T2V demo](../../../demos/t2v.md) as feature complete examples.
