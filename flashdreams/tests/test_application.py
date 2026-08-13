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

from __future__ import annotations

from collections.abc import Sequence

import pytest

from flashdreams.demo import (
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    InputSink,
    NullInputSink,
    NullOutputSink,
    OutputSink,
)
from flashdreams.demo import application as application_module

pytestmark = pytest.mark.ci_cpu


class _Session(IFlashDreamsApplicationSession):
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.step_index = 0

    def init(self) -> None:
        self.initialized = True

    def step(self, input_src: InputSink, output_sink: OutputSink) -> bool:
        del input_src
        output_sink.write(self.step_index)
        self.step_index += 1
        return self.step_index < 2

    def close(self) -> None:
        self.closed = True


class _Application(IFlashDreamsApplication):
    def __init__(self) -> None:
        self.args: list[str] = []
        self.session = _Session()

    def init(
        self,
        commandline_args: Sequence[str],
        input_src: InputSink,
        output_sink: OutputSink,
    ) -> None:
        del input_src, output_sink
        self.args = list(commandline_args)

    def create_session(
        self,
        input_src: InputSink,
        output_sink: OutputSink,
    ) -> IFlashDreamsApplicationSession:
        del input_src, output_sink
        return self.session


class _EntryPoint:
    name = "t2v-cosmos-predict2"
    value = "test:createApp"

    @staticmethod
    def load() -> object:
        return _Application


def test_concrete_slug_loads_exact_integration_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        application_module,
        "entry_points",
        lambda *, group: [_EntryPoint()],
    )

    application, args = application_module.create_application("t2v-cosmos-predict2")

    assert isinstance(application, _Application)
    assert args == []


def test_host_runs_session_loop_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _Application()
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    output = NullOutputSink(store_outputs=True)

    application_module.run_application(
        "t2v-cosmos-predict2",
        ["--prompt", "A waterfall"],
        input_src=NullInputSink(),
        output_sink=output,
    )

    assert application.args == [
        "--prompt",
        "A waterfall",
    ]
    assert application.session.initialized
    assert application.session.closed
    assert output.outputs == [0, 1]


def test_console_entrypoint_accepts_explicit_run_verb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        application_module,
        "run_application",
        lambda slug, args: captured.append((slug, list(args))),
    )

    application_module.entrypoint(["run", "t2v-self-forcing", "--prompt", "A city"])

    assert captured == [
        ("t2v-self-forcing", ["--prompt", "A city"]),
    ]
