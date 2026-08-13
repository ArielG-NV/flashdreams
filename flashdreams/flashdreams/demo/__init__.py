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

"""Transport-neutral FlashDreams application hosting and I/O API."""

from flashdreams.demo.application import (
    APPLICATION_ENTRY_POINT_GROUP,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    create_application,
    run_application,
)
from flashdreams.demo.factories import (
    CallableIOFactory,
    LocalWindowIOFactory,
    Mp4IOFactory,
    NullInputSink,
    ProvidedIOFactory,
)
from flashdreams.demo.io import (
    InputSink,
    IOFactory,
    OutputDecision,
    OutputSink,
    SessionInfo,
)
from flashdreams.demo.outputs import (
    BenchmarkStatsOutputSink,
    CompositeOutputSink,
    CompositeOutputSinkError,
    LocalWindowOutputSink,
    Mp4OutputSink,
    NullOutputSink,
    build_benchmark_output_sink,
)

__all__ = [
    "APPLICATION_ENTRY_POINT_GROUP",
    "BenchmarkStatsOutputSink",
    "CallableIOFactory",
    "CompositeOutputSink",
    "CompositeOutputSinkError",
    "IFlashDreamsApplication",
    "IFlashDreamsApplicationSession",
    "IOFactory",
    "InputSink",
    "LocalWindowIOFactory",
    "Mp4IOFactory",
    "Mp4OutputSink",
    "NullInputSink",
    "NullOutputSink",
    "OutputDecision",
    "OutputSink",
    "ProvidedIOFactory",
    "SessionInfo",
    "LocalWindowOutputSink",
    "build_benchmark_output_sink",
    "create_application",
    "run_application",
]
