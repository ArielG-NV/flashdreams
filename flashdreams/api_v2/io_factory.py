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

"""Application I/O factory protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.core_v2.session_info import SessionInfo

from .input_handler import InputHandler
from .output_sink import OutputSink


@runtime_checkable
class IOFactory(Protocol):
    """Create isolated application input handling and output delivery."""

    def __init__(self, session_info: SessionInfo) -> None:
        """Initialize the factory according to session information."""
        ...

    @abstractmethod
    def create_input_handler(self) -> InputHandler:
        """Create a handler for the application-declared canonical inputs."""
        ...

    @abstractmethod
    def create_output_sink(self) -> OutputSink:
        """Create the output sink for one application run."""
        ...
