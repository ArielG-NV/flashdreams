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

"""Canonical application input handling protocol."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flashdreams.core_v2.user_input_events import UserInputEvents


@runtime_checkable
class InputHandler(Protocol):
    """Provide time-windowed canonical input state to the host."""

    @abstractmethod
    def open(self) -> None:
        """Enable reading."""
        ...

    @abstractmethod
    def current_inputs(self) -> UserInputEvents:
        """Return the latest user input events."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Disable further reading."""
        ...
