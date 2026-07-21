# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small adapter implementation shared by manifest-driven demos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ManifestDemoAdapter:
    """Manifest demo adapter."""

    input_styles: tuple[str, ...]
    """Input styles accepted by the demo."""

    settings: tuple[str, ...]
    """Fixture setting names accepted by the demo."""

    _configured_settings: dict[str, Any] = field(default_factory=dict, init=False)
    """Validated settings retained for a subsequent launch."""

    def valid_settings(self) -> Sequence[str]:
        """Return the fixture setting names accepted by this demo."""
        return self.settings

    def valid_values(self, setting: str) -> Sequence[Any]:
        """Return no values because manifest fixture values are free-form."""
        self._validate_setting(setting)
        return ()

    def set_setting(self, setting: str, value: Any) -> None:
        """Store one validated fixture setting for a subsequent demo launch."""
        self._validate_setting(setting)
        self._configured_settings[setting] = value

    def supported_input_styles(self) -> Sequence[str]:
        """Return the input styles the demo can consume."""
        return self.input_styles

    def _validate_setting(self, setting: str) -> None:
        if setting not in self.settings:
            raise ValueError(
                f"Unknown demo setting {setting!r}; known settings: "
                f"{list(self.settings)!r}"
            )
