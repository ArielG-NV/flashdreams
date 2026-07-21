# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter protocols for manifest-addressable backends, decoders, and demos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QueryableSettingsAdapter(Protocol):
    """Common settings surface exposed by manifest adapter modules.

    Concrete integrations keep checkpoint paths, environment checks, and
    integration-specific validation in their own package. The manifest layer
    only needs this small query/update contract to enumerate legal
    configurations without importing model internals into ``core`` or
    ``infra``.
    """

    def valid_settings(self) -> Sequence[str]:
        """Return the names of settings this adapter accepts."""
        ...

    def valid_values(self, setting: str) -> Sequence[Any]:
        """Return legal values for ``setting``.

        Return an empty sequence when the setting is free-form but supported.
        Raise ``ValueError`` for an unknown setting.
        """
        ...

    def set_setting(self, setting: str, value: Any) -> None:
        """Set one supported setting on this adapter instance."""
        ...


@runtime_checkable
class ModelBackendAdapter(QueryableSettingsAdapter, Protocol):
    """Manifest adapter for a model backend under ``integrations/<model>``."""


@runtime_checkable
class DecoderAdapter(QueryableSettingsAdapter, Protocol):
    """Manifest adapter for a decoder under ``decoders/<decoder>``."""


@runtime_checkable
class DemoAdapter(QueryableSettingsAdapter, Protocol):
    """Manifest adapter for launching one demo against backend variants."""

    def supported_input_styles(self) -> Sequence[str]:
        """Return demo input styles such as ``prompt`` or ``keyboard-inputs``."""
        ...
