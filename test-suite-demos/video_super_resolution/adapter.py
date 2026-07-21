# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the video-super-resolution manifest demo."""

from test_suite_demos._adapter import ManifestDemoAdapter

_adapter = ManifestDemoAdapter(
    input_styles=("video",),
    settings=("input-video", "scale", "seed"),
)
valid_settings = _adapter.valid_settings
valid_values = _adapter.valid_values
set_setting = _adapter.set_setting
supported_input_styles = _adapter.supported_input_styles
