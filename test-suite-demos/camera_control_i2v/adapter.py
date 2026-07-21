# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the camera-control image-to-video manifest demo."""

from test_suite_demos._adapter import ManifestDemoAdapter

_adapter = ManifestDemoAdapter(
    input_styles=("prompt", "first-frame", "camera-controls", "action-controls"),
    settings=(
        "first-frame", "prompt", "camera_path", "actions", "seed",
        "chunk_size", "num_steps",
    ),
)
valid_settings = _adapter.valid_settings
valid_values = _adapter.valid_values
set_setting = _adapter.set_setting
supported_input_styles = _adapter.supported_input_styles
