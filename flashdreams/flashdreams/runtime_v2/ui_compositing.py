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

"""Prepare model frames for UI composition."""

from torch import Tensor
from torch.nn import functional as F


def prepare_ui_back_buffer(
    back_buffer: Tensor | None,
    overlay: Tensor,
) -> Tensor | None:
    """Match a UI back buffer to its rendered overlay.

    Args:
        back_buffer: Optional model frame using the ``StepResult`` value range.
        overlay: Rendered UI frame that supplies the target device, dtype, and
            dimensions.

    Returns:
        The prepared back buffer, or ``None`` when no model frame was supplied.
    """
    if back_buffer is None:
        return None
    if not back_buffer.is_floating_point() and overlay.is_floating_point():
        back_buffer = (
            back_buffer.to(
                device=overlay.device,
                dtype=overlay.dtype,
                non_blocking=True,
            )
            .mul_(2.0 / 255.0)
            .sub_(1.0)
        )
    elif back_buffer.device != overlay.device:
        back_buffer = back_buffer.to(
            device=overlay.device,
            non_blocking=True,
        )
    if back_buffer.shape[1:] != overlay.shape[1:]:
        back_buffer = F.interpolate(
            back_buffer.unsqueeze(0),
            size=overlay.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return back_buffer


__all__ = ["prepare_ui_back_buffer"]
