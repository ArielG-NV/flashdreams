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

"""CPU tests for the deterministic NULL model."""

import pytest
import torch

from flashdreams.core_v2.video_tensor import VideoTensorLayout
from null_model import NULL_MODEL_CONFIG

pytestmark = pytest.mark.ci_cpu


def test_output_equals_scalar_input_plus_step() -> None:
    """Emit a layout-valid tensor filled with the input plus its AR step."""
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()

    assert NULL_MODEL_CONFIG.output_layout is VideoTensorLayout.bcthw

    for step, input_value in enumerate((2.5, -3.0)):
        output = pipeline.generate(
            step,
            cache,
            input=torch.tensor([[input_value]], dtype=torch.float32),
        )

        assert output.shape == (1, 3, 1, 1, 1)
        torch.testing.assert_close(
            output,
            torch.full_like(output, input_value + step),
        )
        pipeline.finalize(step, cache)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1), (2, 1)])
def test_rejects_input_that_is_not_one_by_one(shape: tuple[int, ...]) -> None:
    """Reject input tensors whose shape is not ``[1, 1]``."""
    pipeline = NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()

    with pytest.raises(AssertionError, match=r"expected input tensor shape \(1, 1\)"):
        pipeline.generate(0, cache, input=torch.zeros(shape))
