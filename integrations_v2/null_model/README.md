<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams NULL Model

A deterministic, CPU-safe FlashDreams integration for tests and examples. Each
step accepts a tensor with shape `[1, 1]` and fills the RGB output with its
single value plus the zero-based autoregressive step index. The model declares
the emitted `(1, 3, 1, 1, 1)` tensor as `VideoTensorLayout.bcthw`.
