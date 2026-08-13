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

"""Local-window output sink for generated video tensors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoWindowOutputSink:
    """Present generated video tensors in a local Tk window."""

    title: str = "FlashDreams"
    """Window title."""

    fps: float = 16.0
    """Playback rate for frames within each generated chunk."""

    _root: Any | None = field(default=None, init=False, repr=False)
    """Lazily-created Tk root."""

    _label: Any | None = field(default=None, init=False, repr=False)
    """Tk label that owns the currently displayed frame."""

    _photo: Any | None = field(default=None, init=False, repr=False)
    """Strong reference to the current Tk photo."""

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be greater than zero.")

    def write(self, output: object) -> None:
        """Display every frame from one generated video tensor."""
        frames = _video_frames(output)
        self._ensure_window()
        frame_delay_s = 1.0 / self.fps
        for frame in frames:
            started_at = time.perf_counter()
            self._display_frame(frame)
            remaining_s = frame_delay_s - (time.perf_counter() - started_at)
            if remaining_s > 0:
                time.sleep(remaining_s)

    def close(self) -> None:
        """Close the local window when it exists."""
        if self._root is not None:
            self._root.destroy()
        self._root = None
        self._label = None
        self._photo = None

    def _ensure_window(self) -> None:
        if self._root is not None:
            return
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError(
                "The default output requires tkinter. Pass a different OutputSink "
                "when running without a local display."
            ) from exc
        try:
            self._root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(
                "The default output could not open a local window. Pass a different "
                "OutputSink when running headlessly."
            ) from exc
        self._root.title(self.title)
        self._label = tk.Label(self._root)
        self._label.pack()

    def _display_frame(self, frame: Any) -> None:
        import tkinter as tk

        assert self._root is not None
        assert self._label is not None
        height, width, _channels = frame.shape
        ppm = f"P6 {width} {height} 255\n".encode() + frame.tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self._label.configure(image=self._photo)
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError as exc:
            raise KeyboardInterrupt("FlashDreams output window was closed.") from exc


def _video_frames(output: object) -> Any:
    """Convert a generated tensor into uint8 ``[T, H, W, C]`` frames."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("VideoWindowOutputSink requires PyTorch.") from exc
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "VideoWindowOutputSink expected a torch.Tensor, got "
            f"{type(output).__name__}."
        )

    frames = output.detach().float().cpu()
    while frames.ndim > 4:
        frames = frames[0]
    if frames.ndim == 3:
        if frames.shape[0] in {1, 3, 4}:
            frames = frames.permute(1, 2, 0).unsqueeze(0)
        elif frames.shape[-1] in {1, 3, 4}:
            frames = frames.unsqueeze(0)
        else:
            raise ValueError(
                f"Cannot infer video layout from shape {tuple(frames.shape)}."
            )
    elif frames.ndim == 4:
        if frames.shape[1] in {1, 3, 4}:
            frames = frames.permute(0, 2, 3, 1)
        elif frames.shape[0] in {1, 3, 4}:
            frames = frames.permute(1, 2, 3, 0)
        elif frames.shape[-1] not in {1, 3, 4}:
            raise ValueError(
                f"Cannot infer video layout from shape {tuple(frames.shape)}."
            )
    else:
        raise ValueError(
            f"Expected a 3D or 4D video tensor, got shape {tuple(frames.shape)}."
        )

    if frames.shape[-1] == 1:
        frames = frames.expand(*frames.shape[:-1], 3)
    elif frames.shape[-1] == 4:
        frames = frames[..., :3]
    if float(frames.min()) < 0:
        frames = (frames + 1.0) / 2.0
    elif float(frames.max()) > 1:
        frames = frames / 255.0
    return frames.clamp(0, 1).mul(255).to(torch.uint8).numpy()


__all__ = ["VideoWindowOutputSink"]
