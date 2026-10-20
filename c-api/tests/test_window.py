# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes as ct
import subprocess
import sys
import time
from ctypes import wintypes as wt
from pathlib import Path

import pytest


@pytest.mark.manual
def test_window():
    if sys.platform != "win32":
        pytest.skip("This desktop test uses Win32 window messages.")

    from PIL import ImageGrab

    api_dir = Path(__file__).resolve().parents[1]
    user32 = ct.WinDLL("user32", use_last_error=True)
    user32.SetProcessDPIAware()
    user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
    user32.FindWindowW.restype = wt.HWND
    user32.GetClientRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
    user32.ClientToScreen.argtypes = [wt.HWND, ct.POINTER(wt.POINT)]
    user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.SetWindowPos.argtypes = [
        wt.HWND,
        wt.HWND,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        wt.UINT,
    ]
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ct.POINTER(wt.DWORD)]

    def wait_for(check):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            assert process.poll() is None, "Launcher exited early; see window-check.log"
            result = check()
            if result:
                return result
            time.sleep(0.1)
        raise AssertionError("Timed out waiting for native window/pixels")

    def pixels_are(color):
        rect, point = wt.RECT(), wt.POINT()
        assert user32.GetClientRect(window, ct.byref(rect))
        assert user32.ClientToScreen(window, ct.byref(point))
        image = ImageGrab.grab(
            (point.x, point.y, point.x + rect.right, point.y + rect.bottom)
        )
        interior = image.crop((10, 10, image.width - 10, image.height - 10)).convert(
            "RGB"
        )
        if interior.getextrema() != tuple((value, value) for value in color):
            return False
        image.save(api_dir / "build" / f"window-{color[0]}.png")
        return True

    with (api_dir / "build" / "window-check.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(api_dir / "build" / "Release" / "flashdreams_example.exe"),
                sys.executable,
                "red_screen.app",
                "--mode",
                "c",
                "--layout",
                "bcthw",
                "--pixel-width",
                "640",
                "--pixel-height",
                "480",
                "--timeout",
                "90",
            ],
            cwd=api_dir / "build",
            stdout=log,
            stderr=log,
        )
        try:
            window = wait_for(lambda: user32.FindWindowW(None, "FlashDreams C window"))
            owner = wt.DWORD()
            user32.GetWindowThreadProcessId(window, ct.byref(owner))
            assert owner.value == process.pid
            wait_for(lambda: pixels_are((0, 0, 0)))
            assert user32.PostMessageW(window, 0x100, ord("R"), 0x00130001)
            wait_for(lambda: pixels_are((255, 0, 0)))
            assert user32.SetWindowPos(window, None, 0, 0, 800, 600, 0x0006)
            wait_for(lambda: pixels_are((255, 0, 0)))
            assert user32.PostMessageW(window, 0x101, ord("R"), 0xC0130001)
            wait_for(lambda: pixels_are((0, 0, 0)))
            assert user32.PostMessageW(window, 0x10, 0, 0)
            assert process.wait(timeout=20) == 0
            print(
                "PASS: native window, black/red pixels, resize, key release, close, and clean exit"
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    test_window()
