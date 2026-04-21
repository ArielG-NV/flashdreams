"""Roundtrip tests for streaming_ws.protocol."""

import pytest

import numpy as np

from streaming_ws.protocol import (
    CTRL_MAGIC,
    FRME_MAGIC,
    CameraCtrl,
    pack_ctrl,
    pack_frme,
    unpack_ctrl,
    unpack_frme,
)


def test_pack_unpack_ctrl_roundtrip() -> None:
    msg = pack_ctrl(7, {"dx": 1, "dy": -2, "keys": ["a"]})
    assert int.from_bytes(msg[:4], "big") == CTRL_MAGIC
    c = unpack_ctrl(msg)
    assert c.seq == 7
    assert c.control == {"dx": 1, "dy": -2, "keys": ["a"]}
    assert c.camera is None


def test_pack_unpack_ctrl_camera_roundtrip() -> None:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [0.1, -0.2, 1.5]
    cam = CameraCtrl(fov=0.9, aspect=16 / 9, c2w=c2w)
    msg = pack_ctrl(3, {"phase": "tick"}, camera=cam)
    c = unpack_ctrl(msg)
    assert c.seq == 3
    assert c.control == {"phase": "tick"}
    assert c.camera is not None
    assert c.camera.fov == 0.9
    assert c.camera.aspect == pytest.approx(16 / 9)
    np.testing.assert_allclose(c.camera.c2w, c2w)


def test_camera_ctrl_get_K() -> None:
    cam = CameraCtrl(fov=np.pi / 4, aspect=2.0, c2w=np.eye(4))
    K = cam.get_K((640, 480))
    assert K.shape == (3, 3)
    assert K[0, 2] == 320 and K[1, 2] == 240


def test_pack_unpack_frme_roundtrip() -> None:
    frames = [b"a", b"bb", b"ccc"]
    blob = pack_frme(n_frames=3, width=640, height=360, batch_id=99, frames=frames)
    assert int.from_bytes(blob[:4], "big") == FRME_MAGIC
    f = unpack_frme(blob)
    assert f.n_frames == 3
    assert f.width == 640
    assert f.height == 360
    assert f.batch_id == 99
    assert list(f.frames) == frames


def test_unpack_ctrl_bad_magic() -> None:
    bad = (0xDEADBEEF).to_bytes(4, "big") + b"\x00" * 8
    with pytest.raises(ValueError, match="magic"):
        unpack_ctrl(bad)


def test_unpack_frme_trailing_bytes() -> None:
    blob = pack_frme(n_frames=1, width=1, height=1, batch_id=0, frames=[b"x"])
    with pytest.raises(ValueError, match="trailing"):
        unpack_frme(blob + b"extra")
