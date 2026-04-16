"""Binary wire format: CTRL (client → server), FRME (server → client).

All multi-byte integers are **big-endian** (network order).

CTRL layout::

    u32 magic | u32 seq | u32 payload_len | payload_len bytes UTF-8 JSON object

The JSON object may include arbitrary keys plus an optional ``"camera"`` object
(see :class:`CameraCtrl`) for view-dependent rendering on the server.

FRME layout::

    u32 magic | u8 version | u8 n_frames | u16 width | u16 height | u32 batch_id
    then n_frames times: u32 len_i | len_i bytes (e.g. WebP)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

CTRL_MAGIC = 0x4354524C
FRME_MAGIC = 0x46524D45
PROTO_VERSION = 1

# CTRL: magic, client monotonic seq (opaque to server), JSON byte length
_CTRL_HEADER_STRUCT = struct.Struct(">III")
# FRME: magic, protocol version, frame count, dimensions, server batch counter
_FRME_HEADER_STRUCT = struct.Struct(">IBBHHI")


@dataclass(frozen=True)
class CameraCtrl:
    """Pinhole camera intrinsics + extrinsics for server-side rendering.

    ``c2w`` is camera-to-world (OpenGL-style: ``+Y`` up, ``-Z`` forward), matching
    typical Viser client camera matrices.
    """

    fov: float
    aspect: float
    c2w: np.ndarray

    def __post_init__(self) -> None:
        c2w = np.asarray(self.c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must have shape (4, 4), got {c2w.shape}")
        object.__setattr__(self, "c2w", c2w)

    def get_K(self, img_wh: tuple[int, int]) -> np.ndarray:
        """Intrinsic matrix for resolution ``(W, H)`` (``x`` right, ``y`` down)."""
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(float(self.fov) / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return K

    def to_wire(self) -> dict[str, Any]:
        return {
            "fov": float(self.fov),
            "aspect": float(self.aspect),
            "c2w": self.c2w.astype(np.float64).tolist(),
        }

    @classmethod
    def from_wire(cls, obj: object) -> CameraCtrl | None:
        """Parse the ``"camera"`` JSON value; ``None`` or JSON ``null`` → ``None``."""
        if obj is None:
            return None
        if not isinstance(obj, dict):
            raise ValueError('CTRL "camera" must be a JSON object or null')
        try:
            fov = float(obj["fov"])
            aspect = float(obj["aspect"])
            c2w_raw = obj["c2w"]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('CTRL "camera" requires fov, aspect, c2w') from e
        c2w = np.asarray(c2w_raw, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError('CTRL "camera" c2w must be a 4×4 array')
        if (
            not np.isfinite(c2w).all()
            or not np.isfinite(fov)
            or not np.isfinite(aspect)
        ):
            raise ValueError('CTRL "camera" contains non-finite values')
        if fov <= 0.0 or aspect <= 0.0:
            raise ValueError('CTRL "camera" fov and aspect must be positive')
        return cls(fov=fov, aspect=aspect, c2w=c2w)


@dataclass(frozen=True)
class CtrlMessage:
    """Decoded client control."""

    seq: int
    control: dict
    camera: CameraCtrl | None = None


@dataclass(frozen=True)
class FrmeMessage:
    """Decoded frame batch."""

    version: int
    n_frames: int
    width: int
    height: int
    batch_id: int
    frames: tuple[bytes, ...]


def pack_ctrl(seq: int, control: dict, camera: CameraCtrl | None = None) -> bytes:
    """Pack one WebSocket **binary** control message (not text WS frames)."""
    payload_obj: dict[str, Any] = dict(control)
    if camera is not None:
        payload_obj["camera"] = camera.to_wire()
    payload = json.dumps(payload_obj, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return _CTRL_HEADER_STRUCT.pack(CTRL_MAGIC, seq, len(payload)) + payload


def unpack_ctrl(data: bytes) -> CtrlMessage:
    """Unpack a control message; raises ValueError on bad magic or JSON."""
    if len(data) < _CTRL_HEADER_STRUCT.size:
        raise ValueError("CTRL message too short")
    magic, seq, payload_len = _CTRL_HEADER_STRUCT.unpack_from(data, 0)
    if magic != CTRL_MAGIC:
        raise ValueError(f"bad CTRL magic {magic:#x}")
    end = _CTRL_HEADER_STRUCT.size + payload_len
    if end > len(data):
        raise ValueError("CTRL payload length out of range")
    if end < len(data):
        # One CTRL message must exactly fill one WS binary frame for this minimal stack.
        raise ValueError("CTRL trailing bytes")
    raw = data[_CTRL_HEADER_STRUCT.size : end]
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError("CTRL invalid JSON") from e
    if not isinstance(obj, dict):
        raise ValueError("CTRL JSON must be an object")
    cam_raw = obj.pop("camera", None)
    camera = CameraCtrl.from_wire(cam_raw)
    return CtrlMessage(seq=seq, control=obj, camera=camera)


def pack_frme(
    *,
    n_frames: int,
    width: int,
    height: int,
    batch_id: int,
    frames: list[bytes],
) -> bytes:
    """Pack one FRME WebSocket binary message."""
    if len(frames) != n_frames:
        raise ValueError("n_frames must match len(frames)")
    if not (0 <= n_frames <= 255):
        raise ValueError("n_frames must fit in u8")
    parts: list[bytes] = [
        _FRME_HEADER_STRUCT.pack(
            FRME_MAGIC,
            PROTO_VERSION,
            n_frames,
            width & 0xFFFF,
            height & 0xFFFF,
            batch_id & 0xFFFFFFFF,
        )
    ]
    for blob in frames:
        parts.append(struct.pack(">I", len(blob) & 0xFFFFFFFF))
        parts.append(blob)
    return b"".join(parts)


def unpack_frme(data: bytes) -> FrmeMessage:
    """Unpack one FRME message."""
    pos = 0
    if len(data) < _FRME_HEADER_STRUCT.size:
        raise ValueError("FRME message too short")
    magic, version, n_frames, width, height, batch_id = _FRME_HEADER_STRUCT.unpack_from(
        data, pos
    )
    pos += _FRME_HEADER_STRUCT.size
    if magic != FRME_MAGIC:
        raise ValueError(f"bad FRME magic {magic:#x}")
    if version != PROTO_VERSION:
        raise ValueError(f"unsupported FRME version {version}")
    frames: list[bytes] = []
    for _ in range(n_frames):
        if pos + 4 > len(data):
            raise ValueError("FRME truncated frame length")
        (ln,) = struct.unpack_from(">I", data, pos)
        pos += 4
        if pos + ln > len(data):
            raise ValueError("FRME truncated frame bytes")
        frames.append(bytes(data[pos : pos + ln]))
        pos += ln
    if pos != len(data):
        raise ValueError("FRME trailing bytes")
    return FrmeMessage(
        version=version,
        n_frames=n_frames,
        width=width,
        height=height,
        batch_id=batch_id,
        frames=tuple(frames),
    )
