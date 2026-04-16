"""CPU stub: encode WebP frames (no torch).

Used by the demo server to stand in for a real decoder/video sink. ``ctrl`` is
``None`` for **prefill** batches (no client input yet); otherwise use ``ctrl.seq``
(and optional ``ctrl.camera``) so batches differ on the wire.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from projects.streaming_ws.protocol import CtrlMessage


def encode_stub_batch(
    *,
    ctrl: CtrlMessage | None,
    batch_id: int,
    width: int,
    height: int,
    n_frames: int,
    base_frame: int,
) -> list[bytes]:
    """Return ``n_frames`` WebP blobs for one FRME message."""
    seq = 0 if ctrl is None else int(ctrl.seq)  # prefill vs CTRL-driven batches
    cam_hint = ""
    if ctrl is not None and ctrl.camera is not None:
        t = ctrl.camera.c2w[:3, 3]
        cam_hint = f" tx{t[0]:.1f} ty{t[1]:.1f} tz{t[2]:.1f}"
    out: list[bytes] = []
    for i in range(n_frames):
        idx = base_frame + i
        im = Image.new(
            "RGB",
            (width, height),
            color=((idx * 17 + batch_id * 3) % 220, 60, (seq * 11) % 220),
        )
        draw = ImageDraw.Draw(im)
        margin = max(8, min(width, height) // 128)
        draw.text(
            (margin, margin),
            f"b{batch_id} f{idx} s{seq}{cam_hint}",
            fill=(255, 255, 255),
        )
        buf = BytesIO()
        # Slightly lower quality on large frames to keep encode time predictable.
        q = 68 if width * height > 640 * 360 else 72
        im.save(buf, format="WEBP", quality=q, method=0)
        out.append(buf.getvalue())
    return out
