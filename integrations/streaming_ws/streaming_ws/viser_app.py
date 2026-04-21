"""Viser-only mock stream: **camera updates** trigger new batches (camera → :class:`CameraCtrl`).

Playout pushes RGB into each client's **scene background** via
:meth:`viser.SceneApi.set_background_image` (not a GUI image panel), optionally with a
constant-depth plane for compositing tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections import deque
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from streaming_ws.protocol import CameraCtrl, CtrlMessage
from streaming_ws.stub_frames import encode_stub_batch


def _wxyz_to_R(wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation (same convention as Viser SO3)."""
    w, x, y, z = (float(v) for v in np.asarray(wxyz, dtype=np.float64).reshape(4))
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def camera_ctrl_from_viser_camera(camera) -> CameraCtrl:
    """Build :class:`CameraCtrl` from a Viser ``CameraHandle`` (``wxyz``, ``position``, …)."""
    R = _wxyz_to_R(camera.wxyz)
    pos = np.asarray(camera.position, dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    return CameraCtrl(
        fov=float(camera.fov),
        aspect=float(camera.aspect),
        c2w=c2w,
    )


def _webp_to_rgb(data: bytes) -> np.ndarray:
    im = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def _flat_depth_m(h: int, w: int, depth_m: float) -> np.ndarray:
    """``(H, W)`` float depth in meters, as expected by Viser for background compositing."""
    return np.full((h, w), float(depth_m), dtype=np.float32)


def _discover_ipv4_for_remote_urls() -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(ip: str) -> None:
        ip = ip.strip()
        if not ip or ip in seen or ip.startswith("127."):
            return
        seen.add(ip)
        out.append(ip)

    for probe in ("8.8.8.8", "192.0.2.1", "198.51.100.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect((probe, 80))
                add(s.getsockname()[0])
            finally:
                s.close()
            break
        except OSError:
            continue

    for name_fn in (socket.getfqdn, socket.gethostname):
        try:
            _hn, _alias, ips = socket.gethostbyname_ex(name_fn())
            for ip in ips:
                add(ip)
        except OSError:
            continue

    return out


@dataclass(frozen=True)
class ViserOnlyConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    frames_per_batch: int = 8
    frame_width: int = 1280
    frame_height: int = 720
    prefill_batches: int = 2
    stub_latency_ms: float = 800.0
    display_fps: float = 15.0
    buffer_max_frames: int = 120
    verbose: bool = True
    background_jpeg_quality: int = 70
    # If set, pass ``depth=`` a constant plane (meters) like a real renderer would.
    flat_background_depth_m: float | None = None


def run_viser(cfg: ViserOnlyConfig) -> None:
    try:
        import viser
    except ImportError as e:
        raise SystemExit(
            "viser is required. Install: pip install -e '.[streaming_viser]'"
        ) from e

    buffer: deque[np.ndarray] = deque(maxlen=cfg.buffer_max_frames)
    trigger_q: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
    produce_lock = asyncio.Lock()
    latest_camera: dict[str, object | None] = {"cam": None}
    state: dict[str, int] = {"batch_id": 0, "base_frame": 0, "seq": 0}
    prefill_done = False
    background_started = False

    server = viser.ViserServer(host=cfg.host, port=cfg.port, verbose=cfg.verbose)

    async def produce_one(*, ctrl: CtrlMessage | None, apply_latency: bool) -> None:
        async with produce_lock:
            if apply_latency and cfg.stub_latency_ms > 0:
                await asyncio.sleep(cfg.stub_latency_ms / 1000.0)
            bi = state["batch_id"]
            bf = state["base_frame"]
            frames = encode_stub_batch(
                ctrl=ctrl,
                batch_id=bi,
                width=cfg.frame_width,
                height=cfg.frame_height,
                n_frames=cfg.frames_per_batch,
                base_frame=bf,
            )
            state["batch_id"] = bi + 1
            state["base_frame"] = bf + cfg.frames_per_batch
            for webp in frames:
                buffer.append(_webp_to_rgb(webp))

    async def producer_loop() -> None:
        while True:
            await trigger_q.get()
            state["seq"] += 1
            cam = latest_camera["cam"]
            cc = camera_ctrl_from_viser_camera(cam) if cam is not None else None
            await produce_one(
                ctrl=CtrlMessage(seq=state["seq"], control={}, camera=cc),
                apply_latency=True,
            )

    async def playout_loop() -> None:
        interval = 1.0 / max(cfg.display_fps, 0.1)
        while True:
            await asyncio.sleep(interval)
            if not buffer:
                continue
            img = buffer.popleft()
            h, w = img.shape[0], img.shape[1]
            depth = (
                None
                if cfg.flat_background_depth_m is None
                else _flat_depth_m(h, w, cfg.flat_background_depth_m)
            )
            for client in server.get_clients().values():
                client.scene.set_background_image(
                    img,
                    format="jpeg",
                    jpeg_quality=cfg.background_jpeg_quality,
                    depth=depth,
                )

    async def on_camera_update(camera) -> None:
        latest_camera["cam"] = camera
        try:
            trigger_q.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                trigger_q.get_nowait()
            trigger_q.put_nowait(None)

    @server.on_client_connect
    async def _on_client(client) -> None:
        nonlocal prefill_done, background_started
        if not background_started:
            background_started = True
            asyncio.create_task(producer_loop())
            asyncio.create_task(playout_loop())
        if not prefill_done:
            prefill_done = True
            for _ in range(cfg.prefill_batches):
                await produce_one(ctrl=None, apply_latency=False)
        client.camera.on_update(on_camera_update)

    print(
        f"[viser] mock stream — host={cfg.host!r} port={cfg.port} "
        f"stub_latency_ms={cfg.stub_latency_ms} display_fps={cfg.display_fps}",
        flush=True,
    )
    if cfg.host in ("0.0.0.0", "", "::"):
        guessed = _discover_ipv4_for_remote_urls()
        if guessed:
            for ip in guessed:
                print(f"[viser]   http://{ip}:{cfg.port}", flush=True)
        else:
            print(
                "[viser]   (no non-loopback IPv4 found; try hostname -I or SSH port-forward)",
                flush=True,
            )
        print(f"[viser]   http://127.0.0.1:{cfg.port}", flush=True)
    else:
        print(f"[viser]   http://{cfg.host}:{cfg.port}", flush=True)

    server.sleep_forever()


def main_viser(cfg: ViserOnlyConfig) -> None:
    run_viser(cfg)
