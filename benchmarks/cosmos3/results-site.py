# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Cosmos3 performance tables and comparison videos."""

import argparse
import html
import re
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer

from benchmark import HERE

VIDEOS = {
    f"results/three-seconds/{model}/{backend}/output.mp4": f"videos/{model}-{backend}.mp4"
    for model in ("bimanual", "auto")
    for backend in ("flashdreams", "vllm")
}


def cell_html(cell: str) -> str:
    match = re.fullmatch(r"\[Watch\]\(([^)]+)\)", cell.strip())
    if match and match[1] in VIDEOS:
        return (
            f'<video controls muted loop playsinline preload="metadata" '
            f'src="{VIDEOS[match[1]]}" aria-label="{VIDEOS[match[1]]}"></video>'
        )
    return html.escape(cell.strip())


def table(markdown: str) -> str:
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    return (
        "<table>"
        + "".join(
            "<tr>"
            + "".join(
                f"<{'th' if index == 0 else 'td'}>{cell_html(cell)}</{'th' if index == 0 else 'td'}>"
                for cell in row.strip("|").split("|")
            )
            + "</tr>"
            for index, row in enumerate(rows)
            if index != 1
        )
        + "</table>"
    )


def build() -> str:
    result = (HERE / "RESULTS.md").read_text(encoding="utf-8")
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    for source, target in VIDEOS.items():
        if f"[Watch]({source})" in result:
            destination = HERE / "website" / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / source, destination)
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Cosmos3 benchmark</title><style>"
        "body{font:16px/1.5 system-ui;max-width:1800px;margin:32px auto;padding:0 20px}"
        "video{width:360px;max-width:80vw}"
        "table{border-collapse:collapse;display:block;overflow-x:auto;margin:24px 0}"
        "td,th{padding:8px 16px;border:1px solid #bbb}tr:first-child{font-weight:bold}"
        "</style><h1>Cosmos3: FlashDreams vs. vLLM</h1>"
        f"<p>{html.escape(result.splitlines()[0])}</p>"
        + table(result)
        + "<p>Videos include the initial frame and 96 generated frames at 30 FPS, seed 0. "
        "Auto/AV: one RGB camera, no LiDAR. Eager denoising; CUDA graphs off. "
        "Quality equivalence is unverified.</p>"
        + "<p>Startup uses the saved benchmark.json startup_s: model/backend initialization "
        "before the first warmup request. Process launch, input preparation and request "
        "warmup are excluded; cache state can affect startup time.</p>"
        + "<h2>Matched settings</h2>"
        + table(readme)
        + "<p>FPS includes input encoding, generation, full VAE decode and CPU pixels. "
        "Loading and preparation are excluded. Measurements on a shared GPU can vary.</p></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    content = build().encode("utf-8")
    output = HERE / "website/index.html"
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(content)
    if args.build_only:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body, content_type = content, "text/html; charset=utf-8"
            elif self.path.removeprefix("/") in VIDEOS.values():
                path = HERE / "website" / self.path.removeprefix("/")
                if not path.is_file():
                    self.send_error(404)
                    return
                body, content_type = path.read_bytes(), "video/mp4"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(
            f"Open http://localhost:{server.server_port} (Ctrl+C to stop)", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
