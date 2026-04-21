"""CLI: ``python -m streaming_ws {server|client|viser}``.

Tyro expects flags first; we consume the subcommand token so tyro sees only
``--port``-style arguments for that command.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("server", "client", "viser"):
        print(
            "usage: python -m streaming_ws {server|client|viser} [options...]",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = sys.argv[1]
    del sys.argv[1]  # shift argv for tyro

    import tyro

    if cmd == "server":
        from streaming_ws.server import ServerConfig, main_server

        main_server(tyro.cli(ServerConfig))
    elif cmd == "client":
        from streaming_ws.client import ClientConfig, main_client

        main_client(tyro.cli(ClientConfig))
    else:
        from streaming_ws.viser_app import ViserOnlyConfig, main_viser

        main_viser(tyro.cli(ViserOnlyConfig))


if __name__ == "__main__":
    main()
