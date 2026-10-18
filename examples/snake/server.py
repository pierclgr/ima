"""Serves the screenshot Snake demo page on http://127.0.0.1:8081.

Usage:
    python examples/snake/server.py

This module uses only the Python standard library. It serves the static
page only; the browser calls the separate llama-server API directly.
"""

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ADDRESS: tuple[str, int] = ("127.0.0.1", 8081)


def main() -> None:
    """Serves this directory until interrupted."""
    handler = functools.partial(
        SimpleHTTPRequestHandler, directory=str(Path(__file__).parent)
    )
    with ThreadingHTTPServer(ADDRESS, handler) as server:
        print(f"Snake demo: http://{ADDRESS[0]}:{ADDRESS[1]}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
