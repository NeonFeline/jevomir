"""Serve the jevomir-plays-Tetris page and proxy scoring calls to the API.

The page (index.html) runs the unmodified game from game/ in an iframe and asks the API
which placement to play. The key is added here, so it never reaches the browser.

    echo 'jev_...' > tetris/.api-key      # or: export JEVOMIR_API_KEY=jev_...
    python tetris/server.py --api-url https://....ngrok-free.dev
"""

from __future__ import annotations

import argparse
import http.client
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = {"/": "index.html", "/index.html": "index.html",
          "/game/index.html": "game/index.html", "/game/stats.js": "game/stats.js",
          "/game/texture.jpg": "game/texture.jpg"}
ROUTES = {"/api/info": ("GET", "/v1/info"), "/api/score": ("POST", "/v1/score")}
MAX_BODY = 30 * 1024 * 1024


class RateLimiter:
    """Blocks until a call fits in `per_minute` calls per sliding 60 s window.

    A free ngrok tunnel drops connections after about 100 requests a minute, so the
    game waits here instead of failing.
    """

    def __init__(self, per_minute: int):
        self.per_minute, self.calls, self.lock = per_minute, deque(), threading.Lock()

    def wait(self):
        if self.per_minute <= 0:
            return
        with self.lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                time.sleep(60 - (now - self.calls[0]))


def make_handler(api_url: str, api_key: str, limiter: RateLimiter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data: bytes, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _proxy(self, method, path, body=None):
            limiter.wait()
            request = urllib.request.Request(api_url + path, data=body, method=method, headers={
                "X-API-Key": api_key, "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "1"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self._send(response.status, response.read())
            except urllib.error.HTTPError as error:
                self._send(error.code, error.read())
            except (OSError, http.client.HTTPException) as error:  # URLError, resets, timeouts
                self._send(502, f'{{"detail": "API unreachable: {getattr(error, "reason", error)}"}}'.encode())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in STATIC:
                file = HERE / STATIC[path]
                self._send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            elif ROUTES.get(path, ("",))[0] == "GET":
                self._proxy("GET", ROUTES[path][1])
            else:
                self._send(404, b'{"detail": "not found"}')

        def do_POST(self):
            if ROUTES.get(self.path, ("",))[0] != "POST":
                return self._send(404, b'{"detail": "not found"}')
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self._send(413, b'{"detail": "Request body must be 1 byte to 30 MB"}')
            self._proxy("POST", ROUTES[self.path][1], self.rfile.read(length))

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--api-url", default=os.environ.get("JEVOMIR_API_URL", "http://127.0.0.1:8100"),
                        help="API base URL (env JEVOMIR_API_URL; default: api_server.py on this machine)")
    parser.add_argument("--max-per-minute", type=int, default=90,
                        help="API calls per minute, under ngrok's free limit of ~100; 0 = no limit")
    parser.add_argument("--key-file", type=Path, default=HERE / ".api-key")
    args = parser.parse_args()
    api_key = os.environ.get("JEVOMIR_API_KEY") or (
        args.key_file.read_text().strip() if args.key_file.is_file() else "")
    if not api_key:
        parser.error(f"set JEVOMIR_API_KEY or put the key in {args.key_file}")
    handler = make_handler(args.api_url.rstrip("/"), api_key, RateLimiter(args.max_per_minute))
    print(f"jevomir plays Tetris on http://{args.host}:{args.port} -> {args.api_url}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
