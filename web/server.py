"""Tiny web UI for the jevomir scoring API (api_server.py). See web/README.md.

Serves index.html and proxies /api/* to the API, adding the API key on the server
side, so the key never reaches the browser (and the API needs no CORS).

    echo 'jev_...' > web/.api-key      # or: export JEVOMIR_API_KEY=jev_...
    python web/server.py --api-url https://....ngrok-free.dev
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BODY = 30 * 1024 * 1024  # 2 images x 10 MB, base64-encoded, plus text
ROUTES = {"/api/info": ("GET", "/v1/info"), "/api/score": ("POST", "/v1/score")}


def make_handler(api_url: str, api_key: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data: bytes, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status, message):
            self._send(status, json.dumps({"detail": message}).encode())

        def _proxy(self, method, path, body=None):
            request = urllib.request.Request(
                api_url + path,
                data=body,
                method=method,
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    self._send(response.status, response.read())
            except urllib.error.HTTPError as error:
                self._send(error.code, error.read())
            except (urllib.error.URLError, TimeoutError) as error:
                self._error(502, f"API unreachable: {getattr(error, 'reason', error)}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html")
            elif ROUTES.get(self.path, ("",))[0] == "GET":
                self._proxy("GET", ROUTES[self.path][1])
            else:
                self._error(404, "not found")

        def do_POST(self):
            if ROUTES.get(self.path, ("",))[0] != "POST":
                return self._error(404, "not found")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self._error(413, "Request body must be 1 byte to 30 MB")
            self._proxy("POST", ROUTES[self.path][1], self.rfile.read(length))

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-url", default=os.environ.get("JEVOMIR_API_URL", "http://127.0.0.1:8100"),
                        help="API base URL (env JEVOMIR_API_URL; default: api_server.py on this machine)")
    parser.add_argument("--key-file", type=Path, default=HERE / ".api-key")
    args = parser.parse_args()
    api_key = os.environ.get("JEVOMIR_API_KEY") or (
        args.key_file.read_text().strip() if args.key_file.is_file() else ""
    )
    if not api_key:
        parser.error(f"set JEVOMIR_API_KEY or put the key in {args.key_file}")
    handler = make_handler(args.api_url.rstrip("/"), api_key)
    print(f"jevomir web UI on http://{args.host}:{args.port} -> {args.api_url}", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
