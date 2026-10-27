"""Serve a local page on 127.0.0.1: the first free port from a given one, then the browser.

The dashboard and the counting page of long recordings start, answer and refuse
requests the same way.
"""

import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORTS_TRIED = 20
# Fixed types of the page files. mimetypes takes them from the Windows registry, where
# .js is text/plain on some machines; with nosniff a browser refuses to run such a script.
STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class LocalServer(ThreadingHTTPServer):
    """One page per port.

    On Windows the default SO_REUSEADDR lets a second process bind a port that is
    already served. The old page then keeps answering, and a restarted one seems to
    have no effect. An exclusive bind makes the second start fail instead.
    """

    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class LocalHandler(BaseHTTPRequestHandler):
    """Answers, page files and refusals shared by the local pages.

    A subclass sets `static` (the folder of the page files), `index` (the file
    served at /) and, if needed, `body_limit`.
    """

    static: Path
    index: str
    body_limit = 64 * 1024

    def send_bytes(self, payload: bytes, content_type: str, status=200, download=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value, status=200, download=None):
        self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8",
                        status, download)

    def send_page_file(self, path: str) -> bool:
        """Send a file of `static` named by the URL path; False if there is none.

        Only names listed in the folder are served, so no path can lead outside it.
        """
        name = self.index if path == "/" else path[1:]
        files = {file.name for file in self.static.iterdir() if file.suffix in STATIC_TYPES}
        if name not in files:
            return False
        file = self.static / name
        self.send_bytes(file.read_bytes(), STATIC_TYPES[file.suffix] + "; charset=utf-8")
        return True

    def refuse_foreign_request(self) -> bool:
        """Answer 403 to a request that does not come from this page.

        The server listens on 127.0.0.1 only, but a web page can still reach it
        through DNS rebinding: its own host name, resolved to 127.0.0.1. Such a
        request carries that foreign name in Host. A page of another site may also
        post to the real local address; its request carries that site in Origin.
        The pages read local folders and start work, so every request is checked.
        """
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.send_json({"error": "Localhost Host header required"}, 403)
            return True
        origin = self.headers.get("Origin")
        if self.command == "POST" and origin and origin != f"http://{self.headers.get('Host')}":
            self.send_json({"error": "Same-origin request required"}, 403)
            return True
        return False

    def json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= self.body_limit:
            raise ValueError(f"Expected a JSON body of at most {self.body_limit // 1024} KiB")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        return body

    def log_message(self, format, *args):
        # The pages poll their status all the time; errors reach the page instead.
        pass


def first_free_server(make_server, port: int) -> LocalServer:
    """`make_server(port)` on the first port from `port` that no other program holds."""
    for candidate in range(port, port + PORTS_TRIED):
        try:
            return make_server(candidate)
        except OSError:
            continue
    raise OSError(f"No free port from {port} to {port + PORTS_TRIED - 1}")


def serve_page(server: LocalServer, title: str, open_browser: bool = True) -> None:
    """Print the address, open it in the browser and serve until Ctrl+C."""
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"{title}: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
