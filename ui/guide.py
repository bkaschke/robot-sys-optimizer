"""User guide in the browser: a tiny HTTP server for docs/USER_GUIDE.html and the panel link.

Viser serves only its own client files, so the guide gets its own port (the app's port + 1
by default). The server answers GET for "/" and "/USER_GUIDE.html" with that one file (read
on every request, so a rebuilt guide shows without a restart) and 404 for anything else;
no directory listing, no other files.

The panel link is HTML with an onclick that swaps in the hostname the browser used to reach
the app, so the guide also opens from another PC on the network. The plain href
(localhost) is the fallback. "Download user guide" sends the file itself and works always.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import viser

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.html"
PATHS = ("/", "/USER_GUIDE.html", "/index.html")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path not in PATHS or not GUIDE.is_file():
            self.send_error(404, "Not found")
            return
        data = GUIDE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


class _ExclusiveServer(ThreadingHTTPServer):
    """No SO_REUSEADDR: on Windows it lets two servers bind the same port, so a busy port
    would not be detected. SO_EXCLUSIVEADDRUSE (Windows) makes the bind fail instead."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class GuideServer:
    """Serves the guide on the first free port from ``port`` (up to 20 tries)."""

    def __init__(self, host: str, port: int):
        last: OSError | None = None
        for p in range(port, port + 20):
            try:
                self.httpd = _ExclusiveServer((host, p), _Handler)
                break
            except OSError as err:
                last = err
        else:
            raise OSError(f"no free port for the user guide from {port}") from last
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def link_html(port: int, query: str = "") -> str:
    """Link to the guide; with ``query`` it opens the guide searching for it (v8 T4)."""
    from html import escape
    from urllib.parse import quote

    q = f"?q={quote(query)}" if query else ""
    fallback = f"http://localhost:{port}/{q}"
    js = f"this.href=location.protocol+'//'+location.hostname+':{port}/{q}'"
    label = f"🔎 Search the guide for “{escape(query)}”" if query else "📖 Open user guide"
    return (
        f'<div style="padding:2px 0 6px"><a href="{fallback}" target="_blank" rel="noopener" '
        f'onclick="{js}" style="font-weight:600">{label}</a> '
        f'<span style="opacity:.6;font-size:12px">(new tab, port {port})</span></div>'
    )


def add_guide_panel(server: viser.ViserServer, port: int | None) -> dict:
    """'User guide' folder at the top of the panel: link (when the guide server runs) and a
    download button (always)."""
    out: dict = {}
    with server.gui.add_folder("User guide", expand_by_default=False):
        if port is not None:
            out["link"] = server.gui.add_html(link_html(port))
            out["search"] = box = server.gui.add_text(
                "Search the guide",
                "",
                hint="type a word, then open the link below (Enter in the guide = next)",
            )
            out["search_link"] = link = server.gui.add_html("")

            @box.on_update
            def _(_e) -> None:
                q = box.value.strip()
                link.content = link_html(port, q) if q else ""

        out["download"] = btn = server.gui.add_button("Download user guide (HTML)")

    @btn.on_click
    def _(e) -> None:
        if e.client is not None and GUIDE.is_file():
            e.client.send_file_download("USER_GUIDE.html", GUIDE.read_bytes())

    return out
