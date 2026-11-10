"""Message log (v6 feedback): a floating panel at the bottom centre of the 3D view with errors,
finished processes, the setup steps that are not confirmed yet, and why points could not be
reached.

``log(msg, level)`` works from anywhere (any thread); messages logged before the panel exists
(or in tests, without one) are kept and shown once it is built. ui.widgets.notify logs every
notification, so errors shown as toasts also land here.
"""

from __future__ import annotations

import threading
import time
from collections import Counter

import viser

ICONS = {"error": "✘", "warn": "⚠", "done": "✔", "info": "•"}
MAX_KEEP = 300  # entries kept
MAX_SHOW = 60  # entries shown (newest first)
WIDTH, HEIGHT = 620.0, 210.0
MIN_HEIGHT = 64.0  # minimized: the newest message only

# why a point is unreachable (core.feasibility stages, opt.sites checks), in plain words
REASONS = {
    "no_ik": "out of reach: no arm pose reaches it at any allowed tool angle",
    "joint_limits": "only reachable beyond the joint limits (incl. the margin)",
    "collision": "every arm pose collides (part, carrier, stay-out zone or floor)",
    "": "not covered by any stop",
}

_lock = threading.Lock()
_entries: list[tuple[float, str, str]] = []
_panel: LogPanel | None = None


def took(since: float) -> str:
    """Elapsed time since ``time.monotonic()`` value ``since``, e.g. "12 s" or "2 min 5 s"."""
    s = max(0.0, time.monotonic() - since)
    return f"{s:.0f} s" if s < 60 else f"{int(s // 60)} min {s % 60:.0f} s"


def log(msg: str, level: str = "info", since: float | None = None) -> None:
    """Add a message (level: error, warn, done or info) and refresh the panel if there is one.
    ``since`` (a ``time.monotonic()`` start time) appends how long the run took."""
    if since is not None:
        msg = f"{msg} _({took(since)})_"
    with _lock:
        _entries.append((time.time(), level if level in ICONS else "info", str(msg)))
        del _entries[:-MAX_KEEP]
    if _panel is not None:
        _panel.render()


def entries() -> list[tuple[float, str, str]]:
    with _lock:
        return list(_entries)


def clear() -> None:
    with _lock:
        _entries.clear()
    if _panel is not None:
        _panel.render()


def reason_text(code: str) -> str:
    return REASONS.get(code, code.replace("_", " "))


def unreached_report(results: dict, ids, what: str) -> str | None:
    """One message on the points of ``ids`` that are unreachable in ``results`` (point id ->
    PointResult), grouped by reason with up to 12 point ids each; None when all are reached."""
    bad = [i for i in ids if (r := results.get(i)) is None or r.status == "unreachable"]
    if not bad:
        return None
    why = Counter((getattr(results.get(i), "reason", "") or "") for i in bad)
    parts = []
    for code, n in why.most_common():
        pts = [i for i in bad if (getattr(results.get(i), "reason", "") or "") == code]
        more = f" … +{len(pts) - 12}" if len(pts) > 12 else ""
        parts.append(f"{n} × {reason_text(code)} (points {', '.join(map(str, pts[:12]))}{more})")
    return f"{what}: **{len(bad)} point(s) unreachable**: " + "; ".join(parts)


class LogPanel:
    """The floating log window. One per app (build_app)."""

    def __init__(self, server: viser.ViserServer):
        global _panel
        self.server = server
        gui = server.gui
        self.panel = gui.add_panel()
        with self.panel.add_tab("Log"):
            self.setup = gui.add_markdown("")
            self.body = gui.add_markdown("")
            self.clear_btn = gui.add_button("Clear log", hint="remove all messages")
            self.min_btn = gui.add_button("Minimize log", hint="show only the newest message")
        self.clear_btn.on_click(lambda e: clear())
        self.min_btn.on_click(lambda e: self.minimize(not self.minimized))
        self.minimized = False
        self._x = 8.0
        self.panel.float(y=-12, width=WIDTH, height=HEIGHT)
        self._width: dict[int, int] = {}
        server.on_client_connect(self._on_connect)
        _panel = self
        self.render()

    def _on_connect(self, client) -> None:
        client.camera.on_update(lambda cam: self._center(client.client_id, cam.image_width))

    def _center(self, cid: int, width: int) -> None:
        """Keep the window centred at the bottom of the 3D view (re-placed when it resizes)."""
        if width <= 0 or self._width.get(cid) == width:
            return
        self._width[cid] = width
        self._x = max(8.0, (width - WIDTH) / 2)
        self.panel.float(x=self._x, y=-12)

    def minimize(self, on: bool) -> None:
        """Shrink the window to the newest message (on) or show everything again."""
        self.minimized = on
        self.setup.visible = self.clear_btn.visible = not on
        self.min_btn.label = "Show full log" if on else "Minimize log"
        self.panel.float(x=self._x, y=-12, width=WIDTH, height=MIN_HEIGHT if on else HEIGHT)
        self.render()

    def set_setup(self, todo: list[str]) -> None:
        """Setup steps that are not confirmed (ui.workflow), shown above the messages."""
        self.setup.content = (
            f"⚠ **Setup not confirmed:** {', '.join(todo)}" if todo else "✔ All setup steps confirmed."
        )

    def render(self) -> None:
        rows = entries()[::-1][: 1 if self.minimized else MAX_SHOW]
        if not rows:
            self.body.content = "_No messages yet._"
            return
        self.body.content = "  \n".join(
            f"`{time.strftime('%H:%M:%S', time.localtime(t))}` {ICONS[lv]} {msg}" for t, lv, msg in rows
        )


def plan_log(s, best, cancelled: bool = False, since: float | None = None) -> None:
    """Station plan finished: its best solution, and why its unreached points failed."""
    if best is None:
        log("Station planning finished: **no solution found**.", "warn", since)
        return
    head = "cancelled, best so far" if cancelled else "finished, best"
    log(
        f"Station planning {head}: **{best.label}**, cost {best.cost:.2f} ({len(s.results)} plan(s) listed).",
        "warn" if cancelled else "done",
        since,
    )
    msg = unreached_report(s.result_points.get(best.label, {}), best.unreached, best.label)
    if msg:
        log(msg, "warn")
