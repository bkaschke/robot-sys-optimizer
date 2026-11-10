"""Start-up popup "Open a project" (v7 round 2, T9).

The app opens on an empty cell (the robot, no part, tool or targets). The first browser that
connects gets this popup: **Reopen last project** (folder and date, hidden when there is
none), **Recent projects** (the last five), **Examples**, **New empty cell** and **Close**.
A successful load, New and Close close it; a failed load shows the error in the popup (and
the log) and keeps it open.

``OpenPopup(gui, project)`` builds the content inside ``gui.add_modal`` of whatever ``gui``
it gets: a client's (the app) or the server's (tests); the handlers are plain methods.
"""

from __future__ import annotations

import time
from pathlib import Path

from ui.project_panel import ProjectPanel, list_examples
from ui.session import recent_projects

NONE = "—"


def _label(d: Path) -> str:
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "project.json").stat().st_mtime))
    except OSError:
        stamp = "?"
    return f"{d.name}  ({stamp})"


class OpenPopup:
    def __init__(self, gui, project: ProjectPanel):
        self.project = project
        self.closed = False
        self.recent = recent_projects()
        self._by_label = {_label(d): d for d in self.recent}
        last = self.recent[0] if self.recent else None
        self.modal = gui.add_modal("Open a project")
        with self.modal:
            self.reopen_btn = gui.add_button(
                f"Reopen last project: {_label(last)}" if last else "Reopen last project",
                color="green",
                visible=last is not None,
            )
            has = bool(self.recent)  # hidden when there is nothing to reopen
            self.recent_pick = gui.add_dropdown(
                "Recent projects", tuple(self._by_label) or (NONE,), visible=has
            )
            self.recent_btn = gui.add_button("Open recent", visible=has)
            self.example_pick = gui.add_dropdown("Examples", tuple(list_examples()) or (NONE,))
            self.example_btn = gui.add_button("Open example")
            self.new_btn = gui.add_button("New empty cell")
            self.close_btn = gui.add_button("Close")
            self.error = gui.add_markdown("")
        self.reopen_btn.on_click(lambda e: self.reopen(e))
        self.recent_btn.on_click(lambda e: self.open_recent(e))
        self.example_btn.on_click(lambda e: self.open_example(e))
        self.new_btn.on_click(lambda e: self.new(e))
        self.close_btn.on_click(lambda e: self.close())

    # --- handlers (True = done, the popup closed) --------------------------------------------

    def reopen(self, e=None) -> bool:
        return self._open(self.recent[0], e) if self.recent else False

    def open_recent(self, e=None) -> bool:
        d = self._by_label.get(self.recent_pick.value)
        return self._open(d, e) if d is not None else False

    def open_example(self, e=None) -> bool:
        name = self.example_pick.value
        ok = name in list_examples() and self.project.load_example(name, e)
        return self._done(ok, f"could not open the example {name}")

    def new(self, e=None) -> bool:
        self.project.new_cell(e)
        return self._done(True, "")

    def close(self) -> bool:
        if not self.closed:
            self.closed = True
            self.modal.close()
        return True

    def _open(self, d: Path, e) -> bool:
        return self._done(self.project._load(e, d), f"could not open `{d}`")

    def _done(self, ok: bool, msg: str) -> bool:
        if ok:
            return self.close()
        from ui.log_panel import log

        self.error.content = f"⚠ **Load failed:** {msg}. See the log for details."
        log(f"Open a project: {msg}.", "error")
        return False
