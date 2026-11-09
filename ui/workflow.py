"""Guided setup (v2 phase 2): step headers, the robot + carrier lock, out-of-date banners.

The tabs are steps 1 Cell → 2 External Axis → 3 Tool → 4 Targets → 5 Optimize → 6 Results.
Each setup step has a status line and **Confirm**; any tab can be opened at any time (user
decision: guided, free navigation). Confirming the External Axis step locks the carrier
setup (no editing, the robot rides on it); **Edit setup** unlocks it. In movable mode the
Cell tab never lets the robot base be moved off the carrier.

``refresh()`` recomputes everything from the session (ui.staleness fingerprints) and only
touches widgets when something changed; the app calls it from a 1 s watcher thread and
after every confirm / load.
"""

from __future__ import annotations

import threading
import time

import viser

from ui.session import Session
from ui.staleness import STEP_COMPONENTS, confirm_step, feasibility_changes, solution_changes, step_state

STEPS = tuple(STEP_COMPONENTS)  # setup steps, in tab order
TITLES = {
    "Cell": "Robot & cell",
    "External Axis": "External axis",
    "Tool Setup": "Tool",
    "Targets": "Targets",
}
ICON = {
    "confirmed": "✔ confirmed",
    "changed": "✎ changed since you confirmed it",
    "todo": "○ not confirmed yet",
}


class Workflow:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server = server
        self.s = session
        self.headers: dict[str, tuple] = {}
        self.opt_md = None
        self.results_md = None
        self.cell = None  # set by app.py
        self.carrier = None
        self.results = None
        self._last = None
        self._todo: list[str] | None = None  # unconfirmed steps last reported to the log
        self._lock = threading.Lock()
        self.hooks: list = []  # extra callables run by the watcher (e.g. object visibility)

    # --- widgets (created inside each tab, before its panel) ------------------------------------

    def header(self, step: str) -> None:
        gui = self.server.gui
        md = gui.add_markdown("")
        label = "Confirm setup" if step == "External Axis" else "Confirm step"
        ok = gui.add_button(label, color="green", hint="Mark this step as done")
        edit = gui.add_button("Edit setup", visible=False) if step == "External Axis" else None
        ok.on_click(lambda e: self.confirm(step))
        if edit is not None:
            edit.on_click(lambda e: self.edit(step))
        self.headers[step] = (md, ok, edit)

    def optimize_header(self) -> None:
        self.opt_md = self.server.gui.add_markdown("")

    def results_header(self) -> None:
        self.results_md = self.server.gui.add_markdown("")

    # --- actions ------------------------------------------------------------------------------

    def confirm(self, step: str) -> None:
        confirm_step(self.s, step)
        self.refresh(force=True)

    def edit(self, step: str) -> None:
        self.s.steps.pop(step, None)
        self.refresh(force=True)

    # --- state -> widgets ---------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            s = self.s
            states = {st: step_state(s, st) for st in STEPS}
            self._log_setup(
                [f"{i + 1} {TITLES[st]}" for i, st in enumerate(STEPS) if states[st] != "confirmed"]
            )
            locked = states["External Axis"] != "todo"
            mounted = s.base.mode == "movable"
            sol = self.results.selected() if self.results is not None else None
            sol_changes = solution_changes(s, sol) if sol is not None else []
            feas = feasibility_changes(s) if s.feasibility else []
            key = (
                tuple(states.items()),
                locked,
                mounted,
                sol.label if sol else None,
                tuple(sol_changes),
                tuple(feas),
            )
            if key == self._last and not force:
                return
            self._last = key
            for i, step in enumerate(STEPS):
                if step not in self.headers:
                    continue
                md, ok, edit = self.headers[step]
                nxt = (
                    f" · next: **{i + 2} {TITLES[STEPS[i + 1]]}**"
                    if i + 1 < len(STEPS)
                    else " · next: **5 Optimize**"
                )
                md.content = f"**Step {i + 1} of {len(STEPS)}: {TITLES[step]}** · {ICON[states[step]]}{nxt}"
                ok.disabled = states[step] == "confirmed"
                if edit is not None:
                    ok.visible, edit.visible = not locked, locked
            if self.carrier is not None:
                self.carrier.set_locked(locked)
            if self.cell is not None:
                self.cell.set_mounted(mounted)
            if self.opt_md is not None:
                todo = [f"{i + 1} {TITLES[st]}" for i, st in enumerate(STEPS) if states[st] != "confirmed"]
                lines = [
                    f"⚠ Setup steps not confirmed: **{', '.join(todo)}**. "
                    "Results can change when you edit them."
                    if todo
                    else "✔ Setup confirmed."
                ]
                if feas:
                    lines.append(
                        f"⚠ The feasibility check is **out of date** ({', '.join(feas)} changed): "
                        "check again."
                    )
                self.opt_md.content = "  \n".join(lines)
            if self.results_md is not None:
                self.results_md.content = (
                    f"⚠ **Out of date:** {', '.join(sol_changes)} changed since this result was computed. "
                    "Re-run it (Optimize) to update; exports ask first."
                    if sol_changes
                    else ""
                )

    def _log_setup(self, todo: list[str]) -> None:
        """Unconfirmed setup steps: the log panel's status line, and a message when they change."""
        from ui import log_panel

        if log_panel._panel is not None:
            log_panel._panel.set_setup(todo)
        if todo == self._todo:
            return
        if self._todo is not None or todo:
            if todo:
                log_panel.log(f"Setup steps not confirmed: **{', '.join(todo)}**.", "warn")
            else:
                log_panel.log("All setup steps confirmed.", "done")
        self._todo = todo

    def start_watch(self, period: float = 1.0) -> None:
        """Background refresh (real app only): edits anywhere update the step status and the
        out-of-date banners within about a second."""

        def loop() -> None:
            while True:
                time.sleep(period)
                for fn in (self.refresh, *self.hooks):
                    try:
                        fn()
                    except Exception:  # never let the watcher die (a mid-edit state may not hash)
                        pass

        threading.Thread(target=loop, daemon=True).start()
