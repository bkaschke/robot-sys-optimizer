"""Optimize -> Setup check: look for common setup mistakes before a long plan (core.preflight).

**Check setup** runs the checks in a thread and lists the findings; ``on_plan_start`` runs them
quietly when a planner starts and only writes errors and warnings to the log.
"""

from __future__ import annotations

import threading

import viser

from core import preflight as pf
from ui.log_panel import log
from ui.preflight_run import findings_for

INTRO = (
    "_Looks for common mistakes: wrong part units, normals pointing into the part, points off "
    "the surface or inside a stay-out zone, more than a fixed base can cover. Advice only: "
    "nothing is changed._"
)


def findings_md(findings) -> str:
    if not findings:
        return "✔ **Nothing found.** (This does not check reach or collisions: run the feasibility check.)"
    rows = [f"**{pf.headline(findings)}**"]
    for f in findings:
        rows.append(f"{pf.ICONS.get(f.level, '•')} {f.text}" + (f" _{f.fix}_" if f.fix else ""))
    return "\n\n".join(rows)


class SetupCheckPanel:
    def __init__(self, server: viser.ViserServer, session):
        self.s = session
        self.findings: list[pf.Finding] = []
        self._thread: threading.Thread | None = None
        gui = server.gui
        with gui.add_folder("Setup check", expand_by_default=False):
            gui.add_markdown(INTRO)
            self.run_btn = gui.add_button("Check setup", hint="a few seconds on a large part")
            self.result = gui.add_markdown("_Not run yet._")
        self.run_btn.on_click(lambda e: self.run())

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, block: bool = False, quiet: bool = False) -> None:
        """Run the checks (in a thread unless ``block``). ``quiet``: the panel keeps its text when
        nothing is wrong, and only errors and warnings go to the log."""
        if self.busy:
            return
        if not quiet:
            self.result.content = "_Checking..._"

        def work() -> None:
            try:
                found = findings_for(self.s)
            except Exception as err:  # noqa: BLE001 - advice must never break the panel
                self.result.content = f"⚠ The setup check failed: {type(err).__name__}: {err}"
                return
            self.findings = found
            serious = [f for f in found if f.level in ("error", "warn")]
            if not quiet or serious:
                self.result.content = findings_md(found)
            for f in serious:
                log(f"Setup check: {f.text} {f.fix}".strip(), f.level)
            if not quiet and not serious:
                log(pf.headline(found), "done")

        if block:
            return work()
        self._thread = threading.Thread(target=work, daemon=True, name="setup-check")
        self._thread.start()

    def on_plan_start(self) -> None:
        """A planner started: check in the background, speak up only when something is wrong."""
        self.run(quiet=True)
