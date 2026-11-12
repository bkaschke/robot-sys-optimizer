"""Results -> Robot (v9, issue #6): view each robot of Optimize -> Compare robots at its stops,
with its plans, and build its paths, without changing the project.

Viewing a compared robot puts it in the cell (the 3D robot, the jog, the paths use it) and lists
its plans in Results; the project's own robot, home, joint limits and plans are kept aside in
``session.view_stash`` and come back with **Back to the project's robot**. Saving meanwhile saves
the project's own robot and plans (``Session.snapshot``). **Use this robot in the project** makes
the switch permanent (the project then uses this robot and its plans), after a confirmation.
"""

from __future__ import annotations

from collections.abc import Callable

import viser

from ui.log_panel import log
from ui.widgets import notify

STASH = ("robot", "robot_name", "robot_urdf", "q", "home_deg", "results", "result_points", "stamps")


class CompareView:
    def __init__(self, server: viser.ViserServer, session) -> None:
        self.s = session
        self.cell = self.results = self.sim = None  # set by app.py
        self.on_changed: Callable[[], None] | None = None  # app.py: e.g. the Pareto view
        self._quiet = False  # set while refresh() moves the dropdown itself
        gui = server.gui
        self.folder = gui.add_folder("Compared robots", expand_by_default=True, visible=False)
        with self.folder:
            self.choice = gui.add_dropdown(
                "Robot", ("(project)",), hint="a robot of Optimize → Compare robots"
            )
            self.banner = gui.add_markdown("")
            self.back = gui.add_button("Back to the project's robot", visible=False)
            self.use = gui.add_button("Use this robot in the project", color="green", visible=False)
        self.choice.on_update(lambda e: self.view(self.choice.value, e))
        self.back.on_click(lambda e: self.view(None, e))
        self.use.on_click(lambda e: self.confirm_use(e))
        self.refresh()

    def project_label(self) -> str:
        stash = self.s.view_stash
        name = stash["robot_name"] if stash else self.s.robot_name
        return f"{name or 'project robot'} (project)"

    def refresh(self) -> None:
        """The dropdown lists the project's robot and the compared ones."""
        names = [n for n in self.s.compared if self.s.compared[n].get("results")]
        self.folder.visible = bool(names)
        opts = (self.project_label(), *names)
        viewing = self.s.robot_name if self.s.view_stash else None
        if tuple(self.choice.options) != opts:
            self.choice.options = opts
        want = viewing if viewing in names else opts[0]
        if self.choice.value != want:
            self._quiet = True
            try:
                self.choice.value = want
            finally:
                self._quiet = False
        self._banner()

    def _banner(self) -> None:
        stash = self.s.view_stash
        self.back.visible = self.use.visible = bool(stash)
        if stash:
            m = self.s.compared.get(self.s.robot_name, {}).get("metrics", {})
            cyc = f", cycle {m['cycle_s']:.0f} s" if m.get("cycle_s") is not None else ""
            n = m.get("stations", "?")
            self.banner.content = (
                f"⚠ **Viewing the {self.s.robot_name}** from Compare robots ({n} stations{cyc}). "
                f"The project still uses the **{stash['robot_name']}**: saving saves that one."
            )
        else:
            self.banner.content = "_Pick a compared robot to see it at its stops, with its plans and paths._"

    def _apply(self) -> None:
        if self.cell is not None:
            self.cell.refresh()
        if self.results is not None:
            self.results.refresh()
        if self.sim is not None:
            self.sim.reset("the robot shown in Results changed")
        if self.on_changed is not None:
            self.on_changed()
        self.refresh()

    def view(self, name: str | None, event=None) -> None:
        """Show compared robot ``name`` (None or the project entry: back to the project's robot)."""
        if self._quiet:
            return
        s = self.s
        if name not in s.compared:
            name = None
        current = s.robot_name if s.view_stash else None
        if name == current:
            return self.refresh()
        if s.view_stash:  # back to the project first
            for k, v in s.view_stash.items():
                setattr(s, k, v)
            s.view_stash = None
        if name is not None:
            c = s.compared[name]
            s.view_stash = {k: getattr(s, k) for k in STASH}
            s.load_robot_from_registry(name)
            s.results = list(c["results"])
            s.result_points = {label: dict(pts) for label, pts in c["points"].items()}
            s.stamps = {
                **{k: v for k, v in s.view_stash["stamps"].items() if k == "feasibility"},
                **{k: dict(v) for k, v in c["stamps"].items()},
            }
            log(f"Results shows the {name} from Compare robots (the project keeps its robot).", "info")
        self._apply()

    def use_robot(self) -> str | None:
        """Make the viewed robot the project's (its plans become the project's plans)."""
        s = self.s
        if not s.view_stash:
            return None
        old = s.view_stash["robot_name"]
        s.view_stash = None
        log(f"The project now uses the {s.robot_name} (was the {old}).", "done")
        self.refresh()
        return s.robot_name

    def confirm_use(self, event) -> None:
        client = getattr(event, "client", None)
        if client is None:
            self.use_robot()
            return
        name, old = self.s.robot_name, (self.s.view_stash or {}).get("robot_name")
        with client.gui.add_modal("Use this robot in the project?") as modal:
            client.gui.add_markdown(
                f"The project switches from the **{old}** to the **{name}**, with the {name}'s plans from "
                "Compare robots. The project's current plans are dropped."
            )
            go = client.gui.add_button(f"Use the {name}", color="green")
            stop = client.gui.add_button("Cancel")

        @go.on_click
        def _(e) -> None:
            modal.close()
            self.use_robot()
            notify(e, "Robot changed", f"The project now uses the {name}", ok=True)

        stop.on_click(lambda _e: modal.close())
