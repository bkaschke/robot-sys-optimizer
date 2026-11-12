"""Optimize tab layout (v9, issue #10): one **Optimizer** dropdown at the top. Only the chosen
optimizer's folder is shown (with its one Run / Cancel and progress bar); the settings they share
(max stations, costs & requirements, cycle time, cable & safety) are in one **Settings** folder at
the bottom of the tab.
"""

from __future__ import annotations

import viser

OPTIMIZERS = {
    "Station planner": "Finds the fewest robot stations (base / carrier positions) that reach every "
    "point, refines them against the exact checks, groups each stop's points by joint configuration. "
    "Mode: cost based, or rules only.",
    "Quick station plan": "A first look in seconds: the station planner without refinement, zones or "
    "station areas.",
    "Feasibility check": "Checks every target from the current robot base (static) or from a plan's "
    "stops: reachable, marginal or unreachable, and why.",
    "Base placement map": "A heatmap of how many targets a robot base reaches from each floor position "
    "(reach maps); move the base to the best one.",
    "Tool geometry": "Optimizes the tool (pose ranges, length and bends) together with the stations.",
    "Compare robots": "Plans the same study with several robots and lists stations, reach, cycle time "
    "and cost side by side.",
}


class OptimizerSelect:
    """The dropdown (built first, at the top of the tab); ``bind`` gives it each optimizer's
    folders once the panels exist."""

    def __init__(self, server: viser.ViserServer) -> None:
        gui = server.gui
        names = tuple(OPTIMIZERS)
        self.choice = gui.add_dropdown("Optimizer", names, initial_value=names[0])
        self.about = gui.add_markdown("")
        self.folders: dict[str, list] = {}
        self.choice.on_update(lambda _e: self.show())

    def bind(self, folders: dict[str, list]) -> None:
        self.folders = folders
        self.show()

    def show(self, name: str | None = None) -> None:
        """Show ``name`` (default: the dropdown's choice) and hide the other optimizers."""
        if name is not None and name != self.choice.value:
            self.choice.value = name  # its on_update calls show() again
            return
        pick = self.choice.value
        for key, handles in self.folders.items():
            for h in handles:
                h.visible = key == pick
        text = f"_{OPTIMIZERS.get(pick, '')}_"
        if self.about.content != text:
            self.about.content = text
