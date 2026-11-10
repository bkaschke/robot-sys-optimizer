"""Optimize tab -> Station planner -> Costs & requirements (v4 plan item 5): every cost weight
is editable, and two rules can be made hard requirements ("must be met"):

  All points reachable                     plans with unreached points are rejected
  No configuration change within a group   zoning is on; plans with a wrist flip inside a
                                           group are rejected (a stop may hold several groups)

With a requirement ticked, the planner keeps adding stops until a plan meets it (up to Max
stations). Rejected plans stay in the Results list, marked ✗, below the valid ones. The
weights live in the project (``Project.optimizer.weights``). The tool-geometry panel has its
own copy (``CostPanel(field="tool_weights")``, ``Project.optimizer.tool_weights``; until edited
it starts from the planner's weights)."""

from __future__ import annotations

import viser

from core.schema import Optimizer, Weights
from ui.session import Session

# (Weights field, label, hint); the count costs first
FIELDS = (
    ("stations", "Cost per stop", "each base / carrier position beyond the first (fewer base changes)"),
    ("group", "Cost per group", "each extra joint configuration at a stop (a configuration change)"),
    ("motion", "Cost of arm motion", "mean largest joint move between neighbouring points, per 90°"),
    ("wrist_flip", "Cost per wrist flip", "a joint 4 / joint 6 jump > 180° inside a group's path (zoning)"),
    ("unreached", "Cost per unreached point", "keep it large: reaching points comes first"),
    ("travel", "Carrier travel (per m)", "rail |Δs|, gantry max axis, mobile straight line"),
    ("L", "Tool length", "times tool length / reference length"),
    ("manip", "Manipulability", "mean 1 / (1 + μ): low = near singular"),
    ("clearance", "Clearance", "mean shortfall below 50 mm to the part"),
    ("limits", "Joint limits", "mean closeness to a joint limit (within 30°)"),
    ("standoff", "Preferred standoff", "carrier closer than the preferred standoff"),
)


class CostPanel:
    def __init__(self, server: viser.ViserServer, session: Session, field: str = "weights"):
        self.s = session
        self.field = field  # Optimizer field holding these weights
        self._busy = False
        gui = server.gui
        title = "Costs & requirements" + (" (tool optimizer)" if field == "tool_weights" else "")
        self.folder = gui.add_folder(title, expand_by_default=False)
        with self.folder:
            gui.add_markdown("**Must be met**")
            self.require_all = gui.add_checkbox("All points reachable", False)
            self.require_config = gui.add_checkbox(
                "No configuration change within a group",
                False,
                hint="turns zoning on; a stop may still hold several groups",
            )
            gui.add_markdown(
                "_Ticked: the planner adds stops until a plan meets them (up to Max stations); plans "
                "that break one are marked ✗ and not ranked._  \n**Costs** (J = sum of weight × term)"
            )
            w = self.weights()
            self.fields = {
                key: gui.add_number(label, float(getattr(w, key)), min=0.0, step=0.1, hint=hint)
                for key, label, hint in FIELDS
            }
            self.reset = gui.add_button("Default costs")
        for f in self.fields.values():
            f.on_update(lambda e: None if self._busy else self.commit())
        self.reset.on_click(lambda e: self.set_weights(Weights()))

    def _optimizer(self) -> Optimizer:
        return self.s.extra.get("optimizer") or Optimizer()

    def weights(self) -> Weights:
        opt = self._optimizer()
        return getattr(opt, self.field) or opt.weights

    def commit(self) -> Weights:
        """GUI values -> the project's optimizer weights."""
        w = Weights(**{**self.weights().model_dump(), **{k: float(f.value) for k, f in self.fields.items()}})
        self.s.extra["optimizer"] = self._optimizer().model_copy(update={self.field: w})
        return w

    def set_weights(self, w: Weights) -> None:
        self.s.extra["optimizer"] = self._optimizer().model_copy(update={self.field: w})
        self.load()

    def load(self) -> None:
        """Project -> GUI (after a project is opened)."""
        w = self.weights()
        self._busy = True
        try:
            for k, f in self.fields.items():
                f.value = float(getattr(w, k))
        finally:
            self._busy = False

    def requirements(self) -> dict:
        return {
            "require_all": bool(self.require_all.value),
            "require_one_config": bool(self.require_config.value),
        }
