"""Robot home pose (v5 testing feedback): saved per project (``schema.Robot.home_deg``), set by
jogging the arm and taking the current pose, or by typing the joint values. Used by *Go to
home* and by the paths (*Home between stops*: the arm waits there while the carrier moves).
Without a saved home the robot's ``spec.yaml`` ``home_deg`` applies (else all zeros)."""

from __future__ import annotations

import numpy as np
import viser

from ui.widgets import notify


def parse_joints(text: str, n: int) -> list[float]:
    """'0, -90, 90, ...' (deg) -> n floats; ValueError otherwise."""
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != n or not all(np.isfinite(vals)):
        raise ValueError(f"enter {n} joint values in degrees, separated by commas")
    return vals


class HomePanel:
    def __init__(self, server: viser.ViserServer, cell) -> None:
        self.cell, self.s = cell, cell.s
        self._busy = False
        with server.gui.add_folder("Home position", expand_by_default=False):
            self.info = server.gui.add_markdown("")
            self.joints = server.gui.add_text(
                "Home joints (deg)", "", hint="one value per joint, comma separated"
            )
            self.apply = server.gui.add_button("Apply typed values")
            self.take = server.gui.add_button(
                "Set home to current pose", hint="jog the arm with the joint sliders first"
            )
            self.default = server.gui.add_button("Use robot default", hint="the robot's spec.yaml home")

        @self.apply.on_click
        def _(e) -> None:
            self.apply_text(e)

        @self.take.on_click
        def _(e) -> None:
            self.take_current(e)

        @self.default.on_click
        def _(_e) -> None:
            self.set_home(None)

    def take_current(self, event=None) -> bool:
        """Save the current (jogged) joint pose as home."""
        if self.s.robot is None or self.s.q is None:
            notify(event, "No robot", "Choose a robot first")
            return False
        self.set_home(list(np.degrees(self.s.q)))
        notify(event, "Home saved", "the current pose is the robot's home", ok=True)
        return True

    def apply_text(self, event=None) -> bool:
        """Save the typed joint values as home (a bad entry is reported and reset)."""
        if self.s.robot is None:
            return False
        try:
            vals = parse_joints(self.joints.value, self.s.robot.nq)
        except ValueError as err:
            self.refresh()
            notify(event, "Home not changed", str(err))
            return False
        self.set_home(vals)
        return True

    def set_home(self, deg: list[float] | None) -> None:
        """Save the home (deg, or None = robot default); values are clipped to the limits."""
        r = self.s.robot
        if r is not None and deg is not None:
            q = np.clip(np.radians(np.asarray(deg, float)), r.lower, r.upper)
            deg = [round(float(v), 4) for v in np.degrees(q)]
        self.s.home_deg = deg
        self.refresh()

    def refresh(self) -> None:
        self._busy = True
        try:
            q = self.s.home_q()
            if q is None:
                self.joints.value, self.info.content = "", "_No robot._"
                return
            self.joints.value = ", ".join(f"{v:g}" for v in np.round(np.degrees(q), 3))
            src = "saved in this project" if self.s.home_deg is not None else "robot default"
            self.info.content = f"Home ({src}). The paths use it between stops."
        finally:
            self._busy = False
