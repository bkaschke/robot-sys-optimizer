"""Robot joint limits (v6): per joint (min, max) in degrees, saved per project
(``schema.Robot.joint_limits_deg``) and kept inside the URDF limits. Every check uses them:
IK solutions, the planner, paths, joint sliders and the home pose (``Robot.set_limits``; the
reachability map keeps joint 1's URDF range, see opt.reachability.map_limits)."""

from __future__ import annotations

import numpy as np
import viser

from ui.widgets import notify


class LimitsPanel:
    def __init__(self, server: viser.ViserServer, cell) -> None:
        self.server, self.cell, self.s = server, cell, cell.s
        self.fields: list = []
        self._sig = None
        self.folder = server.gui.add_folder("Joint limits", expand_by_default=False)
        with self.folder:
            self.info = server.gui.add_markdown("")
            self.rows = server.gui.add_folder("Min / max per joint (deg)")
            self.apply = server.gui.add_button("Apply joint limits")
            self.reset = server.gui.add_button("Use URDF limits", hint="the robot's own joint ranges")

        @self.apply.on_click
        def _(e) -> None:
            self.apply_fields(e)

        @self.reset.on_click
        def _(e) -> None:
            self.set_limits(None, e)

    def values(self) -> list[tuple[float, float]]:
        return [(float(f.value[0]), float(f.value[1])) for f in self.fields]

    def apply_fields(self, event=None) -> bool:
        """Apply the (min, max) fields; a bad entry is reported and the fields reset."""
        return self.set_limits(self.values(), event)

    def set_limits(self, limits_deg: list | None, event=None) -> bool:
        r = self.s.robot
        if r is None:
            notify(event, "No robot", "Choose a robot first")
            return False
        try:
            r.set_limits(limits_deg)
        except ValueError as err:
            self.refresh()
            notify(event, "Joint limits not changed", str(err))
            return False
        if limits_deg is not None:  # store what applies (clipped to the URDF), rounded
            deg = np.degrees(np.c_[r.lower, r.upper])
            r.set_limits([(round(float(a), 4), round(float(b), 4)) for a, b in deg])
        if self.s.q is not None:
            self.s.q = np.clip(self.s.q, r.lower, r.upper)
        self.cell.refresh()  # sliders, home and readout follow the new limits
        notify(event, "Joint limits applied", "re-run the planner and rebuild the paths", ok=True)
        return True

    def refresh(self) -> None:
        r = self.s.robot
        sig = None if r is None else (id(r), r.nq)
        if sig != self._sig:  # one field per joint of this robot
            for f in self.fields:
                f.remove()
            self.fields = []
            if r is not None:
                with self.rows:
                    for name in r.joint_names:
                        self.fields.append(self.server.gui.add_vector2(name, (0.0, 0.0), step=0.1))
            self._sig = sig
        if r is None:
            self.info.content = "_No robot._"
            return
        for f, lo, hi in zip(self.fields, r.lower, r.upper, strict=True):
            f.value = (round(float(np.degrees(lo)), 4), round(float(np.degrees(hi)), 4))
        own = r.spec.limits_deg is not None
        self.info.content = (
            "Limits **saved in this project** (inside the URDF limits)."
            if own
            else "The robot's **URDF limits**. Narrow them here: every check uses them."
        )
