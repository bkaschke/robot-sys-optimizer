"""Object tree (v2 phase 2): show / hide the scene's objects, choose which one has its drag
gizmo, and jog the robot (or the carrier it rides on) or the part with arrow buttons.

Jog: with a carrier, the arrows move carrier DOFs (rail s clamped to its travel, gantry x / y
clamped, mobile x / y / yaw unlimited); a static robot base and the part move in world x / y
/ z and turn about world z.
"""

from __future__ import annotations

import numpy as np
import viser

from core.transforms import make_T, rpy_to_matrix
from ui.session import Session
from ui.widgets import GizmoStyle

OBJECTS = ("Robot", "Carrier", "Tool", "Part", "Targets", "Results")
GIZMOS = ("none", "Robot base", "Part", "Tool mount", "Carrier body")
MOVE = ("Robot / carrier", "Part")


class ObjectsPanel:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.s = session
        self.cell = self.carrier = self.tool = self.targets = self.results = None  # set by app.py
        gui = server.gui
        with gui.add_folder("Objects", expand_by_default=False):
            self.show = {
                name: gui.add_checkbox(name, True, hint=f"show / hide {name.lower()}") for name in OBJECTS
            }
            self.gizmo = gui.add_dropdown("Drag gizmo on", GIZMOS, initial_value="Robot base")
            self.gizmo_size = gui.add_slider("Gizmo size", 0.25, 4.0, 0.25, GizmoStyle.size)
            self.gizmo_top = gui.add_checkbox(
                "Gizmos always on top", GizmoStyle.on_top, hint="drawn through bodies, so they can be grabbed"
            )
            with gui.add_folder("Jog", expand_by_default=True):
                self.move = gui.add_dropdown("Move", MOVE)
                self.step = gui.add_number("Step (mm)", 50.0, min=0.1, step=10.0)
                self.angle = gui.add_number("Turn step (deg)", 5.0, min=0.1, step=1.0)
                self.jx = gui.add_button_group("X", ("−X", "+X"))
                self.jy = gui.add_button_group("Y", ("−Y", "+Y"))
                self.jz = gui.add_button_group("Z", ("−Z", "+Z"))
                self.jr = gui.add_button_group("Turn", ("−Rz", "+Rz"))
                self.info = gui.add_markdown("")
        for w in self.show.values():
            w.on_update(lambda e: self.apply_visibility())
        self.gizmo.on_update(lambda e: self.apply_gizmo())
        self.gizmo_size.on_update(lambda e: GizmoStyle.apply(size=self.gizmo_size.value))
        self.gizmo_top.on_update(lambda e: GizmoStyle.apply(on_top=self.gizmo_top.value))
        for axis, group in (("x", self.jx), ("y", self.jy), ("z", self.jz), ("rz", self.jr)):
            group.on_click(lambda e, a=axis: self.jog(a, -1.0 if e.target.value.startswith("−") else 1.0))

    # --- visibility ------------------------------------------------------------------------------

    def _handles(self, name: str) -> list:
        c, car, t, tg, r = self.cell, self.carrier, self.tool, self.targets, self.results
        if name == "Robot":
            return [c.view.root] if c is not None and c.view is not None else []
        if name == "Carrier":
            return list(car._handles) if car is not None else []
        if name == "Tool":
            return [t.view.root] if t is not None else []
        if name == "Part":
            return [c.part_handle] if c is not None and c.part_handle is not None else []
        if name == "Targets":
            return [tg._handles[k] for k in ("points", "normals") if tg is not None and k in tg._handles]
        if name == "Results":
            sim = list(self.sim._handles) if getattr(self, "sim", None) is not None else []
            return (list(r._handles) + list(r._ghosts) if r is not None else []) + sim
        return []

    def apply_visibility(self) -> None:
        """Apply the checkboxes (also re-run by the watcher: panels re-create their nodes)."""
        for name, box in self.show.items():
            for h in self._handles(name):
                if h is not None and h.visible != box.value:
                    h.visible = box.value

    # --- gizmos ------------------------------------------------------------------------------------

    def apply_gizmo(self) -> None:
        """Only the chosen object has drag handles (fewer accidental drags)."""
        g = self.gizmo.value
        mounted = self.s.base.mode == "movable"
        if self.cell is not None:
            self.cell.base.show.value = g == "Robot base" and not mounted
            self.cell.part_pose.show.value = g == "Part"
        if self.tool is not None:
            self.tool.mount.show.value = g == "Tool mount"
        if self.carrier is not None:
            ed = self.carrier.editor
            ed.drag.value = g == "Carrier body" and not self.carrier.locked and ed.index() is not None

    # --- jog -----------------------------------------------------------------------------------------

    def jog(self, axis: str, sign: float) -> str:
        """One arrow press. Returns what moved (also shown in the panel)."""
        d = sign * (np.radians(self.angle.value) if axis == "rz" else self.step.value / 1000)
        if self.move.value == "Part":
            msg = self._jog_part(axis, d)
        elif self.s.base.mode == "movable" and self.carrier is not None:
            msg = self._jog_carrier(axis, d)
        else:
            msg = self._jog_base(axis, d)
        self.info.content = f"_{msg}_"
        return msg

    def _jog_carrier(self, axis: str, d: float) -> str:
        kind = self.s.base.carrier
        dof = {
            ("rail", "x"): "s",
            ("gantry", "x"): "x",
            ("gantry", "y"): "y",
            ("mobile", "x"): "x",
            ("mobile", "y"): "y",
            ("mobile", "rz"): "yaw",
        }.get((kind, axis))
        if axis == "z":
            dof = "lift"
        if dof is None or not self.carrier.jog_dof(dof, d):
            return f"the {kind} carrier has no {axis.upper()} axis"
        limit = "" if kind == "mobile" else " (within its travel)"
        return f"{kind} {dof} moved{limit}; the robot rides along"

    def _jog_base(self, axis: str, d: float) -> str:
        T = self.s.T_world_base.copy()
        self.cell.set_base(_moved(T, axis, d))
        return "robot base moved"

    def _jog_part(self, axis: str, d: float) -> str:
        if self.s.part is None:
            return "no part loaded"
        self.cell.part_pose._apply(_moved(self.s.part.T_world_part.copy(), axis, d), source="code")
        return "part moved"


def _moved(T: np.ndarray, axis: str, d: float) -> np.ndarray:
    """T translated along a world axis, or turned about the vertical through its origin."""
    if axis == "rz":
        out = T.copy()
        out[:3, :3] = rpy_to_matrix((0.0, 0.0, d)) @ T[:3, :3]
        return out
    return make_T(p=d * np.eye(3)["xyz".index(axis)]) @ T
