"""Object tree (v2 phase 2; a panel on the left since the v3 feedback): show / hide the scene's
objects and their parts (carrier bodies, stay-out zones, targets, result overlays), choose which
one has its drag gizmo, and jog the robot (or the carrier it rides on) or the part.

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

GIZMOS = ("none", "Robot base", "Part", "Tool mount", "Carrier body", "Stay-out zone", "Reduced-speed zone")
MOVE = ("Robot / carrier", "Part")
# the object tree: top-level entries and their fixed children ("*": one child per carrier body /
# stay-out zone, rebuilt when that list changes)
TREE = {
    "Robot": ("Tool",),
    "External axis": "*bodies",
    "Part": ("Targets", "Selection"),
    "Stay-out zones": "*zones",
    "Reduced-speed zones": "*slow",
    "Results": ("Stops and lines", "Ghost robots", "Station areas", "Paths"),
}
INDENT = " └ "  # child entries


class ObjectsPanel:
    """Lives in its own panel docked on the left (app.py), v3 feedback item 1."""

    def __init__(self, server: viser.ViserServer, session: Session):
        self.server, self.s = server, session
        self.cell = self.carrier = self.tool = self.targets = self.results = None  # set by app.py
        self.sim = self.stayouts = self.slowzones = None
        self.mover = None  # ui.move_mode.MoveMode, set by app.py
        gui = server.gui
        self.show: dict = {}  # "Robot", "Robot/Tool", "External axis/<body>", ...
        self._dyn: dict[str, list] = {}  # top entry -> names of its current dynamic children
        self._folders: dict = {}
        with gui.add_folder("Show / hide", expand_by_default=True):
            for top, kids in TREE.items():
                self.show[top] = gui.add_checkbox(top, True, hint=f"show / hide {top.lower()}")
                self._folders[top] = gui.add_folder(f"{top} items", expand_by_default=False)
                if isinstance(kids, tuple):
                    with self._folders[top]:
                        for k in kids:
                            self.show[f"{top}/{k}"] = gui.add_checkbox(INDENT + k, True)
        with gui.add_folder("Gizmos", expand_by_default=True):
            self.gizmo = gui.add_dropdown("Drag gizmo on", GIZMOS, initial_value="Robot base")
            self.gizmo_size = gui.add_slider("Gizmo size", 0.25, 4.0, 0.25, GizmoStyle.size)
            self.gizmo_top = gui.add_checkbox(
                "Gizmos always on top", GizmoStyle.on_top, hint="drawn through bodies, so they can be grabbed"
            )
            # v7 round 2 (T11): move one object with its gizmo, then Confirm / Cancel (also Alt+click)
            self.move_pick = gui.add_dropdown("Move object", ("Robot base",))
            self.move_btn = gui.add_button("Move…", hint="Alt+click an object in 3D does the same")
        for w in self.show.values():
            w.on_update(lambda e: self.apply_visibility())
        self.gizmo.on_update(lambda e: self.apply_gizmo())
        self.move_btn.on_click(lambda e: self.start_move(self.move_pick.value, e))
        self.gizmo_size.on_update(lambda e: GizmoStyle.apply(size=self.gizmo_size.value))
        self.gizmo_top.on_update(lambda e: GizmoStyle.apply(on_top=self.gizmo_top.value))

    def build_jog(self) -> None:
        """The Jog widgets; app.py calls this inside the Jog panel docked below the tree (v7
        round 2, T6)."""
        gui = self.server.gui
        self.jog_folder = gui.add_folder("Jog", expand_by_default=True)
        with self.jog_folder:
            self.move = gui.add_dropdown("Move", MOVE)
            self.step = gui.add_number("Step (mm)", 50.0, min=0.1, step=10.0)
            self.angle = gui.add_number("Turn step (deg)", 5.0, min=0.1, step=1.0)
            self.jx = gui.add_button_group("X", ("−X", "+X"))
            self.jy = gui.add_button_group("Y", ("−Y", "+Y"))
            self.jz = gui.add_button_group("Z", ("−Z", "+Z"))
            self.jr = gui.add_button_group("Turn", ("−Rz", "+Rz"))
            self.info = gui.add_markdown("")
            # filled by the External Axis panel (carrier DOF sliders) and the Cell tab (arm jog)
            self.carrier_slot = gui.add_folder("Carrier position", expand_by_default=True)
            self.arm_slot = gui.add_folder("Robot arm", expand_by_default=True)
        for axis, group in (("x", self.jx), ("y", self.jy), ("z", self.jz), ("rz", self.jr)):
            group.on_click(lambda e, a=axis: self.jog(a, -1.0 if e.target.value.startswith("−") else 1.0))

    # --- tree ---------------------------------------------------------------------------------

    def _children(self, top: str) -> list[str]:
        """Current dynamic children: carrier bodies, stay-out zones."""
        if top == "External axis" and self.carrier is not None:
            return list(self.carrier._body_handles)
        if top == "Stay-out zones":
            return [z.name for z in self.s.stayouts if z.speed_pct is None]
        if top == "Reduced-speed zones":
            return [z.name for z in self.s.stayouts if z.speed_pct is not None]
        return []

    def start_move(self, key: str, event=None) -> bool:
        """Move mode for one object (Objects -> Move…, Alt+click). False if it can't move."""
        from ui.move_targets import move_target

        target = move_target(self, key)
        if target is None or self.mover is None:
            return False
        self.mover.start(key, target, event)
        return True

    def _move_options(self) -> None:
        from ui.move_targets import move_keys

        keys = tuple(move_keys(self))
        if keys != tuple(self.move_pick.options):
            self.move_pick.options = keys

    def rebuild_tree(self) -> None:
        """Add / remove the per-body and per-zone entries when those lists changed."""
        self._move_options()
        for top, kids in TREE.items():
            if not isinstance(kids, str):
                continue
            names = self._children(top)
            if names == self._dyn.get(top):
                continue
            for old in self._dyn.get(top, []):
                box = self.show.pop(f"{top}/{old}", None)
                if box is not None:
                    box.remove()
            with self._folders[top]:
                for n in names:
                    box = self.server.gui.add_checkbox(INDENT + n, True)
                    box.on_update(lambda e: self.apply_visibility())
                    self.show[f"{top}/{n}"] = box
            self._dyn[top] = names

    def _handles(self, key: str) -> list:
        c, car, t, tg, r, sim, so = (
            self.cell,
            self.carrier,
            self.tool,
            self.targets,
            self.results,
            self.sim,
            self.stayouts,
        )
        if key == "Robot":
            return [c.view.root] if c is not None and c.view is not None else []
        if key == "Robot/Tool":
            return [t.view.root] if t is not None else []
        if key == "External axis":  # shells and keep-outs (bodies are their own entries); by identity:
            if car is None:  # handles compare their arrays with ==
                return []
            bodies = {id(h) for h in car._body_handles.values()}
            return [h for h in car._handles if id(h) not in bodies]
        if key.startswith("External axis/"):
            h = car._body_handles.get(key.split("/", 1)[1]) if car is not None else None
            return [h] if h is not None else []
        if key == "Part":
            return [c.part_handle] if c is not None and c.part_handle is not None else []
        if key == "Part/Targets":
            if tg is None:
                return []
            return [
                h for k, h in tg._handles.items() if k in ("points", "normals") or k.startswith("unreached")
            ]
        if key == "Part/Selection":
            return [tg._handles["sel"]] if tg is not None and "sel" in tg._handles else []
        if key.startswith("Stay-out zones/"):
            h = so._handles.get(key.split("/", 1)[1]) if so is not None else None
            return [h] if h is not None else []
        if key.startswith("Reduced-speed zones/"):
            sz = self.slowzones
            h = sz._handles.get(key.split("/", 1)[1]) if sz is not None else None
            return [h] if h is not None else []
        if key == "Results/Stops and lines":
            return list(r._handles) if r is not None else []
        if key == "Results/Ghost robots":
            # the ghost frames: hiding one hides its whole robot (ViserUrdf has no .visible)
            return [h for h in r._ghosts if hasattr(h, "visible")] if r is not None else []
        if key == "Results/Station areas":
            return list(r._areas) if r is not None else []
        if key == "Results/Paths":
            return list(sim._handles) if sim is not None else []
        return []

    def shown(self, key: str) -> bool:
        """An entry is shown when it and its parent are ticked."""
        top = key.split("/", 1)[0]
        box = self.show.get(key)
        return bool(self.show[top].value and (box is None or box.value))

    def apply_visibility(self) -> None:
        """Apply the checkboxes (also re-run by the watcher: panels re-create their nodes)."""
        self.rebuild_tree()
        for key in list(self.show):
            on = self.shown(key)
            for h in self._handles(key):
                if h is not None and h.visible != on:
                    h.visible = on

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
        if self.stayouts is not None:
            self.stayouts.drag.value = g == "Stay-out zone" and self.stayouts.index() is not None
        if self.slowzones is not None:
            self.slowzones.drag.value = g == "Reduced-speed zone" and self.slowzones.index() is not None

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
