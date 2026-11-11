"""Drag gizmos for the left panel's Jog (2026-11-11).

``ArmGizmo``: a transform control at the jog point (flange or a TCP), its axes those of the jog
frame chosen in *Robot arm* (World, Robot base, Part, Tool; Joint uses world axes). Dragging
moves the point by the gizmo's motion (rotations about the point) through
``JogPanel.move_point``: IK nearest the current joints, refused out of reach, across a
configuration flip or into a collision. The gizmo snaps back onto the point when a drag ends.

``CarrierGizmo``: a transform control at the robot's mount parent (carriage + lift). Only the
carrier's axes are offered: rail s, gantry x / y, mobile x / y + turn about z, + z for a lift.
A dragged pose is projected onto those DOFs (``dofs_for``) and applied with
``CarrierPanel.set_values`` (rail / gantry within travel, a mobile base anywhere).
"""

from __future__ import annotations

import numpy as np
import viser

from core.transforms import make_T, matrix_to_wxyz, wxyz_to_matrix
from ui.widgets import GizmoStyle, Latest

ARM_NODE = "/jog/arm_gizmo"
CARRIER_NODE = "/jog/carrier_gizmo"


def _pose(h) -> np.ndarray:
    return make_T(wxyz_to_matrix(np.asarray(h.wxyz, float)), np.asarray(h.position, float))


class _Gizmo:
    """Shared drag bookkeeping: ``place()`` is ignored while the user drags."""

    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self.handle = None
        self.visible = False
        self.dragging = False

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        self.place()

    def _remove(self) -> None:
        if self.handle is not None:
            GizmoStyle._items = [(h, b) for h, b in GizmoStyle._items if h is not self.handle]
            self.handle.remove()
            self.handle = None

    def _put(self, T: np.ndarray) -> None:
        h = self.handle
        h.wxyz = matrix_to_wxyz(T[:3, :3])
        h.position = T[:3, 3]

    def _wire(self) -> None:
        h = self.handle
        drag = Latest(self._dragged)

        @h.on_update
        def _(e) -> None:
            phase = getattr(e, "phase", "update")
            if phase == "start" or not self.dragging:
                self.dragging = True
                self._start(_pose(h))
            if phase == "end":
                drag(_pose(h))
                self.dragging = False
                self.place()  # back onto where the robot / carrier actually is
            else:
                drag(_pose(h))

    def _start(self, G: np.ndarray) -> None: ...

    def _dragged(self, G: np.ndarray) -> None: ...

    def place(self) -> None: ...


class ArmGizmo(_Gizmo):
    def __init__(self, server: viser.ViserServer, jog) -> None:
        super().__init__(server)
        self.jog = jog
        self._G0 = self._T0 = None

    def frame_pose(self) -> np.ndarray | None:
        """Gizmo pose: at the jog point, axes of the chosen frame."""
        T = self.jog.T_world_point()
        if T is None:
            return None
        R = None if self.jog.frame.value == "Joint" else self.jog.R_frame(T)
        return make_T(np.eye(3) if R is None else R, T[:3, 3])

    def place(self) -> None:
        if self.dragging:
            return
        G = self.frame_pose() if self.visible else None
        if G is None:
            self._remove()
            return
        if self.handle is None:
            self.handle = self.server.scene.add_transform_controls(ARM_NODE, scale=0.25, depth_test=False)
            GizmoStyle.register(self.handle, 0.25)
            self._wire()
        self._put(G)

    def _start(self, G: np.ndarray) -> None:
        self._G0, self._T0 = G, self.jog.T_world_point()

    def _dragged(self, G: np.ndarray) -> bool:
        """Move the jog point by the gizmo's motion since the drag started."""
        if self._G0 is None or self._T0 is None:
            return False
        delta = G @ np.linalg.inv(self._G0)  # world motion; the gizmo origin is the point
        return self.jog.move_point(delta @ self._T0, quiet=True)


class CarrierGizmo(_Gizmo):
    def __init__(self, server: viser.ViserServer, carrier) -> None:
        super().__init__(server)
        self.carrier = carrier
        self._sig = None

    def _model(self):
        b = self.carrier.s.base
        return None if b.mode != "movable" else self.carrier._carrier()

    def mount_parent(self, c, v: dict) -> np.ndarray:
        return c.T_world_carriage(v) @ make_T(p=(0, 0, v.get("lift", 0.0)))

    def place(self) -> None:
        if self.dragging:
            return
        c = self._model() if self.visible else None
        if c is None:
            self._remove()
            self._sig = None
            return
        lift = "lift" in c.dofs
        sig = (c.kind, lift)
        if sig != self._sig:  # the offered axes depend on the carrier kind
            self._remove()
            self._sig = sig
            mobile, gantry = c.kind == "mobile", c.kind == "gantry"
            self.handle = self.server.scene.add_transform_controls(
                CARRIER_NODE,
                scale=0.6,
                depth_test=False,
                active_axes=(True, mobile or gantry, lift),
                disable_rotations=not mobile,
                rotation_limits=((0.0, 0.0), (0.0, 0.0), (-1000.0, 1000.0)),
            )
            GizmoStyle.register(self.handle, 0.6)
            self._wire()
        self._put(self.mount_parent(c, self.carrier.current(c)))

    def dofs_for(self, c, G: np.ndarray) -> dict:
        """Carrier DOF values whose mount parent is closest to the gizmo pose ``G``."""
        if c.kind == "mobile":
            v = {"x": float(G[0, 3]), "y": float(G[1, 3]), "yaw": float(np.arctan2(G[1, 0], G[0, 0]))}
            z = float(G[2, 3])
        else:
            local = np.linalg.inv(c.T_frame) @ G
            v = {"s": float(local[0, 3])} if c.kind == "rail" else {"x": local[0, 3], "y": local[1, 3]}
            z = float(local[2, 3])
        if "lift" in c.dofs:
            v["lift"] = z
        return {k: float(x) for k, x in v.items()}

    def _dragged(self, G: np.ndarray) -> bool:
        c = self._model()
        if c is None:
            return False
        self.carrier.set_values(self.dofs_for(c, G))
        return True
