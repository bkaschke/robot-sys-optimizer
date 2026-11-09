"""Jog the robot (v6, Cell -> Robot -> Jog): per joint, or the TCP along / about the axes of
the world, the robot base, the part or the tool frame (core.jog), step by step with - / +
buttons, like a teach pendant. The joints come from IK nearest the current pose; a step that
would need a configuration flip (a joint moving more than 30 deg) or leave the reach is
refused. The readout shows the jogged point's pose in the chosen frame."""

from __future__ import annotations

import numpy as np
import viser

from core.jog import AXES, MAX_STEP, jog_pose, nearest_ik
from core.transforms import matrix_to_rpy
from ui.widgets import notify

FRAMES = ("Joint", "World", "Robot base", "Part", "Tool")
FLANGE = "(flange)"


class JogPanel:
    def __init__(self, server: viser.ViserServer, cell) -> None:
        self.cell, self.s = cell, cell.s
        self._ik = self._ik_for = None
        gui = server.gui
        with gui.add_folder("Jog", expand_by_default=False):
            self.frame = gui.add_dropdown("Frame", FRAMES, initial_value="World")
            self.point = gui.add_dropdown("Jog point", (FLANGE,), hint="the TCP that moves")
            self.step = gui.add_number("Step (mm)", 10.0, min=0.1, step=1.0)
            self.turn = gui.add_number("Turn step (deg)", 5.0, min=0.1, step=1.0)
            self.joint_step = gui.add_number("Joint step (deg)", 5.0, min=0.1, step=1.0)
            self.minus = gui.add_button_group("Jog −", AXES)
            self.plus = gui.add_button_group("Jog +", AXES)
            self.j_minus = gui.add_button_group("Joint −", ("J1", "J2", "J3", "J4", "J5", "J6"))
            self.j_plus = gui.add_button_group("Joint +", ("J1", "J2", "J3", "J4", "J5", "J6"))
            self.readout = gui.add_markdown("")
        self.minus.on_click(lambda e: self.jog(e.target.value, -1, e))
        self.plus.on_click(lambda e: self.jog(e.target.value, +1, e))
        self.j_minus.on_click(lambda e: self.jog(e.target.value, -1, e))
        self.j_plus.on_click(lambda e: self.jog(e.target.value, +1, e))
        self.frame.on_update(lambda _e: self.refresh())
        self.point.on_update(lambda _e: self.update_readout())
        self.refresh()

    # --- state ---------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Show the buttons and steps of the chosen frame; TCP list from the current tool."""
        joint = self.frame.value == "Joint"
        for w in (self.minus, self.plus, self.step, self.turn, self.point):
            w.visible = not joint
        for w in (self.j_minus, self.j_plus, self.joint_step):
            w.visible = joint
        names = (FLANGE, *(t.name for t in self.s.tool.tcps)) if self.s.tool is not None else (FLANGE,)
        if tuple(self.point.options) != names:
            keep = self.point.value if self.point.value in names else names[-1]
            self.point.options = names
            self.point.value = keep
        self.update_readout()

    def _T_flange_point(self) -> np.ndarray:
        tm, name = self.s.tool_model, self.point.value
        if tm is None or name == FLANGE or name not in [t.name for t in tm.ee.tcps]:
            return np.eye(4)
        return tm.T_flange_tcp(name)

    def T_world_point(self) -> np.ndarray | None:
        s = self.s
        if s.robot is None or s.q is None:
            return None
        return np.asarray(s.T_world_base) @ s.robot.fk(s.q) @ self._T_flange_point()

    def R_frame(self, T_world_point: np.ndarray) -> np.ndarray | None:
        f = self.frame.value
        if f == "World":
            return np.eye(3)
        if f == "Robot base":
            return np.asarray(self.s.T_world_base)[:3, :3]
        if f == "Part":
            return None if self.s.part is None else np.asarray(self.s.part.T_world_part)[:3, :3]
        return T_world_point[:3, :3]  # Tool

    # --- jog -------------------------------------------------------------------------------------

    def jog(self, axis: str, sign: int, event=None) -> bool:
        """One step: joint ``J1``.. by the joint step, or the jog point along / about ``axis``
        (X Y Z Rx Ry Rz) of the chosen frame."""
        s = self.s
        if s.robot is None or s.q is None:
            notify(event, "No robot", "Choose a robot first")
            return False
        r, q = s.robot, np.asarray(s.q, float)
        if axis.startswith("J"):
            j = int(axis[1:]) - 1
            if j >= r.nq:
                return False
            qn = q.copy()
            qn[j] = np.clip(q[j] + sign * np.radians(self.joint_step.value), r.lower[j], r.upper[j])
            return self._go(qn)
        if self.frame.value == "Joint":
            return False
        T = self.T_world_point()
        R = self.R_frame(T)
        if R is None:
            notify(event, "No part", "Load a part to jog in its frame")
            return False
        amount = sign * (self.step.value / 1000 if axis in AXES[:3] else np.radians(self.turn.value))
        T_new = jog_pose(T, R, axis, amount)
        T_bf = np.linalg.inv(np.asarray(s.T_world_base)) @ T_new @ np.linalg.inv(self._T_flange_point())
        qn = nearest_ik(self._solver(), r, T_bf, q)
        if qn is None:
            notify(event, "Cannot jog", "that pose is out of reach or outside the joint limits")
            return False
        if np.max(np.abs(qn - q)) > MAX_STEP:
            notify(event, "Cannot jog", "the arm would have to change configuration (near a singularity)")
            return False
        return self._go(qn)

    def _solver(self):
        from core.ik import make_ik

        if self._ik_for is not self.s.robot:
            self._ik, self._ik_for = make_ik(self.s.robot), self.s.robot
        return self._ik

    def _go(self, q: np.ndarray) -> bool:
        self.cell.set_q(q)
        self.update_readout()
        return True

    def update_readout(self) -> None:
        T = self.T_world_point()
        if T is None:
            self.readout.content = "_No robot._"
            return
        f = self.frame.value
        if f == "Joint":
            deg = ", ".join(f"{v:.1f}" for v in np.degrees(self.s.q))
            self.readout.content = f"Joints (deg): {deg}"
            return
        ref = {"Robot base": np.asarray(self.s.T_world_base)}
        if f == "Part" and self.s.part is not None:
            ref["Part"] = np.asarray(self.s.part.T_world_part)
        T_f = ref.get(f, np.eye(4))
        P = np.linalg.inv(T_f) @ T
        where = "world" if f in ("World", "Tool") else f.lower()
        p, rpy = P[:3, 3] * 1000, np.degrees(matrix_to_rpy(P[:3, :3]))
        self.readout.content = (
            f"**{self.point.value}** in {where}: xyz mm {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}  \n"
            f"rpy deg {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}"
        )
