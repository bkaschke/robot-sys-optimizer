"""Robot preview in the package builder window (2026-11-11): a robot from the library, posed
with joint sliders, with the package on it: a tool on the flange, or the robot on the mount of a
rail / gantry / mobile base. Only the builder window shows it; the project's robot, tool and
external axis are not touched (*Use in this project* is the only way the package gets there).
"""

from __future__ import annotations

import numpy as np
import viser

from core.robot import Robot, list_robots
from core.transforms import make_T, matrix_to_wxyz, pose_to_matrix

ROBOT_ROOT = "/builder_robot"


class RobotPreview:
    def __init__(self, server: viser.ViserServer, builder, session) -> None:
        self.server, self.b, self.s = server, builder, session
        self.robot: Robot | None = None
        self.view = None
        self._name = None
        names = tuple(list_robots())
        start = session.robot_name if session.robot_name in names else names[0]
        gui = server.gui
        with gui.add_folder("Preview on robot", expand_by_default=True):
            self.on = gui.add_checkbox(
                "Show on a robot", True, hint="the tool on the flange, or the robot on the carrier's mount"
            )
            self.pick = gui.add_dropdown("Robot", names, initial_value=start)
            self.joints = [gui.add_slider(f"J{i + 1} (deg)", -180.0, 180.0, 1.0, 0.0) for i in range(6)]
            self.home = gui.add_button("Home pose")
        self.frame = server.scene.add_frame(ROBOT_ROOT, show_axes=False)
        self.on.on_update(lambda _e: self.b.draw())
        self.pick.on_update(lambda _e: self._load())
        for s in self.joints:
            s.on_update(lambda _e: self.b.draw())
        self.home.on_click(lambda _e: self.set_home())
        self._load()

    def _load(self) -> None:
        name = self.pick.value
        if name == self._name and self.robot is not None:
            return
        from ui.robot_view import RobotView

        if self.view is not None:
            self.view.remove()
            self.view = None
        self.robot = Robot.from_registry(name)
        self._name = name
        self.view = RobotView(
            self.server, self.robot, self.robot.spec.folder / self.robot.spec.urdf, ROBOT_ROOT
        )
        lo, hi = np.degrees(self.robot.lower), np.degrees(self.robot.upper)
        for k, s in enumerate(self.joints):
            on = k < len(lo)
            s.visible = on
            if on:
                s.min, s.max = float(max(lo[k], -360)), float(min(hi[k], 360))
        self.set_home()

    def set_home(self) -> None:
        q = np.degrees(self.robot.home_q())
        for k, s in enumerate(self.joints[: len(q)]):
            s.value = float(np.clip(q[k], s.min, s.max))
        self.b.draw()

    def q(self) -> np.ndarray:
        n = len(self.robot.lower)
        return np.radians([s.value for s in self.joints[:n]])

    @property
    def active(self) -> bool:
        return bool(self.on.value) and self.robot is not None and self.b.pkg.kind != "part"

    def root_pose(self, at: np.ndarray) -> np.ndarray:
        """World pose of the package root: the flange (tool), else ``at`` (the floor point)."""
        if self.active and self.b.pkg.kind == "tool":
            return self.robot.fk(self.q())  # the robot base sits at the world origin
        return at

    def place(self, T_root: np.ndarray) -> None:
        """Pose the robot: base at the origin (tool), or on the package's robot mount."""
        on = self.active
        self.frame.visible = on
        if not on:
            return
        self.view.set_q(self.q())
        if self.b.pkg.kind == "tool":
            T = np.eye(4)
        else:
            T = T_root @ pose_to_matrix(self.b.pkg.mount)
        self.frame.wxyz = matrix_to_wxyz(T[:3, :3])
        self.frame.position = tuple(T[:3, 3])


def at_pose(at_mm) -> np.ndarray:
    return make_T(p=np.asarray(at_mm, float) / 1000)
