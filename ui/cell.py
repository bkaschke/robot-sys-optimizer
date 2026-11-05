"""Cell tab: robot (registry or uploaded URDF), mount, base pose, joint jog, part.

The GUI works in mm and degrees; the session and scene in m and rad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import viser

from core.golden_parts import GOLDEN_PARTS
from core.robot import list_robots
from core.transforms import (
    make_T,
    matrix_to_rpy,
    rpy_to_matrix,
)
from core.urdf_import import URDFImportError, import_urdf_zip
from ui.robot_view import RobotView
from ui.session import MESH_EXTS, MOUNT_RPY, UNIT_SCALE, PartState, Session, load_mesh
from ui.widgets import MAX_UPLOAD, PoseWidgets
from ui.widgets import notify as _notify
from ui.widgets import safe_name as _safe_name

BASE_NODE = "/cell/base"
PART_NODE = "/cell/part"
UPLOADED = "(uploaded URDF)"


def _yaw_about_world_z(R: np.ndarray) -> np.ndarray:
    """The world-z yaw part of R (heading of R's x axis projected on the floor)."""
    yaw = np.arctan2(R[1, 0], R[0, 0]) if abs(R[2, 0]) < 1 - 1e-9 else np.arctan2(-R[0, 1], R[1, 1])
    return rpy_to_matrix((0.0, 0.0, yaw))


class CellTab:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server = server
        self.s = session
        self.view: RobotView | None = None
        self.sliders: list = []
        self.part_handle = None
        self._busy = False  # set while the code (not the user) updates widgets
        self.listeners: list = []  # called with T_world_flange after any robot move
        self.part_listeners: list = []  # called after the part mesh/units change
        self.on_part_click = None  # set by the Targets tab: fn(event)

        with server.gui.add_folder("Robot"):
            robots = (*list_robots(), UPLOADED)
            self.robot_choice = server.gui.add_dropdown(
                "Robot", robots, initial_value=session.robot_name or UPLOADED
            )
            self.robot_upload = server.gui.add_upload_button(
                "Upload URDF (.zip)", mime_type="application/zip", hint="URDF + meshes in one zip"
            )
            self.mount = server.gui.add_dropdown("Mount", tuple(MOUNT_RPY) + ("custom",))
            self.base = PoseWidgets(server, BASE_NODE, session.T_world_base, self._on_base)
            self.joint_folder = server.gui.add_folder("Joints (deg)")
            self.home = server.gui.add_button("Home")
            self.readout = server.gui.add_markdown("")
        with server.gui.add_folder("Part"):
            self.part_choice = server.gui.add_dropdown(
                "Part", (*GOLDEN_PARTS, "(uploaded mesh)", "(none)"), initial_value="(none)"
            )
            self.part_upload = server.gui.add_upload_button("Upload mesh (.stl/.obj/.ply)")
            self.part_units = server.gui.add_dropdown("Units", tuple(UNIT_SCALE), initial_value="m")
            self.part_pose = PoseWidgets(server, PART_NODE, np.eye(4), self._on_part_pose)

        self._wire()
        self.refresh()

    # --- callbacks --------------------------------------------------------------

    def _wire(self) -> None:
        @self.robot_choice.on_update
        def _(e) -> None:
            if self._busy:
                return
            if self.robot_choice.value != UPLOADED and self.robot_choice.value != self.s.robot_name:
                self.s.load_robot_from_registry(self.robot_choice.value)
                self.refresh()

        @self.robot_upload.on_upload
        def _(e) -> None:
            f = self.robot_upload.value
            if len(f.content) > MAX_UPLOAD:
                return _notify(e, "Upload too large", "Limit is 200 MB")
            dest = self.s.project_dir / "assets" / "robots" / _safe_name(Path(f.name).stem)
            try:
                info = import_urdf_zip(f.content, dest)
                self.s.load_robot_from_urdf(info.path)
            except (URDFImportError, ValueError, OSError) as err:
                return _notify(e, "URDF import failed", str(err))
            self.refresh()
            _notify(e, "Robot loaded", f"{info.name}: {len(info.actuated_joints)} joints", ok=True)

        @self.mount.on_update
        def _(_e) -> None:
            if self._busy:
                return
            if self.mount.value in MOUNT_RPY:
                T = self.s.T_world_base.copy()
                T[:3, :3] = _yaw_about_world_z(T[:3, :3]) @ rpy_to_matrix(MOUNT_RPY[self.mount.value])
                self.s.T_world_base = T
                self.base.set(T)
                self._update_readout()
            self.s.mount = self.mount.value

        @self.home.on_click
        def _(_e) -> None:
            from ui.session import HOME_Q

            q = HOME_Q.get(self.s.robot_name or "", np.zeros(self.s.robot.nq))
            self.set_q(np.clip(q, self.s.robot.lower, self.s.robot.upper))

        @self.part_choice.on_update
        def _(_e) -> None:
            if self._busy:
                return
            v = self.part_choice.value
            if v in GOLDEN_PARTS:
                T = self.s.part.T_world_part if self.s.part else make_T(p=(1.0, 0.0, 0.5))
                self.s.part = PartState(f"assets/parts/{v}.stl", "m", T)
            elif v == "(none)":
                self.s.part = None
            self.refresh_part()

        @self.part_upload.on_upload
        def _(e) -> None:
            f = self.part_upload.value
            name = _safe_name(f.name)
            if Path(name).suffix.lower() not in MESH_EXTS or len(f.content) > MAX_UPLOAD:
                return _notify(e, "Unsupported mesh", f"Use {', '.join(MESH_EXTS)} under 200 MB")
            path = self.s.project_dir / "assets" / "parts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f.content)
            try:
                load_mesh(path, self.part_units.value)
            except Exception as err:  # trimesh raises many types on bad files
                return _notify(e, "Mesh load failed", str(err))
            T = self.s.part.T_world_part if self.s.part else make_T(p=(1.0, 0.0, 0.5))
            self.s.part = PartState(self.s.relative(path), self.part_units.value, T)
            self.refresh_part()

        @self.part_units.on_update
        def _(_e) -> None:
            if self._busy:
                return
            if self.s.part is not None and self.s.part.units != self.part_units.value:
                self.s.part.units = self.part_units.value
                self.refresh_part()

    def _on_base(self, T: np.ndarray) -> None:
        self.s.T_world_base = T
        preset = MOUNT_RPY.get(self.mount.value)
        if preset is not None:
            R_rel = T[:3, :3] @ rpy_to_matrix(preset).T
            if abs(R_rel[2, 2] - 1) > 1e-6:  # more than a yaw about world z: not the preset any more
                self.mount.value = "custom"
                self.s.mount = "custom"
        self._update_readout()

    def set_base(self, T: np.ndarray) -> None:
        """Programmatic base move (placement optimizer): same path as a user edit."""
        self.base._apply(T, source="code")

    def _on_part_pose(self, T: np.ndarray) -> None:
        if self.s.part is not None:
            self.s.part.T_world_part = T

    # --- scene refresh ------------------------------------------------------------

    def set_q(self, q: np.ndarray) -> None:
        """Programmatic jog (also used by tests): updates sliders, scene and readout."""
        for sl, v in zip(self.sliders, q, strict=True):
            sl.value = float(np.degrees(v))  # fires the slider callbacks

    def _on_slider(self) -> None:
        if self._busy:
            return
        self.s.q = np.radians([sl.value for sl in self.sliders])
        self.view.set_q(self.s.q)
        self._update_readout()

    def _update_readout(self) -> None:
        T = self.s.T_world_flange()
        for fn in self.listeners:
            fn(T)
        p = T[:3, 3] * 1000
        r = np.degrees(matrix_to_rpy(T[:3, :3]))
        self.readout.content = (
            f"**Flange `{self.s.robot.spec.flange_link}` in world (Pinocchio FK)**  \n"
            f"xyz mm: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}  \nrpy deg: {r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}"
        )

    def refresh(self) -> None:
        """Rebuild robot display + sliders from the session (after robot or project change)."""
        s = self.s
        if s.robot is None:
            s.load_robot_from_registry("ur20")
        self._busy = True
        try:
            self._rebuild()
        finally:
            self._busy = False

    def _rebuild(self) -> None:
        s = self.s
        if self.view is not None:
            self.view.remove()
        self.view = RobotView(self.server, s.robot, s.robot_urdf_path, BASE_NODE)
        for sl in self.sliders:
            sl.remove()
        self.sliders = []
        with self.joint_folder:
            for name, lo, hi, v in zip(s.robot.joint_names, s.robot.lower, s.robot.upper, s.q, strict=True):
                sl = self.server.gui.add_slider(
                    name, float(np.degrees(lo)), float(np.degrees(hi)), 0.1, float(np.degrees(v))
                )
                sl.on_update(lambda _e: self._on_slider())
                self.sliders.append(sl)
        self.robot_choice.value = s.robot_name or UPLOADED
        self.mount.value = s.mount
        self.base.set(s.T_world_base)
        self.view.set_q(s.q)
        self._update_readout()
        self._rebuild_part()

    def refresh_part(self) -> None:
        self._busy = True
        try:
            self._rebuild_part()
        finally:
            self._busy = False

    def _rebuild_part(self) -> None:
        if self.part_handle is not None:
            self.part_handle.remove()
            self.part_handle = None
        part = self.s.part
        self.part_pose.gizmo.visible = part is not None and self.part_pose.show.value
        if part is None:
            self.part_choice.value = "(none)"
            for fn in self.part_listeners:
                fn()
            return
        golden = Path(part.mesh_path).stem
        self.part_choice.value = (
            golden
            if part.mesh_path == f"assets/parts/{golden}.stl" and golden in GOLDEN_PARTS
            else "(uploaded mesh)"
        )
        self.part_units.value = part.units
        self.part_pose.set(part.T_world_part)
        self.part_handle = self.server.scene.add_mesh_trimesh(f"{PART_NODE}/mesh", self.s.part_mesh())
        self.part_handle.on_click(self._part_clicked)
        for fn in self.part_listeners:
            fn()

    def _part_clicked(self, event) -> None:
        if self.on_part_click is not None:
            self.on_part_click(event)
