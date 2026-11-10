"""Robot display in Viser: URDF meshes under a base frame, driven by joint values.

Scene tree:  <parent> (base_link pose in world)
               └─ <parent>/urdf_root (URDF root link, offset so base_link lands on <parent>)
                    └─ ViserUrdf link frames ...
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pinocchio as pin
import viser
from viser.extras import ViserUrdf

from core.robot import Robot
from core.transforms import make_T, matrix_to_wxyz, wxyz_to_matrix


def _handle_T(h) -> np.ndarray:
    return make_T(wxyz_to_matrix(h.wxyz), h.position)


class RobotView:
    def __init__(self, server: viser.ViserServer, robot: Robot, urdf_path: Path, parent: str):
        self.robot = robot
        self.parent = parent
        # URDF root -> base_link offset (identity for the UR files)
        pin.framesForwardKinematics(robot.model, robot.data, pin.neutral(robot.model))
        T_root_base = np.asarray(robot.data.oMf[robot.base_frame_id].homogeneous)
        T_base_root = np.linalg.inv(T_root_base)
        self.root = server.scene.add_frame(
            f"{parent}/urdf_root",
            show_axes=False,
            wxyz=matrix_to_wxyz(T_base_root[:3, :3]),
            position=T_base_root[:3, 3],
        )
        self.urdf = ViserUrdf(server, Path(urdf_path), root_node_name=f"{parent}/urdf_root")
        names = list(self.urdf.get_actuated_joint_names())
        missing = set(robot.joint_names) - set(names)
        if missing:
            raise ValueError(f"viewer and model disagree on joints: {sorted(missing)}")
        self._order = [names.index(n) for n in robot.joint_names]  # pinocchio -> viser order
        self._viser_names = names
        n_joints = len(self.urdf._urdf.joint_map)
        self._link_frame = {
            j.child: (j.parent, h)
            for j, h in zip(
                self.urdf._joint_map_values[:n_joints], self.urdf._joint_frames[:n_joints], strict=True
            )
        }

    def joint_frame(self, joint: str):
        """The scene frame of URDF joint ``joint`` (its child link's frame, moves with q)."""
        n_joints = len(self.urdf._urdf.joint_map)
        for j, h in zip(
            self.urdf._joint_map_values[:n_joints], self.urdf._joint_frames[:n_joints], strict=True
        ):
            if j.name == joint:
                return h
        return None

    def set_q(self, q: np.ndarray) -> None:
        cfg = np.zeros(len(self._viser_names))
        cfg[self._order] = q
        self.urdf.update_cfg(cfg)

    def link_pose_in_parent(self, link: str) -> np.ndarray:
        """Pose of ``link`` in the <parent> frame, composed from the scene handles."""
        T = np.eye(4)
        while link in self._link_frame:
            parent, h = self._link_frame[link]
            T = _handle_T(h) @ T
            link = parent
        return _handle_T(self.root) @ T

    def remove(self) -> None:
        quiet_remove(self.urdf)
        quiet_remove(self.root)


def quiet_remove(handle) -> None:
    """Remove a scene handle without Viser's "Attempted to remove already removed node"
    warnings: ViserUrdf removes child nodes after their parent frame is already gone,
    which is harmless but floods the console on every project load."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attempted to remove already removed node")
        handle.remove()
