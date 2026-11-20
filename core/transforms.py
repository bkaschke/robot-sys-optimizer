"""Small SE(3) helpers on 4x4 numpy matrices. No viewer imports.

rpy is URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (scipy ``"xyz"``).
Quaternions are (w, x, y, z), as used by Viser.

v12 #16: the conversions from a matrix and the quaternion ones use
``scipy.spatial.transform.Rotation``. ``rpy_to_matrix`` stays the closed form: it is on hot paths
(a carrier pose per planner candidate, the tool frames) where scipy is about 7x slower (6 vs 43 us
per call, measured). ``matrix_to_rpy`` keeps its gimbal-lock convention (all rotation into yaw,
roll 0; scipy would put it into roll): saved project poses round-trip through it.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from core.schema import Pose


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    R = np.array(R, float)
    sp = -R[2, 0]
    if abs(sp) > 1 - 1e-12:  # gimbal lock: put all rotation into yaw
        pitch = np.copysign(np.pi / 2, sp)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        return 0.0, float(pitch), float(yaw)
    r, p, y = Rotation.from_matrix(R).as_euler("xyz")
    return float(r), float(p), float(y)


def make_T(R: np.ndarray | None = None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def pose_to_matrix(pose: Pose) -> np.ndarray:
    return make_T(rpy_to_matrix(pose.rpy), pose.xyz)


def matrix_to_pose(T: np.ndarray) -> Pose:
    return Pose(xyz=tuple(float(v) for v in T[:3, 3]), rpy=matrix_to_rpy(T[:3, :3]))


def wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()  # normalises


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) with w >= 0."""
    x, y, z, w = Rotation.from_matrix(np.array(R, float)).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q
