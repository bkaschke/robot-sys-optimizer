"""Small SE(3) helpers on 4x4 numpy matrices. Pure numpy, no viewer imports.

rpy is URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
Quaternions are (w, x, y, z), as used by Viser.
"""

from __future__ import annotations

import numpy as np

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
    R = np.asarray(R, float)
    sp = -R[2, 0]
    if abs(sp) > 1 - 1e-12:  # gimbal lock: put all rotation into yaw
        pitch = np.copysign(np.pi / 2, sp)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        return 0.0, float(pitch), float(yaw)
    return (
        float(np.arctan2(R[2, 1], R[2, 2])),
        float(np.arcsin(np.clip(sp, -1, 1))),
        float(np.arctan2(R[1, 0], R[0, 0])),
    )


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
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q
