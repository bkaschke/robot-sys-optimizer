"""Userframe export (v4 plan item 7): for every stop, the transform from the robot's base frame
to the part frame, i.e. the user frame a robot program at that stop would teach the points in.

The columns follow the loaded robot (``RobotSpec.frame_convention``) so the numbers can be typed
into its controller: UR arms ``x y z`` (mm) and the rotation vector ``rx ry rz`` (rad, as in a
UR pose ``p[...]``); FANUC-style arms ``X Y Z W P R`` (mm, deg; R = Rz(R) Ry(P) Rx(W)). Every
robot also gets the 3 x 3 rotation matrix, which any controller can convert.
"""

from __future__ import annotations

import csv
import io

import numpy as np
from scipy.spatial.transform import Rotation

CONVENTION_COLUMNS = {
    "ur": ("rx_rad", "ry_rad", "rz_rad"),
    "fanuc": ("w_deg", "p_deg", "r_deg"),
    "matrix": (),
}
MATRIX_COLUMNS = tuple(f"r{i}{j}" for i in range(1, 4) for j in range(1, 4))


def base_to_part(T_world_base: np.ndarray, T_world_part: np.ndarray) -> np.ndarray:
    """Pose of the part frame in the robot base frame (the user frame)."""
    return np.linalg.inv(np.asarray(T_world_base, float)) @ np.asarray(T_world_part, float)


def orientation(R: np.ndarray, convention: str) -> tuple[float, ...]:
    if convention == "ur":
        return tuple(float(v) for v in Rotation.from_matrix(R).as_rotvec())
    if convention == "fanuc":  # extrinsic x-y-z: R = Rz(r) Ry(p) Rx(w)
        return tuple(float(v) for v in Rotation.from_matrix(R).as_euler("xyz", degrees=True))
    return ()


def userframe_rows(
    sol, base, T_world_base0: np.ndarray, T_world_part: np.ndarray, convention: str
) -> list[dict]:
    """One row per stop of solution ``sol``."""
    from core.export import station_pose

    rows = []
    for k, st in enumerate(sol.stations):
        T = base_to_part(station_pose(base, st, T_world_base0), T_world_part)
        row = {
            "station": k + 1,
            "carrier": " ".join(f"{d}={v:.6f}" for d, v in st.carrier.items()),
            "x_mm": float(T[0, 3] * 1000),
            "y_mm": float(T[1, 3] * 1000),
            "z_mm": float(T[2, 3] * 1000),
        }
        row.update(zip(CONVENTION_COLUMNS[convention], orientation(T[:3, :3], convention), strict=True))
        row.update(zip(MATRIX_COLUMNS, (float(v) for v in T[:3, :3].ravel()), strict=True))
        rows.append(row)
    return rows


def userframe_csv(rows: list[dict], convention: str) -> str:
    cols = ["station", "carrier", "x_mm", "y_mm", "z_mm", *CONVENTION_COLUMNS[convention], *MATRIX_COLUMNS]
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return buf.getvalue()
