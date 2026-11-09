"""Path CSV export (core.paths): one row per path sample."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CSV_COLUMNS = (
    "station",
    "kind",
    "carrier",
    "zone",
    "from_point",
    "to_point",
    "s",
    "tcp",
    "x_mm",
    "y_mm",
    "z_mm",
    "qw",
    "qx",
    "qy",
    "qz",
    "j1_deg",
    "j2_deg",
    "j3_deg",
    "j4_deg",
    "j5_deg",
    "j6_deg",
    "issue",
)


def path_csv(plan, T_world_part: np.ndarray) -> str:
    """One row per sample: TCP pose in the world (mm, quaternion wxyz), joints, issue."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    carriers = {st.station: st.carrier for st in plan.stations}
    for p in plan.samples:
        T = T_world_part @ p.T_part_tcp
        dofs = p.carrier if p.carrier is not None else carriers.get(p.station) or {}
        x, y, z, qw = Rotation.from_matrix(T[:3, :3]).as_quat()[[0, 1, 2, 3]]
        joints = [""] * 6 if p.q is None else [f"{v:.4f}" for v in np.degrees(p.q)]
        w.writerow(
            [
                p.station + 1,
                p.kind or "surface",
                " ".join(f"{d}={v:.6f}" for d, v in dofs.items()),
                "" if p.zone is None else p.zone + 1,
                p.a,
                p.b,
                f"{p.s:.4f}",
                p.tcp,
                *(f"{v * 1000:.3f}" for v in T[:3, 3]),
                f"{qw:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                *joints,
                p.issue,
            ]
        )
    return buf.getvalue()
