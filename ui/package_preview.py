"""Package builder preview (ui.package_builder): the parts coloured by stage, the TCP or
robot-mount frames, under one scene node placed clear of the cell."""

from __future__ import annotations

import numpy as np

from core.carrier import primitive_mesh
from core.transforms import matrix_to_wxyz, pose_to_matrix

COLORS = {"tool": (90, 140, 220), "fixed": (150, 150, 150), "bridge": (200, 160, 60)}
COLORS.update(carriage=(70, 170, 120), chassis=(70, 170, 120), lift=(200, 110, 70))
ROOT = "/package_preview"


def draw(b) -> None:
    """Redraw the preview of builder ``b``'s package under ROOT (at *Preview at*)."""
    for h in b._handles:
        h.remove()
    b._handles = []
    b.root.visible = bool(b.show.value)
    T_root = b.root_pose()
    b.root.wxyz = matrix_to_wxyz(T_root[:3, :3])
    b.root.position = tuple(T_root[:3, 3])
    if not b.show.value:
        return
    sc = b.server.scene
    for k, part in enumerate(b.pkg.parts):
        try:
            m = primitive_mesh(b.primitive(part))
        except (OSError, ValueError):
            continue
        T = pose_to_matrix(part.pose)
        b._handles.append(
            sc.add_mesh_simple(
                f"{ROOT}/part_{k}",
                np.asarray(m.vertices, np.float32),
                np.asarray(m.faces, np.uint32),
                color=COLORS.get(part.stage, (160, 160, 160)),
                opacity=1.0 if k == b.sel else 0.75,
                wxyz=matrix_to_wxyz(T[:3, :3]),
                position=tuple(T[:3, 3]),
            )
        )
    frames = [(f"tcp_{t.name}", t.pose, 0.08) for t in b.pkg.tcps] if b.pkg.kind == "tool" else []
    if b.pkg.kind != "tool":
        frames.append(("robot_mount", b.pkg.mount, 0.2))
    for name, pose, size in frames:
        T = pose_to_matrix(pose)
        b._handles.append(
            sc.add_frame(
                f"{ROOT}/{name}",
                axes_length=size,
                axes_radius=size / 25,
                wxyz=matrix_to_wxyz(T[:3, :3]),
                position=tuple(T[:3, 3]),
            )
        )
