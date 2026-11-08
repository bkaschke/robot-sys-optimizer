"""Targets tab -> Display -> Preview tool at point: a ghost of the tool posed at a target."""

from __future__ import annotations

import numpy as np

from core import targets as tg
from core.transforms import make_T, matrix_to_wxyz

PREVIEW = "/cell/preview"


def draw_preview(tab) -> None:
    """Ghost of the tool posed at the chosen point (``tab`` = the TargetsTab)."""
    for k in [k for k in tab._handles if k.startswith("pv")]:
        tab._drop(k)
    ids, P, N, G = tg.as_arrays(tab.s.targets)
    tm = tab.s.tool_model
    if not tab.preview_on.value or tm is None or len(P) == 0 or tab.s.part is None:
        return
    i = int(np.clip(tab.preview_idx.value, 0, len(P) - 1))
    T_world_tcp = tab.s.part.T_world_part @ target_tcp_pose(P[i], N[i], standoff_of(tab, G[i]))
    tcp = tm.ee.tcps[0].name
    group = next((g for g in tab.s.targets.groups if g.name == G[i]), None)
    if group and group.tcps:
        tcp = group.tcps[0]
    T_world_flange = T_world_tcp @ np.linalg.inv(tm.T_flange_tcp(tcp))
    root = tab.server.scene.add_frame(
        PREVIEW,
        show_axes=False,
        wxyz=matrix_to_wxyz(T_world_flange[:3, :3]),
        position=T_world_flange[:3, 3],
    )
    tab._handles["pv_root"] = root
    ext = tm.extension_mesh()
    if ext is not None:
        tab._handles["pv_ext"] = tab.server.scene.add_mesh_simple(
            f"{PREVIEW}/ext", ext.vertices, ext.faces, color=(90, 140, 220), opacity=0.5
        )
    T_fh = tm.T_flange_head()
    for hl in tm.links:
        T = T_fh @ hl.T_head_link
        tab._handles[f"pv_{hl.name}"] = tab.server.scene.add_mesh_simple(
            f"{PREVIEW}/{hl.name}",
            hl.mesh.vertices,
            hl.mesh.faces,
            color=(200, 200, 210),
            opacity=0.5,
            wxyz=matrix_to_wxyz(T[:3, :3]),
            position=T[:3, 3],
        )
    tab._handles["pv_tcp"] = tab.server.scene.add_frame(
        f"{PREVIEW}/tcp",
        axes_length=0.05,
        axes_radius=0.002,
        wxyz=matrix_to_wxyz(tm.T_flange_tcp(tcp)[:3, :3]),
        position=tm.T_flange_tcp(tcp)[:3, 3],
    )


def standoff_of(tab, group: str) -> float:
    g = next((x for x in tab.s.targets.groups if x.name == group), None)
    if g is None:
        return 0.0
    return float(np.mean(g.standoff_m)) if isinstance(g.standoff_m, tuple) else float(g.standoff_m)


def target_tcp_pose(p: np.ndarray, n: np.ndarray, standoff: float = 0.0) -> np.ndarray:
    """TCP pose (part frame) that reaches target (p, n): origin at p + standoff*n and
    z = -n (tool points into the surface). Spin about the normal is chosen arbitrarily."""
    z = -np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), np.asarray(p) + standoff * np.asarray(n))
