"""The floor grid sized to the cell: it covers everything placed in the study.

Covered: the part, the robot's reach around its base (and around every stop of the plans in the
Results list), the carrier (fixed bodies, and the moving bodies at both ends of every travel
axis), and the stay-out and reduced-speed zones; plus ``MARGIN`` all round. The grid is centred
on that area (snapped to whole metres), at least ``MIN_SIZE`` square, in whole metres, and is
only re-sent when its size or centre changes (``FloorGrid.update``; the app's watcher calls it
every second, and after a project loads).
"""

from __future__ import annotations

import numpy as np

MIN_SIZE = 10.0  # m
MAX_SIZE = 200.0  # m: a guard against a runaway mobile range
MARGIN = 1.0  # m around the cell
NODE = "/world/grid"


def _box_xy(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, float).reshape(-1, 3)
    return np.array([V[:, :2].min(axis=0), V[:, :2].max(axis=0)])


def cell_points(s) -> list[np.ndarray]:
    """(2, 2) xy boxes [min, max] of everything in the study (world, m)."""
    out = []
    if s.part is not None:
        try:
            m = s.part_mesh()
            T = s.part.T_world_part
            out.append(_box_xy(m.bounds[[0, 1]] @ T[:3, :3].T + T[:3, 3]))
            out.append(
                _box_xy(np.asarray(m.vertices)[:: max(1, len(m.vertices) // 2000)] @ T[:3, :3].T + T[:3, 3])
            )
        except Exception:  # a part file that doesn't load: nothing to cover
            pass
    reach = 0.0
    if s.robot is not None:
        reach = float(getattr(s.robot.spec, "reach_m", None) or 1.0)
    bases = [np.asarray(s.T_world_base, float)[:3, 3]]
    for sol in getattr(s, "results", []) or []:
        bases += [np.asarray(st.xyz, float) for st in sol.stations]
    for p in bases:
        out.append(np.array([p[:2] - reach, p[:2] + reach]))
    try:
        c = s.carrier() if s.base.mode == "movable" else None
    except Exception:
        c = None
    if c is not None:
        for b in c.static_bodies():
            out.append(_box_xy(np.asarray(b.mesh.bounds) @ b.T[:3, :3].T + b.T[:3, 3]))
        if len(c.dofs):
            lo, hi = (
                np.clip(c.bounds[:, 0], -MAX_SIZE, MAX_SIZE),
                np.clip(c.bounds[:, 1], -MAX_SIZE, MAX_SIZE),
            )
            for ends in (lo, hi):
                v = dict(zip(c.dofs, (float(x) for x in ends), strict=True))
                try:
                    for b in c.moving_bodies(v):
                        out.append(_box_xy(np.asarray(b.mesh.bounds) @ b.T[:3, :3].T + b.T[:3, 3]))
                    T = c.T_world_base(v) if hasattr(c, "T_world_base") else None
                    if T is not None:
                        p = np.asarray(T)[:3, 3]
                        out.append(np.array([p[:2] - reach, p[:2] + reach]))
                except Exception:
                    continue
    from core.carrier import primitive_mesh
    from core.transforms import pose_to_matrix

    for z in getattr(s, "stayouts", []) or []:
        try:
            m = primitive_mesh(z, s.resolve)
        except (ValueError, OSError):
            continue
        T = pose_to_matrix(z.pose)
        out.append(_box_xy(np.asarray(m.bounds) @ T[:3, :3].T + T[:3, 3]))
    return out


def grid_layout(boxes: list[np.ndarray]) -> tuple[tuple[float, float], float, float]:
    """(centre xy, width, height) of the grid covering ``boxes`` + MARGIN, whole metres."""
    if not boxes:
        return (0.0, 0.0), MIN_SIZE, MIN_SIZE
    B = np.array(boxes)
    lo, hi = B[:, 0].min(axis=0) - MARGIN, B[:, 1].max(axis=0) + MARGIN
    c = np.round((lo + hi) / 2)
    half = np.maximum(np.abs(hi - c), np.abs(lo - c))
    size = np.clip(np.ceil(2 * half), MIN_SIZE, MAX_SIZE)
    return (float(c[0]), float(c[1])), float(size[0]), float(size[1])


class FloorGrid:
    def __init__(self, server):
        self.server = server
        self.dark = False
        self.layout = ((0.0, 0.0), MIN_SIZE, MIN_SIZE)
        self._sent = None

    def draw(self, dark: bool | None = None) -> None:
        if dark is not None:
            self.dark = dark
        key = (self.layout, self.dark)
        if key == self._sent:
            return
        (cx, cy), w, h = self.layout
        cell, section = ((80, 84, 92), (120, 126, 138)) if self.dark else ((200, 200, 200), (140, 140, 140))
        self.server.scene.add_grid(
            NODE,
            width=w,
            height=h,
            cell_size=0.1,
            section_size=1.0,
            cell_color=cell,
            section_color=section,
            position=(cx, cy, 0.0),
        )
        self._sent = key

    def update(self, s) -> tuple:
        """Fit the grid to the session's cell; re-send it only when that changed."""
        try:
            self.layout = grid_layout(cell_points(s))
        except Exception:  # mid-edit state: keep the last grid
            return self.layout
        self.draw()
        return self.layout
