"""Reduced-speed zones: volumes where the robot moves slower, never obstacles.

A zone is a ``schema.Primitive`` with ``speed_pct`` set (Robot & cell -> *Reduced-speed
zones*). It does not collide with anything (core.stayout skips it); a move whose TCP ends
inside it runs at ``speed_pct`` % of the programmed speed, so it takes 100 / speed_pct times
as long. Where zones overlap the slowest one counts. Only the TCP position is tested (the
usual speed-monitoring point), not the arm's links.

``SpeedZone`` holds the zone mesh in world coordinates as plain arrays (picklable: the
planner settings carry it into the worker and the check processes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpeedZone:
    name: str
    vertices: np.ndarray  # (n, 3) world
    faces: np.ndarray  # (m, 3)
    share: float  # speed_pct / 100
    _mesh: object = field(default=None, repr=False, compare=False)

    def mesh(self):
        if self._mesh is None:
            import trimesh

            self._mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        return self._mesh

    def contains(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        inside = np.all((P >= lo - 1e-9) & (P <= hi + 1e-9), axis=1)
        if inside.any():
            idx = np.flatnonzero(inside)
            inside[idx] = self.mesh().contains(P[idx])
        return inside

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mesh"] = None  # rebuilt on demand
        return d


def speed_zones(prims, resolve=None) -> list[SpeedZone]:
    """The reduced-speed zones among ``prims`` (stay-out list of the project), world meshes."""
    from core.carrier import primitive_mesh
    from core.transforms import pose_to_matrix

    out = []
    for p in prims:
        if p.speed_pct is None:
            continue
        m = primitive_mesh(p, resolve)
        T = pose_to_matrix(p.pose)
        V = np.asarray(m.vertices, float) @ T[:3, :3].T + T[:3, 3]
        out.append(SpeedZone(p.name, V, np.asarray(m.faces, np.int64), float(p.speed_pct) / 100.0))
    return out


def share_at(zones: list[SpeedZone], P) -> np.ndarray:
    """Speed share (1 = full speed) at each world point of ``P`` (n, 3)."""
    P = np.atleast_2d(np.asarray(P, float))
    f = np.ones(len(P))
    for z in zones or ():
        f = np.where(z.contains(P), np.minimum(f, z.share), f)
    return f
