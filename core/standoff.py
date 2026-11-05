"""Floor-plane (2D) standoff and keep-out checks with shapely (plan § Standoff distances).

The part footprint is the convex hull of its vertices projected on the floor: a
conservative (never smaller) outline, cheap for any mesh. Distances are world-xy metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union

from core.schema import Base


def footprint(mesh: trimesh.Trimesh, T_world: np.ndarray) -> Polygon:
    v = mesh.vertices @ T_world[:3, :3].T + T_world[:3, 3]
    return MultiPoint(v[:, :2]).convex_hull


@dataclass
class ZoneCheck:
    ok: bool
    part_distance: float  # carrier footprint to part footprint
    penalty: float  # 0 at/after the preferred standoff, -> 1 when touching the minimum
    reason: str = ""


class Zones:
    def __init__(self, base: Base, part_fp: Polygon | None):
        so = base.standoff_m
        self.min = so.carrier_part_min
        self.pref = (
            so.carrier_part_preferred if so.carrier_part_preferred is not None else so.carrier_part_min
        )
        self.env = so.carrier_env_min
        self.part = part_fp
        self.keepouts = [box(*k.min_xy, *k.max_xy) for k in base.keepouts]
        self.keepout_zone = (
            unary_union([k.buffer(self.env) for k in self.keepouts]) if self.keepouts else None
        )

    def check(self, carrier_fp: Polygon) -> ZoneCheck:
        d = float(carrier_fp.distance(self.part)) if self.part is not None else float("inf")
        if self.keepout_zone is not None and carrier_fp.intersects(self.keepout_zone):
            return ZoneCheck(False, d, 1.0, "keep-out")
        if d < self.min - 1e-6:  # 1 um tolerance: mesh-vertex float noise at exactly the minimum
            return ZoneCheck(False, d, 1.0, "standoff")
        span = self.pref - self.min
        pen = 0.0 if d >= self.pref else (1.0 if span <= 0 else (self.pref - d) / span)
        return ZoneCheck(True, d, float(pen))

    def shell(self, distance: float) -> np.ndarray | None:
        """Outline (n, 2) of the part footprint grown by ``distance`` (for drawing)."""
        if self.part is None:
            return None
        g = self.part.buffer(distance)
        return np.asarray(g.exterior.coords)


def distance_3d(bodies, part_mesh: trimesh.Trimesh, T_world_part: np.ndarray) -> float:
    """Smallest 3D distance between carrier bodies (core.carrier.Body) and the part (Coal)."""
    import coal

    def bvh(m: trimesh.Trimesh):
        b = coal.BVHModelOBBRSS()
        b.beginModel(len(m.faces), len(m.vertices))
        b.addVertices(np.ascontiguousarray(m.vertices, dtype=float))
        b.addTriangles(np.ascontiguousarray(m.faces, dtype=np.int64))
        b.endModel()
        return b

    def tf(T: np.ndarray):
        return coal.Transform3s(np.asarray(T[:3, :3]), np.asarray(T[:3, 3]))

    part = bvh(part_mesh)
    best = float("inf")
    for body in bodies:
        req, res = coal.DistanceRequest(), coal.DistanceResult()
        d = coal.distance(bvh(body.mesh), tf(body.T), part, tf(T_world_part), req, res)
        best = min(best, float(d))
    return best


def carrier_ok(
    carrier, zones: Zones, v: dict, part_mesh: trimesh.Trimesh | None, T_world_part: np.ndarray | None
) -> ZoneCheck:
    """2D footprint check; a standoff failure is confirmed in 3D (carriage above the part,
    e.g. a gantry, is fine). Keep-outs are floor zones: 2D only."""
    fp = carrier.footprint(v)
    if fp is None:
        return ZoneCheck(True, float("inf"), 0.0)
    zc = zones.check(fp)
    if zc.ok or zc.reason != "standoff" or part_mesh is None:
        return zc
    d3 = distance_3d(carrier.moving_bodies(v) + carrier.static_bodies(), part_mesh, T_world_part)
    if d3 < zones.min - 1e-6:
        return ZoneCheck(False, d3, 1.0, "standoff")
    span = zones.pref - zones.min
    pen = 0.0 if d3 >= zones.pref else (1.0 if span <= 0 else (zones.pref - d3) / span)
    return ZoneCheck(True, d3, float(pen))
