"""Stay-out zones (v3 feedback item 5): world-fixed boxes, cylinders and STL meshes that the
arm, the tool and the carrier must keep clear of.

A zone is a ``schema.Primitive`` (its ``attach`` and ``kg`` are ignored) whose pose is in the
world frame. The arm and tool avoid them through ``CollisionScene(obstacles=...)`` (part
clearance); the carrier through ``carrier_clear`` (keep-out clearance), with a cheap 2D
footprint test first and the exact 3D distance only for near misses (a zone hanging above the
floor does not block a carrier passing under it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon

from core.carrier import primitive_mesh, safe_name
from core.schema import Primitive
from core.transforms import pose_to_matrix


@dataclass
class Obstacle:
    name: str
    mesh: trimesh.Trimesh  # in the zone frame
    T: np.ndarray  # world pose of the zone frame
    footprint: Polygon  # world floor projection (convex hull)


def obstacles(zones: list[Primitive], resolve=None) -> list[Obstacle]:
    out = []
    names: set[str] = set()
    for i, p in enumerate(zones):
        m = primitive_mesh(p, resolve)
        T = pose_to_matrix(p.pose)
        V = m.vertices @ T[:3, :3].T + T[:3, 3]
        name = safe_name(p.name or f"zone{i + 1}")
        while name in names:
            name += "_"
        names.add(name)
        out.append(Obstacle(name, m, T, MultiPoint(V[:, :2]).convex_hull))
    return out


def scene_entries(obs: list[Obstacle]) -> list[tuple]:
    """(name, mesh, T_world) for ``CollisionScene(obstacles=...)``."""
    return [(o.name, o.mesh, o.T) for o in obs]


def carrier_clear(bodies, fp: Polygon | None, obs: list[Obstacle], clearance: float) -> tuple[bool, str]:
    """Carrier bodies (core.carrier.Body, world poses) and their floor footprint against the
    zones: ok when the footprints are ``clearance`` apart, else when the 3D distance is."""
    from core.standoff import distance_3d

    for o in obs:
        if fp is not None and fp.distance(o.footprint) >= clearance - 1e-9:
            continue
        if distance_3d(bodies, o.mesh, o.T) < clearance - 1e-6:
            return False, f"stay-out zone {o.name}"
    return True, ""
