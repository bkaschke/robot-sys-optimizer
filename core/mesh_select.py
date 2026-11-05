"""Surface picking on triangle meshes: ray pick + CAD-like region growing.

Shared by contact-face picking on tools (M3) and surface selection on parts (M4).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import trimesh


def pick_face(mesh: trimesh.Trimesh, origin, direction) -> int | None:
    """Index of the first face hit by the ray (mesh frame), or None."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    locs, _, tris = mesh.ray.intersects_location([origin], [d], multiple_hits=True)
    if len(tris) == 0:
        return None
    dist = (locs - np.asarray(origin, float)) @ d
    return int(tris[np.argmin(np.where(dist >= 0, dist, np.inf))])


def grow_region(mesh: trimesh.Trimesh, seed: int, max_angle_deg: float = 15.0) -> np.ndarray:
    """Faces reachable from ``seed`` across edges whose dihedral angle is <= max_angle_deg.

    Smooth surfaces (a flat face, a cylinder wall made of narrow facets) grow as one
    region; sharp CAD edges stop the growth.
    """
    if not 0 <= seed < len(mesh.faces):
        raise IndexError(f"face {seed} out of range")
    adj = mesh.face_adjacency
    ok = mesh.face_adjacency_angles <= np.radians(max_angle_deg)
    neighbors: dict[int, list[int]] = {}
    for a, b in adj[ok]:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    seen = {seed}
    queue = deque([seed])
    while queue:
        f = queue.popleft()
        for g in neighbors.get(f, ()):
            if g not in seen:
                seen.add(g)
                queue.append(g)
    return np.array(sorted(seen), dtype=int)
