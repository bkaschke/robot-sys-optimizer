"""Split a face selection into the sides it covers (v7: points on opposite sides of a part).

The directional generators (rows/columns grid, lattice, projected grid, raster lines) look
at the selection from one direction, so a selection that wraps around a part (top and
bottom, or a whole closed solid) failed or left the far sides empty. ``facing_sides``
groups the selected faces into sides that each face one way:

1. smooth patches: connected faces whose shared edge bends less than ``smooth_deg``
   (a curved panel stays one patch, a box splits into its faces);
2. a patch that still faces several ways (e.g. a whole cylinder) is split by the nearest
   of the six part-frame axis directions, then into connected pieces again;
3. patches facing the same way (mean normals within ``merge_deg``) are merged, so separate
   islands on one side share one grid;
4. sides smaller than ``min_area`` (e.g. a thin chamfer strip) are left out.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SMOOTH_DEG = 40.0
MERGE_DEG = 30.0
CONSISTENT = 0.5  # |sum(a n)| / sum(a) below this: the patch faces several ways
AXES = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    """Connected-component label per node 0..n-1 given edge pairs (local indices)."""
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _mean_normal(mesh: trimesh.Trimesh, faces: np.ndarray) -> tuple[np.ndarray, float]:
    """(unit area-weighted mean normal, facing consistency 0..1)."""
    a = mesh.area_faces[faces]
    s = (mesh.face_normals[faces] * a[:, None]).sum(0)
    norm = float(np.linalg.norm(s))
    total = float(a.sum())
    if norm < 1e-12 or total <= 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return s / norm, norm / total


def facing_sides(
    mesh: trimesh.Trimesh,
    faces,
    smooth_deg: float = SMOOTH_DEG,
    merge_deg: float = MERGE_DEG,
    min_area: float = 0.0,
) -> list[np.ndarray]:
    """The selected faces grouped into sides (face index arrays), largest side first."""
    faces = np.unique(np.asarray(faces, int))
    if len(faces) == 0:
        return []
    local = -np.ones(len(mesh.faces), int)
    local[faces] = np.arange(len(faces))
    adj = mesh.face_adjacency
    both = (local[adj[:, 0]] >= 0) & (local[adj[:, 1]] >= 0)
    pairs = local[adj[both]]
    smooth = mesh.face_adjacency_angles[both] < np.radians(smooth_deg)
    lab = _components(len(faces), pairs[smooth])
    patches: list[np.ndarray] = []
    N = mesh.face_normals[faces]
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        _, spread = _mean_normal(mesh, faces[idx])
        if spread >= CONSISTENT:
            patches.append(idx)
            continue
        axis = np.argmax(N[idx] @ AXES.T, axis=1)  # split a wrap-around patch by axis direction
        for k in np.unique(axis):
            sub = idx[axis == k]
            keep = np.zeros(len(faces), bool)
            keep[sub] = True
            p2 = pairs[keep[pairs[:, 0]] & keep[pairs[:, 1]]]
            lab2 = _components(len(faces), p2)[sub]
            patches.extend(sub[lab2 == j] for j in np.unique(lab2))
    # merge patches facing the same way (largest first, greedy)
    info = sorted(
        ((idx, *_mean_normal(mesh, faces[idx]), float(mesh.area_faces[faces[idx]].sum())) for idx in patches),
        key=lambda t: -t[3],
    )
    cos = np.cos(np.radians(merge_deg))
    sides: list[list] = []  # [face idx list, normal, area]
    for idx, n, _, area in info:
        for side in sides:
            if n @ side[1] >= cos:
                side[0].append(idx)
                break
        else:
            sides.append([[idx], n, area])
    out = [np.sort(faces[np.concatenate(s[0])]) for s in sides]
    out = [f for f in out if mesh.area_faces[f].sum() >= min_area]
    return sorted(out, key=lambda f: -mesh.area_faces[f].sum())
