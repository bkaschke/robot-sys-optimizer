"""More target tools (part frame, meters; no viewer imports): mesh refinement, a row/column
lattice, brush patches, duplicate removal and nearest-point lookup.

Lattice: points sit at ``anchor + i*col*u + j*row*v`` in a plane perpendicular to the ray
direction d, with u the projected ``row_dir`` (points along a row are ``col`` apart, rows
are ``row`` apart). Rays are cast along d onto the faces. Because the lattice is anchored
(default: the part origin), separate brush dabs with the same direction land on one
consistent grid, and ``dedupe`` drops the points that overlapping dabs repeat.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import trimesh

from core.targets import _frame_from_normal, mean_normal, surface_normals

MAX_REFINED_FACES = 2_000_000


def refine_mesh(mesh: trimesh.Trimesh, max_edge: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Subdivide until no edge is longer than ``max_edge`` (m). The surface is unchanged
    (flat subdivision). Returns the new mesh and, per new face, the index of the original
    face it came from (to carry a face selection over)."""
    if max_edge <= 0:
        raise ValueError("max edge length must be positive")
    est = float((mesh.area_faces / (0.433 * max_edge**2)).clip(min=1).sum())  # equilateral area
    if est > MAX_REFINED_FACES:
        raise ValueError(f"about {est:,.0f} faces at {max_edge * 1000:.1f} mm; use a larger edge length")
    v, f, idx = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge, max_iter=20, return_index=True
    )
    return trimesh.Trimesh(v, f, process=False), np.asarray(idx)


def lattice_grid(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    row: float,
    col: float,
    direction=None,
    row_dir=(1.0, 0.0, 0.0),
    anchor=(0.0, 0.0, 0.0),
    normals: str = "interpolated",
) -> tuple[np.ndarray, np.ndarray]:
    """Row/column grid cast along ``direction`` (default: against the mean normal of the
    faces) onto ``faces``. Ordered row by row, serpentine."""
    if row <= 0 or col <= 0:
        raise ValueError("row and column spacing must be positive")
    faces = np.asarray(faces)
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    d = -mean_normal(mesh, faces) if direction is None else np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    u, v = _frame_from_normal(-d, row_dir)
    a = np.asarray(anchor, float)
    verts = mesh.vertices[np.unique(mesh.faces[faces])] - a
    pu, pv, pd = verts @ u, verts @ v, verts @ d
    i = np.arange(np.floor(pu.min() / col), np.ceil(pu.max() / col) + 1)
    j = np.arange(np.floor(pv.min() / row), np.ceil(pv.max() / row) + 1)
    if len(i) * len(j) > 4_000_000:
        raise ValueError("too many grid points: increase the spacing")
    gi, gj = np.meshgrid(i, j, indexing="xy")  # rows of constant j
    gi = np.where((gj - j[0])[:, :1] % 2 == 1, gi[:, ::-1], gi)  # serpentine
    start = pd.min() - 0.01 - max(mesh.extents)
    origins = a + (gi.reshape(-1, 1) * col) * u + (gj.reshape(-1, 1) * row) * v + start * d
    locs, rays, tris = mesh.ray.intersects_location(
        origins, np.tile(d, (len(origins), 1)), multiple_hits=False
    )
    keep = np.isin(tris, faces) & ((mesh.face_normals[tris] @ d) < 0)
    order = np.argsort(rays[keep], kind="stable")
    pts, tri = locs[keep][order], tris[keep][order]
    return pts, surface_normals(mesh, pts, tri, faces, normals)


def brush_faces(
    mesh: trimesh.Trimesh, face: int, center: np.ndarray, radius: float, max_angle_deg: float = 60.0
) -> np.ndarray:
    """Faces connected to ``face`` that come within ``radius`` of ``center`` and turn less
    than ``max_angle_deg`` from the clicked face (so a dab does not spill over an edge)."""
    c = np.asarray(center, float)
    n0 = mesh.face_normals[face]
    cos_lim = np.cos(np.radians(max_angle_deg))
    tri = mesh.triangles
    # distance from the centre to each triangle's bounding sphere (cheap, conservative)
    cen = tri.mean(axis=1)
    rad = np.linalg.norm(tri - cen[:, None, :], axis=2).max(axis=1)
    near = np.linalg.norm(cen - c, axis=1) - rad <= radius
    ok = near & (mesh.face_normals @ n0 >= cos_lim)
    adj = mesh.face_adjacency
    nbrs: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in adj[ok[adj[:, 0]] & ok[adj[:, 1]]]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen = {int(face)}
    queue = deque([int(face)])
    while queue:
        f = queue.popleft()
        for g in nbrs[f]:
            if g not in seen:
                seen.add(g)
                queue.append(g)
    return np.array(sorted(seen), dtype=int)


def brush_dab(
    mesh: trimesh.Trimesh,
    face: int,
    center: np.ndarray,
    radius: float,
    row: float,
    col: float,
    direction=None,
    row_dir=(1.0, 0.0, 0.0),
    max_angle_deg: float = 60.0,
    normals: str = "interpolated",
) -> tuple[np.ndarray, np.ndarray]:
    """Lattice points inside a sphere of ``radius`` around a clicked surface point. The
    default ray direction is against the clicked face normal."""
    faces = brush_faces(mesh, face, center, radius, max_angle_deg)
    d = -mesh.face_normals[face] if direction is None else direction
    P, N = lattice_grid(mesh, faces, row, col, d, row_dir, normals=normals)
    keep = np.linalg.norm(P - np.asarray(center), axis=1) <= radius + 1e-9
    return P[keep], N[keep]


def dedupe(
    P_new: np.ndarray, N_new: np.ndarray, P_old: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop new points within ``tol`` of an existing point (or of an earlier new point)."""
    if len(P_new) == 0:
        return P_new, N_new
    from scipy.spatial import cKDTree

    keep = np.ones(len(P_new), bool)
    if len(P_old):
        dist, _ = cKDTree(P_old).query(P_new, k=1)
        keep &= dist > tol
    tree = cKDTree(P_new)
    for i, j in sorted(tree.query_pairs(tol)):
        if keep[i]:
            keep[j] = False
    return P_new[keep], N_new[keep]


def nearest_point(P: np.ndarray, p: np.ndarray, max_dist: float) -> int | None:
    """Index of the point closest to ``p`` if it is within ``max_dist``."""
    if len(P) == 0:
        return None
    d = np.linalg.norm(P - np.asarray(p), axis=1)
    k = int(np.argmin(d))
    return k if d[k] <= max_dist else None
