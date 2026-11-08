"""Surface selection painting and the rows/columns surface grid (v4 plan item 1), after the
sand-optimizer's ``selection.ts`` / ``surfaceGrid.ts``. Part frame, meters; no viewer imports.

Surface grid: the grid lies in the plane perpendicular to the selection's mean facing
direction n. Rows run along the part X axis projected onto that plane (Y if X is nearly
parallel to n), turned by ``angle_deg``; points along a row are ``col`` apart and rows are
``row`` apart, centred on the selection's extent. Each node is cast along -n onto the
selected faces (outermost hit). Points closer than ``inset`` to the selection's boundary are
dropped. Normals: smoothed over the selection, flipped to agree with n.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from core.targets import surface_normals


@dataclass
class GridInfo:
    rows: int = 0
    cols: int = 0
    area_m2: float = 0.0
    dropped_at_edge: int = 0
    spread: float = 1.0  # |sum(n * area)| / area: 1 = flat, -> 0 = folded
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    warnings: list[str] = field(default_factory=list)


def grid_frame(n: np.ndarray, angle_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Row direction u and row-to-row direction v in the plane perpendicular to n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    px = np.array([1.0, 0, 0]) - n[0] * n
    if np.linalg.norm(px) <= 0.3:
        px = np.array([0, 1.0, 0]) - n[1] * n
    u0 = px / np.linalg.norm(px)
    v0 = np.cross(n, u0)
    a = np.radians(angle_deg)
    u = np.cos(a) * u0 + np.sin(a) * v0
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L2 = (ab * ab).sum(-1)
    t = np.clip(((p - a) * ab).sum(-1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    return np.linalg.norm(a + ab * t[..., None] - p, axis=-1)


def boundary_edges(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    """(E, 2, 3) edges used by exactly one selected face."""
    e = np.sort(mesh.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    return mesh.vertices[uniq[count == 1]]


def near_boundary(P: np.ndarray, edges: np.ndarray, inset: float) -> np.ndarray:
    """Mask of points closer than ``inset`` to any boundary edge."""
    out = np.zeros(len(P), bool)
    if inset <= 0 or len(P) == 0 or len(edges) == 0:
        return out
    from scipy.spatial import cKDTree

    mid = edges.mean(axis=1)
    half = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).max() / 2
    for i, cand in enumerate(cKDTree(mid).query_ball_point(P, inset + half)):
        if cand:
            c = np.asarray(cand)
            out[i] = bool((_seg_dist(P[i], edges[c, 0], edges[c, 1]) < inset - 1e-12).any())
    return out


def surface_grid(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    row: float,
    col: float,
    inset: float = 0.0,
    angle_deg: float = 0.0,
    normals: str = "interpolated",
) -> tuple[np.ndarray, np.ndarray, GridInfo]:
    """Rows/columns grid on the selected faces. Points ordered row by row."""
    faces = np.unique(np.asarray(faces, int))
    if len(faces) == 0:
        raise ValueError("No surface selected: paint or fill a region of the part first.")
    if row <= 0 or col <= 0:
        raise ValueError("Row and column spacing must be greater than 0.")
    if inset < 0:
        raise ValueError("Edge inset must be 0 or more.")
    fa = mesh.area_faces[faces]
    mean = (mesh.face_normals[faces] * fa[:, None]).sum(0)
    area = float(fa.sum())
    if np.linalg.norm(mean) < 1e-9 * max(area, 1e-12):
        raise ValueError(
            "The selected faces point in opposite directions (e.g. a whole closed solid). "
            "Select one side of the part."
        )
    n = mean / np.linalg.norm(mean)
    info = GridInfo(area_m2=area, normal=n, spread=float(np.linalg.norm(mean) / area))
    if info.spread < 0.5:
        info.warnings.append(
            f"The selection is strongly curved or folded (facing consistency {info.spread:.0%}): "
            "points are projected along one direction, so steep areas get sparse points. "
            "Select one side at a time for best results."
        )
    u, v = grid_frame(n, angle_deg)
    V = mesh.vertices[np.unique(mesh.faces[faces])]
    a, b, h = V @ u, V @ v, V @ n
    cols = int(np.floor((a.max() - a.min()) / col + 1e-9)) + 1
    rows = int(np.floor((b.max() - b.min()) / row + 1e-9)) + 1
    if rows * cols > 4_000_000:
        raise ValueError("Too many grid points: increase the spacing.")
    a0 = (a.min() + a.max()) / 2 - (cols - 1) * col / 2
    b0 = (b.min() + b.max()) / 2 - (rows - 1) * row / 2
    jj, ii = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")  # row-major
    lift = h.max() + 0.01 + float(mesh.scale) * 0.01
    origins = (a0 + ii.reshape(-1, 1) * col) * u + (b0 + jj.reshape(-1, 1) * row) * v + lift * n
    sub = mesh.submesh([faces], append=True)
    locs, ray, tri = sub.ray.intersects_location(origins, np.tile(-n, (len(origins), 1)), multiple_hits=False)
    order = np.argsort(ray, kind="stable")
    P, ray, tri = locs[order], ray[order], faces[tri[order]]
    drop = near_boundary(P, boundary_edges(mesh, faces), inset)
    info.dropped_at_edge = int(drop.sum())
    P, ray, tri = P[~drop], ray[~drop], tri[~drop]
    N = surface_normals(mesh, P, tri, faces, normals) if len(P) else np.zeros((0, 3))
    N[(N @ n) < 0] *= -1  # agree with the facing direction (inconsistent winding)
    info.rows = len(np.unique(ray // cols))
    info.cols = cols
    return P, N, info


class FacePainter:
    """Face centroids in a KD-tree (brush) and connected components (full surface)."""

    def __init__(self, mesh: trimesh.Trimesh):
        from scipy.spatial import cKDTree

        self.mesh = mesh
        tri = mesh.triangles
        self.tree = cKDTree(mesh.triangles_center)
        self.reach = float(np.linalg.norm(tri - mesh.triangles_center[:, None], axis=2).max())
        self._labels = None

    def brush(self, center, normal, radius: float, max_angle_deg: float = 90.0) -> np.ndarray:
        """Faces that come within ``radius`` of ``center`` and whose normal is within
        ``max_angle_deg`` of ``normal`` (keeps the brush off the back of thin parts). A coarse
        triangle that the brush touches is taken whole."""
        c = np.asarray(center, float)
        idx = np.asarray(self.tree.query_ball_point(c, radius + self.reach), int)
        if len(idx) == 0:
            return idx
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        ok = self.mesh.face_normals[idx] @ n >= np.cos(np.radians(max_angle_deg)) - 1e-9
        idx = idx[ok]
        if len(idx):
            q = trimesh.triangles.closest_point(self.mesh.triangles[idx], np.tile(c, (len(idx), 1)))
            idx = idx[np.linalg.norm(q - c, axis=1) <= radius]
        return np.sort(idx)

    def stroke(self, points, normals, radius: float, max_angle_deg: float = 90.0) -> np.ndarray:
        out = [self.brush(p, n, radius, max_angle_deg) for p, n in zip(points, normals, strict=True)]
        return np.unique(np.concatenate(out)) if out else np.zeros(0, int)

    def component(self, face: int) -> np.ndarray:
        """Every face of the connected body containing ``face``."""
        if self._labels is None:
            self._labels = trimesh.graph.connected_component_labels(
                self.mesh.face_adjacency, node_count=len(self.mesh.faces)
            )
        return np.flatnonzero(self._labels == self._labels[face])

    def project(self, p) -> tuple[np.ndarray, np.ndarray, int]:
        """Closest surface point, its face normal and face index."""
        c, _, tri = trimesh.proximity.closest_point(self.mesh, [np.asarray(p, float)])
        return c[0], self.mesh.face_normals[int(tri[0])], int(tri[0])
