"""Target-point tools on a part mesh (part frame, meters). No viewer imports.

A *selection* is an array of face indices. Generators return ``(points, normals)``
arrays of shape (n, 3) with outward unit normals, ordered for processing (raster lines
run serpentine). Normals default to barycentric interpolation of area-weighted vertex
normals computed **from the selected faces only**, so a selection's border does not
blend in normals from neighbouring faces across a sharp edge.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import trimesh

from core.mesh_select import grow_region, pick_face

__all__ = [
    "grow_region",
    "pick_face",
    "select_by_normal",
    "select_in_screen_rect",
    "surface_normals",
    "projected_grid",
    "poisson_disk",
    "raster",
    "edge_paths",
    "parse_csv",
]

MAX_CSV_ROWS = 200_000


# --- selection ---------------------------------------------------------------------------


def select_by_normal(mesh: trimesh.Trimesh, direction, max_angle_deg: float) -> np.ndarray:
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    cos = mesh.face_normals @ d
    return np.flatnonzero(cos >= np.cos(np.radians(max_angle_deg)))


def select_in_screen_rect(
    mesh: trimesh.Trimesh,
    T_cam_mesh: np.ndarray,
    fov_y: float,
    aspect: float,
    screen_min,
    screen_max,
    front_facing: bool = True,
) -> np.ndarray:
    """Faces whose centroid projects inside a screen rectangle (0..1, OpenCV image coords).

    ``T_cam_mesh`` maps mesh coordinates into an OpenCV camera frame (x right, y down,
    z forward). Occlusion is not tested; ``front_facing`` drops faces turned away.
    """
    c = mesh.triangles_center @ T_cam_mesh[:3, :3].T + T_cam_mesh[:3, 3]
    z = c[:, 2]
    t = np.tan(fov_y / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 0.5 + 0.5 * c[:, 0] / (z * t * aspect)
        v = 0.5 + 0.5 * c[:, 1] / (z * t)
    (u0, v0), (u1, v1) = screen_min, screen_max
    ok = (z > 0) & (u >= min(u0, u1)) & (u <= max(u0, u1)) & (v >= min(v0, v1)) & (v <= max(v0, v1))
    if front_facing:
        n_cam = mesh.face_normals @ T_cam_mesh[:3, :3].T
        ok &= (n_cam * c).sum(axis=1) < 0  # normal points back toward the camera
    return np.flatnonzero(ok)


# --- normals -----------------------------------------------------------------------------


def _region_vertex_normals(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    vn = np.zeros((len(mesh.vertices), 3))
    weighted = mesh.face_normals[faces] * mesh.area_faces[faces, None]
    for k in range(3):
        np.add.at(vn, mesh.faces[faces, k], weighted)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, norm, out=np.zeros_like(vn), where=norm > 0)


def surface_normals(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    tri: np.ndarray,
    faces: np.ndarray | None = None,
    mode: str = "interpolated",
) -> np.ndarray:
    """Unit normals at ``points`` lying on triangles ``tri``.

    mode "interpolated": barycentric blend of region vertex normals; "face": flat normals.
    """
    if mode == "face":
        return mesh.face_normals[tri].copy()
    region = np.arange(len(mesh.faces)) if faces is None else np.asarray(faces)
    vn = _region_vertex_normals(mesh, region)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], points)
    n = np.einsum("ij,ijk->ik", bary, vn[mesh.faces[tri]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # guard against flipped/degenerate vertex normals: never disagree with the face
    bad = (n * mesh.face_normals[tri]).sum(axis=1) <= 0
    n[bad] = mesh.face_normals[tri][bad]
    return n


def _frame_from_normal(n: np.ndarray, u_hint=None) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (u, v) spanning the plane with normal n; u follows u_hint if given."""
    n = n / np.linalg.norm(n)
    a = (
        np.asarray(u_hint, float)
        if u_hint is not None
        else (np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0]))
    )
    u = a - (a @ n) * n
    if np.linalg.norm(u) < 1e-9:
        return _frame_from_normal(n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def mean_normal(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    n = (mesh.face_normals[faces] * mesh.area_faces[faces, None]).sum(axis=0)
    return n / np.linalg.norm(n)


# --- generators ------------------------------------------------------------------------------


def projected_grid(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    spacing: float,
    direction=None,
    u_hint=None,
    normals: str = "interpolated",
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of rays with ``spacing`` (m) cast along ``direction`` onto the selection.

    Default direction: against the selection's mean normal (looking straight at it).
    Only first hits on selected faces that face the ray are kept.
    """
    faces = np.asarray(faces)
    d = -mean_normal(mesh, faces) if direction is None else np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    u, v = _frame_from_normal(-d, u_hint)
    verts = mesh.vertices[np.unique(mesh.faces[faces])]
    pu, pv, pd = verts @ u, verts @ v, verts @ d
    us = np.arange(pu.min(), pu.max() + 1e-12, spacing)
    vs = np.arange(pv.min(), pv.max() + 1e-12, spacing)
    # centre the grid on the selection so leftover margin is symmetric
    us += (pu.max() - us[-1] - (us[0] - pu.min())) / 2
    vs += (pv.max() - vs[-1] - (vs[0] - pv.min())) / 2
    gu, gv = np.meshgrid(us, vs, indexing="ij")
    start = pd.min() - 0.01 - max(mesh.extents)
    origins = gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v + start * d
    locs, rays, tris = mesh.ray.intersects_location(
        origins, np.tile(d, (len(origins), 1)), multiple_hits=False
    )
    keep = np.isin(tris, faces) & ((mesh.face_normals[tris] @ d) < 0)
    order = np.argsort(rays[keep], kind="stable")
    pts, tri = locs[keep][order], tris[keep][order]
    return pts, surface_normals(mesh, pts, tri, faces, normals)


def poisson_disk(
    mesh: trimesh.Trimesh, faces: np.ndarray, spacing: float, seed: int = 0, normals: str = "interpolated"
) -> tuple[np.ndarray, np.ndarray]:
    """Even surface sampling: no two points closer than ``spacing`` (dart throwing)."""
    rng = np.random.default_rng(seed)
    faces = np.asarray(faces)
    area = mesh.area_faces[faces].sum()
    n_cand = int(min(2_000_000, max(100, 20 * area / spacing**2)))
    prob = mesh.area_faces[faces] / area
    tri = faces[rng.choice(len(faces), n_cand, p=prob)]
    r = rng.random((n_cand, 2))
    flip = r.sum(axis=1) > 1
    r[flip] = 1 - r[flip]
    t = mesh.triangles[tri]
    pts = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
    # dart throwing on a hash grid of cell size = spacing (pure-python floats: ~10x faster
    # than per-pair numpy for this sequential loop)
    r2 = spacing * spacing
    plist = pts.tolist()
    klist = [tuple(k) for k in np.floor(pts / spacing).astype(np.int64).tolist()]
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    cells: dict[tuple, list[tuple]] = {}
    keep: list[int] = []
    for i in range(n_cand):
        x, y, z = plist[i]
        kx, ky, kz = klist[i]
        ok = True
        for dx, dy, dz in offsets:
            for qx, qy, qz in cells.get((kx + dx, ky + dy, kz + dz), ()):
                if (qx - x) ** 2 + (qy - y) ** 2 + (qz - z) ** 2 < r2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            cells.setdefault(klist[i], []).append((x, y, z))
            keep.append(i)
    keep_arr = np.array(keep, dtype=int)
    p, tr = pts[keep_arr], tri[keep_arr]
    return p, surface_normals(mesh, p, tr, faces, normals)


def _chain(segments: np.ndarray, tol: float = 1e-7) -> list[np.ndarray]:
    """Join unordered line segments (n, 2, 3) into polylines."""
    if len(segments) == 0:
        return []
    keys = np.round(segments.reshape(-1, 3) / tol).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 2)
    pos = segments.reshape(-1, 3)
    coords = {int(i): pos[k] for k, i in enumerate(ids.reshape(-1))}
    adj: dict[int, list[int]] = {}
    for a, b in ids:
        if a != b:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
    used: set[tuple[int, int]] = set()
    lines = []
    starts = [n for n, nb in adj.items() if len(nb) == 1] + list(adj)
    for s in starts:
        for nxt in adj[s]:
            if (s, nxt) in used:
                continue
            path = [s]
            prev, cur = s, nxt
            used.update({(prev, cur), (cur, prev)})
            path.append(cur)
            while True:
                cand = [n for n in adj[cur] if (cur, n) not in used]
                if not cand:
                    break
                prev, cur = cur, cand[0]
                used.update({(prev, cur), (cur, prev)})
                path.append(cur)
            lines.append(np.array([coords[i] for i in path]))
    return lines


def _resample(line: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return line[:1]
    n = max(1, int(np.floor(s[-1] / step + 1e-9)))
    margin = (s[-1] - n * step) / 2  # centre the samples on the line
    targets = margin + step * np.arange(n + 1)
    return np.column_stack([np.interp(targets, s, line[:, k]) for k in range(3)])


def raster(
    mesh: trimesh.Trimesh,
    faces: np.ndarray,
    stepover: float,
    step: float,
    direction=None,
    normals: str = "interpolated",
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel section lines across the selection, sampled every ``step`` along the surface.

    Lines run along ``direction`` (projected into the selection's mean plane); successive
    lines are ``stepover`` apart and alternate direction (serpentine).
    """
    faces = np.asarray(faces)
    sub = mesh.submesh([faces], append=True)
    n = mean_normal(mesh, faces)
    u, v = _frame_from_normal(n, direction)  # lines along u, stepping along v
    pv = sub.vertices @ v
    heights = np.arange(pv.min() + stepover / 2, pv.max(), stepover)
    if len(heights) == 0:
        heights = np.array([(pv.min() + pv.max()) / 2])
    pts_all = []
    for k, h in enumerate(heights):
        segs = trimesh.intersections.mesh_plane(sub, v, v * h)  # (n, 2, 3) in 3D
        line_pts = []
        for line in _chain(np.asarray(segs)):
            if line[-1] @ u < line[0] @ u:
                line = line[::-1]
            line_pts.append(_resample(line, step))
        if not line_pts:
            continue
        row = np.vstack(sorted(line_pts, key=lambda a: a[0] @ u))
        pts_all.append(row if k % 2 == 0 else row[::-1])
    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts = np.vstack(pts_all)
    tri = _faces_of(mesh, pts, faces)
    return pts, surface_normals(mesh, pts, tri, faces, normals)


def _faces_of(mesh: trimesh.Trimesh, pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Selected face under each point (points are on the selection's surface)."""
    faces = np.asarray(faces)
    sub = mesh.submesh([faces], append=True)  # face i of sub is faces[i]
    _, _, t_sub = sub.nearest.on_surface(pts)
    return faces[t_sub]


def edge_paths(mesh: trimesh.Trimesh, faces: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points along the selection's boundary edges (deburring); normal = selection normal there."""
    faces = np.asarray(faces)
    edges = np.sort(mesh.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    pts_all = [_resample(line, step) for line in _chain(mesh.vertices[boundary])]
    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts = np.vstack(pts_all)
    return pts, surface_normals(mesh, pts, _faces_of(mesh, pts, faces), faces, "interpolated")


# --- CSV ---------------------------------------------------------------------------------------


def parse_csv(text: str, scale: float = 1.0) -> list[tuple[np.ndarray, np.ndarray, str | None]]:
    """Rows ``x,y,z,nx,ny,nz[,group]``; an optional header row is skipped.

    Positions are multiplied by ``scale`` (e.g. 1e-3 for mm). Raises ValueError with the
    offending line number on bad input.
    """
    rows = []
    reader = csv.reader(io.StringIO(text))
    for lineno, row in enumerate(reader, start=1):
        row = [c.strip() for c in row]
        if not row or all(not c for c in row) or row[0].startswith("#"):
            continue
        try:
            nums = [float(c) for c in row[:6]]
        except ValueError:
            if lineno == 1:
                continue  # header
            raise ValueError(f"line {lineno}: expected 6 numbers") from None
        if len(nums) != 6 or len(row) > 7 or not np.all(np.isfinite(nums)):
            raise ValueError(f"line {lineno}: expected x,y,z,nx,ny,nz[,group]")
        n = np.array(nums[3:])
        if np.linalg.norm(n) < 1e-9:
            raise ValueError(f"line {lineno}: zero normal")
        rows.append(
            (np.array(nums[:3]) * scale, n / np.linalg.norm(n), row[6] if len(row) == 7 and row[6] else None)
        )
        if len(rows) > MAX_CSV_ROWS:
            raise ValueError(f"more than {MAX_CSV_ROWS} rows")
    return rows


# --- target-set editing (schema.Targets in, new schema.Targets out) ------------------------


def as_arrays(targets) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(ids, positions, normals, groups) of a schema.Targets, in stored order."""
    pts = targets.points
    if not pts:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3)), []
    return (
        np.array([p.id for p in pts]),
        np.array([p.p for p in pts], float),
        np.array([p.n for p in pts], float),
        [p.group for p in pts],
    )


def with_points(targets, ids, P, N, groups):
    """Rebuild a schema.Targets from arrays (validates every point)."""
    from core.schema import TargetPoint, Targets

    points = [
        TargetPoint(id=int(i), p=tuple(map(float, p)), n=tuple(map(float, n)), group=g)
        for i, p, n, g in zip(ids, P, N, groups, strict=True)
    ]
    return Targets(groups=targets.groups, points=points)


def append_points(targets, P: np.ndarray, N: np.ndarray, group: str | list[str]):
    ids, P0, N0, G0 = as_arrays(targets)
    start = int(ids.max()) + 1 if len(ids) else 0
    groups = [group] * len(P) if isinstance(group, str) else list(group)
    return with_points(
        targets,
        np.concatenate([ids, start + np.arange(len(P))]),
        np.vstack([P0, P]),
        np.vstack([N0, N]),
        G0 + groups,
    )


def points_on_faces(mesh: trimesh.Trimesh, P: np.ndarray, faces: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Mask of points lying (within ``tol``) on any of ``faces``."""
    if len(P) == 0 or len(faces) == 0:
        return np.zeros(len(P), bool)
    sub = mesh.submesh([np.asarray(faces)], append=True)
    _, dist, _ = sub.nearest.on_surface(P)
    return dist <= tol
