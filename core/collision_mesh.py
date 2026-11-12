"""Low-detail collision geometry from full-detail visual meshes (2026-11-11).

The viewer always draws the full meshes; collision checks (core.collision) use a simpler copy
of every mesh that is not a primitive: the tool's head links, the part, external-axis bodies
and imported static objects. A collision check costs about the log of the triangle count per
pair, but building and placing large BVHs and the narrow phase near contact grow with the
triangle count, so CAD meshes of 100k+ triangles slow planning and paths several times over.

Modes (``Geom.mode``):
    auto     ``reduce`` to ``faces`` when the mesh has more than ``AUTO_FACES`` triangles,
             else the mesh as it is
    visual   the visual mesh as it is
    reduce   quadric decimation to about ``faces`` triangles (fast_simplification)
    hull     one convex hull around the whole mesh (at most about ``faces`` triangles)
    wrap     a closed shell ``wrap_mm`` outside the surface (a level set of the distance to
             the mesh, manifold3d), reduced to ``faces``: many small components (bolts,
             cables, brackets) and gaps become one simple, slightly larger body. Like an
             alpha wrap: conservative (never inside the visual surface by more than the
             reduction error), watertight
``keep`` (face ids of the visual mesh): these faces are copied exactly and only the rest is
simplified, e.g. the part surfaces the tool works on, where a few mm matter.

Results are cached on disk (``CACHE``) by mesh content and settings, so a large mesh is
simplified once per PC, also across worker processes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

MODES = ("auto", "visual", "reduce", "hull", "wrap")
MODE_LABELS = {
    "auto": "Auto (reduce large meshes)",
    "visual": "Same as visual",
    "reduce": "Reduce triangles",
    "hull": "Convex hull",
    "wrap": "Wrap (offset shell)",
}
AUTO_FACES = 20000
CACHE = Path(__file__).resolve().parents[1] / "cache" / "collision"
VERSION = 1  # bump when the algorithms change (cache keys)
WRAP_CELLS = 140  # most grid cells along the longest side of a wrap


@dataclass(frozen=True)
class Geom:
    mode: str = "auto"
    faces: int = 4000  # target triangle count (reduce, wrap, auto)
    wrap_mm: float = 5.0  # wrap offset outside the surface

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _digest(mesh: trimesh.Trimesh, extra: str) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mesh.vertices, np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, np.int64).tobytes())
    h.update(f"{VERSION}|{extra}".encode())
    return h.hexdigest()


def reduce(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """Quadric decimation to about ``faces`` triangles (the mesh itself when it is already
    small). Falls back to the convex hull without fast_simplification."""
    if len(mesh.faces) <= faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(faces))
    except (ImportError, ModuleNotFoundError):
        return hull(mesh)
    return out if len(out.faces) >= 4 else hull(mesh)


def hull(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        return mesh.convex_hull
    except Exception:  # flat or degenerate (qhull): its bounding box
        return trimesh.creation.box(bounds=mesh.bounds)


def _distances(mesh: trimesh.Trimesh, cell: float, G: np.ndarray) -> np.ndarray:
    """Unsigned distance from points ``G`` to the surface (sampled densely, cKDTree)."""
    from scipy.spatial import cKDTree

    spacing = cell / 3
    count = int(np.clip(mesh.area / spacing**2, 2000, 600000))
    pts, _ = trimesh.sample.sample_surface_even(mesh, count, seed=0)
    pts = np.vstack([pts, mesh.vertices])
    d, _ = cKDTree(pts).query(G, workers=-1)
    return d


def _grid(lo: np.ndarray, cell: float, n: np.ndarray, shift: float) -> np.ndarray:
    axes = [lo[i] + cell * (np.arange(n[i]) + shift) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def wrap(mesh: trimesh.Trimesh, offset: float, faces: int) -> trimesh.Trimesh:
    """A closed shell ``offset`` (mesh units) outside the surface, reduced to ``faces``. The
    grid has at most ``WRAP_CELLS`` cells along the longest side, and a cell is at most 0.7 x
    the offset (else the shell gets holes): small offsets on big meshes are raised to fit."""
    import manifold3d as mf

    ext = float(np.max(mesh.extents))
    offset = max(float(offset), 1e-9, ext / WRAP_CELLS / 0.7)
    cell = 0.7 * offset
    lo = mesh.bounds[0] - 2 * offset - cell
    n = np.ceil((mesh.bounds[1] + 2 * offset + cell - lo) / cell).astype(int)
    # manifold3d's level set samples a body-centred cubic grid: the nodes lo + i * cell and the
    # cell centres lo + (i - 0.5) * cell; both are computed up front (vectorized)
    n1, n2 = n + 1, n + 2
    G = np.vstack([_grid(lo, cell, n1, 0.0), _grid(lo, cell, n2, -0.5)])
    d = offset - _distances(mesh, cell, G)  # > 0 inside the shell
    k = int(np.prod(n1))
    A = d[:k].reshape(*n1).tolist()
    B = d[k:].reshape(*n2).tolist()
    x0, y0, z0 = (float(v) for v in lo)
    inv = 1.0 / cell
    m1 = [int(v) - 1 for v in n1]
    m2 = [int(v) - 1 for v in n2]

    def sdf(x: float, y: float, z: float) -> float:
        fx, fy, fz = (x - x0) * inv, (y - y0) * inv, (z - z0) * inv
        i, j, k = round(fx), round(fy), round(fz)
        if abs(fx - i) < 0.25 and abs(fy - j) < 0.25 and abs(fz - k) < 0.25:
            return A[min(max(i, 0), m1[0])][min(max(j, 0), m1[1])][min(max(k, 0), m1[2])]
        i, j, k = round(fx + 0.5), round(fy + 0.5), round(fz + 0.5)
        return B[min(max(i, 0), m2[0])][min(max(j, 0), m2[1])][min(max(k, 0), m2[2])]

    hi = lo + cell * n
    M = mf.Manifold.level_set(sdf, [*lo.tolist(), *hi.tolist()], cell, 0.0)
    m = M.to_mesh()
    raw = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    if len(raw.faces) == 0:
        return hull(mesh)
    # the offset of a closed surface also has an inner shell (a void): keep the outer ones
    outer = None
    for c in raw.split(only_watertight=False):
        if c.volume <= 0:
            continue
        piece = mf.Manifold(mf.Mesh(np.asarray(c.vertices, np.float32), np.asarray(c.faces, np.uint32)))
        outer = piece if outer is None else outer + piece
    M = outer if outer is not None and not outer.is_empty() else M
    tol = cell / 4  # simplify inside the manifold (stays watertight) until small enough
    S = M.simplify(tol)
    while S.num_tri() > faces and tol * 1.6 <= 0.5 * offset:  # never eats into the offset
        tol *= 1.6
        S = M.simplify(tol)
    m = S.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def _simplify(mesh: trimesh.Trimesh, g: Geom, scale: float) -> trimesh.Trimesh:
    mode = g.mode
    if mode == "auto":
        mode = "reduce" if len(mesh.faces) > AUTO_FACES else "visual"
    if mode == "visual":
        return mesh
    if mode == "reduce":
        return reduce(mesh, g.faces)
    if mode == "hull":  # a round body's hull is as fine as the body: reduce it, hull again
        h = hull(mesh)
        return hull(reduce(h, g.faces)) if len(h.faces) > g.faces else h
    if mode == "wrap":
        return wrap(mesh, g.wrap_mm / 1000 / scale, g.faces)
    raise ValueError(f"unknown collision mode {g.mode!r}")


_MEMO: dict = {}  # digest -> mesh (this process)


def collision_mesh(
    mesh: trimesh.Trimesh,
    g: Geom | None = None,
    keep=None,
    scale: float = 1.0,
    cache: bool = True,
    auto_faces: int | None = None,
) -> trimesh.Trimesh:
    """The collision copy of ``mesh`` (see the module docstring). ``scale``: metres per mesh
    unit (for ``wrap_mm``); ``keep``: face ids copied exactly; ``auto_faces``: the "auto"
    threshold (default ``AUTO_FACES``)."""
    g = g or Geom()
    auto = AUTO_FACES if auto_faces is None else int(auto_faces)
    if g.mode == "auto":
        if len(mesh.faces) <= auto:
            return mesh
        g = Geom("reduce", min(g.faces, auto), g.wrap_mm)
    keep = np.unique(np.asarray(keep if keep is not None else [], np.int64))
    keep = keep[(keep >= 0) & (keep < len(mesh.faces))]
    if g.mode == "visual":
        return mesh
    path = None
    digest = _digest(mesh, g.key() + f"|{scale}|" + ",".join(map(str, keep.tolist())))
    if digest in _MEMO:
        return _MEMO[digest]
    if cache:
        path = CACHE / f"{digest}.npz"
        if path.exists():
            try:
                with np.load(path) as z:
                    out = trimesh.Trimesh(z["v"], z["f"], process=False)
                _MEMO[digest] = out
                return out
            except Exception:
                pass
    if len(keep):
        rest = np.setdiff1d(np.arange(len(mesh.faces)), keep)
        kept = mesh.submesh([keep], append=True)
        out = kept
        if len(rest):
            out = trimesh.util.concatenate([kept, _simplify(mesh.submesh([rest], append=True), g, scale)])
    else:
        out = _simplify(mesh, g, scale)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, v=np.asarray(out.vertices, float), f=np.asarray(out.faces, np.int64))
        except OSError:
            pass
    if len(_MEMO) > 256:
        _MEMO.clear()
    _MEMO[digest] = out
    return out
