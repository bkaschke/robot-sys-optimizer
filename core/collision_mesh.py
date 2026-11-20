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

"reduce" (and so "auto" on large meshes) is NOT conservative: the copy can lie inside the real
surface (a bossed plate lost bosses: up to 9 mm against a 10 mm clearance, review R11). Its
deviation (``deviation``: largest distance of the real surface from the copy) is measured when
the copy is made, cached with it, and returned by ``collision_mesh_info`` (``MeshInfo``);
``deviation_warning`` turns a scene's report into a warning. The collision checks are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from core import data_paths

MODES = ("auto", "visual", "reduce", "hull", "wrap")
MODE_LABELS = {
    "auto": "Auto (reduce large meshes)",
    "visual": "Same as visual",
    "reduce": "Reduce triangles",
    "hull": "Convex hull",
    "wrap": "Wrap (offset shell)",
}
AUTO_FACES = 20000
CACHE = data_paths.cache_dir("collision")
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
    return _reduce(mesh, faces)[0]


def _reduce(mesh: trimesh.Trimesh, faces: int) -> tuple[trimesh.Trimesh, str]:
    """``reduce`` and what it did: "visual" (already small), "reduce" or "hull" (fallback)."""
    if len(mesh.faces) <= faces:
        return mesh, "visual"
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(faces))
    except (ImportError, ModuleNotFoundError):
        return hull(mesh), "hull"
    return (out, "reduce") if len(out.faces) >= 4 else (hull(mesh), "hull")


def inflate(mesh: trimesh.Trimesh, d: float) -> trimesh.Trimesh:
    """The mesh pushed out by about ``d`` (mesh units): each vertex moves along its normal far
    enough that every face around it moves out by at least ``d`` (at most 3 d at sharp corners)."""
    if d <= 0:
        return mesh
    m = mesh.copy()
    m.merge_vertices()
    vn = np.asarray(m.vertex_normals)
    fn = np.asarray(m.face_normals)
    cos = np.ones(len(m.vertices))
    for k in range(3):  # the smallest normal . face-normal over the faces at each vertex
        np.minimum.at(cos, m.faces[:, k], np.einsum("ij,ij->i", vn[m.faces[:, k]], fn))
    step = d / np.clip(cos, 1 / 3, 1.0)
    m.vertices = np.asarray(m.vertices) + vn * step[:, None]
    return m


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


def _simplify(mesh: trimesh.Trimesh, g: Geom, scale: float) -> tuple[trimesh.Trimesh, str]:
    """(collision copy, the mode actually used: visual / reduce / hull / wrap)."""
    mode = g.mode
    if mode == "auto":
        mode = "reduce" if len(mesh.faces) > AUTO_FACES else "visual"
    if mode == "visual":
        return mesh, "visual"
    if mode == "reduce":
        return _reduce(mesh, g.faces)
    if mode == "hull":  # a round body's hull is as fine as the body: reduce it, hull again
        h = hull(mesh)
        return (hull(reduce(h, g.faces)) if len(h.faces) > g.faces else h), "hull"
    if mode == "wrap":
        return wrap(mesh, g.wrap_mm / 1000 / scale, g.faces), "wrap"
    raise ValueError(f"unknown collision mode {g.mode!r}")


# R11: how far the real surface lies from a collision copy. Only "reduce" (also what "auto" does
# to large meshes) can cut into the real surface; "hull" and "wrap" enclose it (conservative).
DEV_SAMPLES = 20000  # area-weighted surface samples of the original, besides its vertices
DEV_CHUNK = 50000  # closest-point queries per call (memory)


def deviation(original: trimesh.Trimesh, low: trimesh.Trimesh, samples: int = DEV_SAMPLES) -> float:
    """Largest distance (mesh units) from the original surface to the collision copy ``low``:
    exact closest points on ``low`` (trimesh.proximity) from every original vertex plus
    ``samples`` area-weighted surface samples (seeded). Real bumps between the samples can
    stick out a little further, so it is a close lower bound of the true maximum. About 1.6 s
    for a 128k-triangle part (86k points against 4000 triangles); done once per mesh and cached."""
    if len(low.faces) == 0 or len(original.faces) == 0:
        return 0.0
    pts = np.asarray(original.vertices, float)
    if samples > 0 and original.area > 0:
        s, _ = trimesh.sample.sample_surface(original, int(samples), seed=0)
        pts = np.vstack([pts, s])
    best = 0.0
    for i in range(0, len(pts), DEV_CHUNK):
        _, d, _ = trimesh.proximity.closest_point(low, pts[i : i + DEV_CHUNK])
        best = max(best, float(np.max(d)))
    return best


@dataclass(frozen=True)
class MeshInfo:
    """What a collision copy is, for reports (``CollisionScene.geometry_report``).

    ``mode``: the requested ``Geom.mode``; ``used``: what was done (visual / reduce / hull /
    wrap); ``kind``: "exact" (the mesh itself), "conservative" (hull / wrap: encloses the real
    surface) or "approximate" (reduce: may cut into it); ``deviation_m``: the measured largest
    distance of the real surface from the copy (``deviation``), 0 for exact and conservative
    copies; ``kept``: faces copied exactly (``keep``)."""

    mode: str
    used: str
    kind: str
    faces_before: int
    faces_after: int
    deviation_m: float = 0.0
    kept: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


KIND = {"visual": "exact", "reduce": "approximate", "hull": "conservative", "wrap": "conservative"}

_MEMO: dict = {}  # digest -> (mesh, used, deviation in mesh units) (this process)
_LOCK = threading.Lock()  # one simplification at a time (fast_simplification is not thread-safe)


def _info(mode: str, mesh, out, used: str, dev: float, scale: float, kept: int) -> MeshInfo:
    kind = KIND.get(used, "approximate")
    dm = float(dev) * float(scale) if kind == "approximate" else 0.0
    return MeshInfo(mode, used, kind, len(mesh.faces), len(out.faces), dm, kept)


def _save(path: Path, out: trimesh.Trimesh, used: str, dev: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            v=np.asarray(out.vertices, float),
            f=np.asarray(out.faces, np.int64),
            used=np.array(used),
            dev=np.array(float(dev)),
        )
    except OSError:
        pass


def collision_mesh_info(
    mesh: trimesh.Trimesh,
    g: Geom | None = None,
    keep=None,
    scale: float = 1.0,
    cache: bool = True,
    auto_faces: int | None = None,
) -> tuple[trimesh.Trimesh, MeshInfo]:
    """``collision_mesh`` plus what it is (``MeshInfo``). An approximate ("reduce") copy's
    deviation is measured when the copy is made and cached with it (disk and memory)."""
    g = g or Geom()
    requested = g.mode
    auto = AUTO_FACES if auto_faces is None else int(auto_faces)
    if g.mode == "auto":
        if len(mesh.faces) <= auto:
            return mesh, _info(requested, mesh, mesh, "visual", 0.0, scale, 0)
        g = Geom("reduce", min(g.faces, auto), g.wrap_mm)
    keep = np.unique(np.asarray(keep if keep is not None else [], np.int64))
    keep = keep[(keep >= 0) & (keep < len(mesh.faces))]
    if g.mode == "visual":
        return mesh, _info(requested, mesh, mesh, "visual", 0.0, scale, 0)
    path = None
    digest = _digest(mesh, g.key() + f"|{scale}|" + ",".join(map(str, keep.tolist())))
    hit = _MEMO.get(digest)
    if hit is None and cache:
        path = CACHE / f"{digest}.npz"
        if path.exists():
            try:
                with np.load(path) as z:
                    out = trimesh.Trimesh(z["v"], z["f"], process=False)
                    used = str(z["used"]) if "used" in z.files else g.mode
                    dev = float(z["dev"]) if "dev" in z.files else None
                if dev is None:  # a file from before R11: measure once, write it back
                    dev = deviation(mesh, out) if KIND.get(used) == "approximate" else 0.0
                    _save(path, out, used, dev)
                hit = (out, used, dev)
                _MEMO[digest] = hit
            except Exception:
                hit = None
    if hit is None:
        with _LOCK:
            hit = _MEMO.get(digest)  # another thread may just have made it
            if hit is None:
                used = "visual"
                if len(keep):
                    rest = np.setdiff1d(np.arange(len(mesh.faces)), keep)
                    out = mesh.submesh([keep], append=True)
                    if len(rest):
                        low, used = _simplify(mesh.submesh([rest], append=True), g, scale)
                        out = trimesh.util.concatenate([out, low])
                else:
                    out, used = _simplify(mesh, g, scale)
        if hit is None:
            dev = deviation(mesh, out) if KIND.get(used) == "approximate" else 0.0
            if path is not None:
                _save(path, out, used, dev)
            if len(_MEMO) > 256:
                _MEMO.clear()
            hit = (out, used, dev)
            _MEMO[digest] = hit
    out, used, dev = hit
    return out, _info(requested, mesh, out, used, dev, scale, len(keep))


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
    return collision_mesh_info(mesh, g, keep, scale, cache, auto_faces)[0]


# the Robot & cell tab's dropdowns (ui.collision_settings): component -> "Simplify <name>"
SETTING_NAMES = {"robot": "robot", "tool": "tool", "part": "part", "carrier": "external axis"}


def deviation_warning(report: dict, clearance_m: float) -> str | None:
    """A user-facing warning when an approximate collision copy in ``report`` (a
    ``CollisionScene.geometry_report()``: name -> dict with at least ``component``, ``kind``
    and ``deviation_m``) lies further than half the clearance from the real surface, else
    None. Names the worst body and its component's setting."""
    limit = 0.5 * float(clearance_m)
    bad = [
        (float(r.get("deviation_m", 0.0)), name, r)
        for name, r in report.items()
        if r.get("kind") == "approximate" and float(r.get("deviation_m", 0.0)) > limit
    ]
    if not bad:
        return None
    dev, name, r = max(bad, key=lambda b: b[0])
    comp = r.get("component", "part")
    what = "part" if comp == "part" else f"{comp} ({name})"
    more = f" ({len(bad) - 1} more bodies too)" if len(bad) > 1 else ""
    setting = SETTING_NAMES.get(comp, comp)
    return (
        f"Simplified {what} collision mesh deviates up to {dev * 1000:.1f} mm from the real surface "
        f"(clearance {clearance_m * 1000:g} mm){more}: collision results may be optimistic. "
        f"Set 'Simplify {setting}' (Robot & cell -> Collision geometry) to "
        f"'{MODE_LABELS['wrap']}' or '{MODE_LABELS['visual']}' for this {comp}."
    )
