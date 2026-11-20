"""Tool-surface detection, part 1b: features per smooth region (T1).

``region_features(mesh, labels, up)`` -> ``list[RegionFeatures]`` (index = region label).
Every feature is curvature-sign agnostic except ``convex_frac`` / ``concave_frac`` (inside a
region, informational only) and the boundary ``kind`` (convex break vs concave / tangent
joint), which the steep-wall rule in T2 needs. ``region_graph`` gives the adjacency graph.
No viser / ui imports; SI units (lengths m, areas m^2, angles in degrees in the outputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.tool_surface import edge_table, weighted_quantile

UP_DOT = 0.3  # n.up above this counts as up-facing (frac_up)
FLAT_DEG = 1.0  # internal edges flatter than this are neither convex nor concave
TANGENT_DEG = 8.0  # a region boundary with a mean dihedral below this is "tangent"
FRONT_DOT = 0.05  # a sample faces a view direction when n.d is above this
CONE_RINGS = ((27.5, 6), (55.0, 8))  # (angle from up in deg, directions): + up = 15 rays


@dataclass
class Boundary:
    """The shared boundary of a region with one neighbour region."""

    neighbor: int
    length: float  # m
    angle_deg: float  # length-weighted mean dihedral across it
    max_angle_deg: float
    convex_frac: float  # length share of convex edges (the rest is concave)
    kind: str  # "convex" | "concave" | "tangent"


@dataclass
class RegionFeatures:
    label: int
    n_faces: int
    area: float
    area_frac: float
    mean_normal: tuple  # area-weighted, unit (zeros if it cancels out)
    mean_up: float  # area-weighted mean n.up
    frac_up: float  # area share with n.up > UP_DOT
    height_mean: float  # 0 = part bottom, 1 = part top, along up
    height_min: float
    height_max: float
    smooth_mean_deg: float  # area-weighted dihedral between adjacent faces inside
    smooth_p95_deg: float
    convex_frac: float  # informational only: never use for the decision
    concave_frac: float
    visible_frac: float  # area share visible from at least one cone direction
    visible_up_frac: float  # ... from straight up
    visible_mean: float  # mean over cone directions of the visible share
    n_vis_samples: int
    boundary_length: float  # m: neighbours + open + non-manifold
    sharp_frac: float  # share with dihedral > feature angle (non-manifold counts as sharp)
    open_frac: float  # share that is open mesh boundary
    face_density: float  # faces per m^2
    n_boundary_loops: int
    n_holes: int  # boundary loops other than the longest, each < 25 % of the boundary
    neighbors: dict = field(default_factory=dict)  # neighbour label -> Boundary
    faces: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)

    def summary(self) -> str:
        return (
            f"r{self.label}: {self.area_frac:5.1%} area, {self.n_faces} faces, up {self.mean_up:+.2f}"
            f" ({self.frac_up:.0%}), h {self.height_mean:.2f}, vis {self.visible_frac:.0%}"
            f"/{self.visible_up_frac:.0%}, smooth {self.smooth_mean_deg:.1f}/{self.smooth_p95_deg:.1f} deg,"
            f" sharp {self.sharp_frac:.0%}, open {self.open_frac:.0%}, holes {self.n_holes},"
            f" nbrs {len(self.neighbors)}"
        )


def region_graph(features) -> dict:
    """``{label: {neighbour: Boundary}}``, symmetric."""
    return {f.label: dict(f.neighbors) for f in features}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("up must be a non-zero vector")
    return v / n


def view_directions(up, rings=CONE_RINGS) -> np.ndarray:
    """Unit directions from the surface toward the viewer: up plus rings around it."""
    u = _unit(up)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    out = [u]
    for ri, (ang, n) in enumerate(rings):
        t = np.radians(ang)
        for k in range(n):
            phi = 2 * np.pi * (k + 0.5 * (ri % 2)) / n  # odd rings staggered
            out.append(np.cos(t) * u + np.sin(t) * (np.cos(phi) * e1 + np.sin(phi) * e2))
    return np.array(out)


def _sample(labels, area_faces, k, budget, per_region, rng):
    """Area-weighted face samples per region: (face index, region) arrays."""
    area = np.bincount(labels, weights=area_faces, minlength=k)
    nf = np.bincount(labels, minlength=k)
    tot = area.sum()
    lo = 8
    n_r = np.minimum(per_region, np.maximum(np.round(budget * area / max(tot, 1e-300)), lo))
    n_r = np.minimum(n_r, np.maximum(nf, 1) * 4)
    if n_r.sum() > 1.5 * budget:  # very many regions: drop the minimum
        n_r = np.minimum(per_region, np.maximum(np.round(budget * area / max(tot, 1e-300)), 1))
    n_r = np.where(area > 0, n_r, 0).astype(np.int64)
    order = np.argsort(labels, kind="stable")
    cum = np.cumsum(area_faces[order])
    start = np.concatenate([[0], np.cumsum(nf)])[:-1]
    base = np.where(start > 0, cum[np.maximum(start - 1, 0)], 0.0)
    reg = np.repeat(np.arange(k), n_r)
    u = base[reg] + rng.random(len(reg)) * area[reg]
    idx = np.searchsorted(cum, u, side="right")
    idx = np.clip(idx, start[reg], start[reg] + nf[reg] - 1)
    return order[idx], reg


def visibility(mesh, labels, up=(0, 0, 1), budget=20000, per_region=2000, rings=CONE_RINGS, seed=0):
    """Per region: (visible_frac, visible_up_frac, visible_mean, n_samples).

    A sample point on face f is visible along d if it faces d and a ray from outside the
    bounding box toward it hits f first (or a face whose plane holds the point: duplicates).
    Uses embree when trimesh has it; the rtree fallback gets a smaller budget.
    """
    k = int(labels.max()) + 1
    ray = mesh.ray
    if "embree" not in type(ray).__module__:
        budget, per_region = min(budget, 3000), min(per_region, 300)
    rng = np.random.default_rng(seed)
    face, reg = _sample(labels, np.asarray(mesh.area_faces), k, budget, per_region, rng)
    tri = mesh.triangles[face]
    r1, r2 = rng.random(len(face)), rng.random(len(face))
    s1 = np.sqrt(r1)
    w = np.column_stack([1 - s1, s1 * (1 - r2), s1 * r2])
    p = np.einsum("ij,ijk->ik", w, tri)
    nrm = mesh.face_normals[face]
    dirs = view_directions(up, rings)
    ext = mesh.bounds[1] - mesh.bounds[0]
    far = 2.0 * float(np.linalg.norm(ext)) + 1e-3
    eps = 1e-6 * far + 1e-3 * edge_table(mesh).edge_scale
    vis = np.zeros((len(face), len(dirs)), bool)
    fn, v0 = mesh.face_normals, mesh.triangles[:, 0]
    for j, d in enumerate(dirs):
        front = np.flatnonzero(nrm @ d > FRONT_DOT)
        if len(front) == 0:
            continue
        hit = ray.intersects_first(p[front] + far * d, np.tile(-d, (len(front), 1)))
        ok = hit == face[front]
        h = np.flatnonzero(~ok & (hit >= 0))
        if len(h):
            hh = hit[h]
            ok[h] = np.abs(np.einsum("ij,ij->i", fn[hh], p[front[h]] - v0[hh])) < eps
        vis[front, j] = ok
    n = np.bincount(reg, minlength=k).astype(float)
    den = np.maximum(n, 1)
    return (
        np.bincount(reg, weights=vis.any(1), minlength=k) / den,
        np.bincount(reg, weights=vis[:, 0], minlength=k) / den,
        np.bincount(reg, weights=vis.mean(1), minlength=k) / den,
        n.astype(int),
    )


def _loops(k, n_vertices, reg, verts, length):
    """Boundary loops per region: (n_loops, n_small_holes)."""
    if len(reg) == 0:
        return np.zeros(k, int), np.zeros(k, int)
    a = reg * n_vertices + verts[:, 0]
    b = reg * n_vertices + verts[:, 1]
    nodes, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    ia, ib = inv[: len(a)], inv[len(a) :]
    g = coo_matrix((np.ones(len(ia), np.int8), (ia, ib)), shape=(len(nodes),) * 2)
    comp = connected_components(g, directed=False)[1][ia]
    key, cinv = np.unique(reg * (len(nodes) + 1) + comp, return_inverse=True)
    creg = key // (len(nodes) + 1)
    clen = np.bincount(cinv, weights=length)
    n_loops = np.bincount(creg, minlength=k)
    tot = np.bincount(creg, weights=clen, minlength=k)
    longest = np.zeros(k)
    np.maximum.at(longest, creg, clen)
    small = (clen < 0.25 * tot[creg]) & (clen < longest[creg])
    return n_loops, np.bincount(creg, weights=small, minlength=k).astype(int)


def region_features(
    mesh: trimesh.Trimesh,
    labels,
    up=(0.0, 0.0, 1.0),
    feature_angle_deg: float = 25.0,
    visibility_budget: int = 20000,
    per_region_samples: int = 2000,
    rings=CONE_RINGS,
    compute_visibility: bool = True,
    seed: int = 0,
) -> list[RegionFeatures]:
    """Features of each region of ``labels`` (dense ints, one per face); list index = label."""
    labels = np.asarray(labels, np.int64).ravel()
    n = len(mesh.faces)
    if len(labels) != n:
        raise ValueError(f"labels has {len(labels)} entries, mesh has {n} faces")
    if n == 0:
        return []
    if labels.min() < 0:
        raise ValueError("labels must be >= 0")
    k = int(labels.max()) + 1
    u = _unit(up)
    et = edge_table(mesh)
    af = np.asarray(mesh.area_faces, float)
    fn = np.asarray(mesh.face_normals, float)
    area = np.bincount(labels, weights=af, minlength=k)
    nf = np.bincount(labels, minlength=k)
    total = float(af.sum())
    ar = np.maximum(area, 1e-300)
    nsum = np.column_stack([np.bincount(labels, weights=af * fn[:, i], minlength=k) for i in range(3)])
    nup = fn @ u
    mean_up = np.bincount(labels, weights=af * nup, minlength=k) / ar
    frac_up = np.bincount(labels, weights=af * (nup > UP_DOT), minlength=k) / ar
    # heights relative to the part along up
    hv = mesh.vertices @ u
    h0, h1 = float(hv.min()), float(hv.max())
    span = max(h1 - h0, 1e-12)
    hf = (hv[mesh.faces] - h0) / span
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(nf)])[:-1]
    nonempty = nf > 0
    hmin, hmax = np.zeros(k), np.zeros(k)
    if nonempty.any():
        hmin[nonempty] = np.minimum.reduceat(hf.min(1)[order], starts[nonempty])
        hmax[nonempty] = np.maximum.reduceat(hf.max(1)[order], starts[nonempty])
    hmean = np.bincount(labels, weights=af * hf.mean(1), minlength=k) / ar
    # internal edges: smoothness and curvature-sign mix
    la, lb = labels[et.pairs[:, 0]], labels[et.pairs[:, 1]]
    same = la == lb
    deg = np.degrees(et.angle)
    li, wi = la[same], 0.5 * (af[et.pairs[same, 0]] + af[et.pairs[same, 1]])
    wsum = np.maximum(np.bincount(li, weights=wi, minlength=k), 1e-300)
    smooth_mean = np.bincount(li, weights=wi * deg[same], minlength=k) / wsum
    smooth_p95 = weighted_quantile(li, deg[same], wi, k, 0.95)
    bent = same & (deg > FLAT_DEG)
    lb_len = np.bincount(la[bent], weights=et.length[bent], minlength=k)
    cvx = np.bincount(la[bent], weights=et.length[bent] * et.convex[bent], minlength=k)
    convex_frac = np.where(lb_len > 0, cvx / np.maximum(lb_len, 1e-300), 0.0)
    concave_frac = np.where(lb_len > 0, 1.0 - convex_frac, 0.0)
    # boundaries between regions
    x = ~same
    src = np.concatenate([la[x], lb[x]])
    dst = np.concatenate([lb[x], la[x]])
    ln = np.tile(et.length[x], 2)
    ang = np.tile(deg[x], 2)
    cv = np.tile(et.convex[x], 2)
    sharp_e = ang > feature_angle_deg
    open_len = np.bincount(labels[et.open_face], weights=et.open_len, minlength=k)
    nm_len = np.bincount(labels[et.nm_face], weights=et.nm_len, minlength=k)
    nb_len = np.bincount(src, weights=ln, minlength=k)
    blen = nb_len + open_len + nm_len
    bl = np.maximum(blen, 1e-300)
    sharp_frac = np.where(blen > 0, (np.bincount(src, weights=ln * sharp_e, minlength=k) + nm_len) / bl, 0.0)
    open_frac = np.where(blen > 0, open_len / bl, 0.0)
    neighbors = [dict() for _ in range(k)]
    if len(src):
        key, inv = np.unique(src * k + dst, return_inverse=True)
        pl = np.bincount(inv, weights=ln)
        pa = np.bincount(inv, weights=ln * ang) / np.maximum(pl, 1e-300)
        pc = np.bincount(inv, weights=ln * cv) / np.maximum(pl, 1e-300)
        pmax = np.zeros(len(key))
        np.maximum.at(pmax, inv, ang)
        for kk, L, A, M, C in zip(key, pl, pa, pmax, pc, strict=True):
            kind = "tangent" if A < TANGENT_DEG else ("convex" if C >= 0.5 else "concave")
            a_, b_ = int(kk // k), int(kk % k)
            neighbors[a_][b_] = Boundary(b_, float(L), float(A), float(M), float(C), kind)
    # boundary loops: every edge on a region's boundary, per region side
    loop_reg = np.concatenate([src, labels[et.open_face], labels[et.nm_face]])
    loop_v = np.vstack([et.verts[x], et.verts[x], et.open_verts, et.nm_verts])
    loop_len = np.concatenate([ln, et.open_len, et.nm_len])
    n_loops, n_holes = _loops(k, et.n_vertices, loop_reg, loop_v, loop_len)
    if compute_visibility:
        vis, vis_up, vis_mean, nvs = visibility(
            mesh, labels, u, visibility_budget, per_region_samples, rings, seed
        )
    else:
        vis = vis_up = vis_mean = np.full(k, np.nan)
        nvs = np.zeros(k, int)
    face_lists = np.split(order, np.cumsum(nf)[:-1])
    out = []
    for r in range(k):
        mn = nsum[r]
        nn = np.linalg.norm(mn)
        out.append(
            RegionFeatures(
                label=r,
                n_faces=int(nf[r]),
                area=float(area[r]),
                area_frac=float(area[r] / total) if total > 0 else 0.0,
                mean_normal=tuple(float(c) for c in (mn / nn if nn > 1e-12 else np.zeros(3))),
                mean_up=float(mean_up[r]),
                frac_up=float(frac_up[r]),
                height_mean=float(hmean[r]),
                height_min=float(hmin[r]),
                height_max=float(hmax[r]),
                smooth_mean_deg=float(smooth_mean[r]),
                smooth_p95_deg=float(smooth_p95[r]),
                convex_frac=float(convex_frac[r]),
                concave_frac=float(concave_frac[r]),
                visible_frac=float(vis[r]),
                visible_up_frac=float(vis_up[r]),
                visible_mean=float(vis_mean[r]),
                n_vis_samples=int(nvs[r]),
                boundary_length=float(blen[r]),
                sharp_frac=float(sharp_frac[r]),
                open_frac=float(open_frac[r]),
                face_density=float(nf[r] / area[r]) if area[r] > 0 else 0.0,
                n_boundary_loops=int(n_loops[r]),
                n_holes=int(n_holes[r]),
                neighbors=neighbors[r],
                faces=face_lists[r],
            )
        )
    return out
