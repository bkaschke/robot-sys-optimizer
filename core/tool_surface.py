"""Tool-surface detection, part 1: split a mandrel mesh into smooth regions (T1).

``segment`` labels every face with a region id. Regions are smooth patches bounded by edges
whose dihedral angle exceeds a feature angle; tessellation fragments are merged into their
best neighbour. With CAD face ids (STEP import) the CAD faces are the starting regions and
tangent-adjacent ones are merged instead. ``region_features`` (in
``core.tool_surface_features``) then describes each region for scoring (T2).

Works on non-watertight meshes, on several overlapping closed shells, and on triangle soups
(vertices are merged on a private copy so adjacency is found). Face order is never changed.
No viser / ui imports; SI units.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = ["EdgeTable", "edge_table", "segment", "region_features", "analyze"]


class EdgeTable:
    """Face-pair and unpaired-edge data shared by segmentation and features.

    Attributes (all per face-adjacency pair unless noted):
      pairs [m, 2] face indices; angle [m] dihedral (rad, 0 = flat); convex [m] bool;
      length [m] shared edge length; verts [m, 2] edge vertex ids (topology mesh);
      open_face / open_len / open_verts: edges used by one face only (mesh boundary);
      nm_face / nm_len / nm_verts: occurrences of non-manifold edges (3+ faces), which
      trimesh does not pair; they are treated as sharp boundary.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        topo = _topology(mesh)
        self.n_vertices = len(topo.vertices)
        n = len(mesh.faces)
        if n == 0:
            raise ValueError("mesh has no faces")
        self.pairs = np.asarray(topo.face_adjacency, dtype=np.int64).reshape(-1, 2)
        self.angle = np.asarray(topo.face_adjacency_angles, float)
        # projection of the free vertex of B onto A's plane: <= 0 is convex (trimesh convention)
        self.convex = np.asarray(topo.face_adjacency_convex, bool)
        self.verts = np.asarray(topo.face_adjacency_edges, dtype=np.int64).reshape(-1, 2)
        v = topo.vertices
        self.length = np.linalg.norm(v[self.verts[:, 0]] - v[self.verts[:, 1]], axis=1)
        es = np.asarray(topo.edges_sorted, dtype=np.int64)
        key = es[:, 0] * self.n_vertices + es[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        c = cnt[inv]
        elen = np.linalg.norm(v[es[:, 0]] - v[es[:, 1]], axis=1)
        face_of = np.arange(len(es)) // 3
        o, nm = c == 1, c >= 3
        self.open_face, self.open_len, self.open_verts = face_of[o], elen[o], es[o]
        self.nm_face, self.nm_len, self.nm_verts = face_of[nm], elen[nm], es[nm]
        self.edge_scale = float(np.mean(elen)) if len(elen) else 0.0


def _topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """``mesh`` itself, or a vertex-merged copy when it is a triangle soup (same face order)."""
    n = len(mesh.faces)
    if n < 2 or len(mesh.face_adjacency) >= 0.3 * n:
        return mesh
    copy = trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.faces), process=False)
    copy.merge_vertices(merge_tex=True, merge_norm=True)
    if len(copy.faces) != n:  # never expected; keep labels aligned with the input
        return mesh
    return copy


def edge_table(mesh: trimesh.Trimesh) -> EdgeTable:
    """Build (or fetch from the mesh's own data-hash-checked cache) the ``EdgeTable``."""
    cache = getattr(mesh, "_cache", None)
    try:
        hit = cache.get("tool_surface_edge_table") if cache is not None else None
    except Exception:
        hit = None
    if hit is not None:
        return hit
    t = EdgeTable(mesh)
    try:
        cache["tool_surface_edge_table"] = t
    except Exception:
        pass
    return t


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _cross_pairs(labels: np.ndarray, et: EdgeTable):
    """Region pairs across adjacency edges: unique (lo, hi) with summed length and mean angle."""
    la, lb = labels[et.pairs[:, 0]], labels[et.pairs[:, 1]]
    x = la != lb
    if not x.any():
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0), np.zeros(0)
    lo, hi = np.minimum(la[x], lb[x]), np.maximum(la[x], lb[x])
    k = int(labels.max()) + 1
    key, inv = np.unique(lo * k + hi, return_inverse=True)
    ln = np.bincount(inv, weights=et.length[x])
    ang = np.bincount(inv, weights=et.length[x] * et.angle[x]) / np.maximum(ln, 1e-300)
    return key // k, key % k, ln, ang


def _merge_fragments(
    labels, et, area_faces, min_area, max_faces, feature_rad, max_rad, sliver_area, max_iter=20
):
    """Merge tiny regions into their best neighbour (long shared boundary, small angle).

    Only across boundaries with a mean dihedral <= ``max_rad``: a small region behind a
    sharp edge (the side strip of a faceplate, the wall of a hole) is geometry, not noise.
    Slivers below ``sliver_area`` merge whatever the angle (flipped decimation slivers).

    Each tiny region points at one neighbour, so a merge component holds at most one
    non-tiny region: two real regions are never joined through fragments.
    """
    for _ in range(max_iter):
        k = int(labels.max()) + 1
        area = np.bincount(labels, weights=area_faces, minlength=k)
        nf = np.bincount(labels, minlength=k)
        tiny = (area < min_area) & (nf <= max_faces) & (nf > 0)
        if not tiny.any():
            break
        lo, hi, ln, ang = _cross_pairs(labels, et)
        if len(lo) == 0:
            break
        src, dst = np.concatenate([lo, hi]), np.concatenate([hi, lo])
        ln2, ang2 = np.concatenate([ln, ln]), np.concatenate([ang, ang])
        keep = tiny[src]
        if not keep.any():
            break
        src, dst, ln2, ang2 = src[keep], dst[keep], ln2[keep], ang2[keep]
        ok = (ang2 <= max_rad) | (area[src] < sliver_area)
        if not ok.any():
            break
        src, dst, ln2, ang2 = src[ok], dst[ok], ln2[ok], ang2[ok]
        score = ln2 * np.exp(-ang2 / max(feature_rad, 1e-6)) * np.where(tiny[dst], 0.5, 1.0)
        order = np.lexsort((-score, src))
        first = np.ones(len(order), bool)
        first[1:] = src[order][1:] != src[order][:-1]
        best = order[first]
        comp = _components(k, src[best], dst[best])
        labels = comp[labels]
    return labels


def weighted_quantile(group, value, weight, k, q):
    """Weighted q-quantile of ``value`` per group (0 for empty groups)."""
    out = np.zeros(k)
    if len(group) == 0:
        return out
    order = np.lexsort((value, group))
    g, v, w = group[order], value[order], weight[order]
    cw = np.cumsum(w)
    tot = np.bincount(g, weights=w, minlength=k)
    cnt = np.bincount(g, minlength=k)
    end = np.cumsum(cnt)
    start = end - cnt
    before = np.where(start > 0, cw[np.maximum(start - 1, 0)], 0.0)
    has = cnt > 0
    idx = np.searchsorted(cw, before + q * tot, side="left")
    idx = np.clip(idx, start, np.maximum(end - 1, 0))
    out[has] = v[idx[has]]
    return out


def _merge_tangent_cad(labels, et, tangent_deg):
    """Union CAD faces whose shared boundary is tangent (G1), allowing for tessellation."""
    k = int(labels.max()) + 1
    same = labels[et.pairs[:, 0]] == labels[et.pairs[:, 1]]
    # the tessellation step inside each face: a G1 seam bends about as much as that
    internal = weighted_quantile(labels[et.pairs[same, 0]], et.angle[same], et.length[same], k, 0.9)
    lo, hi, _, ang = _cross_pairs(labels, et)
    thr = np.maximum(np.radians(tangent_deg), 1.25 * np.maximum(internal[lo], internal[hi]))
    m = ang <= thr
    return _components(k, lo[m], hi[m])[labels]


def _relabel_by_area(labels: np.ndarray, area_faces: np.ndarray) -> np.ndarray:
    """Dense labels 0..k-1, ordered by decreasing area (ties: lowest first face)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    area = np.bincount(inv, weights=area_faces)
    first = np.full(len(uniq), len(labels))
    np.minimum.at(first, inv, np.arange(len(labels)))
    rank = np.lexsort((first, -area))
    new = np.empty(len(uniq), np.int64)
    new[rank] = np.arange(len(uniq))
    return new[inv]


def segment(
    mesh: trimesh.Trimesh,
    feature_angle_deg: float = 25.0,
    min_region_frac: float = 0.002,
    max_fragment_faces: int = 200,
    max_merge_angle_deg: float = 60.0,
    cad_face_ids=None,
    tangent_deg: float = 5.0,
) -> np.ndarray:
    """Region label per face (int64, 0 = largest region by area).

    Mesh path: connected components across edges with dihedral <= ``feature_angle_deg``;
    regions smaller than ``min_region_frac`` of the total area *and* with at most
    ``max_fragment_faces`` faces are merged into their best neighbour across a boundary of
    at most ``max_merge_angle_deg`` (slivers below 1e-6 of the area merge at any angle).
    CAD path (``cad_face_ids`` given, one id per face): CAD faces are the regions; adjacent
    faces meeting at <= ``tangent_deg`` (or <= 1.25x the 90th-percentile tessellation angle
    inside either face) merge.
    """
    n = len(mesh.faces)
    if n == 0:
        raise ValueError("segment: mesh has no faces")
    et = edge_table(mesh)
    area_faces = np.asarray(mesh.area_faces, float)
    if cad_face_ids is not None:
        ids = np.asarray(cad_face_ids).ravel()
        if len(ids) != n:
            raise ValueError(f"cad_face_ids has {len(ids)} entries, mesh has {n} faces")
        labels = np.unique(ids, return_inverse=True)[1].astype(np.int64)
        labels = _merge_tangent_cad(labels, et, tangent_deg)
    else:
        ok = et.angle <= np.radians(feature_angle_deg)
        labels = _components(n, et.pairs[ok, 0], et.pairs[ok, 1]).astype(np.int64)
        labels = _merge_fragments(
            labels,
            et,
            area_faces,
            min_region_frac * float(area_faces.sum()),
            max_fragment_faces,
            np.radians(feature_angle_deg),
            np.radians(max_merge_angle_deg),
            1e-6 * float(area_faces.sum()),
        )
    return _relabel_by_area(labels, area_faces)


def region_features(mesh, labels, up=(0.0, 0.0, 1.0), **kw):
    """Per-region features; see ``core.tool_surface_features.region_features``."""
    from core.tool_surface_features import region_features as _rf

    return _rf(mesh, labels, up=up, **kw)


def analyze(mesh, up=(0.0, 0.0, 1.0), cad_face_ids=None, segment_kw=None, **feature_kw):
    """``segment`` then ``region_features``; returns ``(labels, features)``."""
    labels = segment(mesh, cad_face_ids=cad_face_ids, **(segment_kw or {}))
    return labels, region_features(mesh, labels, up=up, **feature_kw)
