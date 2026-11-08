"""Path patterns (v4 plan item 2): the order in which a set of surface points is visited,
ported from the sand-optimizer (``src/core/patterns``): zigzag, unidirectional, cross-hatch,
concentric, spiral, Hilbert and shortest path (nearest neighbour + 2-opt). Pure numpy.

Points are projected onto the plane across their mean normal; the raster angle is measured
in that plane from the part X axis (as the surface grid's row direction). Start corners are
"<along>-<across>" in the raster frame: min-min starts on the lowest row at its low end.
``crossing_lifts`` marks the moves that would cross an earlier move ("prevent crossing
paths"): the tool lifts off for them instead of following the surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.surface_grid import grid_frame

PATTERNS = (
    "Zigzag",
    "Unidirectional",
    "Cross-hatch",
    "Concentric",
    "Spiral",
    "Hilbert",
    "Shortest path (TSP)",
)
CORNERS = ("min-min", "max-min", "min-max", "max-max")
ROW_PATTERNS = ("Zigzag", "Unidirectional", "Cross-hatch")
BOUNDARY_GAP = 2 * np.pi / 3
TSP_WORK = 2_000_000  # 2-opt work budget (pair evaluations), fixed so results are repeatable


@dataclass
class PatternSettings:
    pattern: str = "Zigzag"
    angle_deg: float = 0.0  # raster angle
    corner: str = "min-min"
    row_tol: float = 0.0  # m; 0 = automatic (half the point spacing)
    inward: bool = True  # concentric / spiral: outside -> in
    no_crossing: bool = False  # lift off for moves that cross an earlier move


@dataclass
class Ordered:
    order: np.ndarray  # indices into the points (cross-hatch visits each twice)
    rows: np.ndarray  # row / ring / turn number per visit
    uv: np.ndarray  # projected coordinates of the points (m)
    passes: np.ndarray | None = None  # pass per visit (cross-hatch: 0, then 1)


# --- projection and helpers -----------------------------------------------------------------------


def project(P: np.ndarray, N: np.ndarray | None = None) -> np.ndarray:
    """(n, 2) coordinates in the plane across the points' mean normal (u = projected part X)."""
    P = np.asarray(P, float)
    n = None
    if N is not None and len(N):
        m = np.asarray(N, float).sum(0)
        n = m if np.linalg.norm(m) > 1e-9 * len(N) else None
    if n is None:  # best-fit plane
        c = P - P.mean(0)
        n = np.linalg.svd(c, full_matrices=False)[2][-1] if len(P) >= 3 else np.array([0, 0, 1.0])
    u, v = grid_frame(n, 0.0)
    return np.column_stack([P @ u, P @ v])


def spacing_of(P: np.ndarray) -> float:
    """Median nearest-neighbour distance."""
    if len(P) < 2:
        return 0.0
    from scipy.spatial import cKDTree

    d, _ = cKDTree(P).query(P, k=2)
    return float(np.median(d[:, 1]))


def to_raster(uv: np.ndarray, angle_deg: float) -> np.ndarray:
    """(along rows, across rows) coordinates."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([c * uv[:, 0] + s * uv[:, 1], -s * uv[:, 0] + c * uv[:, 1]])


def group_rows(ac: np.ndarray, tol: float) -> list[np.ndarray]:
    """Rows across the raster: a new row starts when the across-gap exceeds ``tol`` or the row
    would span more than 2 tol (oblique rasters on a grid). Members sorted along the row."""
    by = np.lexsort((ac[:, 0], ac[:, 1]))
    rows, cur, prev, start = [], [], -np.inf, -np.inf
    for i in by:
        c = ac[i, 1]
        if cur and (c - prev > tol or c - start > 2 * tol):
            rows.append(cur)
            cur = []
        if not cur:
            start = c
        cur.append(i)
        prev = c
    if cur:
        rows.append(cur)
    return [np.array(sorted(r, key=lambda i: (ac[i, 0], ac[i, 1])), int) for r in rows]


def corner_signs(corner: str) -> tuple[int, int]:
    a, c = corner.split("-")
    return (1 if a == "min" else -1), (1 if c == "min" else -1)


def nearest_corner(ac: np.ndarray, target: np.ndarray) -> str:
    lo, hi = ac.min(0), ac.max(0)
    pts = {
        "min-min": (lo[0], lo[1]),
        "max-min": (hi[0], lo[1]),
        "min-max": (lo[0], hi[1]),
        "max-max": (hi[0], hi[1]),
    }
    return min(CORNERS, key=lambda k: np.hypot(pts[k][0] - target[0], pts[k][1] - target[1]))


def start_index(ac: np.ndarray, corner: str) -> int:
    along, across = corner_signs(corner)
    return int(np.argmin(np.round(along * ac[:, 0] + across * ac[:, 1], 12)))


# --- raster family ------------------------------------------------------------------------------


def _raster(ac: np.ndarray, corner: str, tol: float, alternate: bool, row0: int = 0):
    rows = group_rows(ac, tol)
    along, across = corner_signs(corner)
    if across < 0:
        rows.reverse()
    order, rid = [], []
    for r, members in enumerate(rows):
        fwd = ((r % 2 == 0) == (along > 0)) if alternate else along > 0
        seq = members if fwd else members[::-1]
        order += list(seq)
        rid += [row0 + r] * len(seq)
    return order, rid


# --- shortest path ----------------------------------------------------------------------------------


def nearest_neighbour_tour(P: np.ndarray, start: int) -> list[int]:
    n = len(P)
    used = np.zeros(n, bool)
    tour = [start]
    used[start] = True
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[tour[-1]], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        tour.append(j)
        used[j] = True
    return tour


def two_opt(P: np.ndarray, tour: list[int], budget: int = TSP_WORK) -> list[int]:
    """2-opt on an open path with a fixed first node (reverses tour[i+1..j])."""
    t = np.array(tour, int)
    n = len(t)
    if n < 4:
        return list(t)
    improved = True
    while improved and budget > 0:
        improved = False
        for i in range(n - 2):
            X = P[t]
            j = np.arange(i + 2, n)
            d_i = np.linalg.norm(X[i] - X[i + 1])
            nxt = np.minimum(j + 1, n - 1)
            has = j + 1 < n
            before = d_i + np.where(has, np.linalg.norm(X[j] - X[nxt], axis=1), 0.0)
            after = np.linalg.norm(X[i] - X[j], axis=1) + np.where(
                has, np.linalg.norm(X[i + 1] - X[nxt], axis=1), 0.0
            )
            gain = before - after
            k = int(np.argmax(gain))
            if gain[k] > 1e-10:
                a, b = i + 1, int(j[k])
                t[a : b + 1] = t[a : b + 1][::-1]
                improved = True
            budget -= n - i - 2
            if budget <= 0:
                break
    return list(t)


# --- concentric and spiral ----------------------------------------------------------------------------


def _neighbours(P: np.ndarray, radius: float) -> list[list[int]]:
    from scipy.spatial import cKDTree

    return [[j for j in lst if j != i] for i, lst in enumerate(cKDTree(P).query_ball_point(P, radius))]


def split_components(nb: list[list[int]], idx) -> list[list[int]]:
    left = set(int(i) for i in idx)
    out = []
    for seed in idx:
        if seed not in left:
            continue
        left.discard(seed)
        comp = [int(seed)]
        for k in comp:
            for j in nb[k]:
                if j in left:
                    left.discard(j)
                    comp.append(j)
        out.append(comp)
    return out


def peel_rings(ac: np.ndarray, nb: list[list[int]]) -> list[list[int]]:
    """Boundary layers, outermost first (a point is on the boundary when its remaining
    neighbours leave an angular gap wider than 120 degrees)."""
    remaining = set(range(len(ac)))
    rings = []
    while remaining:
        rem = sorted(remaining)
        ring = []
        for i in rem:
            js = [j for j in nb[i] if j in remaining]
            if len(js) < 3:
                ring.append(i)
                continue
            ang = np.sort(np.arctan2(ac[js, 1] - ac[i, 1], ac[js, 0] - ac[i, 0]))
            gap = max(ang[0] + 2 * np.pi - ang[-1], float(np.diff(ang).max(initial=0.0)))
            if gap > BOUNDARY_GAP:
                ring.append(i)
        layer = ring or rem
        remaining -= set(layer)
        rings.append(layer)
    return rings


def _walk_loop(P: np.ndarray, ac: np.ndarray, loop: list[int], start: int) -> list[int]:
    """Nearest unvisited point; near-ties go counter-clockwise about the loop centre."""
    c = ac[loop].mean(0)
    ang = np.arctan2(ac[:, 1] - c[1], ac[:, 0] - c[0])
    left = [j for j in loop if j != start]
    seq, cur = [start], start
    while left:
        L = np.array(left)
        d = np.linalg.norm(P[L] - P[cur], axis=1)
        near = L[d <= d.min() * 1.05 + 1e-12]
        turn = np.mod(ang[near] - ang[cur] - 1e-9, 2 * np.pi) + 1e-9
        nxt = int(near[np.lexsort((near, turn))[0]])
        left.remove(nxt)
        seq.append(nxt)
        cur = nxt
    return seq


def _concentric(P, ac, nb, inward: bool, start: int):
    rings = peel_rings(ac, nb)
    if not inward:
        rings = rings[::-1]
    order, rid = [], []
    anchor = ac[start]
    group = 0
    for ring in rings:
        loops = split_components(nb, ring)
        while loops:
            dist = [np.linalg.norm(ac[lp] - anchor, axis=1).min() for lp in loops]
            loop = loops.pop(int(np.argmin(dist)))
            s0 = loop[int(np.argmin(np.linalg.norm(ac[loop] - anchor, axis=1)))]
            seq = two_opt(P, _walk_loop(P, ac, loop, s0), budget=200_000)
            order += seq
            rid += [group] * len(seq)
            anchor = ac[seq[-1]]
            group += 1
    return order, rid


def _boundary_radius(ang_r: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Outer-boundary radius at angles a (periodic linear interpolation)."""
    if len(ang_r) == 1:
        return np.full(len(a), ang_r[0, 1])
    s = ang_r[np.argsort(ang_r[:, 0])]
    x = np.concatenate([s[:, 0] - 2 * np.pi, s[:, 0], s[:, 0] + 2 * np.pi])
    y = np.tile(s[:, 1], 3)
    return np.interp(a, x, y)


def _spiral(ac, nb, inward: bool, start: int):
    rings = peel_rings(ac, nb)
    c = ac.mean(0)
    r = np.hypot(ac[:, 0] - c[0], ac[:, 1] - c[1])
    ang = np.arctan2(ac[:, 1] - c[1], ac[:, 0] - c[0])
    loops = split_components(nb, rings[0])
    outer = max(loops, key=lambda lp: r[lp].mean())
    R = np.maximum(_boundary_radius(np.column_stack([ang[outer], r[outer]]), ang), 1e-12)
    turns = max(1, len(rings))
    depth = (1 - np.minimum(1.0, r / R)) * turns
    frac = np.mod((ang - ang[start]) / (2 * np.pi), 1.0)
    frac[frac > 1 - 1e-9] = 0.0
    turn = np.maximum(0, np.round(depth - frac)).astype(int)
    order = list(np.lexsort((np.arange(len(ac)), depth, turn + frac)))
    if not inward:
        order = order[::-1]
    t = turn[order]
    return order, list(t if inward else t.max() - t)


# --- Hilbert ---------------------------------------------------------------------------------------------


def hilbert_index(x: np.ndarray, y: np.ndarray, order: int = 10) -> np.ndarray:
    x, y = x.astype(np.int64).copy(), y.astype(np.int64).copy()
    d = np.zeros(len(x), np.int64)
    s = 1 << (order - 1)
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx
        x[flip], y[flip] = s - 1 - x[flip], s - 1 - y[flip]
        swap = ~ry
        x[swap], y[swap] = y[swap], x[swap].copy()
        s >>= 1
    return d


def _hilbert(ac: np.ndarray, corner: str):
    along, across = corner_signs(corner)
    q = ac * (along, across)
    lo = q.min(0)
    ext = max(float((q.max(0) - lo).max()), 1e-12)
    n = 1 << 10
    xy = np.minimum(n - 1, np.floor((q - lo) / ext * (n - 1) + 0.5)).astype(np.int64)
    d = hilbert_index(xy[:, 0], xy[:, 1])
    return list(np.lexsort((q[:, 1], q[:, 0], d))), [0] * len(ac)


# --- entry points ----------------------------------------------------------------------------------------


def order_points(
    P: np.ndarray, N: np.ndarray | None, st: PatternSettings, corner: str | None = None
) -> Ordered:
    """Visit order of points P (part frame) for pattern ``st``. ``corner`` overrides the start
    corner (later groups start near where the previous one ended)."""
    P = np.asarray(P, float)
    uv = project(P, N) if len(P) else np.zeros((0, 2))
    if len(P) <= 1:
        return Ordered(np.arange(len(P)), np.zeros(len(P), int), uv)
    if st.pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {st.pattern!r}; have {PATTERNS}")
    corner = corner or st.corner
    sp = spacing_of(P)
    tol = st.row_tol if st.row_tol > 0 else max(0.5 * sp, 1e-5)
    ac = to_raster(uv, st.angle_deg)
    if st.pattern == "Zigzag":
        o, r = _raster(ac, corner, tol, True)
    elif st.pattern == "Unidirectional":
        o, r = _raster(ac, corner, tol, False)
    elif st.pattern == "Cross-hatch":
        o, r = _raster(ac, corner, tol, True)
        ac2 = to_raster(uv, st.angle_deg + 90)
        o2, r2 = _raster(ac2, nearest_corner(ac2, ac2[o[-1]]), tol, True, row0=max(r) + 1)
        passes = np.r_[np.zeros(len(o), int), np.ones(len(o2), int)]
        return Ordered(np.asarray(o + o2, int), np.asarray(r + r2, int), uv, passes)
    elif st.pattern == "Hilbert":
        o, r = _hilbert(ac, corner)
    elif st.pattern == "Shortest path (TSP)":
        o = two_opt(P, nearest_neighbour_tour(P, start_index(ac, corner)))
        r = [0] * len(o)
    else:
        nb = _neighbours(P, max(1.75 * sp, 1.01 * tol, 1e-6))
        if st.pattern == "Concentric":
            o, r = _concentric(P, ac, nb, st.inward, start_index(ac, corner))
        else:
            o, r = _spiral(ac, nb, st.inward, start_index(ac, corner))
    return Ordered(np.asarray(o, int), np.asarray(r, int), uv)


def _cross(a, b, c, d) -> np.ndarray:
    """Proper intersection (or collinear overlap) of segment a-b with each segment c[k]-d[k]."""

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    eps = 1e-12
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    proper = (o1 * o2 < -eps) & (o3 * o4 < -eps)
    col = (np.abs(o1) <= eps) & (np.abs(o2) <= eps)
    if col.any():  # collinear: overlap of positive length along the segment direction
        u = b - a
        L = float(u @ u)
        if L > 0:
            t1 = ((c - a) @ u) / L
            t2 = ((d - a) @ u) / L
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            col &= (np.minimum(hi, 1.0) - np.maximum(lo, 0.0)) > 1e-9
        else:
            col[:] = False
    return proper | col


def crossing_lifts(uv: np.ndarray, order: np.ndarray, passes: np.ndarray | None = None) -> np.ndarray:
    """Per move order[k] -> order[k+1]: True when it would cross (or retrace) an earlier
    surface move of the same pass; those moves become lifts and are not obstacles for later
    moves. Touching at a shared end point is not a crossing."""
    m = len(order) - 1
    lift = np.zeros(max(m, 0), bool)
    A, B, cur = [], [], None
    for k in range(m):
        if passes is not None and passes[k + 1] != cur:  # a new pass may cross the old one
            cur, A, B = passes[k + 1], [], []
            if k > 0 and passes[k] != passes[k + 1]:
                continue  # the move from one pass to the next is a link, not checked
        a, b = uv[order[k]], uv[order[k + 1]]
        if A and _cross(a, b, np.array(A), np.array(B)).any():
            lift[k] = True
            continue
        A.append(a)
        B.append(b)
    return lift
