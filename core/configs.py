"""Joint configurations, zones and path continuity (no viewer imports).

A 6R UR arm reaches most poses in up to 8 ways: shoulder, elbow and wrist each have two
branches. Moving between branches needs a large reconfiguration motion, so the planner
groups the points of a station into **zones** that each use one configuration, and counts
**configuration changes** (zone boundaries and big wrist jumps) in the cost.

Labels: ``"S+E-W+"`` with
  S  sign of the horizontal reach a2*c2 + a3*c23 + d5*s234 (shoulder branch, the factor
     that vanishes at the shoulder singularity),
  E  sign of sin(q3)  (elbow branch),
  W  sign of sin(q5)  (wrist flip).
These are the three factors of |det J| (core.ik.ur.ur_manipulability), so a label never
changes without passing through a singularity. For non-UR arms the shoulder letter is
omitted (sign of q3 and q5 only).

Wrist wraps: joints 4 and 6 can often take values 360 deg apart; ``unwrap_path`` picks
the copies that keep a path continuous, and any remaining jump above ``WRAP_JUMP_DEG``
between neighbours counts as a configuration change too (the user's definition).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

WRAP_JOINTS = (3, 5)  # joints 4 and 6 (0-based)
WRAP_JUMP_DEG = 180.0


def _sgn(x: float) -> str:
    return "+" if x >= 0 else "-"


def config_label(ik, q: np.ndarray) -> str:
    q = np.asarray(q, float)
    e, w = _sgn(np.sin(q[2])), _sgn(np.sin(q[4]))
    if not all(hasattr(ik, a) for a in ("a2", "a3", "d5")):
        return f"E{e}W{w}"
    reach = ik.a2 * np.cos(q[1]) + ik.a3 * np.cos(q[1] + q[2]) + ik.d5 * np.sin(q[1] + q[2] + q[3])
    return f"S{_sgn(reach)}E{e}W{w}"


def min_zones(options: dict[int, dict[str, float]]) -> list[tuple[str, list[int]]]:
    """Fewest configurations that cover every point, each point -> its best covered config.

    ``options``: point id -> {config label: quality (higher is better)}. Exact: tries all
    config subsets by increasing size (at most 8 labels), ties broken by total quality.
    Points with no option are left out. Returns [(config, [point ids])], largest zone first.
    """
    pts = {p: o for p, o in options.items() if o}
    if not pts:
        return []
    labels = sorted({c for o in pts.values() for c in o})
    for k in range(1, len(labels) + 1):
        best, best_q = None, -np.inf
        for subset in combinations(labels, k):
            s = set(subset)
            if all(s & o.keys() for o in pts.values()):
                q = sum(max(v for c, v in o.items() if c in s) for o in pts.values())
                if q > best_q:
                    best, best_q = s, q
        if best is not None:
            zones: dict[str, list[int]] = {c: [] for c in best}
            for p, o in pts.items():
                c = max((c for c in o if c in best), key=lambda c: o[c])
                zones[c].append(p)
            return sorted(((c, v) for c, v in zones.items() if v), key=lambda z: -len(z[1]))
    return []  # unreachable: k = len(labels) always covers


def order_path(P: np.ndarray, row: float | None = None) -> np.ndarray:
    """Serpentine order of points: rows along the main direction of the point set (PCA),
    rows ``row`` apart (default: about the median neighbour spacing). Returns indices."""
    P = np.asarray(P, float)
    n = len(P)
    if n <= 2:
        return np.arange(n)
    c = P - P.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u, v = c @ vt[0], c @ vt[1]
    if row is None:
        d = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(n) * 1e9
        row = float(np.median(d.min(axis=1))) or 1e-3
    r = np.round((v - v.min()) / row).astype(int)
    order = []
    for k, rr in enumerate(np.unique(r)):
        idx = np.flatnonzero(r == rr)
        idx = idx[np.argsort(u[idx])]
        order.extend(idx[::-1] if k % 2 else idx)
    return np.array(order)


def unwrap_path(Q: list[np.ndarray], lo: np.ndarray, hi: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Choose +-2*pi copies of every joint (within limits) so consecutive solutions move as
    little as possible (dynamic programming over the copies of each point). Returns the
    chosen joint vectors and the number of remaining jumps > WRAP_JUMP_DEG in joints 4/6."""
    if not Q:
        return [], 0
    cands = []
    for q in Q:
        opts = [np.asarray(q, float)]
        for j in range(len(q)):
            new = []
            for o in opts:
                for k in (-1, 1):
                    x = o.copy()
                    x[j] += 2 * np.pi * k
                    if lo[j] - 1e-9 <= x[j] <= hi[j] + 1e-9:
                        new.append(x)
            opts += new
        cands.append(np.unique(np.round(np.array(opts), 12), axis=0))
    # DP: cost = sum of the largest joint move between neighbours (wrap joints weighted 2x)
    w = np.ones(len(Q[0]))
    w[list(WRAP_JOINTS)] = 2.0
    cost = np.zeros(len(cands[0]))
    back = []
    for a, b in zip(cands[:-1], cands[1:], strict=True):
        step = (np.abs(b[None, :, :] - a[:, None, :]) * w).max(axis=2)  # (A, B)
        tot = cost[:, None] + step
        back.append(tot.argmin(axis=0))
        cost = tot.min(axis=0)
    k = int(cost.argmin())
    path = [cands[-1][k]]
    for i in range(len(back) - 1, -1, -1):
        k = int(back[i][k])
        path.append(cands[i][k])
    path.reverse()
    jumps = 0
    for a, b in zip(path[:-1], path[1:], strict=True):
        if np.degrees(np.abs(b - a)[list(WRAP_JOINTS)]).max() > WRAP_JUMP_DEG:
            jumps += 1
    return path, jumps
