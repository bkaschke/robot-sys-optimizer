"""Collision-free joint moves (to / from home, and configuration changes at a stop).

A straight joint move is used when it is clear; otherwise a bidirectional RRT (RRT-connect)
in joint space finds a way around the part, floor, stay-out zones and carrier bodies, and
the path is shortened by shortcuts. Every edge is checked every ``step`` of the largest
joint. The two end poses themselves are not checked (they are the planned stop poses /
home, flagged elsewhere). Pure numpy: ``collides(q)`` is the only scene access.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def _edge_ok(collides, q1: np.ndarray, q2: np.ndarray, step: float, check_end: bool) -> bool:
    n = max(1, int(np.ceil(np.max(np.abs(q2 - q1)) / step)))
    last = n + 1 if check_end else n
    return not any(collides(q1 + (i / n) * (q2 - q1)) for i in range(1, last))


def densify(path: list[np.ndarray], step: float) -> list[np.ndarray]:
    """Joint samples along the waypoints every ``step`` (start excluded, end included)."""
    out = []
    for q1, q2 in zip(path[:-1], path[1:], strict=True):
        n = max(1, int(np.ceil(np.max(np.abs(q2 - q1)) / step)))
        out += [q1 + (i / n) * (q2 - q1) for i in range(1, n + 1)]
    return out


def plan_joint_path(
    collides: Callable[[np.ndarray], bool],
    qa,
    qb,
    lo,
    hi,
    step: float = np.radians(3.0),
    reach: float = 0.5,
    iters: int = 2000,
    time_limit: float = 4.0,
    seed: int = 0,
) -> list[np.ndarray] | None:
    """Waypoints qa .. qb of a collision-free joint move, or None (none found in time)."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if _edge_ok(collides, qa, qb, step, False):
        return [qa, qb]
    for way in staged(qa, qb):  # one joint group at a time (arm, base, wrist in some order)
        legs = enumerate(zip(way[:-1], way[1:], strict=True))
        if all(_edge_ok(collides, u, v, step, i < len(way) - 2) for i, (u, v) in legs):
            return way
    # sample near the two poses (within half a turn of either), inside the joint limits
    box_lo = np.maximum(lo, np.minimum(qa, qb) - np.pi)
    box_hi = np.minimum(hi, np.maximum(qa, qb) + np.pi)
    rng = np.random.default_rng(seed)
    trees = [([qa], [-1]), ([qb], [-1])]  # (nodes, parent index) from qa and from qb
    t0 = time.monotonic()

    def steer(q_from, q_to):
        d = q_to - q_from
        m = np.max(np.abs(d))
        return q_to if m <= reach else q_from + d * (reach / m)

    def extend(tree, target):
        """Grow ``tree`` toward ``target`` until blocked; index of the last node added."""
        nodes, parent = tree
        i = int(np.argmin([np.max(np.abs(n - target)) for n in nodes]))
        added = None
        while True:
            q = steer(nodes[i], target)
            if collides(q) or not _edge_ok(collides, nodes[i], q, step, False):
                return added
            nodes.append(q)
            parent.append(i)
            i = added = len(nodes) - 1
            if np.max(np.abs(q - target)) < 1e-9:
                return added

    def branch(tree, i):
        nodes, parent = tree
        out = []
        while i != -1:
            out.append(nodes[i])
            i = parent[i]
        return out

    for it in range(iters):
        if time.monotonic() - t0 > time_limit:
            break
        a, b = trees[it % 2], trees[(it + 1) % 2]
        target = rng.uniform(box_lo, box_hi)
        i = extend(a, target)
        if i is None:
            continue
        j = extend(b, a[0][i])
        if j is not None and np.max(np.abs(b[0][j] - a[0][i])) < 1e-9:
            pa, pb = branch(a, i), branch(b, j)
            path = pa[::-1] + pb[1:]  # root of a .. meeting point .. root of b
            if a is trees[1]:
                path = path[::-1]
            return shortcut(collides, path, step, rng)
    return None


def staged(qa: np.ndarray, qb: np.ndarray) -> list[list[np.ndarray]]:
    """Waypoint lists that move one joint group at a time: base (joint 1), arm (2-3) and
    wrist (the rest), in every order (arm first is tried first: it lifts the tool away)."""
    from itertools import permutations

    n = len(qa)
    groups = [(1, 2), (0,), tuple(range(3, n))] if n >= 4 else [(0,), tuple(range(1, n))]
    out = []
    for order in permutations(groups):
        way, q = [qa], qa.copy()
        for g in order:
            q = q.copy()
            q[list(g)] = qb[list(g)]
            if np.max(np.abs(q - way[-1])) > 1e-12:
                way.append(q)
        out.append(way)
    return out


def shortcut(collides, path: list[np.ndarray], step: float, rng, tries: int = 80) -> list[np.ndarray]:
    """Drop waypoints wherever a straight joint move between two of them is clear."""
    path = list(path)
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if _edge_ok(collides, path[i], path[j], step, True):
            path = path[: i + 1] + path[j:]
    return path
