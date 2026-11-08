"""Mobile-base routes between stops around stay-out zones, keep-outs and the part (v5 testing
feedback: the AMR drove straight through stay-out zones between stops).

The base is treated as holonomic (x, y, yaw). ``route`` runs A* on a floor grid (cell ~ a
quarter of the footprint radius) x ``n_yaw`` headings: moves are the 8 neighbours at the same
heading and turning in place by one heading step (cost = footprint radius x angle, the arc
its corners sweep). States and edge midpoints are checked lazily with the caller's exact
``ok(x, y, yaw)`` (footprint against the zones, cached). The grid path is then shortened
(a shortcut is taken only when a densely sampled straight move is ok) and returned as
waypoints from start to goal, both included.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np

Pose = tuple[float, float, float]


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def interpolate(a: Pose, b: Pose, s: float) -> Pose:
    return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), wrap(a[2] + s * wrap(b[2] - a[2])))


def move_length(a: Pose, b: Pose, radius: float) -> float:
    """Travel used to sample a move: translation, or the corners' arc when turning more."""
    return max(float(np.hypot(b[0] - a[0], b[1] - a[1])), radius * abs(wrap(b[2] - a[2])))


class _Checked:
    """``ok`` with a cache on rounded poses (A* revisits states and edge midpoints)."""

    def __init__(self, ok: Callable[[float, float, float], bool]):
        self.ok, self.cache, self.calls = ok, {}, 0

    def __call__(self, p: Pose) -> bool:
        key = (round(p[0], 4), round(p[1], 4), round(p[2], 3))
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = bool(self.ok(*p))
        return self.cache[key]


def segment_ok(ok: _Checked, a: Pose, b: Pose, step: float, radius: float) -> bool:
    n = max(1, int(np.ceil(move_length(a, b, radius) / step)))
    return all(ok(interpolate(a, b, s)) for s in np.arange(1, n + 1) / n)


def route(
    ok: Callable[[float, float, float], bool],
    start: Pose,
    goal: Pose,
    region: tuple[float, float, float, float],
    radius: float,
    cell: float | None = None,
    n_yaw: int = 24,
    max_expand: int = 60_000,
    greed: float = 2.0,
) -> list[Pose] | None:
    """Waypoints start -> goal avoiding poses where ``ok`` is False, or None (no route).
    ``region`` = (xmin, ymin, xmax, ymax) of the floor searched; ``radius`` = footprint radius
    about the base centre (turning cost and check spacing). ``greed`` > 1 weights the heuristic
    (faster, not shortest; the shortcuts afterwards take most of the detour out)."""
    chk = _Checked(ok)
    radius = max(float(radius), 1e-3)
    cell = float(cell or np.clip(radius / 4, 0.05, 0.15))
    fine = cell / 2  # spacing of the exact checks along a straight move
    x0, y0, x1, y1 = region
    nx, ny = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    dyaw = 2 * np.pi / n_yaw

    def pose(s: tuple[int, int, int]) -> Pose:
        return (x0 + s[0] * cell, y0 + s[1] * cell, wrap(-np.pi + s[2] * dyaw))

    def snap(p: Pose) -> list[tuple[int, int, int]]:
        """Grid states near p, nearest first (the start / goal connect to the first ok one)."""
        i, j = (p[0] - x0) / cell, (p[1] - y0) / cell
        h = int(round((wrap(p[2]) + np.pi) / dyaw)) % n_yaw
        out = []
        for di in (0, 1, -1, 2, -2):
            for dj in (0, 1, -1, 2, -2):
                for dh in (0, 1, -1):
                    s = (int(round(i)) + di, int(round(j)) + dj, (h + dh) % n_yaw)
                    if 0 <= s[0] < nx and 0 <= s[1] < ny:
                        out.append(s)
        return sorted(out, key=lambda s: move_length(p, pose(s), radius))

    def link(p: Pose) -> tuple[int, int, int] | None:
        for s in snap(p)[:40]:
            if chk(pose(s)) and segment_ok(chk, p, pose(s), fine, radius):
                return s
        return None

    if not (chk(start) and chk(goal)):
        return None
    if segment_ok(chk, start, goal, fine, radius):
        return [start, goal]
    s0, s1 = link(start), link(goal)
    if s0 is None or s1 is None:
        return None

    def h(s: tuple[int, int, int]) -> float:
        p = pose(s)
        dx, dy = sorted((abs(p[0] - goal[0]), abs(p[1] - goal[1])))
        octile = dy + (np.sqrt(2) - 1) * dx  # 8-connected grid distance
        return greed * (octile + radius * abs(wrap(p[2] - goal[2])))

    moves = [(di, dj, 0, cell * float(np.hypot(di, dj))) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    moves = [m for m in moves if m[0] or m[1]] + [(0, 0, 1, radius * dyaw), (0, 0, -1, radius * dyaw)]
    g = {s0: 0.0}
    parent: dict = {s0: None}
    heap = [(h(s0), 0.0, s0)]
    done: set = set()
    found = False
    while heap and len(done) < max_expand:
        _, gs, s = heapq.heappop(heap)
        if s in done:
            continue
        done.add(s)
        if s == s1:
            found = True
            break
        for di, dj, dh, c in moves:
            t = (s[0] + di, s[1] + dj, (s[2] + dh) % n_yaw)
            if not (0 <= t[0] < nx and 0 <= t[1] < ny) or t in done:
                continue
            gt = gs + c
            if gt >= g.get(t, np.inf):
                continue
            if not chk(pose(t)) or not chk(interpolate(pose(s), pose(t), 0.5)):
                continue
            g[t], parent[t] = gt, s
            heapq.heappush(heap, (gt + h(t), gt, t))
    if not found:
        return None
    states = []
    s = s1
    while s is not None:
        states.append(s)
        s = parent[s]
    path = [start] + [pose(s) for s in reversed(states)] + [goal]
    return shorten(chk, path, fine, radius)


def shorten(chk: _Checked, path: list[Pose], step: float, radius: float) -> list[Pose]:
    """Greedy shortcuts: from each waypoint jump to the farthest one tried (halving the jump)
    that a straight, densely checked move reaches."""
    out, i = [path[0]], 0
    last = len(path) - 1
    while i < last:
        span, j = last - i, i + 1
        while span > 1:
            if segment_ok(chk, path[i], path[i + span], step, radius):
                j = i + span
                break
            span //= 2
        out.append(path[j])
        i = j
    return out


def resample(path: list[Pose], step: float, radius: float) -> list[Pose]:
    """Poses every ``step`` of travel along the waypoints (the start excluded, the goal kept)."""
    out: list[Pose] = []
    for a, b in zip(path, path[1:], strict=False):
        n = max(1, int(np.ceil(move_length(a, b, radius) / step)))
        out += [interpolate(a, b, s) for s in np.arange(1, n + 1) / n]
    return out
