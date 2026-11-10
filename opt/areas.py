"""Station areas (v4 plan item 3): after verification, points are reassigned so that every
stop's points form one connected area on the part and no point of one stop sits among
another stop's points.

The targets get a neighbourhood graph (k nearest within 1.5 x the median point spacing).
Stage 1, region growing: every stop starts at its served point nearest its base and grows
over the graph in rounds, claiming the unclaimed neighbours it reaches exactly (a point two
stops reach in the same round goes to its planned stop, else to the nearer base). Points no
stop grows into keep their planned stop. Stage 2, clean-up rounds propose moves:
  * a point whose neighbours mostly belong to another stop (more of them than to its own)
    moves to that stop;
  * the points of a stop's smaller pieces (connected components other than the largest)
    move to the stop most of their neighbours belong to.
A move happens only when that stop reaches the point exactly (the same checks as
``placement.verify``, localization included). Rounds repeat until nothing moves. Stops that
are still in pieces are reported (``split``). The reassignment is kept only when it leaves
fewer pieces than the planner's own assignment.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.feasibility import FeasibilityChecker, PointResult


def neighbour_graph(P: np.ndarray, k: int = 8, factor: float = 1.5) -> list[set[int]]:
    """Symmetric k-nearest-neighbour graph limited to ``factor`` x the median spacing."""
    n = len(P)
    if n < 2:
        return [set() for _ in range(n)]
    from scipy.spatial import cKDTree

    tree = cKDTree(P)
    d, j = tree.query(P, k=min(k + 1, n))
    radius = factor * float(np.median(d[:, 1]))
    adj = [set() for _ in range(n)]
    for i in range(n):
        for dist, nb in zip(d[i, 1:], j[i, 1:], strict=True):
            if dist <= radius + 1e-12:
                adj[i].add(int(nb))
                adj[int(nb)].add(i)
    return adj


def components(members: list[int], adj: list[set[int]]) -> list[list[int]]:
    left = set(members)
    out = []
    for seed in members:
        if seed not in left:
            continue
        left.discard(seed)
        comp = [seed]
        for c in comp:
            for nb in adj[c]:
                if nb in left:
                    left.discard(nb)
                    comp.append(nb)
        out.append(comp)
    return sorted(out, key=len, reverse=True)


def _proposals(label: np.ndarray, adj: list[set[int]]) -> dict[int, int]:
    """point index -> stop it should move to."""
    out: dict[int, int] = {}
    for i, nbs in enumerate(adj):
        if label[i] < 0:
            continue
        c = Counter(int(label[j]) for j in nbs if label[j] >= 0)
        if not c:
            continue
        best, n_best = max(c.items(), key=lambda kv: (kv[1], -kv[0]))
        if best != label[i] and n_best > c.get(int(label[i]), 0):
            out[i] = best
    for k in np.unique(label[label >= 0]):
        for comp in components(list(np.flatnonzero(label == k)), adj)[1:]:
            cs = set(comp)
            c = Counter(int(label[j]) for i in comp for j in adj[i] if j not in cs and label[j] >= 0)
            if c:
                target = max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                for i in comp:
                    out.setdefault(i, target)
    return out


def split_stops(label: np.ndarray, adj: list[set[int]]) -> list[int]:
    return [
        int(k)
        for k in np.unique(label[label >= 0])
        if len(components(list(np.flatnonzero(label == k)), adj)) > 1
    ]


class _Reach:
    """Exact per-(point, stop) checks, cached; ``set_base`` only when the stop changes."""

    def __init__(self, checker, t, stations, sites, assignment, results, hint=None):
        from opt.placement import _robust

        self.ch, self.t, self.st, self.sites, self._robust = checker, t, stations, sites, _robust
        self.ids = [int(x) for x in t.ids]
        self.cache: dict[tuple[int, int], PointResult | None] = {}
        for i, pid in enumerate(self.ids):
            if pid in assignment:
                self.cache[(i, assignment[pid])] = results[pid]
        self._at = None
        self.hint = hint

    def __call__(self, i: int, k: int) -> PointResult | None:
        key = (i, k)
        if key not in self.cache and self.hint is not None and not self.hint[k, i]:
            return None
        if key not in self.cache:
            if self._at != k:
                self.ch.set_base(self.sites.T(self.st[k]))
                self._at = k
            t = self.t
            r = self.ch.check_point(self.ids[i], t.P_part[i], t.N_part[i], t.groups[i])
            ok = r.status != "unreachable" and self._robust(self.ch, t, i, r, self.st[k], self.sites)[0]
            self.cache[key] = r if ok else None
        return self.cache[key]


def grow(reach: _Reach, P: np.ndarray, base_pos: np.ndarray, label0: np.ndarray, adj) -> np.ndarray:
    """Stage 1: multi-source region growing from each stop's point nearest its base."""
    label = np.full(len(label0), -1)
    frontier: dict[int, list[int]] = {}
    for k in np.unique(label0[label0 >= 0]):
        # seed: the reachable point nearest the base (not only its own points: the planner may
        # give a stop points beyond its neighbour's, and seeds must not start crossed over)
        own = np.flatnonzero(label0 == k)
        seed = int(own[np.argmin(np.linalg.norm(P[own] - base_pos[k], axis=1))])
        for i in np.argsort(np.linalg.norm(P - base_pos[k], axis=1)):
            if label[i] < 0 and label0[i] >= 0 and reach(int(i), int(k)) is not None:
                seed = int(i)
                break
            if i == seed:
                break
        label[seed] = k
        frontier[int(k)] = [seed]
    while any(frontier.values()):
        claims: dict[int, list[int]] = {}
        for k, fr in frontier.items():
            for i in sorted({j for f in fr for j in adj[f] if label[j] < 0 and label0[j] >= 0}):
                if reach(i, k) is not None:
                    claims.setdefault(i, []).append(k)
        frontier = {k: [] for k in frontier}
        for i, ks in claims.items():
            k = (
                int(label0[i])
                if label0[i] in ks
                else min(ks, key=lambda kk: float(np.linalg.norm(P[i] - base_pos[kk])))
            )
            label[i] = k
            frontier[k].append(i)
    left = (label < 0) & (label0 >= 0)
    label[left] = label0[left]
    return label


def _pieces(label: np.ndarray, adj) -> int:
    return sum(len(components(list(np.flatnonzero(label == k)), adj)) for k in np.unique(label[label >= 0]))


def make_areas(
    checker: FeasibilityChecker,
    t,
    stations: list[dict],
    sites,
    assignment: dict[int, int],
    results: dict[int, PointResult],
    max_rounds: int = 6,
    hint: np.ndarray | None = None,
) -> tuple[dict[int, int], dict[int, PointResult], list[int]]:
    """Reassigned (assignment, results) and the stops still split into pieces. ``hint``
    (stops x targets, bool, e.g. the reachability map's coverage) skips the exact check of
    pairs the map rules out (an unreachable point is the slowest check)."""
    ids = [int(x) for x in t.ids]
    label0 = np.array([assignment.get(pid, -1) for pid in ids])
    adj = neighbour_graph(t.P_world)
    if _pieces(label0, adj) == len(np.unique(label0[label0 >= 0])):  # already one area per stop
        return dict(assignment), dict(results), []
    reach = _Reach(checker, t, stations, sites, assignment, results, hint)
    if getattr(checker, "pool", None) is not None:  # v8 T2: every pair the growth may ask for
        from opt.station_checks import _prefetch_point

        pairs = [
            (i, stations[k])
            for k in range(len(stations))
            for i in range(len(ids))
            if (i, k) not in reach.cache and (hint is None or hint[k, i])
        ]
        _prefetch_point(checker, t, pairs, sites)
    base_pos = np.array([sites.T(v)[:3, 3] for v in stations])
    label = grow(reach, t.P_world, base_pos, label0, adj)
    tried: set[tuple[int, int]] = set()
    for _ in range(max_rounds):  # stage 2: clean-up
        prop = {i: k for i, k in _proposals(label, adj).items() if (i, k) not in tried}
        moved = 0
        for i, k in sorted(prop.items(), key=lambda ik: (ik[1], ik[0])):
            if reach(i, k) is not None:
                label[i] = k
                moved += 1
            else:
                tried.add((i, k))
        if not moved:
            break
    if _pieces(label, adj) >= _pieces(label0, adj):  # no better: keep the planner's assignment
        label = label0
    out_a, out_r = {}, dict(results)
    for i, pid in enumerate(ids):
        if label[i] >= 0:
            out_a[pid] = int(label[i])
            out_r[pid] = reach(i, int(label[i]))
    if reach._at is not None:
        checker.set_base(sites.T(stations[reach._at]))
    return out_a, out_r, split_stops(label, adj)
