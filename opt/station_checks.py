"""Exact verification of a station set and the order of the stops (split from
opt.placement, which re-exports these names)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from core.feasibility import STAGES, FeasibilityChecker, PointResult

if TYPE_CHECKING:
    from opt.placement import Targets


class CheckMemo:
    """Per-plan cache of exact point checks (``check_point``, ``check_point_configs``, and so
    the localization re-checks of ``_robust``), keyed by the robot base pose rounded to 0.1 mm
    / 0.01 deg, the point id and the call's options. Station counts and the refinement revisit
    the same stops, so their checks are reused. ``plan_stations`` installs one on the checker
    and removes it at the end: a memo never outlives one plan (the setup can change between
    plans), and point ids stand for fixed points only within a plan.

    Results are copied in and out, because zoning and the paths edit them in place."""

    POS = 1e-4  # m
    ROT = np.radians(0.01)  # rotation matrix entries ~ angle in rad

    def __init__(self):
        self.data: dict = {}
        self.hits = self.misses = 0

    def key(self, T: np.ndarray) -> tuple:
        return (
            tuple(np.round(T[:3, 3] / self.POS).astype(np.int64).tolist()),
            tuple(np.round(T[:3, :3].ravel() / self.ROT).astype(np.int64).tolist()),
        )

    def get(self, checker: FeasibilityChecker, call: tuple, compute):
        k = (self.key(checker.T_world_base), *call)
        if k in self.data:
            self.hits += 1
        else:
            self.misses += 1
            self.data[k] = _copy(compute())
        return _copy(self.data[k])


def _copy(r):
    if isinstance(r, dict):
        return {lab: _copy(v) for lab, v in r.items()}
    return replace(r, q=None if r.q is None else r.q.copy(), notes=list(r.notes))


def _robust(checker, t: Targets, i: int, r: PointResult, v: dict, sites) -> tuple[bool, str]:
    """Extra per-point checks at a station: localization perturbations."""
    perturbed = sites.perturbations(v)
    result = (True, "")
    for vp in perturbed:
        checker.set_base(sites.T(vp))
        r_p = checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i], clearance=False)
        if r_p.status == "unreachable":
            result = (False, "localization")
            break
    if perturbed:
        checker.set_base(sites.T(v))
    return result


def _call(T, t: Targets, i: int, clearance: bool = True) -> tuple:
    return (T, "point", int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i], None, clearance)


def _prefetch_point(checker, t: Targets, pairs: list[tuple[int, dict]], sites) -> None:
    """v8 T2: compute ``check_point`` for (point index, stop) pairs in the checker's pool, then
    the localization re-checks of the points that passed (no-op without a pool)."""
    if getattr(checker, "pool", None) is None or checker.memo is None:
        return
    from opt.parallel import prefetch

    prefetch(checker, [_call(sites.T(v), t, i) for i, v in pairs])
    memo, extra = checker.memo, []
    for i, v in pairs:
        r = memo.data.get((memo.key(sites.T(v)), "point", int(t.ids[i]), None, True))
        if r is not None and r.status != "unreachable":
            extra += [_call(sites.T(vp), t, i, clearance=False) for vp in sites.perturbations(v)]
    prefetch(checker, extra)


def verify(
    checker: FeasibilityChecker, t: Targets, stations: list[dict], sites, preferred: np.ndarray
) -> tuple[dict[int, int], dict[int, PointResult]]:
    """Exact check of every target. Pass 1: each target from its preferred station.
    Pass 2: targets that failed there are tried from every other station."""
    _prefetch_point(
        checker,
        t,
        [(int(i), stations[int(k)]) for i, k in enumerate(preferred) if 0 <= k < len(stations)],
        sites,
    )
    assignment: dict[int, int] = {}
    results: dict[int, PointResult] = {}
    why: dict[int, str] = {}
    notes: dict[int, list[str]] = {}  # what the arm hits, with the "collision" reason

    def attempt(i: int, k: int) -> bool:
        r = checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i])
        if r.status == "unreachable":  # v7 round 2: keep why: the furthest check any stop got to
            prev = why.get(i)
            if prev is None or (prev in STAGES and STAGES.index(r.reason) > STAGES.index(prev)):
                why[i] = r.reason
                notes[i] = list(r.notes)
        else:
            ok, reason = _robust(checker, t, i, r, stations[k], sites)
            if ok:
                results[int(t.ids[i])], assignment[int(t.ids[i])] = r, k
                return True
            why[i] = reason
        return False

    failed: list[int] = []
    for k in range(len(stations)):
        checker.set_base(sites.T(stations[k]))
        failed += [int(i) for i in np.flatnonzero(preferred == k) if not attempt(int(i), k)]
    for k in range(len(stations)):
        if not failed:
            break
        _prefetch_point(checker, t, [(i, stations[k]) for i in failed if preferred[i] != k], sites)
        checker.set_base(sites.T(stations[k]))
        failed = [i for i in failed if preferred[i] == k or not attempt(i, k)]
    for i in failed:
        pid = int(t.ids[i])
        results[pid] = PointResult(pid, "unreachable", why.get(i, "no station"), notes=notes.get(i, []))
    return assignment, results


def path_travel(stations: list[dict], sites) -> float:
    return float(sum(sites.travel(a, b) for a, b in zip(stations[:-1], stations[1:], strict=True)))


def order_by(stations: list[dict], sites, start: dict | None = None) -> list[int]:
    """Shortest open path through stations in the carrier's travel metric."""
    k = len(stations)
    if k <= 1:
        return list(range(k))
    D = np.array([[sites.travel(a, b) for b in stations] for a in stations])
    s0 = np.array([sites.travel(start, b) for b in stations]) if start else np.zeros(k)
    best, best_len = list(range(k)), np.inf
    perms = itertools.permutations(range(k)) if k <= 7 else [tuple(_nn(D))]
    for perm in perms:
        length = s0[perm[0]] + sum(D[a, b] for a, b in zip(perm[:-1], perm[1:], strict=True))
        if length < best_len:
            best, best_len = list(perm), length
    return best


def _nn(D: np.ndarray) -> list[int]:
    left, path = list(range(1, len(D))), [0]
    while left:
        nxt = min(left, key=lambda j: D[path[-1], j])
        path.append(nxt)
        left.remove(nxt)
    return path


def sites_start(sites, T_world_base0: np.ndarray) -> dict | None:
    """Current base as the travel start, expressed in the sites' DOFs (best effort)."""
    if sites.kind == "floor":
        return {"x": float(T_world_base0[0, 3]), "y": float(T_world_base0[1, 3])}
    if hasattr(sites, "start"):  # e.g. the mobile contour (opt.contour): nearest path point
        return sites.start(T_world_base0)
    v = sites.c.values_for_base(T_world_base0)
    return {d: v[d] for d in sites.dofs if d in v} or None
