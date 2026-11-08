"""Exact verification of a station set and the order of the stops (split from
opt.placement, which re-exports these names)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from core.feasibility import FeasibilityChecker, PointResult

if TYPE_CHECKING:
    from opt.placement import Targets


def _robust(checker, t: Targets, i: int, r: PointResult, v: dict, sites) -> tuple[bool, str]:
    """Extra per-point checks at a station: tip-over, then localization perturbations."""
    ok, why = sites.point_ok(v, r, checker, t.N_world[i])
    if not ok:
        return False, why
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


def verify(
    checker: FeasibilityChecker, t: Targets, stations: list[dict], sites, preferred: np.ndarray
) -> tuple[dict[int, int], dict[int, PointResult]]:
    """Exact check of every target. Pass 1: each target from its preferred station.
    Pass 2: targets that failed there are tried from every other station."""
    assignment: dict[int, int] = {}
    results: dict[int, PointResult] = {}
    why: dict[int, str] = {}

    def attempt(i: int, k: int) -> bool:
        r = checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i])
        if r.status != "unreachable":
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
        checker.set_base(sites.T(stations[k]))
        failed = [i for i in failed if preferred[i] == k or not attempt(i, k)]
    for i in failed:
        pid = int(t.ids[i])
        results[pid] = PointResult(pid, "unreachable", why.get(i, "no station"))
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
