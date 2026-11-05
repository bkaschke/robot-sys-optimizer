"""Station placement (plan § Solver strategy 3-4, M7).

Pipeline for a fixed tool geometry:
  1. candidate bases on a floor grid around the targets, minus the part footprint;
  2. map coverage matrix (B x T) from the reachability map;
  3. set cover with CP-SAT for every station count k = k_min .. max_stations
     (min stations, then max sum of best manipulability via an assignment model;
     if k_min > max_stations: max coverage with max_stations);
  4. order stations by a small open TSP;
  5. refine each station with CMA-ES on the exact checker (collision + torque);
  6. final exact verification and cost; each solution is streamed via ``on_solution``.

Base yaw is irrelevant for full-turn first joints (see opt/reachability.py) and is kept.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.feasibility import FeasibilityChecker, PointResult
from core.schema import Solution, Station, TargetGroup, Weights
from core.transforms import make_T
from opt.cost import cost_breakdown, path_length
from opt.reachability import ReachMap, coverage_matrix


@dataclass
class PlannerSettings:
    max_stations: int = 3
    candidate_step: float = 0.1  # m
    search_margin: float = 2.5  # m around the targets' bounding box
    reach_threshold: float = 0.5  # map score counted as "covered"
    footprint_margin: float = 0.25  # m: robot base radius + clearance around the part footprint
    max_candidates: int = 400
    refine: bool = True
    refine_evals: int = 24
    refine_sigma: float = 0.08  # m
    refine_radius: float = 0.3  # m: refined base stays within this of the set-cover cell
    screen_poses: int = 40  # candidate TCP poses per point during refinement
    time_limit_s: float = 10.0


@dataclass
class Targets:
    """Targets in the world frame plus what the exact checker needs (part frame)."""

    ids: np.ndarray
    P_world: np.ndarray
    N_world: np.ndarray
    P_part: np.ndarray
    N_part: np.ndarray
    groups: list[TargetGroup]  # per point
    cones: np.ndarray


# --- 1-2: candidates and coverage ------------------------------------------------------------


def candidate_bases(t: Targets, footprint_xy: np.ndarray | None, z: float, st: PlannerSettings) -> np.ndarray:
    lo = t.P_world[:, :2].min(axis=0) - st.search_margin
    hi = t.P_world[:, :2].max(axis=0) + st.search_margin
    xs = np.arange(lo[0], hi[0] + 1e-9, st.candidate_step)
    ys = np.arange(lo[1], hi[1] + 1e-9, st.candidate_step)
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    if footprint_xy is not None:  # axis-aligned part footprint (lo, hi), grown by the base radius
        flo, fhi = footprint_xy[0] - st.footprint_margin, footprint_xy[1] + st.footprint_margin
        inside = np.all((xy >= flo) & (xy <= fhi), axis=1)
        xy = xy[~inside]
    return np.column_stack([xy, np.full(len(xy), z)])


def prune(covered: np.ndarray, mu: np.ndarray, keep: int) -> np.ndarray:
    """Candidate indices worth giving the solver: best overall plus best per target."""
    score = covered.sum(1) + (mu * covered).mean(1)
    chosen = set(np.argsort(-score)[: keep // 2].tolist())
    for t in range(covered.shape[1]):
        col = np.flatnonzero(covered[:, t])
        if len(col):
            best = col[np.argsort(-(score[col] + mu[col, t]))[:5]]
            chosen.update(best.tolist())
    return np.array(sorted(chosen), dtype=int)


# --- 3: set cover ------------------------------------------------------------------------------


def set_cover(covered: np.ndarray, mu: np.ndarray, k: int, time_limit: float = 10.0) -> list[int] | None:
    """Exactly ``k`` (or fewer) bases maximizing coverage, then total best manipulability.

    Returns base indices, or None if CP-SAT finds nothing in the time limit.
    """
    from ortools.sat.python import cp_model

    B, T = covered.shape
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{b}") for b in range(B)]
    m.Add(sum(x) <= k)
    terms_cov, terms_q = [], []
    for t in range(T):
        bs = np.flatnonzero(covered[:, t])
        if len(bs) == 0:
            continue
        y = {b: m.NewBoolVar(f"y{b}_{t}") for b in bs}
        for b, v in y.items():
            m.AddImplication(v, x[b])
        m.Add(sum(y.values()) <= 1)
        cov = m.NewBoolVar(f"c{t}")
        m.Add(sum(y.values()) == cov)
        terms_cov.append(cov)
        terms_q += [int(round(1000 * mu[b, t])) * v for b, v in y.items()]
    # coverage dominates (weight > any possible quality sum), then quality, then fewer bases
    big = 1000 * T + 1
    m.Maximize(big * 1000 * sum(terms_cov) + 1000 * sum(terms_q) - sum(x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [b for b in range(B) if solver.Value(x[b])]


def greedy_cover(covered: np.ndarray, mu: np.ndarray, k: int) -> list[int]:
    left = covered.any(axis=0)
    chosen: list[int] = []
    while left.any() and len(chosen) < k:
        gain = (covered & left).sum(1) + 1e-3 * (mu * (covered & left)).sum(1)
        b = int(np.argmax(gain))
        if gain[b] < 1:
            break
        chosen.append(b)
        left &= ~covered[b]
    return chosen


def min_stations(covered: np.ndarray) -> int:
    """Lower bound: greedy count needed to cover every coverable target (exactness via k loop)."""
    return max(1, len(greedy_cover(covered, np.zeros_like(covered, float), covered.shape[0])))


# --- 4: ordering -------------------------------------------------------------------------------


def order_stations(xy: np.ndarray, start: np.ndarray | None = None) -> list[int]:
    """Shortest open path through the stations (from the station nearest ``start``)."""
    k = len(xy)
    if k <= 1:
        return list(range(k))
    best, best_len = None, np.inf
    perms = itertools.permutations(range(k)) if k <= 7 else [tuple(_nearest_neighbour(xy))]
    for perm in perms:
        pts = xy[list(perm)]
        length = path_length(pts) + (np.linalg.norm(pts[0] - start) if start is not None else 0.0)
        if length < best_len:
            best, best_len = list(perm), length
    return best


def _nearest_neighbour(xy: np.ndarray) -> list[int]:
    left = list(range(1, len(xy)))
    path = [0]
    while left:
        nxt = min(left, key=lambda j: np.linalg.norm(xy[j] - xy[path[-1]]))
        path.append(nxt)
        left.remove(nxt)
    return path


# --- 5-6: exact refinement and verification ------------------------------------------------------


def _station_T(xy, z: float, R: np.ndarray) -> np.ndarray:
    return make_T(R, (xy[0], xy[1], z))


def exact_station_score(
    checker: FeasibilityChecker,
    t: Targets,
    idx: list[int],
    T_base: np.ndarray,
    max_poses: int | None,
    max_manip: float,
) -> tuple[float, list[PointResult]]:
    checker.set_base(T_base)
    res = [checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i], max_poses) for i in idx]
    ok = [r for r in res if r.status != "unreachable"]
    q = np.mean([r.manipulability for r in ok]) / max_manip if ok else 0.0
    return len(ok) + q, res


def refine_station(
    checker: FeasibilityChecker,
    t: Targets,
    idx: list[int],
    xy0: np.ndarray,
    z: float,
    R: np.ndarray,
    st: PlannerSettings,
    max_manip: float,
    footprint_xy,
) -> np.ndarray:
    import cma

    def f(v: np.ndarray) -> float:
        xy = xy0 + v
        if np.linalg.norm(v) > st.refine_radius:
            return 1e3
        if footprint_xy is not None:
            flo, fhi = footprint_xy[0] - st.footprint_margin, footprint_xy[1] + st.footprint_margin
            if np.all((xy >= flo) & (xy <= fhi)):
                return 1e3
        s, _ = exact_station_score(checker, t, idx, _station_T(xy, z, R), st.screen_poses, max_manip)
        return -s

    es = cma.CMAEvolutionStrategy(
        np.zeros(2), st.refine_sigma, {"maxfevals": st.refine_evals, "verbose": -9, "seed": 1, "popsize": 6}
    )
    best_v, best_f = np.zeros(2), f(np.zeros(2))
    while not es.stop():
        vs = es.ask()
        fs = [f(np.asarray(v)) for v in vs]
        es.tell(vs, fs)
        k = int(np.argmin(fs))
        if fs[k] < best_f:
            best_v, best_f = np.asarray(vs[k]), fs[k]
    return xy0 + best_v


def verify(
    checker: FeasibilityChecker,
    t: Targets,
    stations_xy: np.ndarray,
    z: float,
    R: np.ndarray,
    preferred: np.ndarray,
) -> tuple[dict[int, int], dict[int, PointResult]]:
    """Exact check of every target. Pass 1: each target from its preferred station.
    Pass 2: targets that failed there are tried from every other station."""
    assignment: dict[int, int] = {}
    results: dict[int, PointResult] = {}
    failed: list[int] = []
    for k in range(len(stations_xy)):
        checker.set_base(_station_T(stations_xy[k], z, R))
        for i in np.flatnonzero(preferred == k):
            r = checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i])
            if r.status != "unreachable":
                results[int(t.ids[i])], assignment[int(t.ids[i])] = r, k
            else:
                failed.append(int(i))
    for k in range(len(stations_xy)):
        if not failed:
            break
        checker.set_base(_station_T(stations_xy[k], z, R))
        still = []
        for i in failed:
            if preferred[i] == k:
                still.append(i)
                continue
            r = checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i])
            if r.status != "unreachable":
                results[int(t.ids[i])], assignment[int(t.ids[i])] = r, k
            else:
                still.append(i)
        failed = still
    for i in failed:
        pid = int(t.ids[i])
        results[pid] = PointResult(pid, "unreachable", "no station")
    return assignment, results


# --- driver ----------------------------------------------------------------------------------------


def plan_stations(
    rmap: ReachMap,
    checker: FeasibilityChecker,
    t: Targets,
    T_world_base0: np.ndarray,
    footprint_xy: np.ndarray | None,
    weights: Weights,
    st: PlannerSettings,
    tool_length: float = 0.0,
    L_ref: float = 1.0,
    on_solution: Callable[[Solution, dict[int, PointResult]], None] | None = None,
    progress: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[tuple[Solution, dict[int, PointResult]]]:
    say = progress or (lambda msg, f: None)
    z, R = float(T_world_base0[2, 3]), T_world_base0[:3, :3]
    say("coverage", 0.0)
    bases = candidate_bases(t, footprint_xy, z, st)
    score, mu = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, bases, R)
    covered = score >= st.reach_threshold
    keep = prune(covered, mu / rmap.max_manip, st.max_candidates)
    bases, covered, mu_n = bases[keep], covered[keep], mu[keep] / rmap.max_manip
    k_min = min(min_stations(covered), st.max_stations)
    out = []
    ks = list(range(k_min, st.max_stations + 1))
    for n, k in enumerate(ks):
        if cancelled is not None and cancelled():
            break
        say(f"set cover k={k}", n / len(ks))
        chosen = set_cover(covered, mu_n, k, st.time_limit_s) or greedy_cover(covered, mu_n, k)
        if not chosen:
            continue
        xy = bases[chosen, :2]
        order = order_stations(xy, T_world_base0[:2, 3])
        xy, chosen = xy[order], [chosen[i] for i in order]
        # preferred station per target: best map manipulability among covering stations
        sub_cov, sub_mu = covered[chosen], mu_n[chosen]
        first = np.argmax(np.where(sub_cov, sub_mu + 1, 0), axis=0)
        if st.refine:
            for s_i in range(len(xy)):
                say(f"refine k={k} station {s_i + 1}", (n + (s_i + 1) / len(xy)) / len(ks))
                idx = [i for i in range(len(t.ids)) if first[i] == s_i and sub_cov[s_i, i]]
                if idx:
                    xy[s_i] = refine_station(checker, t, idx, xy[s_i], z, R, st, rmap.max_manip, footprint_xy)
        assignment, results = verify(checker, t, xy, z, R, first)
        unreached = sorted(pid for pid, r in results.items() if r.status == "unreachable")
        cost, breakdown = cost_breakdown(
            list(results.values()), len(unreached), xy, weights, tool_length, L_ref, rmap.max_manip
        )
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        plural = "s" if len(xy) > 1 else ""
        sol = Solution(
            label=f"{len(xy)} station{plural}: {len(t.ids) - len(unreached)}/{len(t.ids)} reached",
            stations=[Station(xyz=(float(p[0]), float(p[1]), z), yaw=yaw) for p in xy],
            assignment=assignment,
            unreached=unreached,
            cost=cost,
            breakdown=breakdown,
        )
        out.append((sol, results))
        if on_solution is not None:
            on_solution(sol, results)
    say("done", 1.0)
    return sorted(out, key=lambda sr: sr[0].cost)
