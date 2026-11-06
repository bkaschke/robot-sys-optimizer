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

Base yaw is irrelevant to reach for full-turn first joints (see opt/reachability.py); it
still matters for a mobile chassis footprint, so mobile yaw is a DOF. The search space
(free floor or a carrier) is an opt.sites object.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.feasibility import CancelledError, FeasibilityChecker, PointResult
from core.schema import Solution, Station, TargetGroup, Weights
from core.transforms import make_T
from opt.cost import cost_breakdown, path_length
from opt.reachability import coverage_matrix


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
    zoning: bool = False  # group each station's points into joint-configuration zones
    zoning_poses: int = 40  # candidate TCP poses per point when looking for configurations


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


def _solve(m, time_limit: float, cancelled: Callable[[], bool] | None):
    """CP-SAT solve that a watcher thread stops (within 0.1 s) when ``cancelled()`` turns true."""
    import threading

    from ortools.sat.python import cp_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    done = threading.Event()

    def watch() -> None:
        while not done.wait(0.1):
            if cancelled():
                solver.stop_search()
                return

    if cancelled is not None:
        threading.Thread(target=watch, daemon=True).start()
    try:
        return solver, solver.Solve(m)
    finally:
        done.set()


def set_cover(
    covered: np.ndarray,
    mu: np.ndarray,
    k: int,
    time_limit: float = 10.0,
    cancelled: Callable[[], bool] | None = None,
) -> list[int] | None:
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
    solver, status = _solve(m, time_limit, cancelled)
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


def min_stations(
    covered: np.ndarray, time_limit: float = 10.0, cancelled: Callable[[], bool] | None = None
) -> int:
    """Fewest bases covering every coverable target: exact minimum set cover (CP-SAT).

    Falls back to the greedy count (an upper bound, not a lower one) if CP-SAT does not
    finish. Using greedy here would make the planner skip feasible smaller station counts.
    """
    from ortools.sat.python import cp_model

    greedy = max(1, len(greedy_cover(covered, np.zeros_like(covered, float), covered.shape[0])))
    cols = [np.flatnonzero(covered[:, t]) for t in range(covered.shape[1])]
    cols = [c for c in cols if len(c)]
    if not cols:
        return 1
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{b}") for b in range(covered.shape[0])]
    for c in cols:
        m.AddBoolOr([x[b] for b in c])
    m.Minimize(sum(x))
    solver, status = _solve(m, time_limit, cancelled)
    if status == cp_model.OPTIMAL:
        return max(1, int(round(solver.ObjectiveValue())))
    return greedy


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
    v0: dict,
    sites,
    st: PlannerSettings,
    max_manip: float,
    w_standoff: float = 1.0,
) -> dict:
    """CMA-ES over the station's carrier DOFs, scored by the exact checker.

    Positions move within ``refine_radius`` of the start (sigma ``refine_sigma``); a yaw DOF
    (mobile) moves with sigma ~0.3 rad. Stations rejected by the zones score 1e3.
    """
    import cma

    names = list(v0)
    scale = np.array([0.3 / st.refine_sigma if n == "yaw" else 1.0 for n in names])
    x0 = np.array([v0[n] for n in names], float)
    lin_mask = np.array([n != "yaw" for n in names])

    def vals(u: np.ndarray) -> dict:
        return dict(zip(names, map(float, x0 + u[: len(names)] * scale), strict=True))

    def f(u: np.ndarray) -> float:
        u = np.asarray(u)[: len(names)]
        if np.linalg.norm(u[lin_mask]) > st.refine_radius:
            return 1e3
        v = vals(u)
        ok, pen = sites.accept(v)
        if not ok:
            return 1e3
        s, _ = exact_station_score(checker, t, idx, sites.T(v), st.screen_poses, max_manip)
        return -s + 0.1 * w_standoff * pen

    dim = max(2, len(names))  # CMA-ES needs >= 2 dims; extra coordinates are ignored
    es = cma.CMAEvolutionStrategy(
        np.zeros(dim), st.refine_sigma, {"maxfevals": st.refine_evals, "verbose": -9, "seed": 1, "popsize": 6}
    )
    best_u, best_f = np.zeros(dim), f(np.zeros(dim))
    while not es.stop():
        us = es.ask()
        fs = [f(u) for u in us]
        es.tell(us, fs)
        k = int(np.argmin(fs))
        if fs[k] < best_f:
            best_u, best_f = np.asarray(us[k]), fs[k]
    return vals(best_u)


def _robust(checker, t: Targets, i: int, r: PointResult, v: dict, sites) -> tuple[bool, str]:
    """Extra per-point checks at a station: tip-over, then localization perturbations."""
    ok, why = sites.point_ok(v, r, checker, t.N_world[i])
    if not ok:
        return False, why
    perturbed = sites.perturbations(v)
    result = (True, "")
    for vp in perturbed:
        checker.set_base(sites.T(vp))
        if checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i]).status == "unreachable":
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
    v = sites.c.values_for_base(T_world_base0)
    return {d: v[d] for d in sites.dofs if d in v} or None


# --- driver ----------------------------------------------------------------------------------------


def plan_stations(
    rmap,
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
    sites=None,
) -> list[tuple[Solution, dict[int, PointResult]]]:
    """``sites`` defaults to free floor positions (FloorSites); pass CarrierSites for carriers."""
    from opt.sites import FloorSites

    say = progress or (lambda msg, f: None)
    sites = sites or FloorSites(T_world_base0, footprint_xy, st.footprint_margin, st.search_margin)
    say("coverage", 0.0)
    try:
        cand = sites.candidates(t.P_world, st.candidate_step, cancelled)
    except CancelledError:
        say("cancelled", 1.0)
        return []
    if not cand:
        say("no admissible station (standoff / keep-outs / travel range)", 1.0)
        return []
    Ts = np.array([sites.T(v) for v in cand])
    score = np.zeros((len(cand), len(t.ids)))
    mu = np.zeros_like(score)
    rots = np.round(Ts[:, :3, :3].reshape(len(cand), -1), 6)
    for key in np.unique(rots, axis=0):
        rows = np.flatnonzero(np.all(rots == key, axis=1))
        sc, m = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, Ts[rows, :3, 3], Ts[rows[0], :3, :3])
        score[rows], mu[rows] = sc, m
    covered = score >= st.reach_threshold
    keep = prune(covered, mu / rmap.max_manip, st.max_candidates)
    cand = [cand[i] for i in keep]
    covered, mu_n = covered[keep], mu[keep] / rmap.max_manip
    out = []
    prev, checker.cancelled = checker.cancelled, cancelled  # every exact check polls Cancel
    try:
        k_min = min(min_stations(covered, st.time_limit_s, cancelled), st.max_stations)
        start = sites_start(sites, T_world_base0)
        ks = list(range(k_min, st.max_stations + 1))
        for n, k in enumerate(ks):
            if cancelled is not None and cancelled():
                break
            say(f"set cover k={k}", n / len(ks))
            chosen = set_cover(covered, mu_n, k, st.time_limit_s, cancelled) or greedy_cover(covered, mu_n, k)
            if not chosen:
                continue
            stations = [dict(cand[i]) for i in chosen]
            order = order_by(stations, sites, start)
            stations, chosen = [stations[i] for i in order], [chosen[i] for i in order]
            sub_cov, sub_mu = covered[chosen], mu_n[chosen]
            first = np.argmax(np.where(sub_cov, sub_mu + 1, 0), axis=0)
            if st.refine:
                for s_i in range(len(stations)):
                    say(f"refine k={k} station {s_i + 1}", (n + (s_i + 1) / len(stations)) / len(ks))
                    idx = [i for i in range(len(t.ids)) if first[i] == s_i and sub_cov[s_i, i]]
                    if idx:
                        stations[s_i] = refine_station(
                            checker, t, idx, stations[s_i], sites, st, rmap.max_manip, weights.standoff
                        )
            assignment, results = verify(checker, t, stations, sites, first)
            zones, changes = [], 0
            if st.zoning:
                from opt.zoning import zone_solution

                say(f"zoning k={k}", (n + 1) / len(ks))
                zones, changes = zone_solution(
                    checker, t, stations, sites, assignment, results, st.zoning_poses
                )
            unreached = sorted(pid for pid, r in results.items() if r.status == "unreachable")
            Tst = [sites.T(v) for v in stations]
            penalty = float(np.mean([sites.accept(v)[1] for v in stations]))
            cost, breakdown = cost_breakdown(
                list(results.values()),
                len(unreached),
                np.array([T[:2, 3] for T in Tst]),
                weights,
                tool_length,
                L_ref,
                rmap.max_manip,
                travel=path_travel(stations, sites),
                standoff_penalty=penalty,
                config_changes=changes,
            )
            full = getattr(sites, "full", lambda v: v)
            plural = "s" if len(stations) > 1 else ""
            reached = len(t.ids) - len(unreached)
            sol = Solution(
                label=f"{len(stations)} {sites.kind} station{plural}: {reached}/{len(t.ids)} reached",
                stations=[
                    Station(
                        xyz=tuple(float(x) for x in T[:3, 3]),
                        yaw=float(np.arctan2(T[1, 0], T[0, 0])),
                        carrier={a: float(b) for a, b in full(v).items()},
                    )
                    for T, v in zip(Tst, stations, strict=True)
                ],
                assignment=assignment,
                unreached=unreached,
                cost=cost,
                breakdown=breakdown,
                zones=zones,
            )
            out.append((sol, results))
            if on_solution is not None:
                on_solution(sol, results)
    except CancelledError:
        say("cancelled", 1.0)
        return sorted(out, key=lambda sr: sr[0].cost)
    finally:
        checker.cancelled = prev
    say("done", 1.0)
    return sorted(out, key=lambda sr: sr[0].cost)
