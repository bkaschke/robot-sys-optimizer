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
from opt.set_cover import _solve, greedy_cover, min_stations, set_cover  # noqa: F401
from opt.station_checks import _nn, _robust, order_by, path_travel, sites_start, verify  # noqa: F401


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
    quick_first: bool = True  # verify every k without refinement first, then refine the best
    refine_top: int = 2  # how many of the quick plans get the (slow) exact refinement
    areas: bool = True  # each stop's points one connected area (opt.areas)
    require_all: bool = False  # must be met: every point reached
    require_one_config: bool = False  # must be met: no configuration change inside a group (zoning)
    patience: int = 2  # stop adding stops after this many counts without a better plan
    # coarse to fine (opt.sampling): set cover and refinement on a spread-out subsample of the
    # targets (a quarter, 60..150 points; all of them below 60); verification uses every point
    sample: bool = True
    sample_fraction: float = 0.25
    sample_max: int = 150
    sample_min: int = 60


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
    res = [  # screening: reachable or not + manipulability; no clearance distance needed
        checker.check_point(int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i], max_poses, clearance=False)
        for i in idx
    ]
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

    if len(names) == 1:  # rail s / mobile contour t: a 1-D grid, then a finer one around the best
        r = st.refine_radius
        best_u, best_f = 0.0, f(np.zeros(1))
        for us in (np.linspace(-r, r, 9), None):
            if us is None:
                us = best_u + np.array([-1.0, 1.0]) * r / 8
            for u in us:
                if abs(u) <= r + 1e-12 and u != 0.0:
                    fu = f(np.array([u]))
                    if fu < best_f:
                        best_u, best_f = float(u), fu
        return vals(np.array([best_u]))

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
    # the map is symmetric about the base z axis (opt.reachability): bases whose z axes
    # agree share one lookup whatever their yaw (a mobile base: one lookup, not one per heading)
    rots = np.round(Ts[:, :3, 2], 6)
    for key in np.unique(rots, axis=0):
        rows = np.flatnonzero(np.all(rots == key, axis=1))
        sc, m = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, Ts[rows, :3, 3], Ts[rows[0], :3, :3])
        score[rows], mu[rows] = sc, m
    covered = score >= st.reach_threshold
    keep = prune(covered, mu / rmap.max_manip, st.max_candidates)
    cand = [cand[i] for i in keep]
    covered, mu_n = covered[keep], mu[keep] / rmap.max_manip
    from opt.sampling import farthest_points, repair_cover, sample_size

    n_t = len(t.ids)
    m = sample_size(n_t, st.sample_fraction, st.sample_max, st.sample_min) if st.sample else n_t
    S = farthest_points(t.P_world, m)  # the coarse targets
    in_s = np.zeros(n_t, bool)
    in_s[S] = True
    cov_s, mu_s = covered[:, S], mu_n[:, S]
    out = []
    prev, checker.cancelled = checker.cancelled, cancelled  # every exact check polls Cancel
    try:
        k_min = min(min_stations(cov_s, st.time_limit_s, cancelled), st.max_stations)
        start = sites_start(sites, T_world_base0)
        ks = list(range(k_min, st.max_stations + 1))
        zoning = st.zoning or st.require_one_config

        def make_job(k: int):
            """(k, stations, preferred station per target, coverage) of one station count."""
            chosen = set_cover(cov_s, mu_s, k, st.time_limit_s, cancelled) or greedy_cover(cov_s, mu_s, k)
            if not chosen:
                return None
            if m < n_t:  # a target the sample missed may need one more stop (map-level repair)
                chosen = repair_cover(covered, mu_n, list(chosen))[: max(k, st.max_stations)]
            stations = [dict(cand[i]) for i in chosen]
            order = order_by(stations, sites, start)
            stations, chosen = [stations[i] for i in order], [chosen[i] for i in order]
            sub_cov, sub_mu = covered[chosen], mu_n[chosen]
            return (k, stations, np.argmax(np.where(sub_cov, sub_mu + 1, 0), axis=0), sub_cov)

        def finish(stations: list[dict], first: np.ndarray, k: int, f: float, cov=None):
            """Exact verification (+ zoning) and cost of one station set."""
            assignment, results = verify(checker, t, stations, sites, first)
            split: list[int] = []
            if st.areas and len(stations) > 1:
                from opt.areas import make_areas

                say(f"station areas k={k}", f)
                assignment, results, split = make_areas(
                    checker, t, stations, sites, assignment, results, hint=cov
                )
            zones = []
            from opt.zoning import violations, zone_counts

            if zoning:
                from opt.zoning import zone_solution

                say(f"zoning k={k}", f)
                zones, _ = zone_solution(checker, t, stations, sites, assignment, results, st.zoning_poses)
            unreached = sorted(pid for pid, r in results.items() if r.status == "unreachable")
            groups, flips = zone_counts(zones)
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
                groups=groups,
                wrist_flips=flips,
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
                split_stops=split,
                violations=violations(len(unreached), zones, zoning, st.require_all, st.require_one_config),
            )
            return sol, results

        def refine(
            k: int, stations: list[dict], first: np.ndarray, sub_cov: np.ndarray, f0: float, f1: float
        ):
            stations = list(stations)
            for s_i in range(len(stations)):
                say(f"refine k={k} station {s_i + 1}", f0 + (f1 - f0) * (s_i + 1) / len(stations))
                idx = [i for i in range(len(t.ids)) if first[i] == s_i and sub_cov[s_i, i] and in_s[i]]
                if idx:
                    stations[s_i] = refine_station(
                        checker, t, idx, stations[s_i], sites, st, rmap.max_manip, weights.standoff
                    )
            return stations

        def emit(sol, results) -> None:
            out[:] = [(a, b) for a, b in out if len(a.stations) != len(sol.stations)] + [(sol, results)]
            if on_solution is not None:
                on_solution(sol, results)

        def rank(sol) -> tuple:
            return (bool(sol.violations), len(sol.unreached), sol.cost)

        # one station count after the other (v1 order: refined; else a quick plan at the set-cover
        # stops). Stop once a plan meets the requirements and `patience` more counts gave nothing
        # better (Max stations 30 does not mean 30 full plans).
        jobs, best, stale = [], None, 0
        slow = st.refine and not st.quick_first
        span = 0.8 if slow else 0.5
        for n, k in enumerate(ks):
            if cancelled is not None and cancelled():
                break
            f0 = span * n / len(ks)
            say(f"set cover k={k}", f0)
            job = make_job(k)
            if job is None:
                continue
            jobs.append(job)
            _, stations, first, sub_cov = job
            if slow:
                stations = refine(k, stations, first, sub_cov, f0, f0 + 0.7 * span / len(ks))
            else:
                say(f"quick plan k={k}", f0)
            sol, res = finish(stations, first, k, f0 + span / len(ks), sub_cov)
            emit(sol, res)
            if best is None or rank(sol) < best:
                best, stale = rank(sol), 0
            else:
                stale += 1
            if not best[0] and stale >= st.patience:
                break
        if not slow:
            if st.refine:
                # refine the most promising plans (fewest unreached, then cost); a refined plan
                # replaces the quick one when it is better
                quick = {len(sol.stations): sol for sol, _ in out}
                order = sorted(jobs, key=lambda j: rank(quick[j[0]]) if j[0] in quick else (True, 1e9, 0))
                best = quick[order[0][0]] if order and order[0][0] in quick else None
                top = [  # skip plans that are clearly worse than the best quick one
                    j
                    for j in order[: st.refine_top]
                    if best is None
                    or j[0] not in quick
                    or (
                        len(quick[j[0]].unreached) <= len(best.unreached)
                        and quick[j[0]].cost <= 1.3 * best.cost + 1.0
                    )
                ]
                for n, (k, stations, first, sub_cov) in enumerate(top):
                    f0 = 0.5 + 0.5 * n / len(top)
                    stations = refine(k, stations, first, sub_cov, f0, f0 + 0.4 / len(top))
                    new = finish(stations, first, k, f0, sub_cov)
                    old = quick.get(k)
                    if old is None or rank(new[0]) < rank(old):
                        emit(*new)
    except CancelledError:
        say("cancelled", 1.0)
        return sorted(out, key=lambda sr: (bool(sr[0].violations), sr[0].cost))
    finally:
        checker.cancelled = prev
    say("done", 1.0)
    return sorted(out, key=lambda sr: (bool(sr[0].violations), sr[0].cost))
