"""Point order over every arm configuration (Results -> Paths -> Point order = Least joint
motion (GTSP)): a generalized TSP whose clusters are the points and whose nodes are each
point's valid joint solutions.

The planner keeps one joint solution per point. At path-build time the same TCP pose is solved
again for every IK branch (``branch_options``): copies inside the joint limits (margin
included), not singular, not colliding, the planned one always among them. The route picks
one solution per point and the order of the points together so the total of the synchronized
joint move times (core.cycle_time) is smallest; a move between two different configurations
also pays ``reconfigure_s`` (the arm backs off and approaches again, ``RETREAT`` each way).

Zoned plans keep their zones: each zone's points may only take the zone's configuration, so
each zone is solved alone (as before, the planned branch). Plans without zones (*best per
point*) are solved per stop over all its points; the route is split into one sequence per run
of the same configuration. The route starts at the pattern's first point (groups and stops
still chain corner to corner) and is kept only when it beats the planner's order and branches.

OR-Tools routing: one disjunction (visit exactly one node) per point, guided local search
with a time limit; at most ``MAX_NODES`` solution nodes per problem (else the fixed-branch
order of core.path_order_gtsp).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.cycle_time import CycleSettings, trapezoid

MAX_NODES = 1200
TIME_LIMIT_S = 3
BIG = 10**9


def move_times(qa: np.ndarray, Q: np.ndarray, cs: CycleSettings, v_limits=None) -> np.ndarray:
    """Synchronized joint move time from ``qa`` to every row of ``Q`` (wrapped differences),
    vectorized core.cycle_time.joint_move_time."""
    d = np.abs(np.mod(np.asarray(Q, float) - qa + np.pi, 2 * np.pi) - np.pi)
    v = np.full(d.shape[1], cs.joint_speed)
    if v_limits is not None:
        v = np.minimum(v, np.asarray(v_limits, float)[: d.shape[1]])
    a = cs.joint_acc
    if a <= 0:
        return np.max(d / v, axis=1)
    t = np.where(d <= v * v / a, 2.0 * np.sqrt(d / a), d / v + v / a)
    return np.max(np.where(d > 0, t, 0.0), axis=1)


def reconfigure_s(cs: CycleSettings) -> float:
    from core.paths import RETREAT

    return 2.0 * trapezoid(RETREAT, cs.approach_speed, cs.linear_acc)


def branch_options(checker, r, label_of) -> list[tuple[str, np.ndarray]]:
    """(configuration label, q) of every valid joint solution of point result ``r``'s TCP pose
    at the checker's current base; the planned solution first."""
    from core.ik import best_copies

    q0 = np.asarray(r.q, float)
    out = [(label_of(q0), q0)]
    T = checker.robot.fk(q0)
    try:
        sols = np.atleast_2d(np.asarray(checker.ik.solve(T), float))
    except Exception:  # a solver that refuses this pose: keep the planned branch
        return out
    if sols.size == 0:
        return out
    qs, inside = best_copies(sols, checker.lo, checker.hi, checker.robot.lower, checker.robot.upper)
    seen = {out[0][0]}
    for q in qs[inside]:
        lab = label_of(q)
        if lab in seen:
            continue
        fk = checker.robot.fk(q)
        if not np.allclose(fk, T, atol=1e-6):  # a bad branch of the solver
            continue
        if float(np.asarray(checker.ik.manipulability(q[None]), float)[0]) < 1e-6:
            continue
        if checker.scene.collides(q, r.tcp):
            continue
        seen.add(lab)
        out.append((lab, q))
    return out


def solve_gtsp(
    clusters: list[list[np.ndarray]],
    labels: list[list[str]],
    cs: CycleSettings,
    v_limits=None,
) -> tuple[list[tuple[int, int]], float]:
    """Route over clusters (points) from cluster 0, one node (solution) per cluster.
    Returns ([(cluster, option)], cost s). Option 0 of each cluster is its planned solution."""
    nodes = [(c, o) for c, opts in enumerate(clusters) for o in range(len(opts))]
    Q = np.array([clusters[c][o] for c, o in nodes], float)
    L = [labels[c][o] for c, o in nodes]
    n = len(nodes)
    rec = reconfigure_s(cs)
    C = np.zeros((n, n))
    for i in range(n):
        C[i] = move_times(Q[i], Q, cs, v_limits) + rec * np.array([lab != L[i] for lab in L])

    def cost(route) -> float:
        idx = {nd: i for i, nd in enumerate(nodes)}
        ids = [idx[x] for x in route]
        return float(sum(C[a, b] for a, b in zip(ids[:-1], ids[1:], strict=True)))

    given = [(c, 0) for c in range(len(clusters))]
    if len(clusters) <= 1 or n > MAX_NODES:
        return given, cost(given)
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    S, E = n, n + 1  # dummy start (to cluster 0 only) and free end
    M = np.zeros((n + 2, n + 2), np.int64)
    M[:n, :n] = np.round(C * 1000).astype(np.int64)
    M[S, :n] = [0 if c == 0 else BIG for c, _ in nodes]
    M[:n, S] = BIG
    M[E, :] = BIG
    manager = pywrapcp.RoutingIndexManager(n + 2, 1, [S], [E])
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(lambda i, j: int(M[manager.IndexToNode(i), manager.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    for c in range(len(clusters)):
        ids = [manager.NodeToIndex(i) for i, (cc, _) in enumerate(nodes) if cc == c]
        routing.AddDisjunction(ids, BIG, 1)  # exactly one solution of every point
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = TIME_LIMIT_S
    sol = routing.SolveWithParameters(p)
    if sol is None:
        return given, cost(given)
    route, i = [], routing.Start(0)
    while not routing.IsEnd(i):
        k = manager.IndexToNode(i)
        if k < n:
            route.append(nodes[k])
        i = sol.Value(routing.NextVar(i))
    if len(route) != len(clusters) or route[0][0] != 0:
        return given, cost(given)
    c_new, c_old = cost(route), cost(given)
    return (route, c_new) if c_new < c_old - 1e-9 else (given, c_old)


def reorder_branches(
    seqs: dict, results: dict, points: dict, checker, base_poses, label_of, cs=None, v_limits=None
) -> tuple[dict, dict]:
    """``seqs`` {stop: [Seq]} -> (new seqs, results with the chosen solutions). Zoned
    sequences (``Seq.zone`` set) keep their configuration; unzoned stops are solved whole."""
    from core.path_order import Seq
    from core.patterns import point_spacing

    cs = cs or CycleSettings()
    P = np.array([v[0] for v in points.values()], float).reshape(-1, 3)
    spacing = point_spacing(P) if len(P) > 1 else 0.0
    res = dict(results)
    out = {}

    def lifts(pids) -> list[bool]:
        X = np.array([points[p][0] for p in pids], float)
        return [bool(g > 1.5 * spacing) for g in np.linalg.norm(np.diff(X, axis=0), axis=1)]

    for k, ss in seqs.items():
        checker.set_base(base_poses[k])
        zoned = any(s.zone is not None for s in ss)
        problems = [[s] for s in ss] if zoned else [ss]
        new = []
        for group in problems:
            pids = [p for s in group for p in s.pids]
            opts = [branch_options(checker, results[p], label_of) for p in pids]
            if zoned:  # the zone's configuration only
                opts = [[o for o in op if o[0] == op[0][0]] for op in opts]
            route, _ = solve_gtsp(
                [[q for _, q in op] for op in opts], [[lab for lab, _ in op] for op in opts], cs, v_limits
            )
            if route == [(c, 0) for c in range(len(pids))]:
                new += group  # nothing better: the planner's order and branches
                continue
            run, run_label = [], None
            for c, o in route:
                lab, q = opts[c][o]
                p = pids[c]
                if o != 0:
                    res[p] = replace(results[p], q=np.asarray(q, float), config=lab)
                if run and lab != run_label:
                    new.append(Seq(group[0].zone, run, lifts(run)))
                    run = []
                run.append(p)
                run_label = lab
            if run:
                new.append(Seq(group[0].zone, run, lifts(run)))
        out[k] = new
    return out, res
