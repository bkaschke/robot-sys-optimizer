"""Least-joint-motion point order (v8 T14, Results -> Paths -> Point order).

The pattern order (core.patterns) follows the surface: rows, serpentines, spirals. For an arm
the cheapest order can differ: points next to each other on the surface can be far apart in
joint space (a wrist swing), and vice versa. This orders the points of each sequence (a zone /
configuration group) so the sum of the joint move times between neighbours is smallest, an
open path from the sequence's first point (the pattern's start, so groups still chain from
corner to corner). Joint move times are synchronized trapezoids (core.cycle_time).

The joint solution of every point is the planned one: its configuration was chosen by the
zones, and a point's other solutions are not kept after planning (a full generalized TSP over
IK branches would need them). Solved with OR-Tools routing (guided local search, a time limit
per sequence) up to ``MAX_POINTS`` points; the result is kept only when it beats the pattern
order. Moves between points that are not surface neighbours lift off the surface.
"""

from __future__ import annotations

import numpy as np

from core.cycle_time import CycleSettings, joint_move_time

MAX_POINTS = 200
TIME_LIMIT_S = 2


def _cost_matrix(Q: np.ndarray, cs: CycleSettings, v_limits) -> np.ndarray:
    n = len(Q)
    C = np.zeros((n, n))
    for i in range(n):
        d = np.mod(Q - Q[i] + np.pi, 2 * np.pi) - np.pi
        C[i] = [joint_move_time(row, cs, v_limits) for row in d]
    return C


def path_cost(order, C: np.ndarray) -> float:
    return float(sum(C[a, b] for a, b in zip(order[:-1], order[1:], strict=True)))


def joint_order(Q: np.ndarray, cs: CycleSettings | None = None, v_limits=None) -> list[int]:
    """Order of the rows of ``Q`` (joint vectors) from row 0, least total joint move time."""
    n = len(Q)
    if n <= 2 or n > MAX_POINTS:
        return list(range(n))
    C = _cost_matrix(Q, cs or CycleSettings(), v_limits)
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    # open path from node 0: a dummy end node n, free to reach from every node
    m = n + 1
    M = np.zeros((m, m), np.int64)
    M[:n, :n] = np.round(C * 1000).astype(np.int64)  # ms
    manager = pywrapcp.RoutingIndexManager(m, 1, [0], [n])
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(lambda i, j: int(M[manager.IndexToNode(i), manager.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = TIME_LIMIT_S
    sol = routing.SolveWithParameters(p)
    if sol is None:
        return list(range(n))
    out, i = [], routing.Start(0)
    while not routing.IsEnd(i):
        out.append(manager.IndexToNode(i))
        i = sol.Value(routing.NextVar(i))
    order = [k for k in out if k < n]
    return order if path_cost(order, C) < path_cost(list(range(n)), C) - 1e-9 else list(range(n))


def reorder(
    seqs: dict, results: dict, points: dict, cs=None, v_limits=None, spacing: float | None = None
) -> dict:
    """``seqs`` {stop: [core.path_order.Seq]} re-ordered by joint motion (new Seq objects).
    A move longer than 1.5 x the point spacing (not to a surface neighbour) lifts."""
    from core.path_order import Seq
    from core.patterns import point_spacing

    if spacing is None:
        P = np.array([v[0] for v in points.values()], float).reshape(-1, 3)
        spacing = point_spacing(P) if len(P) > 1 else 0.0
    out = {}
    for k, ss in seqs.items():
        new = []
        for s in ss:
            Q = np.array([np.asarray(results[p].q, float) for p in s.pids])
            o = joint_order(Q, cs, v_limits)
            pids = [s.pids[i] for i in o]
            if o == list(range(len(o))):
                new.append(s)
                continue
            P = np.array([points[p][0] for p in pids], float)
            gaps = np.linalg.norm(np.diff(P, axis=0), axis=1)
            new.append(Seq(s.zone, pids, [bool(g > 1.5 * spacing) for g in gaps]))
        out[k] = new
    return out
