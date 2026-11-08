"""Set cover for the station planner (CP-SAT, greedy fallback, exact minimum count); split
from opt.placement, which re-exports these names."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
