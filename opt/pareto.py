"""Pareto fronts (v8 T11): plans no other plan beats on every chosen measure.

Measures (``OBJECTIVES``): stations, estimated cycle time, worst clearance, tool length. A
plan *dominates* another when it is no worse on every measure and better on at least one;
the front is the set of plans nobody dominates. Plans that miss more points than the best
plan never make the front (reaching the points comes first).

Station planner: the normal plans of every station count seed the pool; then, per station
count, Optuna's NSGA-II moves the stops (within the refinement radius, linear DOFs) and each
candidate is verified exactly by the planner's own ``finish`` (verify, areas, zones, cost).
The front over the whole pool is returned.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from core.feasibility import CancelledError

# key -> (label, sense): sense +1 = smaller is better, -1 = larger is better
OBJECTIVES = {
    "stations": ("Stations", 1),
    "cycle": ("Cycle time (s)", 1),
    "clearance": ("Worst clearance (mm)", -1),
    "length": ("Tool length (mm)", 1),
}


def measures(sol, res: dict, tool_length: float | None = None) -> dict[str, float]:
    """Raw values of every measure for a plan (``res``: its PointResults)."""
    ok = [r for r in res.values() if r.status != "unreachable"]
    clear = [r.clearance for r in ok if math.isfinite(r.clearance)]
    return {
        "stations": float(len(sol.stations)),
        "cycle": float(sol.cycle_s) if sol.cycle_s is not None else math.inf,
        "clearance": 1000 * float(min(clear)) if clear else 0.0,
        "length": 1000 * float(tool_length) if tool_length is not None else 0.0,
        "unreached": float(len(sol.unreached)),
    }


def vector(m: dict, keys) -> np.ndarray:
    """Minimization vector of the chosen measures."""
    return np.array([OBJECTIVES[k][1] * m[k] for k in keys], float)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + 1e-9) and np.any(a < b - 1e-9))


def front(items: list, keys, measure_of: Callable) -> list:
    """Non-dominated ``items`` (among those with the fewest unreached points), sorted by the
    first measure. ``measure_of(item)`` -> measures dict. Duplicates (same vector) keep one."""
    if not items:
        return []
    ms = [measure_of(it) for it in items]
    least = min(m["unreached"] for m in ms)
    cand = [(it, vector(m, keys)) for it, m in zip(items, ms, strict=True) if m["unreached"] == least]
    out, seen = [], []
    for it, v in cand:
        if any(dominates(w, v) for _, w in cand):
            continue
        if any(np.allclose(v, w) for w in seen):
            continue
        seen.append(v)
        out.append((it, v))
    out.sort(key=lambda iv: tuple(iv[1]))
    return [it for it, _ in out]


def station_front(
    finish,
    sites,
    t,
    plans: list,
    keys,
    radius: float,
    budget: int,
    seed: int = 1,
    say: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list:
    """Front of ``plans`` [(Solution, results)] plus NSGA-II variations of the best plan of
    each station count. ``finish(stops, first, k, f)`` is opt.placement's plan evaluation."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pool = list(plans)
    say = say or (lambda m, f: None)
    best_per_k: dict[int, tuple] = {}
    for sol, res in plans:
        k = len(sol.stations)
        if k not in best_per_k or (len(sol.unreached), sol.cost) < (
            len(best_per_k[k][0].unreached),
            best_per_k[k][0].cost,
        ):
            best_per_k[k] = (sol, res)
    lin = [d for d in sites.dofs if d != "yaw"] if sites.kind != "floor" else ["x", "y"]
    index = {int(p): i for i, p in enumerate(t.ids)}
    directions = ["minimize"] * len(keys)
    for n, (k, (sol, _res)) in enumerate(sorted(best_per_k.items())):
        base = [
            {d: float(s.carrier.get(d, s.xyz[0] if d == "x" else s.xyz[1])) for d in sites.dofs}
            if sites.kind != "floor"
            else {"x": float(s.xyz[0]), "y": float(s.xyz[1])}
            for s in sol.stations
        ]
        first = np.zeros(len(t.ids), int)
        for pid, kk in sol.assignment.items():
            if int(pid) in index:
                first[index[int(pid)]] = kk
        sampler = optuna.samplers.NSGAIISampler(seed=seed + k, population_size=max(4, min(10, budget // 3)))
        study = optuna.create_study(directions=directions, sampler=sampler)
        for e in range(budget):
            if cancelled is not None and cancelled():
                raise CancelledError()
            say(f"Pareto front: {k} stop(s), candidate {e + 1}/{budget}", (n + e / budget) / len(best_per_k))
            trial = study.ask()
            stops = []
            for j, b in enumerate(base):
                v = dict(b)
                for d in lin:
                    v[d] = b[d] + trial.suggest_float(f"s{j}_{d}", -radius, radius)
                stops.append(v)
            if not all(sites.accept(v)[0] for v in stops):
                study.tell(trial, [1e9] * len(keys))
                continue
            cand_sol, cand_res = finish(stops, first, k, (n + e / budget) / len(best_per_k))
            pool.append((cand_sol, cand_res))
            study.tell(trial, list(vector(measures(cand_sol, cand_res), keys)))
    return front(pool, keys, lambda it: measures(*it))
