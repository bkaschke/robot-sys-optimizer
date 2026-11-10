"""Robust objective (v8 T12): plans scored by their worst result under tolerance.

robust cost = nominal cost + w_unreached x (points lost under tolerance), where the points lost
are aggregated over a few tolerance samples (opt.tolerance) as the **worst case** (most points
lost in any sample) or **CVaR 20 %** (the mean of the worst fifth of the samples). A plan whose
points sit at the edge of reach loses some when the part or a stop is off and pays for it.

Station planner: the normal plans of every station count, then per station count a TPE search
(opt.refine_bayes) moves the stops within the refinement radius to lower the robust cost;
every candidate is verified exactly by the planner's own ``finish``. The best plan of each
station count is returned, sorted by robust cost (its tolerance term is in the breakdown as
``robust``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.feasibility import CancelledError
from opt.tolerance import ToleranceSettings, tolerance_check

MEASURES = ("Worst case", "CVaR 20 %")


def aggregate(lost: list[int], measure: str) -> float:
    if not lost:
        return 0.0
    a = np.sort(np.asarray(lost, float))[::-1]
    if measure == "CVaR 20 %":
        return float(a[: max(1, int(np.ceil(0.2 * len(a))))].mean())
    return float(a[0])


def stops_of(sol, sites) -> list[dict]:
    """The planner's DOF dicts of a Solution's stations."""
    if sites.kind == "floor":
        return [{"x": float(s.xyz[0]), "y": float(s.xyz[1])} for s in sol.stations]
    return [{d: float(s.carrier[d]) for d in sites.dofs if d in s.carrier} for s in sol.stations]


def robust_cost(
    checker, t, sol, res, T_stops, tol: ToleranceSettings, measure: str, w_unreached: float, pool=None
):
    """(robust cost, ToleranceResult)."""
    r = tolerance_check(checker, t, sol, res, T_stops, tol, pool=pool)
    return float(sol.cost) + w_unreached * aggregate(r.lost, measure), r


def station_robust(
    finish,
    sites,
    checker,
    t,
    plans: list,
    tol: ToleranceSettings,
    measure: str,
    w_unreached: float,
    radius: float,
    budget: int,
    say: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list:
    """[(Solution, results)] sorted by robust cost (best first); see the module docstring."""
    from opt.refine_bayes import make_search

    say = say or (lambda m, f: None)
    pool = checker.pool
    scored = []

    def score(sol, res, stops) -> float:
        if cancelled is not None and cancelled():
            raise CancelledError()
        c, _ = robust_cost(checker, t, sol, res, [sites.T(v) for v in stops], tol, measure, w_unreached, pool)
        sol.breakdown = {**sol.breakdown, "robust": c - sol.cost}
        scored.append((c, sol, res))
        return c

    best_per_k: dict[int, tuple] = {}
    for sol, res in plans:
        c = score(sol, res, stops_of(sol, sites))
        k = len(sol.stations)
        if k not in best_per_k or c < best_per_k[k][0]:
            best_per_k[k] = (c, sol, res)
    index = {int(p): i for i, p in enumerate(t.ids)}
    lin = [d for d in sites.dofs if d != "yaw"] if sites.kind != "floor" else ["x", "y"]
    for n, (k, (_c, sol, _res)) in enumerate(sorted(best_per_k.items())):
        base = stops_of(sol, sites)
        first = np.zeros(len(t.ids), int)
        for pid, kk in sol.assignment.items():
            if int(pid) in index:
                first[index[int(pid)]] = kk
        d = len(base) * len(lin)
        search = make_search("tpe", np.zeros(d), -radius * np.ones(d), radius * np.ones(d), budget, 1 + k)
        e = 0
        while not search.stop():
            us = search.ask()
            fs = []
            for u in us:
                e += 1
                say(f"robust plan: {k} stop(s), candidate {e}/{budget}", (n + e / budget) / len(best_per_k))
                stops = [dict(b) for b in base]
                for j, v in enumerate(stops):
                    for i, dof in enumerate(lin):
                        v[dof] = base[j][dof] + float(u[j * len(lin) + i])
                if not all(sites.accept(v)[0] for v in stops):
                    fs.append(1e9)
                    continue
                cand_sol, cand_res = finish(stops, first, k, (n + e / budget) / len(best_per_k))
                fs.append(score(cand_sol, cand_res, stops))
            search.tell(us, fs)
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _c, sol, res in scored:  # the best robust plan of each station count
        if len(sol.stations) in seen:
            continue
        seen.add(len(sol.stations))
        out.append((sol, res))
    return out


def rerank_designs(ev, seen: list, st, cancelled=None, n: int = 6) -> list:
    """Tool optimizer (v8 T12): the ``n`` cheapest evaluated designs (every target reached
    first) re-ranked by robust cost, each with its own tool's checker; best three returned."""
    from core.export import station_pose
    from core.schema import Base
    from opt.tool_geometry import TOP_DESIGNS, Design

    r = st.robust or {}
    tol = r.get("tolerance") or ToleranceSettings(samples=10)
    measure = r.get("measure", "Worst case")
    ranked = sorted(
        (c for c in seen if c[1] is not None), key=lambda c: (not c[3], len(c[1].unreached), c[1].cost)
    )
    out, names = [], set()
    for v, sol, res, ok, length in ranked:
        name = f"tool {1000 * length:.0f} mm: {sol.label}"
        if name in names:
            continue
        names.add(name)
        if ev.sites is not None:
            T_stops = [ev.sites.T(s) for s in stops_of(sol, ev.sites)]
        else:
            T_stops = [station_pose(Base(), s, ev.T_base0) for s in sol.stations]
        c, _ = robust_cost(ev.checker(v), ev.t, sol, res, T_stops, tol, measure, float(ev.w.unreached))
        out.append(
            (
                c,
                Design(
                    v, sol.model_copy(update={"tool_values": v, "label": f"Robust: {name}"}), res, bool(ok)
                ),
            )
        )
        if len(out) == n or (cancelled is not None and cancelled()):
            break
    out.sort(key=lambda x: x[0])
    return [d for _, d in out[:TOP_DESIGNS]]
