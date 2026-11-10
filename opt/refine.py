"""Exact refinement of one station (plan step 5, split from opt.placement in v8): CMA-ES (or,
v8 T10, a Bayesian search: opt.refine_bayes) over the station's carrier DOFs, scored by the
exact checker on the station's targets. opt.placement re-exports these names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.feasibility import FeasibilityChecker, PointResult
from core.transforms import make_T

if TYPE_CHECKING:
    from opt.placement import PlannerSettings, Targets


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

    def pre(us) -> None:  # v8 T2: a batch of candidates' checks in the checker's pool
        if getattr(checker, "pool", None) is None:
            return
        from opt.parallel import prefetch

        calls = []
        for u in us:
            u = np.atleast_1d(np.asarray(u, float))[: len(names)]
            if np.linalg.norm(u[lin_mask]) > st.refine_radius or not sites.accept(vals(u))[0]:
                continue
            T = sites.T(vals(u))
            calls += [
                (T, "point", int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i], st.screen_poses, False)
                for i in idx
            ]
        prefetch(checker, calls)

    if st.refine_method in ("tpe", "gp"):  # v8 T10: Bayesian search over the same box
        from opt.refine_bayes import make_search

        r = st.refine_radius
        d = len(names)
        search = make_search(
            st.refine_method, np.zeros(d), -r * np.ones(d), r * np.ones(d), st.refine_evals, 1
        )
        best_u, best_f = np.zeros(d), f(np.zeros(d))
        while not search.stop():
            us = search.ask()
            pre(us)
            fs = [f(u) for u in us]
            search.tell(us, fs)
            k = int(np.argmin(fs))
            if fs[k] < best_f:
                best_u, best_f = np.asarray(us[k]), fs[k]
        return vals(best_u)

    if len(names) == 1:  # rail s / mobile contour t: a 1-D grid, then a finer one around the best
        r = st.refine_radius
        best_u, best_f = 0.0, f(np.zeros(1))
        for us in (np.linspace(-r, r, 9), None):
            if us is None:
                us = best_u + np.array([-1.0, 1.0]) * r / 8
            pre([np.array([u]) for u in us])
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
        pre(us)
        fs = [f(u) for u in us]
        es.tell(us, fs)
        k = int(np.argmin(fs))
        if fs[k] < best_f:
            best_u, best_f = np.asarray(us[k]), fs[k]
    return vals(best_u)
