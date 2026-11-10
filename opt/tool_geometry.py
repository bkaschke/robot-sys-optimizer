"""Tool-geometry optimizer (plan § Solver strategy 5, M8): the outer loop.

Stage A, length: the shortest L in the extension's range for which the station planner
reaches every target with at most ``max_stations``. Coarse scan, then bisection to L_tol.
Coverage vs L is assumed monotone above that minimum (a longer tool reaches deeper); the
coarse samples are kept so the GUI can show the coverage-vs-length curve.

Stage B, shape: with L fixed at the minimum and the stations from stage A kept, CMA-ES
tunes whichever of theta, d and roll have a non-degenerate range, minimizing the full
cost J (exact checker on every target).

Straight tools share one flange reachability map for every L (opt.reachability.OffsetMap);
a bent nominal shape needs its own cached map per L, which is slow the first time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from core.feasibility import CancelledError, PointResult
from core.robot import Robot
from core.schema import EndEffector, ExtensionValues, Solution, Weights
from opt.placement import PlannerSettings, Targets

SHAPE_VARS = ("theta_deg", "d", "roll_deg", "theta2_deg", "d2")


@dataclass
class GeometrySettings:
    L_tol: float = 0.001  # m
    coarse_samples: int = 6
    shape_evals: int = 30
    shape_method: str = "cmaes"  # v8 T10: "cmaes" | "tpe" | "gp" (opt.methods tool_shape)
    objective: str = "weighted"  # v8 T11: "weighted" | "pareto" (opt.methods tool_objective)
    pareto_keys: tuple = ("length", "stations", "cycle")
    loop: str = "fixed"  # v8 T13: "fixed" | "alternate" (opt.methods tool_loop)
    start: str = "nominal"  # v8 T15: "nominal" | "cones" (opt.methods tool_start)
    family: list = field(default_factory=list)  # v8 T16: FamilyPart of the other parts
    family_cost: str = "sum"  # "sum" | "worst" over the parts
    robust: dict | None = None  # v8 T12: {"tolerance": ToleranceSettings, "measure": str}
    planner: PlannerSettings = field(default_factory=lambda: PlannerSettings(refine_evals=12))


@dataclass
class FamilyPart:
    """Another part the tool must also work on (v8 T16, part family): its cell, in the frames of
    its own project."""

    part: trimesh.Trimesh
    T_part: np.ndarray
    T_base0: np.ndarray
    t: Targets
    fp: np.ndarray | None
    sites: object = None
    obstacles: list = field(default_factory=list)
    label: str = ""


@dataclass
class LSample:
    L: float
    feasible: bool
    reached: int
    total: int
    stations: int
    cost: float


@dataclass
class Design:
    """One evaluated tool geometry with its station plan (v6: the GUI offers the best few,
    also when none reaches every target)."""

    values: ExtensionValues
    solution: Solution
    point_results: dict[int, PointResult]
    feasible: bool


TOP_DESIGNS = 3


def top_designs(cands, label, n: int = TOP_DESIGNS) -> list[Design]:
    """The ``n`` best of (values, solution, results, feasible): every target reached first,
    then fewest unreached, then cost. Each solution gets ``tool_values`` and the label
    ``label(values) + ": " + its plan label`` (labels key the GUI's result lists)."""
    ranked = sorted(
        (c for c in cands if c[1] is not None),
        key=lambda c: (not c[3], len(c[1].unreached), c[1].cost),
    )
    out: list[Design] = []
    seen: set[str] = set()
    for v, sol, res, ok in ranked:
        name = f"{label(v)}: {sol.label}"
        if name in seen:
            continue
        seen.add(name)
        out.append(Design(v, sol.model_copy(update={"tool_values": v, "label": name}), res, bool(ok)))
        if len(out) == n:
            break
    return out


@dataclass
class GeometryResult:
    feasible: bool
    values: ExtensionValues
    solution: Solution | None
    point_results: dict[int, PointResult]
    samples: list[LSample]
    message: str = ""
    designs: list[Design] = field(default_factory=list)  # best few, best first (may miss points)


from opt.tool_eval import _Evaluator  # noqa: E402, F401  (v8: split out)


def optimize_tool(*args, **kw) -> GeometryResult:
    """Tool geometry optimization (see the module docstring); with ``st.objective ==
    "pareto"`` (v8 T11) the designs offered are the Pareto front of every evaluated design
    (``st.pareto_keys``) instead of the best three by cost."""
    st = kw.get("st") or GeometrySettings()
    seen: list = []
    holder: dict = {}
    r = _optimize_tool(*args, **{**kw, "seen": seen, "holder": holder})
    if st.objective == "robust" and seen and "ev" in holder:  # v8 T12
        from opt.robust import rerank_designs

        r.designs = rerank_designs(holder["ev"], seen, st, kw.get("cancelled"))
        r.message += "; designs ranked by robust cost (worst case under tolerance)"
    if st.objective == "pareto" and seen:
        from opt.pareto import front, measures

        f = front(seen, st.pareto_keys, lambda it: measures(it[1], it[2], it[4]))
        n = len(f)
        r.designs = [
            Design(
                v,
                sol.model_copy(
                    update={
                        "tool_values": v,
                        "label": f"Pareto {i + 1}/{n}: tool {1000 * length:.0f} mm: {sol.label}",
                        "planner": "tool",
                    }
                ),
                res,
                bool(ok),
            )
            for i, (v, sol, res, ok, length) in enumerate(f)
        ]
        r.message += f"; Pareto front: {n} design(s)"
    return r


def _optimize_tool(
    robot: Robot,
    ee: EndEffector,
    urdf: Path | None,
    part: trimesh.Trimesh,
    T_world_part: np.ndarray,
    T_world_base0: np.ndarray,
    targets: Targets,
    footprint: np.ndarray | None,
    weights: Weights,
    max_stations: int,
    st: GeometrySettings | None = None,
    progress: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    sites=None,
    obstacles: list | None = None,
    seen: list | None = None,
    holder: dict | None = None,
) -> GeometryResult:
    st = st or GeometrySettings()
    say = progress or (lambda m, f: None)
    ev = _Evaluator(
        robot,
        ee,
        urdf,
        part,
        T_world_part,
        T_world_base0,
        targets,
        footprint,
        weights,
        max_stations,
        st,
        sites=sites,
    )
    ev.cancelled = cancelled
    ev.obstacles = list(obstacles or [])
    ev.seen = seen if seen is not None else []  # v8 T11: every evaluated design
    if holder is not None:
        holder["ev"] = ev
    ev.family = []
    for f in st.family:  # v8 T16: the same tool on other parts (their own cell and targets)
        other = _Evaluator(
            robot, ee, urdf, f.part, f.T_part, f.T_base0, f.t, f.fp, weights, max_stations, st, sites=f.sites
        )
        other.cancelled, other.obstacles, other.seen, other.family = cancelled, list(f.obstacles), [], []
        ev.family.append(other)
    x = ee.extension
    if x.type == "pose":  # v2: head pose x y z rx ry rz, best overall cost (opt.tool_pose)
        from opt.tool_pose import POSE_KEYS, optimize_pose, pose_label

        samples: list[LSample] = []
        cands: list = []
        ok, v, sol, res, msg = optimize_pose(ev, st, samples, say, cancelled, cands)
        say("done", 1.0)
        active = [k for k in POSE_KEYS if getattr(x, k)[1] - getattr(x, k)[0] > 1e-9]
        designs = top_designs(cands, lambda u: pose_label(u, active))
        return GeometryResult(ok, v, sol, res, samples, msg, designs)
    base_v = ExtensionValues(
        L=x.nominal.L,
        theta_deg=float(np.clip(0.0, *x.theta_deg)),
        d=float(np.clip(0.0, *x.d)),
        theta2_deg=float(np.clip(0.0, *x.theta2_deg)),
        d2=float(np.clip(0.0, *x.d2)),
        roll_deg=x.nominal.roll_deg,
    )
    n = len(targets.ids)
    samples: list[LSample] = []
    found: dict[float, tuple] = {}

    def at(L: float):
        if L not in found:
            ok, sol, res = ev.plan(base_v.model_copy(update={"L": float(L)}), cancelled)
            found[L] = (ok, sol, res)
            reached = n - len(sol.unreached) if sol else 0
            samples.append(
                LSample(
                    float(L),
                    ok,
                    reached,
                    n,
                    len(sol.stations) if sol else 0,
                    sol.cost if sol else float("inf"),
                )
            )
        return found[L]

    # --- stage A: coarse scan, then bisection --------------------------------------------------
    lo, hi = x.L
    grid = np.linspace(lo, hi, st.coarse_samples)
    first = None
    for k, L in enumerate(grid):
        if cancelled is not None and cancelled():
            break
        say(f"L scan {L * 1000:.0f} mm", 0.6 * k / len(grid))
        if at(float(L))[0]:
            first = k
            break

    def L_label(u: ExtensionValues) -> str:
        return f"L {u.L * 1000:.0f} mm"

    if first is None:
        designs = top_designs(
            [(base_v.model_copy(update={"L": L}), f[1], f[2], f[0]) for L, f in found.items()], L_label
        )
        samples.sort(key=lambda s: s.L)
        best = max(samples, key=lambda s: s.reached) if samples else None
        msg = "no tool length in range reaches every target"
        if cancelled is not None and cancelled():
            msg = "cancelled before a tool length reached every target"
        if best:
            msg += f" (best: {best.reached}/{n} at L = {best.L * 1000:.0f} mm)"
        if designs:
            msg += f"; the best {len(designs)} design(s) are listed to choose from"
        return GeometryResult(False, base_v, None, {}, samples, msg, designs)
    a, b = (grid[first - 1], grid[first]) if first > 0 else (lo, lo)
    while b - a > st.L_tol and not (cancelled is not None and cancelled()):
        mid = (a + b) / 2
        say(f"bisect L {mid * 1000:.1f} mm", 0.6)
        if at(float(mid))[0]:
            b = mid
        else:
            a = mid
    L_min = float(b)
    _, sol, res = at(L_min)
    v_best = base_v.model_copy(update={"L": L_min})
    sol = sol.model_copy(update={"tool_values": v_best})

    # --- stage B: shape (theta, d, roll) ---------------------------------------------------------
    active = [
        k
        for k in SHAPE_VARS
        if getattr(x, k)[1] - getattr(x, k)[0] > 1e-9
        and (
            k == "roll_deg"
            or (x.type in ("bent", "bent2") and k in ("theta_deg", "d"))
            or (x.type == "bent2" and k in ("theta2_deg", "d2"))
        )
    ]
    if active and st.shape_evals > 0 and not (cancelled is not None and cancelled()):
        bounds = np.array([getattr(x, k) for k in active], float)

        def values(u: np.ndarray) -> ExtensionValues:
            u = np.clip(u, 0, 1)
            return v_best.model_copy(
                update={
                    k: float(bounds[i, 0] + u[i] * (bounds[i, 1] - bounds[i, 0]))
                    for i, k in enumerate(active)
                }
            )

        best = [float("inf"), sol, res]  # cost, solution, results: updated as the search runs
        v_start = v_best
        if st.start == "cones":  # v8 T15: try the approach-cone bends first, search from the best
            from opt.approach_cones import shape_seeds

            say("approach cones", 0.6)
            for v in shape_seeds(ev, v_best):
                c, s_new, r_new = ev.cost_at_stations(v, sol)
                if c < best[0]:
                    best[:] = [c, s_new, r_new]
                    v_start = v
        try:
            _shape_search(ev, st, v_start, sol, active, values, say, best)
        except CancelledError:  # keep the best shape found so far
            pass
        if st.loop == "alternate":  # v8 T13: re-plan the stations for the new shape, and again
            from opt.tool_alternate import alternate

            def values_for(v):
                return lambda u: v.model_copy(
                    update={
                        k: float(bounds[i, 0] + np.clip(u, 0, 1)[i] * (bounds[i, 1] - bounds[i, 0]))
                        for i, k in enumerate(active)
                    }
                )

            alternate(ev, st, best, active, values_for, say, cancelled)
        sol, res = best[1], best[2]
        v_best = sol.tool_values
    label = f"L {v_best.L * 1000:.0f} mm" + "".join(
        f", {k.split('_')[0]} {getattr(v_best, k):.1f}" for k in active
    )
    sol = sol.model_copy(update={"label": f"{label}: {sol.label}"})
    samples.sort(key=lambda s: s.L)
    stopped = cancelled is not None and cancelled()
    say("cancelled" if stopped else "done", 1.0)
    msg = f"minimal L = {L_min * 1000:.1f} mm" + (" (cancelled: best found so far)" if stopped else "")
    others = [(base_v.model_copy(update={"L": L}), f[1], f[2], f[0]) for L, f in found.items() if L != L_min]
    best = Design(v_best, sol, res, True)
    designs = [best, *[d for d in top_designs(others, L_label) if d.solution.label != sol.label]][
        :TOP_DESIGNS
    ]
    return GeometryResult(True, v_best, sol, res, samples, msg, designs)


def _shape_search(ev, st, v_best, sol, active, values, say, best: list) -> None:
    """Stage B: CMA-ES over the active shape variables (normalized to [0, 1]) at the stage A
    stations. ``best`` = [cost, solution, results] is updated in place after every
    evaluation, so a CancelledError raised mid-way loses nothing already found."""
    import cma

    def consider(c, s_new, r_new) -> None:
        if c < best[0]:
            best[:] = [c, s_new, r_new]

    consider(*ev.cost_at_stations(v_best, sol))
    x = ev.ee.extension
    lo_hi = [getattr(x, k) for k in active]
    u0 = np.array([(getattr(v_best, k) - a) / (b - a) for k, (a, b) in zip(active, lo_hi, strict=True)])
    if st.shape_method in ("tpe", "gp"):  # v8 T10
        from opt.refine_bayes import make_search

        es = make_search(st.shape_method, u0, np.zeros(len(u0)), np.ones(len(u0)), st.shape_evals, 2)
    else:
        if len(u0) == 1:  # CMA-ES does not support 1-D: pad with a coordinate values() ignores
            u0 = np.append(u0, 0.5)
        es = cma.CMAEvolutionStrategy(
            u0, 0.25, {"bounds": [0, 1], "maxfevals": st.shape_evals, "verbose": -9, "seed": 2, "popsize": 6}
        )
    evals = 0
    while not es.stop():
        us = es.ask()
        costs = []
        for u in us:
            c, s_new, r_new = ev.cost_at_stations(values(np.asarray(u)), sol)
            costs.append(c)
            consider(c, s_new, r_new)
            evals += 1
            say(f"shape {evals}/{st.shape_evals}", 0.6 + 0.4 * evals / st.shape_evals)
        es.tell(us, costs)
