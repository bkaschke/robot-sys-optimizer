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

from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker, PointResult
from core.ik import make_ik
from core.robot import Robot
from core.schema import EndEffector, ExtensionValues, Solution, Weights
from core.tool import ToolModel
from opt.cost import cost_breakdown
from opt.placement import PlannerSettings, Targets, plan_stations, verify
from opt.reachability import map_for_tool
from opt.sites import FloorSites

SHAPE_VARS = ("theta_deg", "d", "roll_deg")


@dataclass
class GeometrySettings:
    L_tol: float = 0.001  # m
    coarse_samples: int = 6
    shape_evals: int = 30
    planner: PlannerSettings = field(default_factory=lambda: PlannerSettings(refine_evals=12))


@dataclass
class LSample:
    L: float
    feasible: bool
    reached: int
    total: int
    stations: int
    cost: float


@dataclass
class GeometryResult:
    feasible: bool
    values: ExtensionValues
    solution: Solution | None
    point_results: dict[int, PointResult]
    samples: list[LSample]
    message: str = ""


class _Evaluator:
    """Builds per-geometry scenes; the head meshes are loaded once and the ee swapped."""

    def __init__(
        self,
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
        st: GeometrySettings,
        sites=None,
    ):
        self.robot, self.ee, self.part = robot, ee, part
        self.sites = sites  # None = free floor (opt.sites.FloorSites)
        self.T_part, self.T_base0 = T_world_part, T_world_base0
        self.t, self.fp, self.w = targets, footprint, weights
        self.max_stations, self.st = max_stations, st
        self.tm = ToolModel(ee, urdf)
        self.ik = make_ik(robot)
        self.tcp = ee.tcps[0].name
        self.L_ref = max(ee.extension.L[1], 1e-6)
        self.max_manip = 1.0  # set from the reachability map on the first plan

    def tool(self, v: ExtensionValues) -> ToolModel:
        self.tm.ee = self.ee.model_copy(
            update={"extension": self.ee.extension.model_copy(update={"nominal": v})}
        )
        return self.tm

    def checker(self, v: ExtensionValues) -> FeasibilityChecker:
        tm = self.tool(v)
        bb, wb = self.sites.scene_bodies() if hasattr(self.sites, "scene_bodies") else ([], [])
        scene = CollisionScene(
            self.robot, tm, self.part, self.T_base0, self.T_part, base_bodies=bb, world_bodies=wb
        )
        return FeasibilityChecker(self.robot, self.ik, tm, scene, self.T_base0, self.T_part)

    def plan(self, v: ExtensionValues, cancelled=None):
        checker = self.checker(v)
        rmap = map_for_tool(self.robot, self.tm.T_flange_tcp(self.tcp))
        self.max_manip = rmap.max_manip
        ps = PlannerSettings(**{**self.st.planner.__dict__, "max_stations": self.max_stations})
        out = plan_stations(
            rmap,
            checker,
            self.t,
            self.T_base0,
            self.fp,
            self.w,
            ps,
            self.tm.length(v),
            self.L_ref,
            cancelled=cancelled,
            sites=self.sites,
        )
        full = [(s, r) for s, r in out if not s.unreached and len(s.stations) <= self.max_stations]
        if full:
            return True, *min(full, key=lambda sr: sr[0].cost)
        if out:
            return False, *min(out, key=lambda sr: (len(sr[0].unreached), sr[0].cost))
        return False, None, {}

    def cost_at_stations(self, v: ExtensionValues, sol: Solution) -> tuple[float, Solution, dict]:
        checker = self.checker(v)
        xy = np.array([s.xyz[:2] for s in sol.stations])
        sites = self.sites or FloorSites(self.T_base0, self.fp, self.st.planner.footprint_margin, 0.0)
        if sites.kind == "floor":
            stations = [{"x": s.xyz[0], "y": s.xyz[1]} for s in sol.stations]
        else:
            stations = [{d: s.carrier[d] for d in sites.dofs} for s in sol.stations]
        pref = np.array([sol.assignment.get(int(i), 0) for i in self.t.ids])
        assignment, res = verify(checker, self.t, stations, sites, pref)
        unreached = sorted(p for p, r in res.items() if r.status == "unreachable")
        cost, terms = cost_breakdown(
            list(res.values()), len(unreached), xy, self.w, self.tm.length(v), self.L_ref, self.max_manip
        )
        new = sol.model_copy(
            update={
                "assignment": assignment,
                "unreached": unreached,
                "cost": cost,
                "breakdown": terms,
                "tool_values": v,
            }
        )
        return cost, new, res


def optimize_tool(
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
    x = ee.extension
    base_v = ExtensionValues(
        L=x.nominal.L,
        theta_deg=float(np.clip(0.0, *x.theta_deg)),
        d=float(np.clip(0.0, *x.d)),
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
    if first is None:
        samples.sort(key=lambda s: s.L)
        best = max(samples, key=lambda s: s.reached) if samples else None
        msg = "no tool length in range reaches every target"
        if best:
            msg += f" (best: {best.reached}/{n} at L = {best.L * 1000:.0f} mm)"
        return GeometryResult(False, base_v, None, {}, samples, msg)
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
        if getattr(x, k)[1] - getattr(x, k)[0] > 1e-9 and (k == "roll_deg" or x.type == "bent")
    ]
    if active and st.shape_evals > 0 and not (cancelled is not None and cancelled()):
        import cma

        bounds = np.array([getattr(x, k) for k in active], float)

        def values(u: np.ndarray) -> ExtensionValues:
            u = np.clip(u, 0, 1)
            return v_best.model_copy(
                update={
                    k: float(bounds[i, 0] + u[i] * (bounds[i, 1] - bounds[i, 0]))
                    for i, k in enumerate(active)
                }
            )

        best_cost, best_sol, best_res = ev.cost_at_stations(v_best, sol)
        u0 = np.array(
            [
                (getattr(v_best, k) - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0])
                for i, k in enumerate(active)
            ]
        )
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
                if c < best_cost:
                    best_cost, best_sol, best_res = c, s_new, r_new
                evals += 1
                say(f"shape {evals}/{st.shape_evals}", 0.6 + 0.4 * evals / st.shape_evals)
            es.tell(us, costs)
        sol, res, v_best = best_sol, best_res, best_sol.tool_values
    label = f"L {v_best.L * 1000:.0f} mm" + "".join(
        f", {k.split('_')[0]} {getattr(v_best, k):.1f}" for k in active
    )
    sol = sol.model_copy(update={"label": f"{label}: {sol.label}"})
    samples.sort(key=lambda s: s.L)
    say("done", 1.0)
    return GeometryResult(True, v_best, sol, res, samples, f"minimal L = {L_min * 1000:.1f} mm")
