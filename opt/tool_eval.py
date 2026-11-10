"""The tool optimizer's evaluator (split from opt.tool_geometry in v8): builds the collision
scene and checker per tool geometry, plans stations for a geometry (``plan``) or scores it at
fixed stations (``cost_at_stations``), records every evaluation (``seen``, for Pareto fronts)
and adds the other parts of a part family (``family``, v8 T16)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from core.collision import CollisionScene
from core.feasibility import CheckSettings, FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from core.schema import EndEffector, ExtensionValues, Solution, Weights, length_ref
from core.tool import ToolModel
from opt.cost import cost_breakdown, wrap_excess
from opt.placement import PlannerSettings, Targets, plan_stations, verify
from opt.sites import FloorSites

if TYPE_CHECKING:
    from opt.tool_geometry import GeometrySettings


class _Evaluator:
    """Builds per-geometry scenes; the head meshes are loaded once and the ee swapped."""

    seen: list = []  # (values, solution, results, feasible, tool length) of every evaluation

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
        self.L_ref = length_ref(ee.extension)
        self.max_manip = 1.0  # set from the reachability map on the first plan
        self.cancelled: Callable[[], bool] | None = None  # polled by every exact check
        self.obstacles: list = []  # stay-out zones (name, mesh, T_world)

    def tool(self, v: ExtensionValues) -> ToolModel:
        self.tm.ee = self.ee.model_copy(
            update={"extension": self.ee.extension.model_copy(update={"nominal": v})}
        )
        return self.tm

    def checker(self, v: ExtensionValues) -> FeasibilityChecker:
        tm = self.tool(v)
        bb, wb = self.sites.scene_bodies() if hasattr(self.sites, "scene_bodies") else ([], [])
        scene = CollisionScene(
            self.robot,
            tm,
            self.part,
            self.T_base0,
            self.T_part,
            base_bodies=bb,
            world_bodies=wb,
            obstacles=self.obstacles,
        )
        checker = FeasibilityChecker(
            self.robot, self.ik, tm, scene, self.T_base0, self.T_part, CheckSettings()
        )
        checker.cancelled = self.cancelled
        return checker

    def plan(self, v: ExtensionValues, cancelled=None, refine: bool | None = None):
        v = self.snap(v)
        checker = self.checker(v)
        pose = self.ee.extension.type in ("pose", "bent2")  # any TCP offset: shared flange map
        from opt.reachability import map_for_tcps

        tcps = [c.name for c in self.ee.tcps]  # v8 T16: all TCPs (two tips)
        rmap = map_for_tcps(self.robot, self.tm, tcps, cancelled=cancelled, any_offset=pose)
        self.max_manip = rmap.max_manip
        extra = {} if refine is None else {"refine": refine}
        ps = PlannerSettings(**{**self.st.planner.__dict__, "max_stations": self.max_stations, **extra})
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
        out = [(self._family(v, s_, cancelled), r_) for s_, r_ in out]  # v8 T16: other parts
        full = [(s, r) for s, r in out if not s.unreached and len(s.stations) <= self.max_stations]
        for s_, r_ in out:  # v8 T11: every plan is a candidate for a Pareto front
            self.seen.append((v, s_, r_, not s_.unreached, self.tm.length(v)))
        if full:
            return True, *min(full, key=lambda sr: sr[0].cost)
        if out:
            return False, *min(out, key=lambda sr: (len(sr[0].unreached), sr[0].cost))
        return False, None, {}

    def snap(self, v: ExtensionValues) -> ExtensionValues:
        """v8 T17: bend angles on the chosen step (inside their ranges)."""
        if not self.st.bend_step:
            return v
        from core.step_export import ANGLE_KEYS, snap_values

        x = self.ee.extension
        return snap_values(v, self.st.bend_step, {k: getattr(x, k) for k in ANGLE_KEYS})

    def cost_at_stations(self, v: ExtensionValues, sol: Solution) -> tuple[float, Solution, dict]:
        v = self.snap(v)
        checker = self.checker(v)
        xy = np.array([s.xyz[:2] for s in sol.stations])
        sites = self.sites or FloorSites(self.T_base0, self.fp, self.st.planner.footprint_margin, 0.0)
        if sites.kind == "floor":
            stations = [{"x": s.xyz[0], "y": s.xyz[1]} for s in sol.stations]
        else:
            stations = [{d: s.carrier[d] for d in sites.dofs if d in s.carrier} for s in sol.stations]
        pref = np.array([sol.assignment.get(int(i), 0) for i in self.t.ids])
        assignment, res = verify(checker, self.t, stations, sites, pref)
        zones, changes = [], 0
        from opt.zoning import zone_counts

        if self.st.planner.zoning:
            from opt.zoning import zone_solution

            zones, changes = zone_solution(
                checker, self.t, stations, sites, assignment, res, self.st.planner.zoning_poses
            )
        unreached = sorted(p for p, r in res.items() if r.status == "unreachable")
        from opt.zoning import arm_motion, config_changes, stop_sequences

        seqs = stop_sequences(checker, self.t, assignment, res, zones)
        groups = zone_counts(zones)[0] if zones else config_changes(seqs)
        from opt.cost import cycle_estimate

        cyc = cycle_estimate(seqs, res, stations, sites, checker.robot, self.st.planner.cycle)["total"]
        cost, terms = cost_breakdown(
            list(res.values()),
            len(unreached),
            xy,
            self.w,
            self.tm.length(v),
            self.L_ref,
            self.max_manip,
            groups=groups,
            wrist_flips=zone_counts(zones)[1],
            motion=arm_motion(seqs, res),
            cycle_s=cyc,
            wrap=wrap_excess(seqs, res, self.st.planner.wrap_limit_deg),
        )
        new = sol.model_copy(
            update={
                "assignment": assignment,
                "unreached": unreached,
                "cost": cost,
                "breakdown": terms,
                "tool_values": v,
                "zones": zones,
                "cycle_s": round(cyc, 1),
            }
        )
        new = self._family(v, new)  # v8 T16: the other parts of the family
        self.seen.append((v, new, res, not new.unreached, self.tm.length(v)))
        return new.cost, new, res

    family: list = []  # v8 T16: _Evaluator per other part (part family), each re-planned per tool

    def _family(self, v: ExtensionValues, sol: Solution, cancelled=None) -> Solution:
        """``sol`` with the other parts' costs added (``st.family_cost``: sum or worst) and their
        unreached points counted (as negative ids, so this part's ids stay recognisable)."""
        if not self.family or sol is None:
            return sol
        costs, extra = [sol.cost], []
        for k, other in enumerate(self.family):
            _ok, s2, _r2 = other.plan(v, cancelled, refine=False)
            costs.append(s2.cost if s2 is not None else 1e9)
            extra += [-(1000 * (k + 1) + i) for i in (s2.unreached if s2 is not None else [0])]
        cost = sum(costs) if self.st.family_cost == "sum" else max(costs)
        terms = {**sol.breakdown, "family": cost - sol.cost}
        return sol.model_copy(
            update={"cost": cost, "breakdown": terms, "unreached": [*sol.unreached, *extra]}
        )
