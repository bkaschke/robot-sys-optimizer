"""Station planner panel (Optimize tab, M7): runs opt.placement in a background thread and
streams each solution into ``session.results`` (best cost first)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import viser

from core import targets as tg
from core.collision import CollisionScene
from core.feasibility import CancelledError, FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from core.schema import Optimizer, TargetGroup, length_ref
from opt.placement import PlannerSettings, Targets, plan_stations
from opt.reachability import map_for_tool
from ui.log_panel import log, plan_log
from ui.session import Session
from ui.staleness import stamp_for
from ui.widgets import notify


def world_targets(s: Session) -> Targets:
    ids, P, N, G = tg.as_arrays(s.targets)
    Tp = s.part.T_world_part
    groups = {g.name: g for g in s.targets.groups}
    per = [groups.get(g) or TargetGroup(name=g) for g in G]
    return Targets(
        ids, P @ Tp[:3, :3].T + Tp[:3, 3], N @ Tp[:3, :3].T, P, N, per, np.array([g.tilt_deg for g in per])
    )


def part_footprint(s: Session) -> np.ndarray:
    """World-xy AABB of the part (lo, hi)."""
    b = s.part_mesh().bounds
    corners = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
    Tp = s.part.T_world_part
    w = corners @ Tp[:3, :3].T + Tp[:3, 3]
    return np.array([w[:, :2].min(axis=0), w[:, :2].max(axis=0)])


ZONING = ("best per point", "zones (minimize changes)")


class StationsPanel:
    def __init__(self, server: viser.ViserServer, session: Session, on_solution: Callable[[], None]):
        self.s = session
        self.on_solution = on_solution
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.on_start: Callable[[], None] | None = None  # set by app.py: e.g. clear old paths
        gui = server.gui
        with gui.add_folder("Station planner"):
            opt = self._optimizer()
            self.max_stations = gui.add_number(
                "Max stations",
                opt.max_stations,
                min=1,
                max=40,
                step=1,
                hint="the planner stops early once more stops no longer give a better plan",
            )
            gui.add_markdown(
                "_Static base mode places one base (k = 1); set Mode = movable with a carrier in the "
                "External Axis tab for several stations._"
            )
            self.step = gui.add_number("Candidate grid (mm)", 100.0, min=25.0, max=500.0, step=25.0)
            self.refine = gui.add_checkbox("Refine with exact check (CMA-ES)", True)
            self.areas = gui.add_checkbox(
                "Connected station areas", True, hint="each stop's points one area on the part (no overlap)"
            )
            # v6: zones by default (best per point mixed all configurations at a stop)
            self.zoning = gui.add_dropdown("Joint configuration", ZONING, initial_value=ZONING[1])
            self.zoning_note = gui.add_markdown(
                "_**zones**: each stop's points are grouped by joint configuration (shoulder/elbow/"
                "wrist) into ordered paths; extra groups and wrist jumps > 180° cost what *Costs & "
                "requirements* says. About 3× slower._"
            )

            def zoning_fields(_e=None) -> None:  # zoning settings only when zoning is on
                self.zoning_note.visible = self.zoning.value == ZONING[1]

            self.zoning.on_update(zoning_fields)
            zoning_fields()
            from ui.costs_ui import CostPanel

            self.costs = CostPanel(server, session)
            self.run = gui.add_button("Plan stations", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("_Results appear in the Results tab as they finish._")
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

    def options(self) -> dict:
        """Planner options shared with the tool-geometry panel."""
        req = self.costs.requirements()
        return {
            "zoning": self.zoning.value == ZONING[1] or req["require_one_config"],
            "areas": bool(self.areas.value),
            **req,
        }

    def _optimizer(self) -> Optimizer:
        return self.s.extra.get("optimizer") or Optimizer()

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "Planning is already running")
        if s.tool_model is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot plan", "Need a tool, a part and targets")
        if s.robot is None:
            return notify(event, "No robot", "Choose a robot first (tab 1 Robot & cell)")
        if not getattr(make_ik(s.robot), "batch", False):
            return notify(
                event,
                "Not supported",
                "Station planning needs an analytic IK solver (spec.yaml: ik ur_analytic or opw). "
                "See the user guide, Adding a robot.",
            )
        sites = s.sites()
        static = sites is None
        settings = PlannerSettings(
            max_stations=1 if static else int(self.max_stations.value),
            candidate_step=self.step.value / 1000,
            refine=self.refine.value,
            zoning=self.options()["zoning"],
            areas=self.options()["areas"],
            require_all=self.options()["require_all"],
            require_one_config=self.options()["require_one_config"],
        )
        robot = Robot(s.robot.spec)
        tcp = s.tool.tcps[0].name
        t = world_targets(s)
        fp = part_footprint(s)
        mesh = s.part_mesh()
        obstacles = s.obstacle_entries()  # stay-out zones
        T_base0, T_part = s.T_world_base.copy(), s.part.T_world_part.copy()
        weights = self.costs.commit()
        L = s.tool_model.length()
        L_ref = length_ref(s.tool.extension)
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        if self.on_start is not None:
            self.on_start()
        s.results, s.result_points = [], {}
        self.on_solution()
        stamp = stamp_for(s, "plan")  # the setup this run plans for (out-of-date tracking)

        def streamed(sol, res) -> None:
            # a refined plan replaces the quick plan with the same number of stations
            for old in [
                r for r in s.results if len(r.stations) == len(sol.stations) and r.tool_values is None
            ]:
                s.result_points.pop(old.label, None)
                s.stamps.pop(old.label, None)
            keep = [
                r for r in s.results if not (len(r.stations) == len(sol.stations) and r.tool_values is None)
            ]
            s.stamps[sol.label] = stamp
            s.result_points[sol.label] = res
            s.results = sorted([*keep, sol], key=lambda x: (bool(x.violations), x.cost))
            self.on_solution()

        def say(msg: str, f: float) -> None:
            self.status.content = f"_{msg}_"
            self.progress.value = 100.0 * f

        def work() -> None:
            try:
                say("loading reachability map (first time per tool length: ~1 min)", 0.0)
                rmap = map_for_tool(
                    robot,
                    s.tool_model.T_flange_tcp(tcp),
                    cancelled=self._cancel.is_set,
                    any_offset=s.tool.extension.type == "pose",  # pose tools: shared flange map
                )
                bb, wb = sites.scene_bodies() if sites is not None else ([], [])
                scene = CollisionScene(
                    robot,
                    s.tool_model,
                    mesh,
                    T_base0,
                    T_part,
                    base_bodies=bb,
                    world_bodies=wb,
                    obstacles=obstacles,
                )
                checker = FeasibilityChecker(robot, make_ik(robot), s.tool_model, scene, T_base0, T_part)
                out = plan_stations(
                    rmap,
                    checker,
                    t,
                    T_base0,
                    fp,
                    weights,
                    settings,
                    L,
                    L_ref,
                    on_solution=streamed,
                    progress=say,
                    cancelled=self._cancel.is_set,
                    sites=sites,
                )
                best = out[0][0] if out else None
                self.status.content = (
                    f"Best: **{best.label}**, cost {best.cost:.2f}.  \n"
                    "Next: **Results → Paths & simulation → Build paths**, or *Inspect result* (above) "
                    "to step through the points."
                    if best
                    else "_No solution found._"
                )
                plan_log(s, best, self._cancel.is_set(), self._t0)
                if self._cancel.is_set():
                    self.status.content = "**Cancelled.** " + (
                        f"Best so far: **{best.label}**, cost {best.cost:.2f}"
                        if best
                        else "_No plan finished._"
                    )
            except CancelledError:  # Cancel while the reachability map was being built
                self.status.content = "**Cancelled** (while building the reachability map)."
                log("Station planning cancelled (while building the reachability map).", "warn")
            except Exception as err:  # report in the GUI instead of dying silently in a thread
                self.status.content = f"⚠ planning failed: {err}"
                log(f"Station planning failed: {err}", "error")
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()
