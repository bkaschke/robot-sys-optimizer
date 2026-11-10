"""Station planner panel (Optimize tab, M7): runs opt.placement in a background thread and
streams each solution into ``session.results`` (best cost first)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import viser

from core import targets as tg
from core.feasibility import CancelledError
from core.ik import make_ik
from core.schema import Optimizer, TargetGroup
from opt.placement import PlannerSettings, Targets
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
            from ui import rules_ui

            rules_ui.build(gui, self)  # v7 round 2 (T15): Mode = Cost based / Rules only
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
            from ui.method_picker import MethodPicker

            # v8 T10: was the checkbox "Refine with exact check (CMA-ES)"; old saved values map on
            self.refine = MethodPicker(gui, "station_refine", "Refinement")
            self.objective_m = MethodPicker(gui, "station_objective", "Objective")  # v8 T11 / T12
            from ui.pareto_ui import pareto_keys_widgets

            self.pareto_keys = pareto_keys_widgets(gui, ("stations", "cycle", "clearance"))
            from opt.robust import MEASURES

            self.robust_measure = gui.add_dropdown(  # v8 T12
                "Robust measure",
                MEASURES,
                visible=False,
                hint="points lost: the worst sample or the worst 20 %",
            )
            self.robust_samples = gui.add_number(
                "Robust samples",
                10,
                min=2,
                max=200,
                step=1,
                visible=False,
                hint="tolerance samples per candidate",
            )
            self.tolerance_settings = None  # app.py: the Results -> Tolerance check offsets
            self.cable = None  # app.py: Optimize -> Cable & safety (v8 T18)
            self.objective_m.dropdown.on_update(lambda _e: self._objective_fields())
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
            rules_ui.show(self)
            self.run = gui.add_button("Plan stations", color="green")
            self.gate_md = gui.add_markdown("", visible=False)  # why Run is disabled (T5)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("_Results appear in the Results tab as they finish._")
        # v7 round 2 (T14): a fast first look, without refinement, zones or station areas
        with gui.add_folder("Quick station plan"):
            gui.add_markdown(
                "_Uses *Max stations* of the Station planner. No refinement, no zones, no station "
                "areas: a first look in seconds._"
            )
            self.quick_run = gui.add_button("Quick plan", color="green")
            self.quick_gate_md = gui.add_markdown("", visible=False)
            self.quick_stop = gui.add_button("Cancel", visible=False)
            self.quick_progress = gui.add_progress_bar(0.0, visible=False)
            self.quick_status = gui.add_markdown("")
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())
        self.quick_run.on_click(lambda e: self.start(e, quick=True))
        self.quick_stop.on_click(lambda e: self._cancel.set())

    def _objective_fields(self) -> None:
        on = self.objective_m.key == "pareto" and self.objective_m.visible
        for cb in self.pareto_keys.values():
            cb.visible = on
        rob = self.objective_m.key == "robust" and self.objective_m.visible
        self.robust_measure.visible = self.robust_samples.visible = rob

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

    def start(self, event=None, block: bool = False, quick: bool = False) -> None:
        """Plan stations in a thread. ``quick`` (T14): the quick plan (no refinement, zones or
        station areas; results labelled "Quick: ...")."""
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "Planning is already running")
        from ui.workflow import setup_reason

        if reason := setup_reason(s):
            return notify(event, "Setup not confirmed", reason)
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
            refine=self.refine.key != "none",
            refine_method=self.refine.key,
            zoning=self.options()["zoning"],
            areas=self.options()["areas"],
            require_all=self.options()["require_all"],
            require_one_config=self.options()["require_one_config"],
            cycle=getattr(s, "cycle", None),  # v8 T6
            wrap_limit_deg=float(self.cable.wrap.value) if self.cable else 360.0,  # v8 T18
        )
        if quick:
            settings = PlannerSettings(
                max_stations=settings.max_stations,
                candidate_step=settings.candidate_step,
                refine=False,
                zoning=False,
                areas=False,
                patience=1,
                quick_first=True,
                cycle=settings.cycle,
                wrap_limit_deg=settings.wrap_limit_deg,
            )
        run, stop, progress, status = (
            (self.quick_run, self.quick_stop, self.quick_progress, self.quick_status)
            if quick
            else (self.run, self.stop, self.progress, self.status)
        )
        from ui import rules_ui

        rules = not quick and rules_ui.rules_mode(self)
        what = "Quick station plan" if quick else ("Rules-only station plan" if rules else "Station planning")
        weights = self.costs.commit()
        from core import collision
        from ui import jobs

        spec = {"settings": settings, "weights": weights, "rules": None, "workers": jobs.CHECK_WORKERS}
        spec["dress_m"] = collision.DRESS_M  # v8 T18: the worker has its own copy
        spec["hose"] = collision.HOSE
        if not quick and not rules:  # v8 T11 / T12: the Objective dropdown (Cost based mode)
            spec["objective"] = self.objective_m.key
            spec["pareto_keys"] = tuple(k for k, cb in self.pareto_keys.items() if cb.value)
            from dataclasses import replace

            from opt.tolerance import ToleranceSettings

            tol = self.tolerance_settings() if self.tolerance_settings else ToleranceSettings()
            n = int(self.robust_samples.value)
            spec["robust"] = {"tolerance": replace(tol, samples=n), "measure": self.robust_measure.value}
        if rules:
            spec["rules"] = {"objective": self.objective.value, "sing_pct": float(self.sing_pct.value)}
        self._cancel.clear()
        run.visible, stop.visible, progress.visible = False, True, True
        if self.on_start is not None:
            self.on_start()
        s.results, s.result_points = [], {}
        self.on_solution()
        stamp = stamp_for(s, "plan")  # the setup this run plans for (out-of-date tracking)

        def streamed(sol, res) -> None:
            if quick and not sol.label.startswith("Quick: "):
                sol.label = f"Quick: {sol.label}"  # in place: the planner's list has the same object
            sol.planner = sol.planner or ("quick" if quick else "cost")
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
            status.content = f"_{msg}_"
            progress.value = 100.0 * f

        def on_msg(kind: str, data) -> None:  # messages from the job (thread or worker)
            if kind == "progress":
                say(*data)
            elif kind == "solution":
                streamed(*data)
            elif kind == "max_manip":
                s.max_manip = float(data)
            elif kind == "warn":
                log(f"{what}: {data}.", "warn")
            elif kind == "replace":  # v8 T11 / T12: the front / robust plans replace the list
                s.results, s.result_points = [], {}
                for sol, res in data:
                    s.stamps[sol.label] = stamp
                    s.result_points[sol.label] = res
                    s.results.append(sol)
                self.on_solution()

        from ui import jobs
        from ui.plan_job import payload_for

        in_process = block or not jobs.USE_WORKER
        payload = payload_for(s, spec, live=in_process)

        def work() -> None:
            try:
                out = jobs.call("ui.plan_job:station_job", payload, on_msg, self._cancel.is_set, in_process)
                rules_status = out["rules_status"]
                best = out["best"]
                if best is not None:  # the worker's copy: use the streamed object of that label
                    best = next(
                        (r for r in s.results if r.label in (best.label, f"Quick: {best.label}")), best
                    )
                for r in s.results:  # the run time, for the Results summary card
                    if r.run_s is None and r.tool_values is None:
                        r.run_s = round(time.monotonic() - self._t0, 1)
                if rules:
                    status.content = rules_status
                    for line in best.violations if best is not None else []:
                        log(f"Rules-only plan: {line}.", "warn")
                elif quick:
                    status.content = (
                        f"Best: **{best.label}**, cost {best.cost:.2f}: _not optimized; run **Plan "
                        "stations** to optimize._"
                        if best
                        else "_No solution found._"
                    )
                else:
                    status.content = (
                        f"Best: **{best.label}**, cost {best.cost:.2f}.  \n"
                        "Next: **Results → Paths & simulation → Build paths**, or *Results → Inspect result* "
                        "to step through the points."
                        if best
                        else "_No solution found._"
                    )
                plan_log(s, best, self._cancel.is_set(), self._t0, what=what)
                if self._cancel.is_set():
                    status.content = "**Cancelled.** " + (
                        f"Best so far: **{best.label}**, cost {best.cost:.2f}"
                        if best
                        else "_No plan finished._"
                    )
            except CancelledError:  # Cancel while the reachability map was being built
                status.content = "**Cancelled** (while building the reachability map)."
                log(f"{what} cancelled (while building the reachability map).", "warn")
            except jobs.JobCancelledError as err:  # the worker was ended after Cancel
                status.content = f"**Cancelled** ({err}); {len(s.results)} plan(s) kept."
                log(f"{what} cancelled: {err}.", "warn", self._t0)
            except Exception as err:  # report in the GUI instead of dying silently in a thread
                status.content = f"⚠ planning failed: {err}"
                log(f"{what} failed: {err}", "error")
                raise
            finally:
                run.visible, stop.visible, progress.visible = True, False, False

        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()
