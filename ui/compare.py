"""Optimize -> Compare robots (v8 T9): the same study planned with several robots.

For each chosen robot the job rebuilds the session from the project snapshot (the live
session is not touched), swaps the robot (keeping the mount, base pose, tool, part, targets,
carrier and stay-out zones; joint limits and home revert to the robot's own), and runs the
station planner with the current planner settings. One row per robot: stations, points
reached, estimated cycle time, worst clearance, lowest manipulability, cost and run time.
v9 #6: every robot's plans (solutions + joint solutions + setup stamps) are kept in
``session.compared`` and saved with the project; Results -> Robot shows one robot at its stops
with its plans (ui.compare_view), and switches the project's robot after a confirmation.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from pathlib import Path

import viser

from core.feasibility import CancelledError
from core.robot import list_robots
from ui.log_panel import log
from ui.widgets import notify


def _metrics(sol, res: dict, run_s: float) -> dict:
    ok = [r for r in res.values() if r.status != "unreachable" and r.q is not None]
    clear = [r.clearance for r in ok if math.isfinite(r.clearance)]
    return {
        "stations": len(sol.stations),
        "reached": len(res) - len(sol.unreached) if res else 0,
        "cycle_s": sol.cycle_s,
        "clear_mm": 1000 * min(clear) if clear else None,
        "manip": min((r.manipulability for r in ok), default=None),
        "cost": sol.cost,
        "violations": len(sol.violations),
        "run_s": run_s,
        "label": sol.label,
    }


def compare_job(payload: dict, emit, cancelled) -> list[dict]:
    """``spec``: robots (registry names), settings (PlannerSettings), weights, workers."""
    from core import schema
    from ui.plan_job import station_job
    from ui.session import Session

    spec = payload["spec"]
    rows = []
    for n, name in enumerate(spec["robots"]):
        if cancelled():
            raise CancelledError()
        emit("progress", (f"planning with {name} ({n + 1} / {len(spec['robots'])})", n / len(spec["robots"])))
        s = Session()
        p = schema.Project.model_validate_json(payload["project"])
        if p.robot is not None:  # another arm: its own limits and home
            p = p.model_copy(
                update={
                    "robot": p.robot.model_copy(
                        update={"name": name, "urdf": None, "joint_limits_deg": None, "home_deg": None}
                    )
                }
            )
        s.apply_project(p, Path(payload["project_dir"]))
        s.T_world_base = payload["T_base"].copy()
        if payload["T_part"] is not None:
            s.part.T_world_part = payload["T_part"].copy()
        t0 = time.monotonic()
        sols: dict = {}
        sub = {
            "settings": spec["settings"],
            "weights": spec["weights"],
            "rules": None,
            "workers": spec["workers"],
        }

        def collect(kind, data, _sols=sols) -> None:
            if kind == "solution":
                _sols[data[0].label] = data
            elif kind == "max_manip":
                _sols["__max_manip"] = float(data)

        out = station_job({"session": s, "spec": sub}, collect, cancelled)
        best = out["best"]
        if best is None:
            rows.append({"robot": name, "label": "no plan"})
            continue
        _, res = sols.get(best.label, (best, {}))
        m = _metrics(best, res, time.monotonic() - t0)
        top = sols.get("__max_manip")
        m["manip_pct"] = 100 * m["manip"] / top if m["manip"] is not None and top else None
        from ui.staleness import stamp_for

        plans = sorted(
            (v for k, v in sols.items() if k != "__max_manip"),
            key=lambda sr: (bool(sr[0].violations), sr[0].cost),
        )
        stamp = stamp_for(s, "plan")  # this robot's setup: its plans are up to date when viewed with it
        rows.append(
            {"robot": name, **m, "plans": plans, "stamps": {sol.label: dict(stamp) for sol, _ in plans}}
        )
    return rows


def keep_plans(session, rows: list[dict]) -> list[dict]:
    """Move each row's plans into ``session.compared``; returns the rows without them (the table)."""
    session.compared = {}
    out = []
    for r in rows:
        r = dict(r)
        plans, stamps = r.pop("plans", []), r.pop("stamps", {})
        if plans:
            session.compared[r["robot"]] = {
                "results": [sol for sol, _ in plans],
                "points": {sol.label: dict(res) for sol, res in plans},
                "stamps": stamps,
                "metrics": dict(r),
            }
        out.append(r)
    return out


def table(rows: list[dict], n_points: int) -> str:
    def f(v, fmt):
        return "—" if v is None else format(v, fmt)

    lines = [
        f"| {r['robot']} | {f(r.get('stations'), 'd')} | {f(r.get('reached'), 'd')}/{n_points} | "
        f"{f(r.get('cycle_s'), '.0f')} | {f(r.get('clear_mm'), '.0f')} | {f(r.get('manip_pct'), '.0f')} | "
        f"{f(r.get('cost'), '.2f')}{' ✗' if r.get('violations') else ''} | {f(r.get('run_s'), '.0f')} |"
        for r in rows
    ]
    return (
        "| robot | stations | reached | cycle (s) | worst clearance (mm) | lowest manip. (%) | cost "
        "| run (s) |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(lines)
    )


class ComparePanel:
    def __init__(self, server: viser.ViserServer, session, stations, use_robot: Callable[[str], None]):
        self.s, self.stations, self.use_robot = session, stations, use_robot
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.rows: list[dict] = []
        gui = server.gui
        self.folder = gui.add_folder("Compare robots", expand_by_default=False)
        with self.folder:
            gui.add_markdown(
                "_Plans the current study with each ticked robot (same tool, part, targets, carrier and "
                "planner settings) and lists the results side by side._"
            )
            self.pick = {
                name: gui.add_checkbox(name, name in ("ur20", "ur10e", "ur30"))
                for name in list_robots()
                if name != "generic_6r"
            }
            self.run = gui.add_button("Compare robots", color="green")
            self.gate_md = gui.add_markdown("", visible=False)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("")
            self.choice = gui.add_dropdown("Robot", ("—",), visible=False)
            self.use = gui.add_button(
                "View in Results", visible=False, hint="this robot at its stops, with its plans (Results tab)"
            )
        self.view_robot: Callable[[str], None] | None = None  # app.py: ui.compare_view.CompareView.view
        self.on_done: Callable[[], None] | None = None  # app.py: Results lists the compared robots
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

        @self.use.on_click
        def _(e) -> None:
            if self.choice.value not in list_robots():
                return
            if self.view_robot is not None:  # v9: view it in Results (the project's robot stays)
                self.view_robot(self.choice.value)
                notify(
                    e,
                    "Viewing a compared robot",
                    f"Results shows the {self.choice.value} and its plans",
                    ok=True,
                )
            else:
                self.use_robot(self.choice.value)

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "The comparison is already running")
        robots = [n for n, cb in self.pick.items() if cb.value]
        if not robots:
            return notify(event, "No robots", "Tick at least one robot")
        if s.view_stash:
            return notify(
                event, "Viewing a compared robot", "Results → Robot: go back to the project's robot first"
            )
        if s.tool_model is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot compare", "Need a tool, a part and targets")
        from opt.placement import PlannerSettings
        from ui import jobs
        from ui.plan_job import payload_for

        st = self.stations
        o = st.options()
        settings = PlannerSettings(
            max_stations=1 if s.sites() is None else int(st.max_stations.value),
            candidate_step=st.step.value / 1000,
            refine=st.refine.key != "none",
            refine_method=st.refine.key,
            zoning=o["zoning"],
            areas=o["areas"],
            require_all=o["require_all"],
            require_one_config=o["require_one_config"],
            cycle=getattr(s, "cycle", None),
        )
        spec = {
            "robots": robots,
            "settings": settings,
            "weights": st.costs.commit(),
            "workers": jobs.CHECK_WORKERS,
        }
        payload = payload_for(s, spec, live=False)  # always a snapshot: the live session stays as it is
        in_process = block or not jobs.USE_WORKER
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        t0 = time.monotonic()
        n = len(s.targets.points)

        def on_msg(kind: str, data) -> None:
            if kind == "progress":
                self.status.content = f"_{data[0]}_"
                self.progress.value = 100.0 * data[1]

        def work() -> None:
            try:
                rows = jobs.call("ui.compare:compare_job", payload, on_msg, self._cancel.is_set, in_process)
                self.rows = keep_plans(s, rows)  # v9 #6: each robot's plans, viewable in Results
                if self.on_done is not None:
                    self.on_done()
                self.status.content = table(self.rows, n)
                names = tuple(r["robot"] for r in self.rows if r.get("stations"))
                self.choice.options = names or ("—",)
                self.choice.visible = self.use.visible = bool(names)
                log(f"Robot comparison finished ({', '.join(robots)}).", "done", t0)
            except (CancelledError, jobs.JobCancelledError):
                self.status.content = "**Cancelled.**"
                log("Robot comparison cancelled.", "warn", t0)
            except Exception as err:
                self.status.content = f"⚠ comparison failed: {err}"
                log(f"Robot comparison failed: {err}", "error")
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()
