"""Optimize -> Compare setups (v8 T9 robots; v10 T17 robots and / or tools): the same study planned
with several setups. A setup is a robot x a tool: the ticked robots (none: the project's) times the
ticked tools (library tools and the project's tool packages; none: the project's tool); the
project's own setup is always planned too (its row is marked, its plans stay the project's).

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
from ui.plan_ids import give_ids, key
from ui.session_lock import LOCK
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


def setups(
    robots: list[str], tools: dict, own_robot: str | None, own_tool: str, axes: dict | None = None
) -> list[dict]:
    """The setups to plan: robots x tools ({name: EndEffector}) x external axes (v11 #8: {name:
    Base}); the project's (``own_robot``, its tool, its axis) first and always. Keys: the robot's
    name, then "· tool" when not the project's tool and "· axis" when not the project's axis."""
    rs = list(robots) or [own_robot]
    ts = list(tools.items()) or [(None, None)]
    xs = list((axes or {}).items()) or [(None, None)]
    out = []
    for r in rs:
        for tn, ee in ts:
            for an, base in xs:
                same = tn is None or tn == own_tool
                key = r if same else f"{r} · {tn}"
                out.append(
                    {
                        "key": key if an is None else f"{key} · {an}",
                        "robot": r,
                        "tool": None if same else ee,
                        "tool_name": own_tool if same else tn,
                        "axis": base,
                        "axis_name": an or "",
                        "project": r == own_robot and same and an is None,
                    }
                )
    if not any(x["project"] for x in out):
        out.insert(
            0,
            {
                "key": own_robot, "robot": own_robot, "tool": None, "tool_name": own_tool,
                "axis": None, "axis_name": "", "project": True,
            },
        )  # fmt: skip
    out.sort(key=lambda x: not x["project"])
    return out


def tool_choices(s) -> dict[str, object]:
    """Tools to compare: {name: EndEffector}: the library's and the project's tool packages."""
    from core.tool import list_tools, load_tool
    from core.urdf_package import read_urdf, to_tool

    out = {}
    for name in list_tools():
        try:
            out[name] = load_tool(name)
        except (OSError, ValueError):
            continue
    pk = s.project_dir / "assets" / "packages"
    for d in sorted(pk.glob("*/package.yaml")) if pk.is_dir() else []:
        if d.parent.name.startswith("_"):
            continue
        try:
            urdf = next(d.parent.glob("*.urdf"))
            pkg = read_urdf(urdf, "tool")
            if pkg.kind == "tool":
                out.setdefault(f"{pkg.name} (package)", to_tool(pkg, s.relative(urdf)))
        except (StopIteration, OSError, ValueError):
            continue
    return out


def compare_job(payload: dict, emit, cancelled) -> list[dict]:
    """``spec``: setups (``setups``), settings (PlannerSettings), weights, workers. Each finished
    row is also sent as a ``row`` message, so a cancel keeps the rows done so far (review R4)."""
    from core import schema
    from ui import plan_job
    from ui.session import Session

    spec = payload["spec"]
    rows = []
    todo = spec["setups"]

    def add(row: dict) -> None:
        rows.append(row)
        emit("row", row)

    for n, su in enumerate(todo):
        name = su["robot"]
        if cancelled():
            raise CancelledError()
        emit("progress", (f"planning {su['key']} ({n + 1} / {len(todo)})", n / len(todo)))
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
        if su.get("tool") is not None:  # v10 T17: another tool (its own TCPs, contacts, ranges)
            s.set_tool(su["tool"])
        tag = {"setup": su["key"], "robot": name, "tool": su.get("tool_name", ""), "project": su["project"]}
        settings = spec["settings"]
        if su.get("axis") is not None:  # v11 #8: another external axis type, placed for this part
            from dataclasses import replace

            from ui.compare_axes import kind_of, place

            base, note = place(s, su["axis"], _rmap(s, cancelled), cancelled)
            if note.startswith("not placed"):  # v11 #12: no usable frame for this part
                add({**tag, "axis": su.get("axis_name", ""), "label": f"no plan ({note})"})
                continue
            s.base = base
            if base.mode != "movable":
                settings = replace(settings, max_stations=1)
            tag.update(axis=su.get("axis_name", ""), axis_kind=kind_of(base), placed=note)
            tag["base"] = base
        t0 = time.monotonic()
        sols: dict = {}
        sub = {
            "settings": settings,
            "weights": spec["weights"],
            "rules": None,
            "workers": spec["workers"],
        }

        def collect(kind, data, _sols=sols, _name=name) -> None:
            if kind == "solution":
                _sols[data[0].label] = data
            elif kind == "max_manip":
                _sols["__max_manip"] = float(data)
            elif kind == "warn":  # e.g. the check pool fell back to serial checks
                emit("warn", f"{_name}: {data}")

        out = plan_job.station_job({"session": s, "spec": sub}, collect, cancelled)
        best = out["best"]
        stopped = bool(cancelled())  # the plan returned its best so far after Cancel
        if best is None:
            add({**tag, "label": "no plan", "stopped": stopped})
        else:
            _, res = sols.get(best.label, (best, {}))
            m = _metrics(best, res, time.monotonic() - t0)
            from core.report_svg import stops_text

            m["stops_xy"] = stops_text(best.stations)  # v10 T22: the report's picture of this setup
            top = sols.get("__max_manip")
            m["manip_pct"] = 100 * m["manip"] / top if m["manip"] is not None and top else None
            from ui.staleness import stamp_for

            plans = sorted(
                (v for k, v in sols.items() if k != "__max_manip"),
                key=lambda sr: (bool(sr[0].violations), sr[0].cost),
            )
            give_ids([sol for sol, _ in plans])  # v12 #8
            stamp = stamp_for(s, "plan")  # this robot's setup: its plans are up to date when viewed with it
            add(
                {
                    **tag,
                    **m,
                    "plans": plans,
                    "stamps": {key(sol): dict(stamp) for sol, _ in plans},
                    "ee": su.get("tool"),
                    "stopped": stopped,
                }
            )
        if stopped:
            raise CancelledError()
    return rows


def _rmap(s, cancelled):
    """The reach map of the session's robot and tool (cached on disk; frame placement needs it)."""
    from core.robot import Robot
    from opt.reachability import map_for_tcps

    return map_for_tcps(
        Robot(s.robot.spec),
        s.tool_model,
        [c.name for c in s.tool.tcps],
        cancelled=cancelled,
        any_offset=s.tool.extension.type in ("pose", "bent2"),
    )


def keep_plans(session, rows: list[dict]) -> list[dict]:
    """Move each row's plans into ``session.compared``; returns the rows without them (the table)."""
    session.compared = {}
    out = []
    for r in rows:
        r = dict(r)
        plans, stamps, ee = r.pop("plans", []), r.pop("stamps", {}), r.pop("ee", None)
        base = r.pop("base", None)  # v11 #8: the setup's placed external axis
        if r.get("project") and r.get("stations"):  # v10 T22: its row (no plans) for the report
            session.compared[f"{r.get('setup', r['robot'])} (project)"] = {
                "robot": r["robot"],
                "tool": None,
                "results": [],
                "points": {},
                "stamps": {},
                "metrics": dict(r),
            }
        elif plans and not r.get("project"):  # the project's own plans stay the project's
            session.compared[r.get("setup", r["robot"])] = {
                "robot": r["robot"],
                "tool": ee,
                "base": base,
                "results": [sol for sol, _ in plans],
                "points": {key(sol): dict(res) for sol, res in plans},
                "stamps": stamps,
                "metrics": dict(r),
            }
        out.append(r)
    return out


def table(rows: list[dict], n_points: int) -> str:
    def f(v, fmt):
        return "—" if v is None else format(v, fmt)

    def name(r):
        label = r.get("setup", r["robot"])
        return f"{label} (stopped early)" if r.get("stopped") else label

    lines = [
        f"| {name(r)}{' (project)' if r.get('project') else ''} | "
        f"{f(r.get('stations'), 'd')} | {f(r.get('reached'), 'd')}/{n_points} | "
        f"{f(r.get('cycle_s'), '.0f')} | {f(r.get('clear_mm'), '.0f')} | {f(r.get('manip_pct'), '.0f')} | "
        f"{f(r.get('cost'), '.2f')}{' ✗' if r.get('violations') else ''} | {f(r.get('run_s'), '.0f')} |"
        for r in rows
    ]
    return (
        "| setup | stations | reached | cycle (s) | worst clearance (mm) | lowest manip. (%) | cost "
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
        self.folder = gui.add_folder("Compare setups", expand_by_default=False)
        with self.folder:
            gui.add_markdown(
                "_Plans the current study with each ticked robot and / or tool (every robot with every "
                "tool; none ticked = the project's), same part, targets, carrier and planner settings, and "
                "lists the results side by side. The project's own setup is always in the table._"
            )
            with gui.add_folder("Robots"):
                self.pick = {
                    name: gui.add_checkbox(name, name in ("ur20", "ur10e", "ur30"))
                    for name in list_robots()
                    if name != "generic_6r"
                }
            self.tools_folder = gui.add_folder("Tools", expand_by_default=False)
            self.tool_pick: dict = {}
            self.refresh_tools(gui)
            self.axes_folder = gui.add_folder("External axes", expand_by_default=False)  # v11 #8
            self.axis_pick: dict = {}
            self.refresh_axes(gui)
            self.run = gui.add_button("Compare setups", color="green")
            self.gate_md = gui.add_markdown("", visible=False)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("")
            self.choice = gui.add_dropdown("Setup", ("—",), visible=False)
            self.use = gui.add_button(
                "View in Results", visible=False, hint="this setup at its stops, with its plans (Results tab)"
            )
        self.view_robot: Callable[[str], None] | None = None  # app.py: ui.compare_view.CompareView.view
        self.on_done: Callable[[], None] | None = None  # app.py: Results lists the compared robots
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

        @self.use.on_click
        def _(e) -> None:
            key = self.choice.value
            if key not in self.s.compared and key not in list_robots():
                return
            if self.view_robot is not None:  # v9: view it in Results (the project's setup stays)
                self.view_robot(key)
                notify(e, "Viewing a compared setup", f"Results shows {key} and its plans", ok=True)
            else:
                self.use_robot(self.s.compared.get(key, {}).get("robot", key))

    def refresh_tools(self, gui=None) -> None:
        """The Tools folder: one checkbox per library tool and project tool package (v10 T17)."""
        gui = gui or self._gui
        self._gui = gui
        names = tuple(tool_choices(self.s))
        if tuple(self.tool_pick) == names:
            return
        for cb in self.tool_pick.values():
            cb.remove()
        with self.tools_folder:
            self.tool_pick = {n: gui.add_checkbox(n, False) for n in names}

    def refresh_axes(self, gui=None) -> None:
        """The External axes folder: one checkbox per axis type (v11 #8, ui.compare_axes)."""
        from ui.compare_axes import axis_choices

        gui = gui or self._gui
        self._gui = gui
        names = tuple(axis_choices(self.s))
        if tuple(self.axis_pick) == names:
            return
        for cb in self.axis_pick.values():
            cb.remove()
        with self.axes_folder:
            self.axis_pick = {n: gui.add_checkbox(n, False) for n in names}

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "The comparison is already running")
        robots = [n for n, cb in self.pick.items() if cb.value]
        all_tools = tool_choices(s)
        tools = {n: all_tools[n] for n, cb in self.tool_pick.items() if cb.value and n in all_tools}
        from ui.compare_axes import axis_choices

        all_axes = axis_choices(s) if any(cb.value for cb in self.axis_pick.values()) else {}
        axes = {n: all_axes[n] for n, cb in self.axis_pick.items() if cb.value and n in all_axes}
        if not robots and not tools and not axes:
            return notify(event, "Nothing to compare", "Tick at least one robot, tool or external axis")
        if s.robot_name is None or s.tool is None:
            return notify(event, "Cannot compare", "The project needs a library robot and a tool")
        if s.view_stash:
            return notify(
                event, "Viewing a compared robot", "Results → Robot: go back to the project's robot first"
            )
        if s.tool_model is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot compare", "Need a tool, a part and targets")
        from ui import jobs
        from ui.plan_job import failure_text, payload_for

        st = self.stations
        todo = setups(robots, tools, s.robot_name, s.tool.name, axes)
        spec = {
            "setups": todo,
            "settings": st.planner_settings(),  # review R12: exactly the station planner's settings
            "weights": st.costs.commit(),
            "workers": jobs.CHECK_WORKERS,
        }
        payload = payload_for(s, spec, live=False)  # always a snapshot: the live session stays as it is
        in_process = block or not jobs.USE_WORKER
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        t0 = time.monotonic()
        n = len(s.targets.points)
        done: list[dict] = []  # rows finished so far (kept on cancel)

        def on_msg(kind: str, data) -> None:
            if kind == "progress":
                self.status.content = f"_{data[0]}_"
                self.progress.value = 100.0 * data[1]
            elif kind == "row":
                done.append(data)
            elif kind == "warn":
                log(f"Setup comparison: {data}.", "warn")

        def show(rows: list[dict]) -> None:
            with LOCK:  # v12 #9
                self.rows = keep_plans(s, rows)  # v9 #6: each robot's plans, viewable in Results
            if self.on_done is not None:
                self.on_done()
            names = tuple(r["setup"] for r in self.rows if r.get("stations") and not r.get("project"))
            self.choice.options = names or ("—",)
            self.choice.visible = self.use.visible = bool(names)

        def finished() -> str:
            return f"{len(done)} of {len(todo)} setup(s) finished"

        def work() -> None:
            try:
                rows = jobs.call("ui.compare:compare_job", payload, on_msg, self._cancel.is_set, in_process)
                show(rows)
                self.status.content = table(self.rows, n)
                log(f"Setup comparison finished ({', '.join(r['setup'] for r in self.rows)}).", "done", t0)
            except (CancelledError, jobs.JobCancelledError):  # any cancel: keep the finished rows
                show(done)
                self.status.content = f"**Cancelled** ({finished()})." + (
                    f"\n\n{table(self.rows, n)}" if done else ""
                )
                log(f"Setup comparison cancelled ({finished()}).", "warn", t0)
            except Exception as err:
                show(done)
                self.status.content = f"⚠ comparison failed: {failure_text('Setup comparison', err)}" + (
                    f" ({finished()})\n\n{table(done, n)}" if done else ""
                )
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()
