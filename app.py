"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser

from core.schema import Optimizer
from core.transforms import make_T
from ui import settings as ui_settings
from ui.carrier_ui import CarrierPanel
from ui.cell import CellTab
from ui.guide import GuideServer, add_guide_panel
from ui.log_panel import LogPanel, log
from ui.objects import ObjectsPanel
from ui.open_popup import OpenPopup
from ui.optimize import INSPECT_SOURCES, OptimizeTab
from ui.placement import PlacementPanel
from ui.project_panel import ProjectPanel
from ui.results import ResultsTab
from ui.session import PartState, Session, empty_cell
from ui.simulate import SimPanel
from ui.staleness import drop_stale_results
from ui.stations import StationsPanel
from ui.stayout_ui import StayoutPanel
from ui.targets_ui import TargetsTab
from ui.tool_opt import ToolOptPanel
from ui.tool_setup import ToolSetupTab
from ui.widgets import Latest
from ui.workflow import Workflow

# tabs in setup order (guided steps, ui.workflow); shown as "1 Robot & cell", ...
TABS = ("Cell", "External Axis", "Tool Setup", "Targets", "Optimize", "Results")
TAB_TITLES = ("Robot & cell", "External axis", "Tool", "Targets", "Optimize", "Results")
UR_NOTICE = (
    "_UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' "
    "Terms and Conditions for Use of Graphical Documentation._"
)


def apply_theme(server: viser.ViserServer, dark: bool) -> None:
    server.gui.configure_theme(dark_mode=dark, control_width="medium", brand_color=(90, 140, 220))
    from ui import log_panel

    log_panel.set_dark(dark)  # the log's row colours follow the theme
    # grid lines: light on a dark background, grey on a light one
    cell, section = ((80, 84, 92), (120, 126, 138)) if dark else ((200, 200, 200), (140, 140, 140))
    server.scene.add_grid(
        "/world/grid",
        width=10.0,
        height=10.0,
        cell_size=0.1,
        section_size=1.0,
        cell_color=cell,
        section_color=section,
    )


def build_app(
    server: viser.ViserServer,
    session: Session | None = None,
    dark: bool = False,
    guide_port: int | None = None,
    remember: bool = False,
    autosave_thread: bool = False,
) -> dict:
    """Populate the scene and GUI. Returns handles for tests and later milestones.
    ``guide_port``: port of the running user-guide server (ui.guide), for the panel link.
    ``remember`` / ``autosave_thread``: record the last project folder and run the autosave
    loop (the real app; tests leave both off and call ``project.autosave_tick``)."""
    if session is None:
        session = Session()
        session.part = PartState("assets/parts/flat_plate.stl", "m", make_T(p=(1.0, 0.0, 0.5)))
        session.load_robot_from_registry("ur20")
        session.load_tool_from_library("orbital_sander")
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (3.0, -3.0, 2.2)
    server.initial_camera.look_at = (0.5, 0.0, 0.6)
    apply_theme(server, dark)
    server.scene.add_frame("/world/origin", axes_length=0.2, axes_radius=0.005)

    state: dict = {"session": session}
    with server.gui.add_folder("Settings & help", expand_by_default=False):  # v7: out of the way
        dark_toggle = server.gui.add_checkbox("Dark mode", dark)
        from ui import jobs

        worker_box = server.gui.add_checkbox(
            "Run planners in a separate process",
            jobs.USE_WORKER,
            hint="keeps the panel responsive; Cancel always ends the run; "
            "the worker's memory is freed after 5 idle minutes",
        )

        def set_worker(_e=None) -> None:
            jobs.USE_WORKER = bool(worker_box.value)

        worker_box.on_update(set_worker)
        workers_num = server.gui.add_number(
            "Check worker processes",
            jobs.CHECK_WORKERS,
            min=1,
            max=16,
            step=1,
            hint="exact checks of the station planner in parallel (1 = off); each holds a robot + scene "
            "(~300 MB)",
        )

        def set_workers(_e=None) -> None:
            jobs.CHECK_WORKERS = int(workers_num.value)

        workers_num.on_update(set_workers)
        guide = add_guide_panel(server, guide_port)
    dark_toggle.on_update(lambda e: apply_theme(server, dark_toggle.value))
    state["worker_box"] = worker_box

    def on_loaded() -> None:
        state["cell"].refresh()
        state["tool"].refresh()
        state["targets"].refresh()
        state["carrier"].load(session.base)
        state["stayouts"].refresh()
        state["slowzones"].refresh()
        state["carrier"].refresh()
        state["results"].refresh()
        restore_work()
        state["workflow"].refresh(force=True)
        state["sim"].reset("another project was opened")
        state["tool_opt"].update_hint()
        state["cable"].follow_robot()  # the robot view was rebuilt: redraw the hose on it

    def restore_work() -> None:
        """Bring back where the user left off: selection, then panel settings."""
        t = state["targets"]
        sel = session.selection
        if sel and session.part is not None and sel[0] == session.part.mesh_path:
            m = t.mesh()
            faces = np.asarray(sel[1], dtype=int)
            if m is not None and (len(faces) == 0 or faces.max() < len(m.faces)):
                t.selection = faces
        # the project's station limit first; saved panel settings (if any) override it
        n = (session.extra.get("optimizer") or Optimizer()).max_stations
        for panel in (state["stations"], state["tool_opt"]):
            panel.max_stations.value = float(min(max(n, 1), 40))
        state["stations"].costs.load()  # the project's cost weights
        state["tool_opt"].costs.load()  # and the tool optimizer's
        ui_settings.apply(state, session.ui)
        t.refresh()

    def collect_work() -> None:
        """before_save hook: copy GUI-side state into the session."""
        t = state["targets"]
        part = session.part
        session.selection = (part.mesh_path, [int(f) for f in t.selection]) if part is not None else None
        session.ui = ui_settings.collect(state)
        opt = session.extra.get("optimizer") or Optimizer()
        session.extra["optimizer"] = opt.model_copy(
            update={"max_stations": int(state["stations"].max_stations.value)}
        )

    session.before_save.append(collect_work)
    project = ProjectPanel(
        server, session, on_loaded=on_loaded, remember=remember, autosave_thread=autosave_thread
    )
    tree = server.gui.add_panel()  # object tree on the left of the 3D view (v3 feedback item 1)
    with tree.add_tab("Objects"):
        objects = ObjectsPanel(server, session)
    tree.dock_left()
    tree.set_width(290)
    jog = server.gui.add_panel()  # Jog at the bottom of the left column (v7 round 2, T6)
    with jog.add_tab("Jog"):
        objects.build_jog()
    jog.dock_below(tree)  # viser: the anchor must be docked first
    objects.jog_panel = jog
    log_panel = LogPanel(server)  # errors, finished runs, setup status, unreached points
    workflow = Workflow(server, session)
    state["workflow"] = workflow
    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(f"{i + 1} {TAB_TITLES[i]}") for i, name in enumerate(TABS)}
    with tab_handles["Cell"]:
        workflow.header("Cell")
        cell = CellTab(server, session, jog_container=objects.arm_slot)
        stayouts = StayoutPanel(server, session)  # arm, tool and carrier keep clear (v3 item 5)
        slowzones = StayoutPanel(server, session, slow=True)  # the robot only moves slower there
    with tab_handles["External Axis"]:
        workflow.header("External Axis")
        carrier = CarrierPanel(server, session, set_base=cell.set_base, dof_container=objects.carrier_slot)
    with tab_handles["Tool Setup"]:
        workflow.header("Tool Setup")
        tool = ToolSetupTab(server, session)
    cell.listeners.append(tool.on_robot_moved)
    with tab_handles["Targets"]:
        workflow.header("Targets")
        targets = TargetsTab(server, session)
    cell.on_part_click = targets.on_part_click
    targets.tools.on_mesh_replaced = cell.refresh_part
    cell.part_listeners.append(targets.on_part_changed)
    cell.part_listeners.append(carrier.draw)

    def part_moved() -> None:
        carrier.realign()  # a track aligned with the part follows it
        carrier.draw()  # standoff shells follow a moved part

    cell.pose_listeners.append(Latest(part_moved))

    def on_results() -> None:
        targets.color_by.value = "feasibility"
        targets.refresh()

    with tab_handles["Optimize"]:
        workflow.optimize_header()
        optimize = OptimizeTab(server, session, on_results, show_q=cell.set_q)
        placement = PlacementPanel(server, session, set_base=cell.set_base)

        def new_results() -> None:
            """A plan or tool result arrived: list it, and inspect the selected plan."""
            state["results"].refresh()
            optimize.refresh_inspect(INSPECT_SOURCES[1])
            if "pareto" in state:
                state["pareto"].refresh()
            if "safety" in state:
                state["safety"].refresh()

        stations = StationsPanel(server, session, on_solution=new_results)

        def apply_values(v) -> None:
            if tool._commit(lambda d: d["extension"]["nominal"].update(v.model_dump())):
                tool._after(full=True)

        tool_opt = ToolOptPanel(server, session, apply_values, on_solution=new_results)
        from ui.tool_ext_mirror import ExtensionMirror

        tool_ext_copy = ExtensionMirror(server, tool, tool_opt.ext_slot)  # 3 Tool -> Extension, here too
        from ui.cycle_ui import CyclePanel

        cycle = CyclePanel(server, session)  # v8 T6
        from ui.safety_ui import CablePanel

        cable = CablePanel(server, session, cell)  # v8 T18 (+ the hose drawn on the robot)
        stations.cable = tool_opt.cable = cable
        from ui.compare import ComparePanel

        compare = ComparePanel(  # v8 T9
            server, session, stations, use_robot=lambda name: setattr(cell.robot_choice, "value", name)
        )
    with tab_handles["Results"]:
        workflow.results_header()
        results = ResultsTab(server, session, set_base=cell.set_base)
        optimize.build_inspect()  # v7 round 2 (T18): Inspect result lives in Results
        sim = SimPanel(  # paths along the selected solution + playback (v2 phase 5)
            server,
            session,
            results.selected,
            move_carrier=carrier.set_values,
            set_base=cell.set_base,
            show_q=cell.show_q,
            set_q=cell.set_q,
            order=results.export.folder.order - 0.5,  # above Export
        )
        from ui.tolerance_ui import TolerancePanel

        tolerance = TolerancePanel(server, session, results.selected)  # v8 T5
        tolerance.on_done = lambda: targets.heat.draw()  # v8 T8: Colour part by -> tolerance
        stations.tolerance_settings = tolerance.settings  # v8 T12: robust objective
        from ui.safety_ui import SafetyView

        safety = SafetyView(server, session, results.selected)  # v8 T18
        tool_opt.tolerance_settings = tolerance.settings
        from ui.pareto_ui import ParetoView

        pareto = ParetoView(  # v8 T11: Results -> Pareto front
            server,
            session,
            tool_length=lambda sol: (
                session.tool_model.length(sol.tool_values)
                if sol.tool_values is not None and session.tool_model is not None
                else None
            ),
            select=results.select_solution,  # a click on a dot selects the plan
            picked=results.selected,
        )
        results.chosen_listeners.append(pareto.refresh)
    state["results"] = results
    tool_opt.planner_options = stations.options
    workflow.gate(stations.run, stations.gate_md)  # optimizers need a confirmed setup (T5)
    workflow.gate(stations.quick_run, stations.quick_gate_md)
    from ui.workflow import feasibility_reason, plan_reason, result_reason

    workflow.gate(optimize.run, optimize.gate_md, feasibility_reason)  # T16: disabled buttons say why
    ex = results.export
    ex.tube_od = tool_opt.tube_od  # v8 T17: the STEP export uses the chosen tube
    ex.sim = sim  # v8 T19: the report shows the built paths' cycle time
    workflow.gate(
        (ex.csv, ex.uframes, ex.urdf, ex.report, ex.pdf, ex.step), ex.gate_md, result_reason(results.selected)
    )
    workflow.gate(sim.build_btn, sim.gate_md, plan_reason(sim.selected))
    workflow.gate(tool_opt.run, tool_opt.gate_md)
    workflow.gate(compare.run, compare.gate_md)
    workflow.gate(tolerance.run, tolerance.gate_md, plan_reason(results.selected))

    def selected_results() -> dict:
        sol = results.selected()
        return session.result_points.get(sol.label, {}) if sol is not None else {}

    targets.selected_results = selected_results
    optimize.solution_points = lambda: (results.selected(), selected_results())
    optimize.move_to_point = results.go_to_point
    optimize.go_station = results.go_to_station  # plan check: carrier to the point's stop

    def on_selection(_e=None) -> None:
        targets._draw_points()  # zone / configuration colours
        optimize.refresh_inspect()  # the inspector follows the Results selection
        workflow.refresh()  # out-of-date banner of the selected result
        sim.on_selection()  # paths belong to one plan
        safety.refresh()  # v8 T18: envelope of the selected plan

    results.choice.on_update(on_selection)
    results.tool_opt = tool_opt
    stations.on_start = lambda: sim.reset("a new station plan started")
    tool_opt.on_start = lambda: sim.reset("the tool optimizer started")
    results.move_carrier = carrier.set_values
    results.set_q = cell.set_q
    workflow.cell, workflow.carrier, workflow.results = cell, carrier, results
    from ui.staleness import step_state

    tool_opt.confirm_tool = lambda: workflow.confirm("Tool Setup")
    tool_opt.tool_state = lambda: step_state(session, "Tool Setup")
    workflow.listeners.append(tool_opt.update_confirm)  # its Confirm tool follows the step's state
    objects.cell, objects.carrier, objects.tool = cell, carrier, tool
    objects.targets, objects.results, objects.sim = targets, results, sim
    objects.stayouts, objects.slowzones = stayouts, slowzones
    from ui.move_mode import MoveMode

    objects.mover = MoveMode(server, objects.gizmo, on_commit=lambda: workflow.refresh(force=True))
    cell.on_part_alt_click = lambda e: objects.start_move("Part", e)
    stayouts.on_alt_click = lambda name, e: objects.start_move(f"Stay-out zone/{name}", e)
    slowzones.on_alt_click = lambda name, e: objects.start_move(f"Reduced-speed zone/{name}", e)
    objects.rebuild_tree()

    fresh: set[str] = set()  # result labels seen up to date (ui.staleness)

    def clear_stale() -> None:
        """v6 feedback: plans and their paths go when the part moves, the tool or the targets
        change (they no longer apply); a run in progress is left alone."""
        busy = any(p._thread is not None and p._thread.is_alive() for p in (stations, tool_opt, sim))
        if busy:
            return
        gone = drop_stale_results(session, fresh)
        if gone:
            what = sorted({c for v in gone.values() for c in v})
            results.refresh()
            log(
                f"Cleared {len(gone)} result(s): the {', '.join(what)} changed since they were planned.",
                "warn",
            )
        if sim.plan is not None and sim.label not in {r.label for r in session.results}:
            sim.reset("the plan they belonged to was cleared")

    workflow.refresh(force=True)
    objects.apply_gizmo()
    if autosave_thread:  # the real app: keep step status, banners and visibility current
        workflow.hooks.append(objects.apply_visibility)
        workflow.hooks.append(tool_opt.update_hint)  # pose-limits hint follows the tool type
        workflow.hooks.append(sim.check_targets)  # paths are cleared when the targets change
        workflow.hooks.append(clear_stale)  # and plans when the part, tool or targets change
        workflow.hooks.append(cable.follow_robot)  # the hose follows a robot change
        workflow.start_watch()
    server.gui.add_markdown(UR_NOTICE)

    state.update(
        tabs=tab_handles,
        cell=cell,
        carrier=carrier,
        stayouts=stayouts,
        slowzones=slowzones,
        objects=objects,
        sim=sim,
        tool=tool,
        targets=targets,
        optimize=optimize,
        placement=placement,
        stations=stations,
        tool_opt=tool_opt,
        tolerance=tolerance,
        cycle=cycle,
        compare=compare,
        pareto=pareto,
        cable=cable,
        tool_ext_copy=tool_ext_copy,
        safety=safety,
        project=project,
        dark_toggle=dark_toggle,
        guide=guide,
        log=log_panel,
        clear_stale=clear_stale,
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot cell & end-effector optimizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dark", action="store_true", help="start in dark mode (also a checkbox in the GUI)")
    parser.add_argument("--guide-port", type=int, default=None, help="user guide port (default: --port + 1)")
    # v7 round 2 (T9): the app always opens empty with a project popup; --fresh is kept so old
    # shortcuts keep working (it does nothing now), --open skips the popup
    parser.add_argument("--fresh", action="store_true", help="(no effect; the app always starts empty)")
    parser.add_argument("--open", default=None, metavar="FOLDER", help="open this project folder (no popup)")
    args = parser.parse_args()

    from ui import jobs

    jobs.USE_WORKER = True  # v8 T1: planners run in a worker process in the real app
    from opt.parallel import default_workers

    jobs.CHECK_WORKERS = default_workers()  # v8 T2
    server = viser.ViserServer(host=args.host, port=args.port, label="Robot Cell Optimizer")
    guide = None
    try:
        guide = GuideServer(args.host, args.guide_port or server.get_port() + 1)
        print(f"User guide: http://localhost:{guide.port}/", flush=True)
    except OSError as err:  # the app still works; the panel keeps the download button
        print(f"User guide server not started: {err}")
    state = build_app(
        server,
        session=empty_cell(),
        dark=args.dark,
        guide_port=guide.port if guide else None,
        remember=True,
        autosave_thread=True,
    )
    project = state["project"]
    project.untitled = True  # the empty start-up cell has no folder: no autosave until one opens
    opened = False
    if args.open:
        opened = project._load(None, Path(args.open).expanduser().resolve())
        print(f"Opened {args.open}" if opened else f"Could not open {args.open}", flush=True)
        if args.dark:
            state["dark_toggle"].value = True  # the flag wins over the saved setting
    shown = [opened]  # the popup: first browser only, and not after --open
    lock = threading.Lock()

    @server.on_client_connect
    def _(client) -> None:
        with lock:
            if shown[0]:
                return
            shown[0] = True
        state["popup"] = OpenPopup(client.gui, project)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        jobs.runner().stop()
        from opt.parallel import shutdown_all

        shutdown_all()
        server.stop()
        if guide is not None:
            guide.stop()


if __name__ == "__main__":
    main()
