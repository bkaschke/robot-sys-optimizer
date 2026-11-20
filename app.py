"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import viser

from core.transforms import make_T
from core.version import app_version
from ui import display_lod
from ui.app_state import build_settings, collect_work, restore_work
from ui.carrier_ui import CarrierPanel
from ui.cell import CellTab
from ui.guide import GuideServer
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
from ui.view_controls import ViewControls
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
    floor_grid(server).draw(dark)  # grid lines: light on a dark background, grey on a light one


def floor_grid(server: viser.ViserServer):
    """The server's floor grid (ui.floor.FloorGrid: sized to the cell)."""
    from ui.floor import FloorGrid

    if getattr(server, "_rso_floor", None) is None:
        server._rso_floor = FloorGrid(server)
    return server._rso_floor


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
    from ui.precision import precise

    precise(server)  # v12 #28: four decimals in every mm / deg field
    server.scene.set_up_direction("+z")
    apply_theme(server, dark)
    server.scene.add_frame("/world/origin", axes_length=0.2, axes_radius=0.005)

    state: dict = {"session": session}
    dark_toggle, worker_box, guide = build_settings(server, dark, guide_port, session)
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
        restore_work(state, session)
        state["workflow"].forget_states()  # the loaded project's confirmed steps: gizmos off
        state["workflow"].refresh(force=True)
        state["sim"].reset("another project was opened")
        state["tool_opt"].update_hint()
        state["cable"].follow_robot()  # the robot view was rebuilt: redraw the hose on it
        state["floor"].update(session)  # the floor fits the new cell
        view.fit(quiet=True)  # the camera frames the loaded cell
        state["collision"].settings_changed()  # the project's part surfaces kept exact
        state["compare_view"].refresh()  # v9 #6: the project's compared robots

    session.before_save.append(lambda: collect_work(state, session))
    project = ProjectPanel(
        server, session, on_loaded=on_loaded, remember=remember, autosave_thread=autosave_thread
    )
    tree = server.gui.add_panel()  # object tree on the left of the 3D view (v3 feedback item 1)
    with tree.add_tab("Objects"):
        state["view"] = view = ViewControls(server, session)  # lighting, shadows, camera (R2)
        objects = ObjectsPanel(server, session)
        from ui.collision_view import CollisionView

        coll_view = CollisionView(server, session)  # show the collision shapes per component
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
    workflow.status_strip()  # v10 #8: each step's state and what is out of date, above the tabs
    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(f"{i + 1} {TAB_TITLES[i]}") for i, name in enumerate(TABS)}
    with tab_handles["Cell"]:
        workflow.header("Cell")
        cell = CellTab(server, session, jog_container=objects.arm_slot)
        from ui.collision_settings import CollisionSettings

        coll = CollisionSettings(server, session)  # simplify for collision, arm padding (2026-11-11)
        coll.view = coll_view
        stayouts = StayoutPanel(server, session)  # arm, tool and carrier keep clear (v3 item 5)
        slowzones = StayoutPanel(server, session, slow=True)  # the robot only moves slower there
    with tab_handles["External Axis"]:
        workflow.header("External Axis")
        carrier = CarrierPanel(server, session, set_base=cell.set_base, dof_container=objects.carrier_slot)
    from ui.joystick import DrivePanel

    drive = DrivePanel(server, carrier, objects.drive_slot)  # carrier gizmo + joystick (left Jog)
    cell.listeners.append(lambda _T: drive.refresh())
    coll_view.cell = cell
    cell.listeners.append(coll_view.on_moved)
    cell.part_listeners.append(coll.settings_changed)  # a new part: its kept faces
    cell.pose_listeners.append(coll_view.on_moved)
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
    cell.part_listeners.append(view.on_part_changed)  # a different part: frame the cell

    def redraw_overlays() -> None:  # the display deviation was measured: re-lift selection and heatmap
        targets.sel_layer._mesh = None
        targets._draw_selection()
        targets._fp.pop("heat", None)
        targets.heat.draw()

    display_lod.on_deviation_ready[:] = [redraw_overlays]

    def part_moved() -> None:
        carrier.realign()  # a track aligned with the part follows it
        carrier.draw()  # standoff shells follow a moved part

    cell.pose_listeners.append(Latest(part_moved))

    def on_results() -> None:
        targets.color_by.value = "feasibility"
        targets.refresh()

    with tab_handles["Optimize"]:
        workflow.optimize_header()
        workflow.confirm_all_button()  # v11 #5
        from ui.preflight_ui import SetupCheckPanel

        # common setup mistakes, before a long plan (above the Optimizer dropdown: it applies to all)
        setup_check = SetupCheckPanel(server, session)
        from ui.optimizer_select import OptimizerSelect

        picker = OptimizerSelect(server)  # v9 #10: one Optimizer dropdown, one panel shown
        settings_folder = server.gui.add_folder("Settings", expand_by_default=False, order=1e6)
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

        stations = StationsPanel(server, session, on_solution=new_results, settings=settings_folder)

        def apply_values(v) -> None:
            if tool._commit(lambda d: d["extension"]["nominal"].update(v.model_dump())):
                tool._after(full=True)

        tool_opt = ToolOptPanel(
            server, session, apply_values, on_solution=new_results, settings=settings_folder
        )
        tool_opt.max_stations.visible = False  # v9: one Max stations (Settings) for every optimizer
        tool_opt.max_stations = stations.max_stations
        from ui.tool_ext_mirror import ExtensionMirror

        tool_ext_copy = ExtensionMirror(
            server, tool, tool_opt.ext_slot, tool_opt
        )  # 3 Tool -> Extension, here too
        from ui.cycle_ui import CyclePanel
        from ui.safety_ui import CablePanel

        with settings_folder:  # v9: shared by every optimizer
            cycle = CyclePanel(server, session)  # v8 T6
            cable = CablePanel(server, session, cell)  # v8 T18 (+ the hose drawn on the robot)
        stations.cable = tool_opt.cable = cable
        from ui.compare import ComparePanel

        compare = ComparePanel(  # v8 T9
            server, session, stations, use_robot=lambda name: setattr(cell.robot_choice, "value", name)
        )
        from ui.frame_ui import FramePanel

        frame = FramePanel(server, session, stations, carrier, on_results=new_results)  # v9 #9
        picker.on_show["Gantry placement"] = frame.refresh
        from ui.batch_ui import BatchPanel

        # v13 #6 + our queue: part packages, part files and the current part, planned by v13's runner
        batch = BatchPanel(server, session, project, targets, stations)
        picker.on_show["Several parts (batch)"] = batch.refresh
        state["batch"] = batch
        picker.bind(
            {
                "Station planner": [stations.folder],
                "Quick station plan": [stations.quick_folder],
                "Feasibility check": [optimize.folder],
                "Base placement map": [placement.folder],
                "Tool geometry": [tool_opt.folder],
                "Compare setups": [compare.folder],
                "Gantry placement": [frame.folder],
                "Several parts (batch)": [batch.folder],
            }
        )
    with tab_handles["Results"]:
        workflow.results_header()
        from ui.compare_view import CompareView

        compare_view = CompareView(server, session)  # v9 #6: Results -> Robot (compared robots)
        from ui.batch_view import BatchView

        batch_view = BatchView(server, session, project)  # v13 #7: the parts of a batch run
        # v7 round 2 (T18): Inspect result lives in Results; v11 #4: right under the solutions
        results = ResultsTab(server, session, set_base=cell.set_base, after_solutions=optimize.build_inspect)
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
        sim.on_home = cell.home_panel.refresh  # v10 #8: the paths' clear home is offered there
        batch_view.build_paths = sim.build
        batch.on_done = batch_view.open_batch
        state["batch_view"] = batch_view
        from ui.tolerance_ui import TolerancePanel

        tolerance = TolerancePanel(server, session, results.selected)  # v8 T5
        tolerance.on_done = lambda: targets.heat.draw()  # v8 T8: Colour part by -> tolerance
        stations.tolerance_settings = tolerance.settings  # v8 T12: robust objective
        from ui.edge_ui import EdgePanel, gap_arrays

        edge = EdgePanel(server, session, results.selected)  # v10 T10
        edge.targets, edge.on_done = targets, targets.refresh
        stations.on_done = edge.after_plan
        targets.edge_result = edge.current
        targets.edge_gaps = lambda: gap_arrays(edge.current())
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
    compare_view.cell, compare_view.results, compare_view.sim = cell, results, sim
    compare_view.tool_tab = tool  # v10 T17: compared tools
    compare_view.carrier = carrier  # v11 #8: compared external axes
    tool.travel.targets = targets  # v11 #7: Use for the target rows
    compare_view.on_changed = lambda: pareto.refresh()
    compare.view_robot, compare.on_done = compare_view.view, compare_view.refresh
    state["compare_view"] = compare_view
    with tabs.add_tab("Package builder"):  # URDF packages for tools, external axes, mobile bases
        from ui.builder_window import BuilderWindow

        builder_window = BuilderWindow(server, session, tool_tab=tool, carrier_panel=carrier)
        builder = builder_window.builder
    builder.on_part = cell.refresh_part  # v9: a part package used in the project
    builder.part_pkg.on_loaded = targets.refresh  # v13 #4: a part package opened for editing
    with tab_handles["Targets"]:
        builder.part_pkg.attach_targets_tab(server.gui)
    builder.collision.get_selection = lambda: targets.selection  # surfaces kept exact
    from ui.builder_window import open_html

    with coll.folder:  # v9 #11: the part's collision geometry is made in the Package builder
        server.gui.add_html(
            open_html(builder_window.port, "Simplify the part in the package builder ↗", False)
        )
    tool_opt.planner_options = stations.options
    workflow.gate(stations.run, stations.gate_md)
    workflow.gate(
        frame.run, frame.gate_md
    )  # v9 #9: like the planner, a confirmed setup  # optimizers need a confirmed setup (T5)
    workflow.gate(stations.quick_run, stations.quick_gate_md)
    from ui.workflow import feasibility_reason, plan_reason, result_reason

    workflow.gate(optimize.run, optimize.gate_md, feasibility_reason)  # T16: disabled buttons say why
    ex = results.export
    ex.tube_od = tool_opt.tube_od  # v8 T17: the STEP export uses the chosen tube
    ex.sim = sim  # v8 T19: the report shows the built paths' cycle time
    ex.frame = frame  # v10 T22: Gantry placement in the report
    workflow.gate(
        (ex.csv, ex.uframes, ex.urdf, ex.report, ex.step, ex.package),
        ex.gate_md,
        result_reason(results.selected),
    )
    workflow.gate(sim.build_btn, sim.gate_md, plan_reason(sim.selected))
    workflow.gate(tool_opt.run, tool_opt.gate_md)
    workflow.gate(compare.run, compare.gate_md)
    workflow.gate(tolerance.run, tolerance.gate_md, plan_reason(results.selected))
    workflow.gate(edge.run, edge.gate_md, plan_reason(results.selected))

    def selected_results() -> dict:
        sol = results.selected()
        return session.result_points.get(sol.key, {}) if sol is not None else {}  # v12 #8

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
    pre = setup_check.on_plan_start if autosave_thread else (lambda: None)  # the real app only
    stations.on_start = lambda: (sim.reset("a new station plan started"), pre())
    tool_opt.on_start = lambda: (sim.reset("the tool optimizer started"), pre())
    results.move_carrier = carrier.set_values
    results.set_q = cell.set_q
    workflow.cell, workflow.carrier, workflow.results = cell, carrier, results
    from ui.staleness import step_state

    tool_opt.confirm_tool = lambda: workflow.confirm("Tool Setup")
    tool_opt.tool_state = lambda: step_state(session, "Tool Setup")
    workflow.listeners.append(tool_opt.update_confirm)  # its Confirm tool follows the step's state

    workflow.on_confirmed.append(objects.step_confirmed)  # its drag gizmos go off
    objects.cell, objects.carrier, objects.tool = cell, carrier, tool
    carrier.on_overlay = objects.apply_visibility  # v9: redrawn overlays keep their show / hide ticks
    objects.targets, objects.results, objects.sim = targets, results, sim
    objects.stayouts, objects.slowzones = stayouts, slowzones
    from ui.move_mode import MoveMode

    objects.mover = MoveMode(server, objects.gizmo, on_commit=lambda: workflow.refresh(force=True))
    cell.on_part_alt_click = lambda e: objects.start_move("Part", e)
    stayouts.on_alt_click = lambda name, e: objects.start_move(f"Stay-out zone/{name}", e)
    slowzones.on_alt_click = lambda name, e: objects.start_move(f"Reduced-speed zone/{name}", e)
    objects.rebuild_tree()

    fresh: set[str] = set()  # plan keys seen up to date (ui.staleness)

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
        if sim.plan is not None and sim.key not in {r.key for r in session.results}:
            sim.reset("the plan they belonged to was cleared")

    workflow.refresh(force=True)
    objects.apply_gizmo()
    if autosave_thread:  # the real app: keep step status, banners and visibility current
        workflow.hooks.append(objects.apply_visibility)
        workflow.hooks.append(tool_opt.update_hint)  # pose-limits hint follows the tool type
        workflow.hooks.append(sim.check_targets)  # paths are cleared when the targets change
        workflow.hooks.append(clear_stale)  # and plans when the part, tool or targets change
        workflow.hooks.append(coll_view.check)  # v10 #5: collision files changed on disk
        workflow.hooks.append(frame.watch)  # v10 #6: the height range resets on carrier changes
        workflow.hooks.append(lambda: picker.sync(session))  # v11 #3: optimizers by axis type
        workflow.hooks.append(cable.follow_robot)  # the hose follows a robot change
        workflow.hooks.append(lambda: floor_grid(server).update(session))  # floor fits the cell
        workflow.start_watch()
    server.gui.add_markdown(UR_NOTICE)

    state["drive"] = drive
    state["collision"] = coll
    state["collision_view"] = coll_view
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
        picker=picker,
        frame=frame,
        placement=placement,
        stations=stations,
        tool_opt=tool_opt,
        tolerance=tolerance,
        edge=edge,
        cycle=cycle,
        compare=compare,
        setup_check=setup_check,
        builder=builder,
        builder_window=builder_window,
        pareto=pareto,
        cable=cable,
        tool_ext_copy=tool_ext_copy,
        floor=floor_grid(server),
        safety=safety,
        project=project,
        dark_toggle=dark_toggle,
        guide=guide,
        log=log_panel,
        clear_stale=clear_stale,
    )
    view.fit(quiet=True)  # start-up camera
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot cell & end-effector optimizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--version", action="version", version=f"Robot Cell Optimizer {app_version()}")
    parser.add_argument("--dark", action="store_true", help="start in dark mode (also a checkbox in the GUI)")
    parser.add_argument("--guide-port", type=int, default=None, help="user guide port (default: --port + 1)")
    # v7 round 2 (T9): the app always opens empty with a project popup; --fresh is kept so old
    # shortcuts keep working (it does nothing now), --open skips the popup
    parser.add_argument("--fresh", action="store_true", help="(no effect; the app always starts empty)")
    parser.add_argument("--open", default=None, metavar="FOLDER", help="open this project folder (no popup)")
    args = parser.parse_args()

    from core import app_log
    from ui import jobs

    log_file = app_log.setup(app_version())  # <data>/logs/app.log, for the support bundle
    print(f"Log file: {log_file}" if log_file else "Log file not written (folder not writable)", flush=True)
    jobs.USE_WORKER = True  # v8 T1: planners run in a worker process in the real app
    from opt.parallel import default_workers

    jobs.CHECK_WORKERS = default_workers()  # v8 T2
    server = viser.ViserServer(host=args.host, port=args.port, label=f"Robot Cell Optimizer {app_version()}")
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
    print(
        f"Robot Cell Optimizer ready: http://localhost:{server.get_port()}/ (version {app_version()})",
        flush=True,
    )  # CI start-up check
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
