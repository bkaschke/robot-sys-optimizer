"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser

from core.schema import Optimizer
from core.transforms import make_T
from ui import settings as ui_settings
from ui.carrier_ui import CarrierPanel
from ui.cell import CellTab
from ui.guide import GuideServer, add_guide_panel
from ui.objects import ObjectsPanel
from ui.optimize import INSPECT_SOURCES, OptimizeTab
from ui.placement import PlacementPanel
from ui.project_panel import ProjectPanel
from ui.results import ResultsTab
from ui.session import PartState, Session, last_project
from ui.simulate import SimPanel
from ui.stations import StationsPanel
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
    dark_toggle = server.gui.add_checkbox("Dark mode", dark)
    dark_toggle.on_update(lambda e: apply_theme(server, dark_toggle.value))
    guide = add_guide_panel(server, guide_port)

    def on_loaded() -> None:
        state["cell"].refresh()
        state["tool"].refresh()
        state["targets"].refresh()
        state["carrier"].load(session.base)
        state["carrier"].refresh()
        state["results"].refresh()
        restore_work()
        state["workflow"].refresh(force=True)
        state["tool_opt"].update_hint()

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
            panel.max_stations.value = float(min(max(n, 1), 12))
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
    objects = ObjectsPanel(server, session)
    workflow = Workflow(server, session)
    state["workflow"] = workflow
    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(f"{i + 1} {TAB_TITLES[i]}") for i, name in enumerate(TABS)}
    with tab_handles["Cell"]:
        workflow.header("Cell")
        cell = CellTab(server, session)
    with tab_handles["External Axis"]:
        workflow.header("External Axis")
        carrier = CarrierPanel(server, session, set_base=cell.set_base)
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
    cell.pose_listeners.append(Latest(carrier.draw))  # standoff shells follow a moved part

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

        stations = StationsPanel(server, session, on_solution=new_results)

        def apply_values(v) -> None:
            if tool._commit(lambda d: d["extension"]["nominal"].update(v.model_dump())):
                tool._after(full=True)

        tool_opt = ToolOptPanel(server, session, apply_values, on_solution=new_results)
    with tab_handles["Results"]:
        workflow.results_header()
        results = ResultsTab(server, session, set_base=cell.set_base)
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
    state["results"] = results
    tool_opt.planner_options = stations.options

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

    results.choice.on_update(on_selection)
    results.tool_opt = tool_opt
    results.move_carrier = carrier.set_values
    results.set_q = cell.set_q
    workflow.cell, workflow.carrier, workflow.results = cell, carrier, results
    objects.cell, objects.carrier, objects.tool = cell, carrier, tool
    objects.targets, objects.results, objects.sim = targets, results, sim
    workflow.refresh(force=True)
    objects.apply_gizmo()
    if autosave_thread:  # the real app: keep step status, banners and visibility current
        workflow.hooks.append(objects.apply_visibility)
        workflow.hooks.append(tool_opt.update_hint)  # pose-limits hint follows the tool type
        workflow.start_watch()
    server.gui.add_markdown(UR_NOTICE)

    state.update(
        tabs=tab_handles,
        cell=cell,
        carrier=carrier,
        objects=objects,
        sim=sim,
        tool=tool,
        targets=targets,
        optimize=optimize,
        placement=placement,
        stations=stations,
        tool_opt=tool_opt,
        project=project,
        dark_toggle=dark_toggle,
        guide=guide,
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot cell & end-effector optimizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dark", action="store_true", help="start in dark mode (also a checkbox in the GUI)")
    parser.add_argument("--guide-port", type=int, default=None, help="user guide port (default: --port + 1)")
    parser.add_argument(
        "--fresh", action="store_true", help="start with the default cell, not the last project"
    )
    args = parser.parse_args()

    server = viser.ViserServer(host=args.host, port=args.port, label="Robot Cell Optimizer")
    guide = None
    try:
        guide = GuideServer(args.host, args.guide_port or server.get_port() + 1)
        print(f"User guide: http://localhost:{guide.port}/", flush=True)
    except OSError as err:  # the app still works; the panel keeps the download button
        print(f"User guide server not started: {err}")
    state = build_app(
        server, dark=args.dark, guide_port=guide.port if guide else None, remember=True, autosave_thread=True
    )
    last = None if args.fresh else last_project()
    if last is not None:
        if state["project"]._load(None, last):
            print(f"Reopened last project: {last}", flush=True)
        else:
            print(f"Could not reopen {last}; starting with the default cell", flush=True)
        if args.dark:
            state["dark_toggle"].value = True  # the flag wins over the saved setting
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()
        if guide is not None:
            guide.stop()


if __name__ == "__main__":
    main()
