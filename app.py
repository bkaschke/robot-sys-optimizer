"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import time

import viser

from core.transforms import make_T
from ui.carrier_ui import CarrierPanel
from ui.cell import CellTab
from ui.optimize import OptimizeTab
from ui.placement import PlacementPanel
from ui.project_panel import ProjectPanel
from ui.results import ResultsTab
from ui.session import PartState, Session
from ui.stations import StationsPanel
from ui.targets_ui import TargetsTab
from ui.tool_opt import ToolOptPanel
from ui.tool_setup import ToolSetupTab

TABS = ("Tool Setup", "Cell", "External Axis", "Targets", "Optimize", "Results")
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


def build_app(server: viser.ViserServer, session: Session | None = None, dark: bool = False) -> dict:
    """Populate the scene and GUI. Returns handles for tests and later milestones."""
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

    def on_loaded() -> None:
        state["cell"].refresh()
        state["tool"].refresh()
        state["targets"].refresh()
        state["carrier"].load(session.base)
        state["carrier"].refresh()
        state["results"].refresh()

    project = ProjectPanel(server, session, on_loaded=on_loaded)
    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(name) for name in TABS}
    with tab_handles["Cell"]:
        cell = CellTab(server, session)
    with tab_handles["External Axis"]:
        carrier = CarrierPanel(server, session, set_base=cell.set_base)
    with tab_handles["Tool Setup"]:
        tool = ToolSetupTab(server, session)
    cell.listeners.append(tool.on_robot_moved)
    with tab_handles["Targets"]:
        targets = TargetsTab(server, session)
    cell.on_part_click = targets.on_part_click
    cell.part_listeners.append(targets.on_part_changed)
    cell.part_listeners.append(carrier.draw)

    def on_results() -> None:
        targets.color_by.value = "feasibility"
        targets.refresh()

    with tab_handles["Optimize"]:
        optimize = OptimizeTab(server, session, on_results, show_q=cell.set_q)
        placement = PlacementPanel(server, session, set_base=cell.set_base)
        stations = StationsPanel(server, session, on_solution=lambda: state["results"].refresh())

        def apply_values(v) -> None:
            if tool._commit(lambda d: d["extension"]["nominal"].update(v.model_dump())):
                tool._after(full=True)

        tool_opt = ToolOptPanel(server, session, apply_values, on_solution=lambda: state["results"].refresh())
    with tab_handles["Results"]:
        results = ResultsTab(server, session, set_base=cell.set_base)
    state["results"] = results
    results.tool_opt = tool_opt
    server.gui.add_markdown(UR_NOTICE)

    state.update(
        tabs=tab_handles,
        cell=cell,
        carrier=carrier,
        tool=tool,
        targets=targets,
        optimize=optimize,
        placement=placement,
        stations=stations,
        tool_opt=tool_opt,
        project=project,
        dark_toggle=dark_toggle,
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot cell & end-effector optimizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dark", action="store_true", help="start in dark mode (also a checkbox in the GUI)")
    args = parser.parse_args()

    server = viser.ViserServer(host=args.host, port=args.port, label="Robot Cell Optimizer")
    build_app(server, dark=args.dark)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
