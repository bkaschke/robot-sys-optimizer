"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import time

import viser

from core.transforms import make_T
from ui.cell import CellTab
from ui.project_panel import ProjectPanel
from ui.session import PartState, Session
from ui.tool_setup import ToolSetupTab

TABS = ("Tool Setup", "Cell", "Optimize", "Results")
UR_NOTICE = (
    "_UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' "
    "Terms and Conditions for Use of Graphical Documentation._"
)


def build_app(server: viser.ViserServer, session: Session | None = None) -> dict:
    """Populate the scene and GUI. Returns handles for tests and later milestones."""
    if session is None:
        session = Session()
        session.part = PartState("assets/parts/flat_plate.stl", "m", make_T(p=(1.0, 0.0, 0.5)))
        session.load_robot_from_registry("ur20")
        session.load_tool_from_library("orbital_sander")
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (3.0, -3.0, 2.2)
    server.initial_camera.look_at = (0.5, 0.0, 0.6)
    server.scene.add_grid("/world/grid", width=10.0, height=10.0, cell_size=0.1, section_size=1.0)
    server.scene.add_frame("/world/origin", axes_length=0.2, axes_radius=0.005)

    state: dict = {"session": session}

    def on_loaded() -> None:
        state["cell"].refresh()
        state["tool"].refresh()

    project = ProjectPanel(server, session, on_loaded=on_loaded)
    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(name) for name in TABS}
    with tab_handles["Cell"]:
        cell = CellTab(server, session)
    with tab_handles["Tool Setup"]:
        tool = ToolSetupTab(server, session)
    cell.listeners.append(tool.on_robot_moved)
    for name in ("Optimize", "Results"):
        with tab_handles[name]:
            server.gui.add_markdown(f"_{name}: not built yet (see CLAUDE.md milestones)._")
    server.gui.add_markdown(UR_NOTICE)

    state.update(tabs=tab_handles, cell=cell, tool=tool, project=project)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot cell & end-effector optimizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = viser.ViserServer(host=args.host, port=args.port, label="Robot Cell Optimizer")
    build_app(server)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
