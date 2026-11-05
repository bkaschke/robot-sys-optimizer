"""Viser server entry point: `python app.py`, then open http://localhost:8080."""

from __future__ import annotations

import argparse
import time

import viser

from core.golden_parts import GOLDEN_PARTS

TABS = ("Tool Setup", "Cell", "Optimize", "Results")


def build_app(server: viser.ViserServer) -> dict:
    """Populate the scene and GUI. Returns handles for tests and later milestones."""
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/world/grid", width=10.0, height=10.0, cell_size=0.1, section_size=1.0)
    server.scene.add_frame("/world/origin", axes_length=0.2, axes_radius=0.005)

    tabs = server.gui.add_tab_group()
    tab_handles = {name: tabs.add_tab(name) for name in TABS}

    state: dict = {"tabs": tab_handles, "part": None}

    with tab_handles["Cell"]:
        part_choice = server.gui.add_dropdown(
            "Test part", options=tuple(GOLDEN_PARTS), initial_value="flat_plate"
        )

    def show_part(name: str) -> None:
        mesh = GOLDEN_PARTS[name]()
        state["part"] = server.scene.add_mesh_trimesh("/world/part", mesh)
        state["part_name"] = name

    @part_choice.on_update
    def _(_event) -> None:
        show_part(part_choice.value)

    for name in ("Tool Setup", "Optimize", "Results"):
        with tab_handles[name]:
            server.gui.add_markdown(f"_{name}: not built yet (see CLAUDE.md milestones)._")

    show_part(part_choice.value)
    state["part_choice"] = part_choice
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
