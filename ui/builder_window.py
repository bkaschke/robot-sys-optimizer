"""The package builder in its own browser window (2026-11-11).

The builder (ui.package_builder) runs on a second viser server (the app's port + 1, or the next
free one), so its parts, frames and gizmos never appear in the project's 3D view, and a robot
preview there (ui.package_robot) never touches the project's robot or tool. The Package builder
tab only has a button that opens that window; *Use in this project* in the builder still loads
the package into the project (Tool / External axis tabs).
"""

from __future__ import annotations

import viser

OPEN_JS = (
    "window.open(location.protocol+'//'+location.hostname+':{port}/','rso_package_builder',"
    "'width=1500,height=950')"
)


def open_html(port: int, label: str = "Open package builder ↗", note: bool = True) -> str:
    js = OPEN_JS.format(port=port)
    hint = (
        '<div style="font-size:11px;opacity:.65;margin-top:4px">'
        f"or browse to port {port} of this address</div>"
        if note
        else ""
    )
    return (
        '<div style="padding:4px 0 8px">'
        f'<button onclick="{js}" style="width:100%;padding:8px;border-radius:6px;border:0;'
        f'background:#2f9e44;color:white;font-weight:600;cursor:pointer">{label}</button>{hint}</div>'
    )


class BuilderWindow:
    def __init__(
        self, server: viser.ViserServer, session, tool_tab=None, carrier_panel=None, port: int | None = None
    ):
        from ui.package_builder import PackageBuilder

        self.main, self.s = server, session
        host = server.get_host()
        want = port if port is not None else server.get_port() + 1
        self.server = viser.ViserServer(host=host, port=want, verbose=False)  # viser: next free port
        self.server.gui.configure_theme(control_width="large")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=6.0, height=6.0, cell_size=0.25, plane="xy")
        self.port = self.server.get_port()

        @self.server.on_client_connect
        def _view(client) -> None:  # close on the package and the robot
            client.camera.position = (1.8, -1.8, 1.4)
            client.camera.look_at = (0.0, 0.0, 0.5)

        with self.server.gui.add_folder("Package builder", expand_by_default=True):
            self.builder = PackageBuilder(
                self.server, session, tool_tab=tool_tab, carrier_panel=carrier_panel, robot_preview=True
            )
        gui = server.gui
        gui.add_markdown(
            "_The package builder opens in its own window, so its parts and the robot preview stay out of "
            "this project's 3D view. **Use in this project** there loads the package here._"
        )
        self.link = gui.add_html(open_html(self.port))

    def stop(self) -> None:
        self.server.stop()
