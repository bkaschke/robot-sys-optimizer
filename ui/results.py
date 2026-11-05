"""Results tab (M7): ranked placement solutions, cost breakdown, stations in the scene with
ghost robots and lines to the points each station serves, and moving the base to a station.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import viser
from viser.extras import ViserUrdf

from core import targets as tg
from core.transforms import make_T, matrix_to_wxyz, rpy_to_matrix
from ui.session import HOME_Q, Session

ROOT = "/results"
STATION_COLORS = [(230, 60, 60), (40, 120, 230), (240, 170, 20), (150, 70, 200), (20, 170, 170), (90, 90, 90)]


def station_T(st, session: Session | None = None) -> np.ndarray:
    """Base pose of a station: exact from the carrier DOFs when a session is given."""
    if session is not None:
        from core.export import station_pose

        return station_pose(session.base, st, session.T_world_base)
    return make_T(rpy_to_matrix((0.0, 0.0, st.yaw)), st.xyz)


class ResultsTab:
    def __init__(self, server: viser.ViserServer, session: Session, set_base: Callable[[np.ndarray], None]):
        self.server = server
        self.s = session
        self.set_base = set_base
        self._handles: list = []
        self._ghosts: list = []
        self._busy = False
        gui = server.gui
        self.choice = gui.add_dropdown("Solution", ("(none)",))
        self.summary = gui.add_markdown("_No results yet: run the station planner (Optimize tab)._")
        self.breakdown = gui.add_markdown("")
        self.station = gui.add_dropdown("Station", ("-",))
        self.goto = gui.add_button("Move base to station")
        self.show_lines = gui.add_checkbox("Lines to served points", True)
        self.show_ghosts = gui.add_checkbox("Ghost robots", True)
        self.clear = gui.add_button("Clear from scene")
        from ui.export_ui import ExportPanel

        self.export = ExportPanel(server, session, self.selected, length_samples=self._length_samples)
        self.tool_opt = None  # set by app.py: the tool-geometry panel (coverage-vs-L samples)

        for w in (self.choice, self.show_lines, self.show_ghosts):
            w.on_update(lambda e: None if self._busy else self.draw())

        @self.goto.on_click
        def _(e) -> None:
            sol = self.selected()
            if sol is None or self.station.value == "-":
                return
            k = int(self.station.value.split()[0][1:]) - 1
            self.set_base(station_T(sol.stations[k], self.s))

        @self.clear.on_click
        def _(e) -> None:
            self._drop()

    def _length_samples(self, sol) -> list:
        r = getattr(self.tool_opt, "result", None)
        if r is not None and r.solution is not None and r.solution.label == sol.label:
            return r.samples
        return []

    def selected(self):
        labels = [self._label(i, s) for i, s in enumerate(self.s.results)]
        if self.choice.value not in labels:
            return None
        return self.s.results[labels.index(self.choice.value)]

    @staticmethod
    def _label(i: int, sol) -> str:
        return f"#{i + 1} {sol.label} (cost {sol.cost:.2f})"

    def refresh(self) -> None:
        """Called whenever the result list changes (streamed from the planner thread)."""
        self._busy = True
        try:
            labels = [self._label(i, s) for i, s in enumerate(self.s.results)] or ["(none)"]
            self.choice.options = tuple(labels)
            self.choice.value = labels[0]
        finally:
            self._busy = False
        self.draw()

    # --- drawing -------------------------------------------------------------------------------

    def _drop(self) -> None:
        for h in self._handles + self._ghosts:
            h.remove()
        self._handles, self._ghosts = [], []

    def draw(self) -> None:
        self._drop()
        sol = self.selected()
        if sol is None:
            self.summary.content = "_No results yet: run the station planner (Optimize tab)._"
            self.breakdown.content = ""
            self.station.options = ("-",)
            return
        n = len(self.s.targets.points)
        self.summary.content = (
            f"**{len(sol.stations)} station(s)**, {n - len(sol.unreached)}/{n} points reached, "
            f"cost **{sol.cost:.2f}**" + (f"  \nUnreached ids: {sol.unreached[:20]}" if sol.unreached else "")
        )
        total = max(sol.cost, 1e-9)
        rows = [
            f"`{k:<9}` {v:7.2f} {'█' * int(round(20 * v / total))}"
            for k, v in sorted(sol.breakdown.items(), key=lambda kv: -kv[1])
            if v > 0
        ]
        self.breakdown.content = "**Cost breakdown**  \n" + "  \n".join(rows)
        self.station.options = tuple(
            f"S{k + 1} ({st.xyz[0] * 1000:.0f}, {st.xyz[1] * 1000:.0f}) mm"
            for k, st in enumerate(sol.stations)
        )
        ids, P, _, _ = tg.as_arrays(self.s.targets)
        Tp = self.s.part.T_world_part if self.s.part else np.eye(4)
        Pw = P @ Tp[:3, :3].T + Tp[:3, 3] if len(P) else P
        where = {int(i): k for k, i in enumerate(ids)}
        res = self.s.result_points.get(sol.label, {})
        for k, st in enumerate(sol.stations):
            T = station_T(st, self.s)
            color = STATION_COLORS[k % len(STATION_COLORS)]
            self._handles.append(
                self.server.scene.add_frame(
                    f"{ROOT}/S{k + 1}",
                    axes_length=0.2,
                    axes_radius=0.008,
                    wxyz=matrix_to_wxyz(T[:3, :3]),
                    position=T[:3, 3],
                )
            )
            self._handles.append(
                self.server.scene.add_label(f"{ROOT}/S{k + 1}/label", f"S{k + 1}", position=(0, 0, 0.3))
            )
            served = [where[p] for p, s_k in sol.assignment.items() if s_k == k and p in where]
            if self.show_lines.value and served:
                seg = np.stack([np.repeat(T[None, :3, 3] + [0, 0, 0.3], len(served), 0), Pw[served]], axis=1)
                self._handles.append(
                    self.server.scene.add_line_segments(
                        f"{ROOT}/S{k + 1}_lines", seg, color, thickness=1.5, thickness_units="screen"
                    )
                )
                self._handles.append(
                    self.server.scene.add_point_cloud(
                        f"{ROOT}/S{k + 1}_pts",
                        Pw[served],
                        color,
                        point_size=0.012,
                        point_shape="circle",
                        precision="float32",
                    )
                )
            if self.show_ghosts.value:
                q = next(
                    (
                        res[p].q
                        for p, s_k in sol.assignment.items()
                        if s_k == k and p in res and res[p].q is not None
                    ),
                    None,
                )
                self._ghost(k, T, q, color)

    def _ghost(self, k: int, T: np.ndarray, q, color) -> None:
        robot = self.s.robot
        name = f"{ROOT}/ghost{k + 1}"
        self._ghosts.append(
            self.server.scene.add_frame(
                name, show_axes=False, wxyz=matrix_to_wxyz(T[:3, :3]), position=T[:3, 3]
            )
        )
        urdf = ViserUrdf(
            self.server,
            robot.spec.folder / robot.spec.urdf,
            root_node_name=name,
            mesh_color_override=(*[c / 255 for c in color], 0.35),
        )
        names = list(urdf.get_actuated_joint_names())
        q = q if q is not None else HOME_Q.get(self.s.robot_name or "", np.zeros(robot.nq))
        cfg = np.zeros(len(names))
        cfg[[names.index(j) for j in robot.joint_names]] = q
        urdf.update_cfg(cfg)
        self._ghosts.append(urdf)
