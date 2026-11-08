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
from ui.robot_view import quiet_remove
from ui.session import HOME_Q, Session

ROOT = "/results"
STATION_COLORS = [(230, 60, 60), (40, 120, 230), (240, 170, 20), (150, 70, 200), (20, 170, 170), (90, 90, 90)]
ZONE_COLORS = [(250, 120, 30), (30, 160, 230), (200, 60, 200), (40, 200, 120), (230, 200, 40), (120, 90, 220)]


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
        self._areas: list = []  # station areas painted on the part
        self._busy = False
        gui = server.gui
        self.choice = gui.add_dropdown("Solution", ("(none)",))
        self.nav = gui.add_button_group("Switch plan", ("◀ previous", "next ▶"))
        self.follow = gui.add_checkbox(
            "Move robot with the plan", True, hint="switching plans moves the robot to the first stop"
        )
        self.compare = gui.add_markdown("")
        self.summary = gui.add_markdown("_No results yet: run the station planner (Optimize tab)._")
        self.breakdown = gui.add_markdown("")
        self.station = gui.add_dropdown("Station", ("-",))
        self.goto = gui.add_button("Move robot to station")
        self.show_lines = gui.add_checkbox("Lines to served points", True)
        self.show_ghosts = gui.add_checkbox("Ghost robots", True)
        self.show_areas = gui.add_checkbox(
            "Station areas on the part", True, hint="the part painted by stop (groups lighter)"
        )
        self.clear = gui.add_button("Clear from scene")
        from ui.export_ui import ExportPanel

        self.export = ExportPanel(server, session, self.selected, length_samples=self._length_samples)
        self.tool_opt = None  # set by app.py: the tool-geometry panel (coverage-vs-L samples)
        self.move_carrier: Callable[[dict], None] | None = None  # set by app.py: carrier DOFs
        self.set_q: Callable[[np.ndarray], None] | None = None  # set by app.py: arm joints

        for w in (self.show_lines, self.show_ghosts, self.show_areas):
            w.on_update(lambda e: None if self._busy else self.draw())
        self.choice.on_update(lambda e: None if self._busy else self._chosen())
        self.nav.on_click(lambda e: self.switch(-1 if e.target.value.startswith("◀") else 1))

        @self.goto.on_click
        def _(e) -> None:
            sol = self.selected()
            if sol is None or self.station.value == "-":
                return
            self.go_to_station(sol, int(self.station.value.split()[0][1:]) - 1)

        @self.clear.on_click
        def _(e) -> None:
            self._drop()

    def switch(self, delta: int) -> None:
        """Arrow: the previous / next plan in the list (best cost first)."""
        labels = list(self.choice.options)
        if self.choice.value not in labels or labels == ["(none)"]:
            return
        i = int(np.clip(labels.index(self.choice.value) + delta, 0, len(labels) - 1))
        self.choice.value = labels[i]  # fires _chosen

    def _chosen(self) -> None:
        """The user picked another plan: draw it, optionally move the robot to its first stop."""
        self.draw()
        sol = self.selected()
        if sol is not None and self.follow.value:
            self.go_to_station(sol, 0)

    def _compare_table(self) -> str:
        """All plans side by side; the selected one in bold, out-of-date ones flagged."""
        from ui.staleness import solution_changes

        if not self.s.results:
            return ""
        cur = self.selected()
        rows = []
        for i, sol in enumerate(self.s.results):
            n = len(sol.assignment) + len(sol.unreached)
            mark = "**▶**" if sol is cur else ""
            old = " ⚠" if solution_changes(self.s, sol) else ""
            zones = f", {len(sol.zones)} zones" if sol.zones else ""
            rows.append(
                f"| {mark}{i + 1} | {len(sol.stations)}{zones} | {len(sol.assignment)}/{n} | "
                f"{sol.cost:.2f}{old} |"
            )
        return "| # | stops | reached | cost |\n|---|---|---|---|\n" + "\n".join(rows)

    def go_to_station(self, sol, k: int) -> None:
        """Move the robot to station k of ``sol``: the carrier DOFs (which carry the base) when
        a carrier is set up, else the base pose; then the arm to the joint solution of the
        first point that station serves (path order when zoned)."""
        st = sol.stations[k]
        if self.move_carrier is not None and st.carrier and self.s.base.mode == "movable":
            self.move_carrier(st.carrier)
        else:
            self.set_base(station_T(st, self.s))
        res = self.s.result_points.get(sol.label, {})
        order = [p for z in sol.zones if z.station == k for p in z.points]
        order += sorted(p for p, s in sol.assignment.items() if s == k)
        for pid in order:
            r = res.get(pid)
            if r is not None and r.q is not None and self.set_q is not None:
                self.set_q(np.asarray(r.q))
                break

    def go_to_point(self, sol, pid: int) -> None:
        """Robot (carrier, base, arm) to point ``pid`` as solved in ``sol``."""
        k = sol.assignment.get(pid)
        r = self.s.result_points.get(sol.label, {}).get(pid)
        if k is not None:
            self.go_to_station(sol, k)
        if r is not None and r.q is not None and self.set_q is not None:
            self.set_q(np.asarray(r.q))

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
        return f"#{i + 1} {'✗ ' if sol.violations else ''}{sol.label} (cost {sol.cost:.2f})"

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
        for h in self._handles + self._ghosts + self._areas:
            quiet_remove(h)
        self._handles, self._ghosts, self._areas = [], [], []

    def draw_areas(self, sol) -> None:
        from ui.station_areas import PATH, area_meshes, display_mesh

        mesh = self.s.part_mesh() if self.s.part else None
        if mesh is None or not self.show_areas.value:
            return
        ids, P, _, _ = tg.as_arrays(self.s.targets)
        mesh = display_mesh(mesh, P)
        for name, v, f, color in area_meshes(mesh, ids, P, sol, STATION_COLORS):
            self._areas.append(
                self.server.scene.add_mesh_simple(
                    f"{PATH}/{name}", v, f, color=color, opacity=0.55, side="double", flat_shading=False
                )
            )

    def draw(self) -> None:
        self.compare.content = self._compare_table()
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
        if sol.violations:
            self.summary.content += "  \n✗ **Breaks a must-be-met rule**: " + "; ".join(sol.violations)
        if sol.split_stops:
            names = ", ".join(f"S{k + 1}" for k in sol.split_stops)
            self.summary.content += (
                f"  \n⚠ Area of {names} is in pieces (no stop could take the points between)."
            )
        self.draw_areas(sol)
        if sol.zones:
            switches = len(sol.zones) - len({z.station for z in sol.zones})
            jumps = sum(z.wrap_jumps for z in sol.zones)
            per = " · ".join(
                f"S{z.station + 1}·Z{i + 1} `{z.config}` {len(z.points)} pts"
                + (f" ({z.wrap_jumps} jumps)" if z.wrap_jumps else "")
                for i, z in enumerate(sol.zones)
            )
            self.summary.content += (
                f"  \n**Zones**: {len(sol.zones)}, configuration changes **{switches + jumps}** "
                f"({switches} zone switches, {jumps} wrist jumps)  \n{per}"
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
            for zi, z in enumerate(sol.zones):  # zone paths: the order the points are processed
                path = [where[p] for p in z.points if p in where]
                if self.show_lines.value and z.station == k and len(path) > 1:
                    seg = np.stack([Pw[path[:-1]], Pw[path[1:]]], axis=1)
                    self._handles.append(
                        self.server.scene.add_line_segments(
                            f"{ROOT}/Z{zi + 1}_path",
                            seg,
                            ZONE_COLORS[zi % len(ZONE_COLORS)],
                            thickness=3.0,
                            thickness_units="screen",
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
        if robot is None:  # empty study: no arm to show
            return
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
