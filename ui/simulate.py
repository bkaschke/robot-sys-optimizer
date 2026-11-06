"""Results tab -> Paths & simulation (v2 phase 5): build the paths of the selected solution
(core.paths), draw them, play / scrub the robot (and carrier) along them, export a path CSV.

The build runs in a thread on its own Robot (Pinocchio data is not thread-safe). Playback
moves the carrier (or base) to each stop and the arm through the samples; it is a preview of
the TCP paths, not a timed trajectory (the moves between stops and reconfigurations are not
planned)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import viser

from core.feasibility import CancelledError
from ui.robot_view import quiet_remove
from ui.session import Session
from ui.widgets import Latest, notify

ROOT = "/results/paths"
ZONE_COLORS = [(250, 120, 30), (30, 160, 230), (200, 60, 200), (40, 200, 120), (230, 200, 40), (120, 90, 220)]
ISSUE_COLOR = (230, 30, 30)


class SimPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        selected: Callable[[], object],
        move_carrier: Callable[[dict], None],
        set_base: Callable[[np.ndarray], None],
        show_q: Callable[[np.ndarray], None],
        set_q: Callable[[np.ndarray], None] | None = None,
    ):
        self.server, self.s = server, session
        self.selected, self.move_carrier, self.set_base, self.show_q = (
            selected,
            move_carrier,
            set_base,
            show_q,
        )
        self.set_q = set_q or show_q  # syncs the joint sliders (scrubbing, end of playback)
        self.plan = None
        self.label = None  # solution the paths belong to
        self._handles: list = []
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._playing = threading.Event()
        self._station = None
        gui = server.gui
        with gui.add_folder("Paths & simulation", expand_by_default=False):
            self.step = gui.add_number("Path sample step (mm)", 20.0, min=1.0, step=5.0)
            self.jump = gui.add_number("Max joint step (deg)", 20.0, min=1.0, step=5.0)
            self.build_btn = gui.add_button("Build paths", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Plan stations first, then build the paths of the selected solution._"
            )
            self.show = gui.add_checkbox("Show paths", True)
            self.play = gui.add_button("▶ Play", disabled=True)
            self.speed = gui.add_slider("Speed (mm/s)", 20.0, 2000.0, 10.0, 250.0)
            self.frame = gui.add_slider("Sample", 0, 1, 1, 0, disabled=True)
            self.status = gui.add_markdown("")
            self.export = gui.add_button("Export path CSV", disabled=True)
        self.build_btn.on_click(lambda e: self.build(e))
        self.stop.on_click(lambda e: self._cancel.set())
        self.show.on_update(lambda e: self.draw())
        self.play.on_click(lambda e: self.toggle())
        self._goto = Latest(self.go_to)
        self.frame.on_update(lambda e: self._goto(int(self.frame.value)))
        self.export.on_click(lambda e: self.export_csv(e))

    # --- build ---------------------------------------------------------------------------------

    def _checker(self):
        from core.collision import CollisionScene
        from core.feasibility import FeasibilityChecker
        from core.ik import make_ik
        from core.robot import Robot

        s = self.s
        robot = Robot(s.robot.spec)  # private Pinocchio data for the thread
        sites = s.sites()
        bb, wb = sites.scene_bodies() if sites is not None else ([], [])
        scene = CollisionScene(
            robot,
            s.tool_model,
            s.part_mesh(),
            s.T_world_base,
            s.part.T_world_part,
            base_bodies=bb,
            world_bodies=wb,
        )
        return FeasibilityChecker(
            robot, make_ik(robot), s.tool_model, scene, s.T_world_base, s.part.T_world_part
        )

    def build(self, event=None, block: bool = False) -> None:
        from core.export import station_pose
        from core.paths import trace

        s, sol = self.s, self.selected()
        if sol is None:
            return notify(event, "No solution", "Plan stations and select a solution in Results first")
        res = s.result_points.get(sol.label)
        if not res:
            return notify(
                event, "No joint solutions", "Re-run the planner: this solution has no stored joints"
            )
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "Already building paths")
        self.pause()
        checker = self._checker()
        pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
        poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
        mesh = s.part_mesh()
        self._cancel.clear()
        self.build_btn.visible, self.stop.visible, self.progress.visible = False, True, True
        self.progress.value = 0.0

        def work() -> None:
            try:
                self.plan = trace(
                    checker,
                    mesh,
                    sol,
                    res,
                    pts,
                    poses,
                    step=self.step.value / 1000,
                    jump_deg=self.jump.value,
                    cancelled=self._cancel.is_set,
                    progress=lambda f: setattr(self.progress, "value", 100.0 * f),
                )
                self.label = sol.label
                self._after_build()
            except CancelledError:
                self.info.content = "**Cancelled.**"
            except Exception as err:  # report in the GUI instead of dying silently in a thread
                self.info.content = f"⚠ path build failed: {err}"
                raise
            finally:
                self.build_btn.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()

    def _after_build(self) -> None:
        plan = self.plan
        c = plan.counts()
        problems = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items() if v) or "none"
        self.info.content = (
            f"**{len(plan.samples)} samples**, {plan.length():.2f} m of TCP path over "
            f"{len(plan.stations)} stop(s) for _{self.label}_.  \nFlagged samples: **{problems}**."
            + (
                "  \n_Reconfigure = the plan switches joint configuration there: plan with Joint "
                "configuration = zones to keep paths continuous._"
                if c["reconfigure"]
                else ""
            )
        )
        n = len(plan.samples)
        self.frame.max = max(1, n - 1)
        self.frame.disabled = self.play.disabled = self.export.disabled = n == 0
        self._station = None
        self.draw()
        if n:
            self.frame.value = 0
            self.go_to(0)

    # --- drawing -------------------------------------------------------------------------------

    def clear(self) -> None:
        for h in self._handles:
            quiet_remove(h)
        self._handles = []

    def draw(self) -> None:
        self.clear()
        if self.plan is None or not self.show.value or self.s.part is None:
            return
        T = self.s.part.T_world_part
        segs, cols, bad = [], [], []
        for st in self.plan.stations:
            for p0, p1 in zip(st.samples[:-1], st.samples[1:], strict=True):
                if p1.issue == "reconfigure" or p1.zone != p0.zone:
                    continue
                a, b = (T @ p0.T_part_tcp)[:3, 3], (T @ p1.T_part_tcp)[:3, 3]
                segs.append((a, b))
                k = p1.zone if p1.zone is not None else p1.station
                cols.append(ZONE_COLORS[k % len(ZONE_COLORS)])
            bad += [(T @ p.T_part_tcp)[:3, 3] for p in st.samples if p.issue]
        if segs:
            seg = np.array(segs, dtype=np.float32)
            col = np.repeat(np.array(cols, dtype=np.uint8)[:, None, :], 2, axis=1)
            self._handles.append(
                self.server.scene.add_line_segments(
                    f"{ROOT}/lines", seg, col, thickness=3.0, thickness_units="screen"
                )
            )
        if bad:
            self._handles.append(
                self.server.scene.add_point_cloud(
                    f"{ROOT}/issues", np.array(bad, dtype=np.float32), ISSUE_COLOR, point_size=0.02
                )
            )

    # --- playback ------------------------------------------------------------------------------

    def go_to(self, i: int) -> None:
        """Show sample i: carrier / base at its stop (when it changes), arm at its joints."""
        if self.plan is None:
            return
        samples = self.plan.samples
        if not samples:
            return
        i = int(np.clip(i, 0, len(samples) - 1))
        p = samples[i]
        if p.station != self._station:
            st = self.plan.stations[[x.station for x in self.plan.stations].index(p.station)]
            if st.carrier and self.s.base.mode == "movable":
                self.move_carrier(st.carrier)
            else:
                self.set_base(st.T_world_base)
            self._station = p.station
        q = p.q
        if q is None:  # unreachable sample: hold the last reachable joints
            q = next((x.q for x in reversed(samples[:i]) if x.q is not None), None)
        if q is not None:
            (self.show_q if self._playing.is_set() else self.set_q)(np.asarray(q))
        zone = "" if p.zone is None else f", zone {p.zone + 1}"
        flag = f" · **{p.issue.replace('_', ' ')}**" if p.issue else ""
        self.status.content = f"Stop S{p.station + 1}{zone}: point {p.a} → {p.b} ({100 * p.s:.0f} %){flag}"

    def toggle(self) -> None:
        if self._playing.is_set():
            return self.pause()
        if self.plan is None or not self.plan.samples:
            return
        self._playing.set()
        self.play.label = "⏸ Pause"

        def run() -> None:
            dt = 0.05
            pos = float(self.frame.value)
            n = len(self.plan.samples)
            while self._playing.is_set():
                pos += self.speed.value / 1000 * dt / self.plan.step  # samples per frame
                if pos >= n - 1:
                    pos = n - 1
                    self._playing.clear()
                self.frame.value = int(pos)  # fires go_to (newest value only)
                time.sleep(dt)
            self.play.label = "▶ Play"
            self.go_to(int(self.frame.value))  # sync the joint sliders with the final pose

        threading.Thread(target=run, daemon=True).start()

    def pause(self) -> None:
        self._playing.clear()
        self.play.label = "▶ Play"

    # --- export --------------------------------------------------------------------------------

    def export_csv(self, event=None):
        from core.paths import path_csv

        if self.plan is None or self.s.part is None:
            return notify(event, "No paths", "Build the paths first")
        text = path_csv(self.plan, self.s.part.T_world_part)
        out = self.s.project_dir / "export"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "paths.csv"
        path.write_text(text, encoding="utf-8")
        client = getattr(event, "client", None)
        if client is not None:
            client.send_file_download(path.name, text.encode())
            notify(event, "Exported", str(path), ok=True)
        return path
