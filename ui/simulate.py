"""Results tab -> Paths & simulation (v2 phase 5): build the paths of the selected solution
(core.paths), draw them, play / scrub the robot (and carrier) along them, export a path CSV.

The build runs in a thread on its own Robot (Pinocchio data is not thread-safe). Playback
moves the carrier (or base) to each stop and the arm through the samples; it is a preview of
the TCP paths, not a timed trajectory. v4 plan items 2 and 4: a path pattern per group, lifts
over crossings, and (optionally) home moves with the carrier travelling between stops."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import viser

from core.feasibility import CancelledError
from ui.log_panel import log, took
from ui.robot_view import quiet_remove
from ui.session import Session
from ui.sim_build import carrier_path as sim_carrier_path
from ui.sim_build import path_checker
from ui.widgets import Latest, notify

ROOT = "/results/paths"
ZONE_COLORS = [(250, 120, 30), (30, 160, 230), (200, 60, 200), (40, 200, 120), (230, 200, 40), (120, 90, 220)]
ISSUE_COLOR = (230, 30, 30)
MOVE_COLOR = (150, 150, 150)  # lifts, home and carrier moves
BETWEEN_LABELS = {"Collision-free joint path": "direct", "Via home": "home"}
PLANNER_ORDER = "Planner order"


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
        order: float | None = None,
    ):
        self.server, self.s = server, session
        self.selected, self.move_carrier = selected, move_carrier
        self.set_base, self.show_q = set_base, show_q
        self.set_q = set_q or show_q  # syncs the joint sliders (scrubbing, end of playback)
        self.plan = None
        self.label = None  # solution the paths belong to
        self._handles: list = []
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._playing = threading.Event()
        self._station = None
        self._targets_key = None
        gui = server.gui
        self.folder = gui.add_folder("Paths & simulation", order=order)  # open, above Export
        with self.folder:
            self.step = gui.add_number("Path sample step (mm)", 20.0, min=1.0, step=5.0)
            self.jump = gui.add_number("Max joint step (deg)", 20.0, min=1.0, step=5.0)
            self._pattern_fields(gui)
            self.build_btn = gui.add_button("Build paths", color="green")
            self.gate_md = gui.add_markdown("", visible=False)  # why Build is disabled (T16)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Plan stations first, then build the paths of the selected solution._"
            )
            self.show = gui.add_checkbox("Show paths", True)
            self.play = gui.add_button("▶ Play", disabled=True)
            self.restart = gui.add_button(
                "⏮ Back to start", disabled=True, hint="playback to the first sample"
            )
            self.speed = gui.add_slider("Speed (mm/s)", 20.0, 2000.0, 10.0, 250.0)
            self.frame = gui.add_slider("Sample", 0, 1, 1, 0, disabled=True)
            self.status = gui.add_markdown("")
            self.export = gui.add_button("Export path CSV", disabled=True)
            self.clear_btn = gui.add_button("Clear paths", disabled=True, hint="remove the built paths")
        self.build_btn.on_click(lambda e: self.build(e))
        self.stop.on_click(lambda e: self._cancel.set())
        self.show.on_update(lambda e: self.draw())
        self.play.on_click(lambda e: self.toggle())
        self.restart.on_click(lambda e: self.rewind())
        self._goto = Latest(self.go_to)
        self.frame.on_update(lambda e: self._goto(int(self.frame.value)))
        self.export.on_click(lambda e: self.export_csv(e))
        self.clear_btn.on_click(lambda e: self.clear_paths())

    def _pattern_fields(self, gui) -> None:
        from core.patterns import CORNERS, PATTERNS

        self.pattern = gui.add_dropdown("Pattern", (*PATTERNS, PLANNER_ORDER), initial_value="Zigzag")
        self.angle = gui.add_number("Raster angle (deg)", 0.0, min=-180.0, max=180.0, step=5.0)
        self.corner = gui.add_dropdown("Start corner", CORNERS, hint="<along rows>-<across rows>")
        self.row_tol = gui.add_number("Row tolerance (mm)", 0.0, min=0.0, step=1.0, hint="0 = automatic")
        self.inward = gui.add_checkbox("Outside → in", True)
        self.no_cross = gui.add_checkbox(
            "Prevent crossing paths", False, hint="lift off for moves that cross"
        )
        self.lift = gui.add_number("Lift height (mm)", 30.0, min=1.0, step=5.0)
        self.home = gui.add_checkbox(
            "Home between stops", True, hint="arm to home before the carrier moves, back after"
        )
        self.between = gui.add_dropdown(  # keys of core.paths.BETWEEN
            "Between groups", tuple(BETWEEN_LABELS), hint="to the next group / configuration at a stop"
        )
        self.drive = gui.add_checkbox(
            "Drive carrier between stops",
            True,
            hint="plan the mobile base / external axis moves between stops (collision checked)",
        )
        from ui.method_picker import MethodPicker

        self.order = MethodPicker(gui, "point_order", "Point order")  # v8 T14
        self.motion = gui.add_checkbox(  # v8 T7
            "Check moves between points",
            True,
            hint="collision check every 2° of joint motion between samples (flags 'motion')",
        )
        for w in (self.pattern, self.no_cross):
            w.on_update(lambda e: self.show_pattern_fields())
        self.show_pattern_fields()

    def show_pattern_fields(self) -> None:
        from core.patterns import ROW_PATTERNS

        p = self.pattern.value
        on = p != PLANNER_ORDER
        self.angle.visible = self.corner.visible = self.no_cross.visible = on
        self.row_tol.visible = p in (*ROW_PATTERNS, "Concentric", "Spiral")
        self.inward.visible = p in ("Concentric", "Spiral")
        self.lift.visible = on and self.no_cross.value

    def pattern_settings(self):
        from core.patterns import PatternSettings

        if self.pattern.value == PLANNER_ORDER:
            return None
        return PatternSettings(
            self.pattern.value,
            self.angle.value,
            self.corner.value,
            self.row_tol.value / 1000,
            self.inward.value,
            self.no_cross.value,
        )

    def _carrier_path(self, sol, step: float, checker=None, home=None):
        """ui.sim_build.carrier_path, the arm held at ``home`` unless the call gives arm_q."""
        path = sim_carrier_path(self.s, sol, step, checker)
        if path is None:
            return None
        return lambda a, b, arm_q=None: path(a, b, home if arm_q is None else arm_q)

    # --- build ---------------------------------------------------------------------------------

    def _checker(self):
        return path_checker(self.s)  # private Robot / Pinocchio data for the thread

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
        if s.robot is None:
            return notify(event, "No robot", "Choose a robot first (tab 1 Robot & cell)")
        self.pause()
        checker = self._checker()
        pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
        poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
        mesh = s.part_mesh()
        targets_key = s.targets.model_dump_json()  # paths belong to these targets
        home = s.home_q() if self.home.value else None  # the project's home (Robot & cell)
        carrier_path = (
            sim_carrier_path(s, sol, max(self.step.value / 1000, 0.05), checker) if self.drive.value else None
        )
        pattern = self.pattern_settings()
        self._cancel.clear()
        self.build_btn.visible, self.stop.visible, self.progress.visible = False, True, True
        self.progress.value = 0.0
        self._t0 = time.monotonic()

        def work() -> None:
            try:
                plan = trace(  # local until done: the 1 s targets watcher must not see it half-built
                    checker,
                    mesh,
                    sol,
                    res,
                    pts,
                    poses,
                    step=self.step.value / 1000,
                    jump_deg=self.jump.value,
                    cancelled=self._cancel.is_set,
                    progress=self._build_progress,
                    pattern=pattern,
                    lift=self.lift.value / 1000,
                    home=home,
                    carrier_path=carrier_path,
                    group_of={p.id: p.group for p in s.targets.points},
                    park=None if home is not None else s.home_q(),  # clear pose to go via
                    order=self.order.key,
                    cycle=s.cycle,
                    between=BETWEEN_LABELS[self.between.value],
                )
                if self.motion.value:  # v8 T7: the arm sweeps between clear samples
                    from core.motion_check import check_moves

                    self._build_progress(1.0)
                    check_moves(checker, plan, cancelled=self._cancel.is_set)
                blocked = []  # stops where the home pose itself collides
                if home is not None:
                    for st in plan.stations:
                        checker.set_base(st.T_world_base)
                        if checker.scene.collides(home):
                            blocked.append(st.station)
                self.label, self._targets_key, self.home_blocked = sol.label, targets_key, blocked
                self.plan = plan  # last: key and label already match it
                self._after_build()
            except CancelledError:
                self.info.content = "**Cancelled.**"
                log("Path build cancelled.", "warn")
            except Exception as err:  # report in the GUI instead of dying silently in a thread
                self.info.content = f"⚠ path build failed: {err}"
                log(f"Path build failed: {err}", "error")
                raise
            finally:
                self.build_btn.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()

    def _build_progress(self, f: float) -> None:
        """Progress bar + a status line with the elapsed time (collision detours can take
        tens of seconds per stop)."""
        self.progress.value = 100.0 * f
        self.info.content = f"_Building paths… {100 * f:.0f} % · {took(self._t0)}_"

    def _cycle_line(self, plan) -> str:
        """v8 T6: cycle time of the built paths (Optimize -> Cycle time speeds)."""
        from core.cycle_time import CycleSettings, path_time
        from ui.safety_ui import reduced_speed_fn

        vlim = self.s.robot.model.velocityLimit if self.s.robot else None
        slow = reduced_speed_fn(self.s, plan)  # v8 T18: reduced-speed zones
        self.cycle = c = path_time(plan, self.s.cycle or CycleSettings(), vlim, slow)
        parts = ", ".join(f"{k} {v:.0f} s" for k, v in c.items() if k != "total" and v > 0.05)
        return f"  \nCycle time: **{c['total']:.0f} s** ({parts})."

    def _after_build(self) -> None:
        plan = self.plan
        c = plan.counts()
        problems = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in c.items() if v) or "none"
        self.info.content = (
            f"**{len(plan.samples)} samples**, {plan.length():.2f} m of TCP path over "
            f"{len(plan.stations)} stop(s) for _{self.label}_ "
            f"({self.pattern.value.lower()}{', home between stops' if self.home.value else ''}"
            f"{', carrier moves planned' if self.drive.value else ''}).  \n"
            f"Flagged samples: **{problems}**."
            + self._cycle_line(plan)
            + (
                "  \n_Reconfigure = the plan switches joint configuration there: plan with Joint "
                "configuration = zones to keep paths continuous._"
                if c["reconfigure"]
                else ""
            )
            + (
                "  \n⚠ **The home pose collides at stop "
                + ", ".join(f"S{k + 1}" for k in self.home_blocked)
                + "**: the moves to and from home cannot avoid it. Set a home that is clear there "
                "(Robot & cell → Robot → *Home position*)."
                if getattr(self, "home_blocked", None)
                else ""
            )
        )
        log(
            f"Paths built for **{self.label}**: {len(plan.samples)} samples, {plan.length():.2f} m; "
            f"flagged: {problems}.",
            "done" if problems == "none" else "warn",
            getattr(self, "_t0", None),
        )
        if getattr(self, "home_blocked", None):
            log(
                "The home pose collides at stop " + ", ".join(f"S{k + 1}" for k in self.home_blocked), "error"
            )
        n = len(plan.samples)
        self.frame.max = max(1, n - 1)
        self.frame.disabled = self.play.disabled = self.restart.disabled = self.export.disabled = (
            self.clear_btn.disabled
        ) = n == 0
        self._station = None
        self.draw()
        if n:
            self.frame.value = 0
            self.go_to(0)

    def reset(self, why: str = "") -> None:
        """Forget the built paths (a new plan / tool run started, or the targets changed)."""
        if self.plan is None and not self._handles:
            return
        self.pause()
        self.plan, self.label, self._targets_key = None, None, None
        self.clear()
        self.frame.disabled = self.play.disabled = self.restart.disabled = self.export.disabled = (
            self.clear_btn.disabled
        ) = True
        self.status.content = ""
        self.info.content = f"_Paths cleared{': ' + why if why else ''}. **Build paths** again._"

    def clear_paths(self) -> None:
        """Clear paths button: drop the built paths (Build paths makes them again)."""
        self.reset("by request")

    def check_targets(self) -> None:
        """Clear the paths when the targets changed since they were built (watcher, 1 s)."""
        if self.plan is not None and self._targets_key != self.s.targets.model_dump_json():
            self.reset("the targets changed")

    def on_selection(self) -> None:
        """Results selection changed: paths of another plan are cleared (not misleadingly
        shown for the new one)."""
        sol = self.selected()
        if self.plan is None or sol is None or sol.label == self.label:
            return
        self.pause()
        self.plan, self.label = None, None
        self.clear()
        self.frame.disabled = self.play.disabled = self.restart.disabled = self.export.disabled = (
            self.clear_btn.disabled
        ) = True
        self.status.content = ""
        self.info.content = f"_Paths were for another plan: **Build paths** for {sol.label}._"

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
        segs, cols, moves, bad = [], [], [], []
        for st in self.plan.stations:
            for p0, p1 in zip(st.samples[:-1], st.samples[1:], strict=True):
                if p1.issue == "reconfigure":
                    continue
                a, b = (T @ p0.T_part_tcp)[:3, 3], (T @ p1.T_part_tcp)[:3, 3]
                if p0.kind or p1.kind:  # lift, home or carrier move: grey
                    moves.append((a, b))
                    continue
                if p1.zone != p0.zone:
                    continue
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
        if moves:
            self._handles.append(
                self.server.scene.add_line_segments(
                    f"{ROOT}/moves",
                    np.array(moves, dtype=np.float32),
                    MOVE_COLOR,
                    thickness=1.5,
                    thickness_units="screen",
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
        if p.kind == "carrier" and p.carrier is not None:  # between stops: the carrier moves
            if self.s.base.mode == "movable":
                self.move_carrier(p.carrier)
            else:
                self.set_base(p.T_world_base)
            self._station = None  # back to the stop's own pose afterwards
        elif p.station != self._station:
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
        ends = ("home" if p.a == 0 else p.a, "home" if p.b == 0 else p.b)
        what = {
            "carrier": "carrier moving, arm parked",
            "home": f"{ends[0]} → {ends[1]} (joint move)",
            "retreat": f"backing off point {p.a}",
            "lift": f"point {p.a} → {p.b} (lifted)",
        }.get(p.kind, f"point {p.a} → {p.b}")
        self.status.content = f"Stop S{p.station + 1}{zone}: {what} ({100 * p.s:.0f} %){flag}"

    def toggle(self) -> None:
        if self._playing.is_set():
            return self.pause()
        if self.plan is None or not self.plan.samples:
            return
        if int(self.frame.value) >= len(self.plan.samples) - 1:  # at the end: play from the start
            self.rewind()
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

        self._player = threading.Thread(target=run, daemon=True)
        self._player.start()

    def pause(self) -> None:
        self._playing.clear()
        self.play.label = "▶ Play"

    def rewind(self) -> None:
        """Back to start: stop playback and show the first sample (carrier at the first stop)."""
        self.pause()
        player = getattr(self, "_player", None)
        if player is not None and player is not threading.current_thread():
            player.join(timeout=1.0)  # its last frame update must not land after the rewind
        if self.plan is None or not self.plan.samples:
            return
        self._station = None
        self.frame.value = 0
        self.go_to(0)

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
