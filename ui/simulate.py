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
from contextlib import nullcontext

import numpy as np
import viser

from core.feasibility import CancelledError
from core.path_spin import plan_needs_spin
from ui import sim_build, sim_fields
from ui.log_panel import log, took
from ui.plan_ids import key as plan_key
from ui.robot_view import quiet_remove
from ui.session import Session
from ui.session_lock import LOCK
from ui.sim_build import carrier_path as sim_carrier_path
from ui.sim_build import path_checker
from ui.sim_fields import BETWEEN_LABELS, PLANNER_ORDER  # noqa: F401 (re-export)
from ui.sim_status import status_text
from ui.widgets import Latest, notify

STATUS_PERIOD = 0.3  # s: shortest gap between status-text updates while playing
ROOT = "/results/paths"
ZONE_COLORS = [(250, 120, 30), (30, 160, 230), (200, 60, 200), (40, 200, 120), (230, 200, 40), (120, 90, 220)]
ISSUE_COLOR = (230, 30, 30)
MOVE_COLOR = (150, 150, 150)  # lifts, home and carrier moves


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
        self.on_home: Callable[[], None] | None = None  # v10 #8: the paths found another home
        self.label = None  # solution the paths belong to (shown)
        self.key = None  # v12 #8: its plan key (ui.plan_ids)
        self._handles: list = []
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._playing = threading.Event()
        self._status_t = 0.0  # last status update (throttled while playing)
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
            self.timed = gui.add_checkbox(  # v10 T14
                "Play at program timing", True, hint="as the cycle time has it; untick: the Speed below"
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
        sim_fields.build(self, gui)

    def show_pattern_fields(self) -> None:
        sim_fields.show(self)

    def pattern_settings(self):
        return sim_fields.pattern_settings(self)

    def _spin(self, res):
        return sim_fields.spin_of(self, res)  # v10 #9

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
        from core.home_search import blocked_stops, pick_home
        from core.paths import trace

        s, sol = self.s, self.selected()
        if sol is None:
            return notify(event, "No solution", "Plan stations and select a solution in Results first")
        res = s.result_points.get(plan_key(sol))
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
        carrier_path = (
            sim_carrier_path(s, sol, max(self.step.value / 1000, 0.05), checker) if self.drive.value else None
        )
        pattern = self.pattern_settings()
        pattern, pattern_of = sim_fields.group_patterns(self, pattern)  # v10 T15
        self._cancel.clear()
        self.build_btn.visible, self.stop.visible, self.progress.visible = False, True, True
        self.progress.value = 0.0
        self._t0 = time.monotonic()

        def work() -> None:
            try:  # the project's home (Robot & cell), or a clear folded pose when it collides
                home_q, self.home_note = pick_home(checker, s.home_q(), sol, res, poses)
                # v10 #8: Robot & cell -> Home position offers it (the home is set in one place)
                s.suggested_home = None if not self.home_note else np.asarray(home_q, float).copy()
                if self.on_home is not None:
                    self.on_home()
                home = home_q if self.home.value else None
                kw = dict(
                    step=self.step.value / 1000,
                    jump_deg=self.jump.value,
                    cancelled=self._cancel.is_set,
                    progress=self._build_progress,
                    pattern=pattern,
                    lift=self.lift.value / 1000,
                    home=home,
                    carrier_path=carrier_path,
                    group_of={p.id: p.group for p in s.targets.points},
                    park=None if home is not None else home_q,  # clear pose to go via
                    order=self.order.key,
                    cycle=s.cycle,
                    between=BETWEEN_LABELS[self.between.value],
                    smooth=bool(self.smooth.value),  # v10 T13
                    pattern_of=pattern_of,
                    direction=sim_fields.direction_of(s),  # v11 #7: directional tools
                )
                # local until done: the 1 s targets watcher must not see it half-built
                from core.motion_check import check_moves

                def checked(p):  # v8 T7: the arm sweeps between clear samples (before comparing:
                    if self.motion.value:  # a sweep collision makes a build worse, v10 T14 fix)
                        self._build_progress(1.0)
                        check_moves(checker, p, cancelled=self._cancel.is_set)
                    return p

                spin, self.spin_used = self._spin(res), None
                if sim_fields.coordinated(self):  # v10 T16: the carrier moves while the arm works
                    plan = checked(sim_build.coordinated_plan(self, checker, mesh, sol, res, pts, home))
                    spin = None
                else:
                    plan = checked(trace(checker, mesh, sol, res, pts, poses, **kw))
                if spin is not None and not plan_needs_spin(plan, checker.lo, checker.hi):
                    spin, self.spin_used = None, False  # v12 #18: clean and clear of the limits
                if spin is not None:  # v10 #9: also with the spin chosen along the path; keep
                    alt = checked(trace(checker, mesh, sol, res, pts, poses, **kw, spin=spin))
                    better = sim_build.score(alt, self._cycle_of(alt)) < sim_build.score(
                        plan, self._cycle_of(plan)
                    )
                    plan, self.spin_used = (alt, True) if better else (plan, False)
                blocked = blocked_stops(checker, home, plan.stations)  # the home pose itself collides
                with LOCK:  # v12 #9: the paths and the plan they belong to change together
                    self.label, self._targets_key, self.home_blocked = sol.label, targets_key, blocked
                    self.key = plan_key(sol)
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

    def _cycle_of(self, plan, times: list | None = None) -> dict:
        from core.cycle_time import CycleSettings, path_time
        from ui.safety_ui import reduced_speed_fn

        vlim = self.s.robot.model.velocityLimit if self.s.robot else None
        cs = self.s.cycle or CycleSettings()
        return path_time(plan, cs, vlim, reduced_speed_fn(self.s, plan), times)

    def _cycle_line(self, plan) -> str:
        """v8 T6: cycle time of the built paths (Optimize -> Cycle time speeds; v8 T18 reduced-speed
        zones). v10 #9: whether the optimized tool spin was used."""
        self.times: list = []  # v10 T14: time of every sample (playback at program timing)
        self.cycle = c = self._cycle_of(plan, self.times)
        sim_fields.keep_cycle(self, c)  # v10 T16: per plan and carrier motion (report)
        parts = ", ".join(f"{k} {v:.0f} s" for k, v in c.items() if k != "total" and v > 0.05)
        used = getattr(self, "spin_used", None)
        spin = ""
        if used is not None:
            spin = (
                " Tool spin: **optimized along the path**." if used else " Tool spin: as planned (no better)."
            )
        return f"  \nCycle time: **{c['total']:.0f} s** ({parts}).{spin}"

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
        if getattr(self, "home_note", ""):
            log(self.home_note, "warn")
            self.info.content += "  \n" + self.home_note
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
        sim_fields.show_motion(self)  # v10 T16: Carrier motion only for a movable rail / gantry
        if self.plan is None and not self._handles:
            return
        self.pause()
        self.plan, self.label, self.key, self._targets_key = None, None, None, None
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
        if self.plan is None or sol is None or plan_key(sol) == self.key:
            return
        self.pause()
        self.plan, self.label, self.key = None, None, None
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
        """Show sample i: carrier / base at its stop, arm at its joints (one atomic batch when playing)."""
        with self.server.atomic() if self._playing.is_set() else nullcontext():
            self._go_to(i)

    def _go_to(self, i: int) -> None:
        if self.plan is None:
            return
        samples = self.plan.samples
        if not samples:
            return
        i = int(np.clip(i, 0, len(samples) - 1))
        p = samples[i]
        if p.carrier is not None:  # between stops, or coordinated (v10 T16): the carrier moves
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
        now = time.monotonic()
        if not self._playing.is_set() or now - self._status_t >= STATUS_PERIOD:  # playing: ~3 Hz
            self._status_t = now
            self.status.content = status_text(p)

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
            times = getattr(self, "times", [])
            clock = times[int(pos)] if len(times) == n else None
            while self._playing.is_set():
                if clock is not None and self.timed.value:  # v10 T14: as the cycle time has it
                    clock += dt
                    pos = float(np.searchsorted(times, clock))
                else:
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
