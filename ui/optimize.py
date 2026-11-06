"""Optimize tab. M5: feasibility check of all targets for the current cell, tool and base.

The check runs in a background thread on its own Robot instance (Pinocchio data is not
thread-safe); results land in ``session.feasibility`` and recolour the target points.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import viser

from core.collision import CollisionScene, Margins
from core.feasibility import CheckSettings, FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from ui.session import Session
from ui.staleness import stamp_for
from ui.widgets import notify

INSPECT_SOURCES = ("feasibility check", "selected plan (Results)")
CHECK_FROM = ("stops of the selected plan", "current robot position")


class OptimizeTab:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        on_results: Callable[[], None],
        show_q: Callable[[np.ndarray], None],
    ):
        self.s = session
        self.on_results = on_results
        self.show_q = show_q
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        gui = server.gui
        with gui.add_folder("Feasibility check"):
            self.clearance = gui.add_number("Clearance (mm)", 10.0, min=0.0, step=1.0)
            self.joint_margin = gui.add_number("Joint-limit margin (deg)", 5.0, min=0.0, step=1.0)
            self.torque = gui.add_number("Torque utilization limit", 0.8, min=0.05, max=1.0, step=0.05)
            self.check_from = gui.add_dropdown(
                "Check from", CHECK_FROM, hint="each point from its stop in the selected plan, or from here"
            )
            self.run = gui.add_button("Check all targets", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.summary = gui.add_markdown("_Not run yet._")
        with gui.add_folder("Inspect result"):
            self.source = gui.add_dropdown("Show", INSPECT_SOURCES, initial_value=INSPECT_SOURCES[1])
            self.pick = gui.add_slider("Point #", 0, 1, 1, 0)
            self.step_pts = gui.add_button_group(
                "Step + move",
                ("◀ previous point", "next point ▶"),
                hint="go to the point and move the robot there",
            )
            self.detail = gui.add_markdown("")
            self.goto = gui.add_button("Move robot to this solution")

        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())
        self.pick.on_update(lambda e: self._show_detail())
        self.source.on_update(lambda e: self.refresh_inspect())
        # set by app.py: () -> (Solution | None, {point id: PointResult}) of the Results selection
        self.solution_points: Callable[[], tuple] | None = None
        self.move_to_point: Callable[[object, int], None] | None = None  # (solution, point id)
        self.go_station: Callable[[object, int], None] | None = None  # (solution, station index)
        self.station_of: dict[int, int] = {}  # plan check: point id -> stop it was checked from
        self.checked_plan = None  # the solution whose stops the last check used

        self.goto.on_click(lambda e: self.move_robot(e))
        self.step_pts.on_click(lambda e: self.step(-1 if e.target.value.startswith("◀") else 1, e))

    def step(self, delta: int, event=None) -> None:
        """Arrow: the previous / next point, and move the robot (and carrier) to it at once."""
        n = len(self._results())
        if n == 0:
            return notify(event, "Nothing to inspect", "Plan stations or check the targets first")
        self.pick.value = int(np.clip(int(self.pick.value) + delta, 0, n - 1))  # updates the detail text
        self.move_robot(event)

    def move_robot(self, e=None) -> None:
        """Move the robot to the inspected point's solution (carrier to its stop first)."""
        r = self._selected()
        if r is None or r.q is None:
            return notify(e, "No solution", "This point has no feasible joint solution")
        sol = self._solution()
        if sol is not None and self.move_to_point is not None:
            self.move_to_point(sol, r.id)  # carrier / base to the point's station, then the arm
            return
        if self.checked_plan is not None and r.id in self.station_of and self.go_station is not None:
            self.go_station(self.checked_plan, self.station_of[r.id])  # carrier to the stop first
        self.show_q(r.q)

    # --- running -------------------------------------------------------------------------

    def _build_checker(self) -> FeasibilityChecker:
        s = self.s
        if s.tool_model is None or s.part is None:
            raise ValueError("load a tool (Tool Setup) and a part (Cell) first")
        if not s.targets.points:
            raise ValueError("no targets: generate points in the Targets tab")
        if s.robot is None:
            raise ValueError("choose a robot first (tab 1 Robot & cell)")
        robot = Robot(s.robot.spec)  # private Pinocchio data for this thread
        settings = CheckSettings(self.joint_margin.value, self.torque.value, self.clearance.value / 1000)
        sites = s.sites()  # carrier bodies collide with the arm and tool too
        bb, wb = sites.scene_bodies() if sites is not None else ([], [])
        scene = CollisionScene(
            robot,
            s.tool_model,
            s.part_mesh(),
            s.T_world_base,
            s.part.T_world_part,
            Margins(clearance=settings.clearance),
            base_bodies=bb,
            world_bodies=wb,
            obstacles=s.obstacle_entries(),  # stay-out zones
        )
        return FeasibilityChecker(
            robot, make_ik(robot), s.tool_model, scene, s.T_world_base, s.part.T_world_part, settings
        )

    def start(self, event=None, block: bool = False) -> None:
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "A check is already running")
        try:
            checker = self._build_checker()
        except (ValueError, OSError) as err:
            return notify(event, "Cannot run", str(err))
        self._cancel.clear()
        targets = self.s.targets
        sol = self.solution_points()[0] if self.solution_points is not None else None
        plan_mode = self.check_from.value == CHECK_FROM[0] and sol is not None
        # a plan check does not depend on where the robot stands now
        stamp = stamp_for(self.s, "plan" if plan_mode else "feasibility")
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        self.progress.value = 0.0

        def progress(k: int, n: int) -> None:
            self.progress.value = 100.0 * k / n

        def work() -> None:
            try:
                if plan_mode:
                    res, station_of = self._check_plan(checker, sol, progress)
                else:
                    res = checker.check_all(targets, progress=progress, cancelled=self._cancel.is_set)
                    station_of = {}
                self.s.feasibility = {r.id: r for r in res}
                self.s.stamps["feasibility"] = stamp
                self.station_of, self.checked_plan = station_of, (sol if plan_mode else None)
                where = (
                    f"Checked each point from its stop in **{sol.label}** ({len(sol.stations)} stops)."
                    if plan_mode
                    else "Checked from the **current robot position**"
                    + (" (no plan selected in Results)." if self.check_from.value == CHECK_FROM[0] else ".")
                )
                self._finish(len(res), len(targets.points), where)
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()

    def _check_plan(self, checker, sol, progress):
        """Each point from the stop the plan assigns it to (points the plan did not reach:
        every stop is tried). Returns (results, point id -> stop index)."""
        from core.export import station_pose
        from core.schema import TargetGroup

        s = self.s
        poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
        groups = {g.name: g for g in s.targets.groups}
        pts = s.targets.points
        res: dict[int, object] = {}
        station_of: dict[int, int] = {}
        done = 0

        def check(pt):
            g = groups.get(pt.group) or TargetGroup(name=pt.group)
            return checker.check_point(pt.id, np.asarray(pt.p), np.asarray(pt.n), g)

        for k, T in enumerate(poses):
            checker.set_base(T)
            for pt in pts:
                if sol.assignment.get(pt.id) == k and not self._cancel.is_set():
                    res[pt.id], station_of[pt.id] = check(pt), k
                    done += 1
                    progress(done, len(pts))
        for pt in pts:
            if pt.id in res or self._cancel.is_set():
                continue
            for k, T in enumerate(poses):
                checker.set_base(T)
                r = check(pt)
                if r.status != "unreachable" or k == len(poses) - 1:
                    res[pt.id], station_of[pt.id] = r, k
                    if r.status != "unreachable":
                        break
            done += 1
            progress(done, len(pts))
        order = {pt.id: i for i, pt in enumerate(pts)}
        return sorted(res.values(), key=lambda r: order[r.id]), station_of

    def _finish(self, done: int, total: int, where: str = "") -> None:
        res = list(self.s.feasibility.values())
        count = {k: sum(r.status == k for r in res) for k in ("reachable", "marginal", "unreachable")}
        reasons: dict[str, int] = {}
        for r in res:
            if r.status == "unreachable":
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        notes = sorted({n for r in res for n in r.notes})
        why = ", ".join(f"{k}: {v}" for k, v in reasons.items()) or "none"
        self.summary.content = (
            (f"{where}  \n" if where else "")
            + f"**{count['reachable']}** reachable · **{count['marginal']}** marginal · "
            f"**{count['unreachable']}** unreachable ({why})"
            + ("" if done == total else f"  \n⚠ cancelled after {done}/{total} points")
            + "".join(f"  \n_{n}_" for n in notes)
        )
        self.source.value = INSPECT_SOURCES[0]
        self.refresh_inspect()
        self.on_results()

    # --- inspection ------------------------------------------------------------------------

    def _solution(self):
        """The Results selection when the inspector shows plan results, else None."""
        if self.source.value != INSPECT_SOURCES[1] or self.solution_points is None:
            return None
        sol, _ = self.solution_points()
        return sol

    def _results(self) -> list:
        if self.source.value == INSPECT_SOURCES[1] and self.solution_points is not None:
            _, res = self.solution_points()
            return [res[k] for k in sorted(res)]
        return list(self.s.feasibility.values())

    def refresh_inspect(self, source: str | None = None) -> None:
        """Re-read the shown results (after a check, a new plan / tool result, or another
        Results selection) so the inspector never shows a stale run."""
        if source is not None and source != self.source.value:
            self.source.value = source  # fires refresh_inspect again
            return
        self.pick.max = max(1, len(self._results()) - 1)
        self._show_detail()

    def _selected(self):
        res = self._results()
        if not res:
            return None
        return res[int(np.clip(self.pick.value, 0, len(res) - 1))]

    def _show_detail(self) -> None:
        r = self._selected()
        if r is None:
            self.detail.content = (
                "_No plan selected in Results._"
                if self.source.value == INSPECT_SOURCES[1]
                else "_Not run yet._"
            )
            return
        sol = self._solution()
        where = ""
        if sol is not None and r.id in sol.assignment:
            where = f" at station **S{sol.assignment[r.id] + 1}**"
            if r.zone is not None:
                where += f", zone {r.zone + 1} #{(r.seq or 0) + 1} ({r.config})"
        elif sol is not None:
            where = f" ({sol.label})"
        elif self.checked_plan is not None and r.id in self.station_of:
            where = f" from stop **S{self.station_of[r.id] + 1}**"
        if r.status == "unreachable":
            self.detail.content = f"Point **{r.id}**{where}: unreachable ({r.reason})"
            return
        q = ", ".join(f"{v:.1f}" for v in np.degrees(r.q))
        self.detail.content = (
            f"Point **{r.id}**{where}: {r.status} with TCP `{r.tcp}`  \n"
            f"spin {r.spin_deg:.0f}°, tilt {r.tilt_deg:.1f}°, standoff {r.standoff * 1000:.0f} mm  \n"
            f"clearance {r.clearance * 1000:.1f} mm · manipulability {r.manipulability:.3f} · "
            f"torque {100 * r.torque_util:.0f}% · limit margin {r.limit_margin_deg:.0f}°  \n"
            f"q (deg): {q}"
        )
