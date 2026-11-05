"""Tool-geometry panel (Optimize tab, M8): shortest tool length, then bend/offset/roll,
with the coverage-vs-length samples and an "apply to tool" action."""

from __future__ import annotations

import threading
from collections.abc import Callable

import viser

from core.robot import Robot
from core.schema import ExtensionValues, Optimizer
from opt.tool_geometry import GeometryResult, GeometrySettings, optimize_tool
from ui.session import Session
from ui.stations import part_footprint, world_targets
from ui.widgets import notify


class ToolOptPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        apply_values: Callable[[ExtensionValues], None],
        on_solution: Callable[[], None],
    ):
        self.planner_options = None  # set by the app: StationsPanel.options
        self.s = session
        self.apply_values = apply_values
        self.on_solution = on_solution
        self.result: GeometryResult | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        gui = server.gui
        with gui.add_folder("Tool geometry"):
            opt = session.extra.get("optimizer") or Optimizer()
            self.max_stations = gui.add_number("Max stations", opt.max_stations, min=1, max=12, step=1)
            self.tol = gui.add_number("Length tolerance (mm)", 1.0, min=0.2, max=20.0, step=0.5)
            self.shape_evals = gui.add_number("Bend/offset/roll evaluations", 30, min=0, max=300, step=5)
            self.run = gui.add_button("Find shortest tool", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Bisection on L within the extension's range (Tool Setup), then CMA-ES on θ / d / roll._"
            )
            self.apply = gui.add_button("Apply to tool", disabled=True)
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

        @self.apply.on_click
        def _(e) -> None:
            if self.result is not None and self.result.feasible:
                self.apply_values(self.result.values)
                notify(e, "Applied", f"tool nominal set to L = {self.result.values.L * 1000:.1f} mm", ok=True)

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "Already running")
        if s.tool is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot optimize", "Need a tool, a part and targets")
        if s.robot.spec.ik != "ur_analytic":
            return notify(event, "Not supported", "Needs a UR-type arm (v1)")
        opts = self.planner_options() if self.planner_options else {"zoning": False, "config_cost": 1.0}
        robot = Robot(s.robot.spec)
        ee = s.tool
        urdf = s.resolve(ee.urdf) if ee.urdf else None
        args = (
            robot,
            ee,
            urdf,
            s.part_mesh(),
            s.part.T_world_part.copy(),
            s.T_world_base.copy(),
            world_targets(s),
            part_footprint(s),
            (s.extra.get("optimizer") or Optimizer()).weights.model_copy(
                update={"config": opts["config_cost"]}
            ),
            int(self.max_stations.value),
        )
        st = GeometrySettings(L_tol=self.tol.value / 1000, shape_evals=int(self.shape_evals.value))
        st.planner.zoning = opts["zoning"]
        sites = s.sites()
        if sites is None:
            args = (*args[:-1], 1)  # static base: one station
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        self.apply.disabled = True

        def say(msg: str, f: float) -> None:
            self.info.content = f"_{msg}_"
            self.progress.value = 100.0 * f

        def work() -> None:
            try:
                self.result = optimize_tool(
                    *args, st=st, progress=say, cancelled=self._cancel.is_set, sites=sites
                )
                self._show()
            except Exception as err:
                self.info.content = f"⚠ tool optimization failed: {err}"
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()

    def _show(self) -> None:
        r = self.result
        rows = "\n".join(
            f"| {x.L * 1000:.1f} | {'✔' if x.feasible else '✘'} | {x.reached}/{x.total} | {x.stations} |"
            for x in r.samples
        )
        table = "| L (mm) | full | reached | stations |\n|---|---|---|---|\n" + rows
        if not r.feasible:
            self.info.content = f"⚠ {r.message}\n\n{table}"
            return
        v = r.values
        self.info.content = (
            f"**{r.message}**  \nθ {v.theta_deg:.1f}°, d {v.d * 1000:.0f} mm, roll {v.roll_deg:.1f}° · "
            f"cost {r.solution.cost:.2f} · {len(r.solution.stations)} station(s)\n\n{table}"
        )
        self.apply.disabled = False
        self.s.result_points[r.solution.label] = r.point_results
        self.s.results = sorted([*self.s.results, r.solution], key=lambda x: x.cost)
        self.on_solution()
