"""Tool-geometry panel (Optimize tab, M8): shortest tool length, then bend/offset/roll,
with the coverage-vs-length samples and an "apply to tool" action."""

from __future__ import annotations

import threading
from collections.abc import Callable

import viser

from core.feasibility import CancelledError
from core.robot import Robot
from core.schema import ExtensionValues, Optimizer
from opt.tool_geometry import GeometryResult, GeometrySettings, optimize_tool
from ui.session import Session
from ui.staleness import stamp_for
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
            self.tol = gui.add_number(
                "Length tolerance (mm)", 1.0, min=0.01, step=0.5
            )  # user-defined, no upper limit
            self.shape_evals = gui.add_number(
                "Shape / pose evaluations",
                30,
                min=0,
                max=500,
                step=5,
                hint="candidate tool geometries to try",
            )
            self.run = gui.add_button("Optimize tool", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Pose tools: the head pose (x y z rx ry rz within its ranges) with the best overall cost. "
                "Straight / bent tools: the shortest L, then θ / d / roll. Payload is not considered._"
            )
            self.legacy_hint = gui.add_markdown(
                "_This tool is the legacy type: the optimizer only changes L / θ / d / roll. For x y z / "
                "W P R limits set **Tool → Extension → Type = pose**._",
                visible=False,
            )
            self.apply = gui.add_button("Apply to tool", disabled=True)
            self.save_name = gui.add_text("Save as tool", "", hint="library name for the optimized tool")
            self.save_lib = gui.add_button("Save optimized tool to library", disabled=True)
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

        @self.apply.on_click
        def _(e) -> None:
            if self.result is not None and self.result.feasible:
                self.apply_values(self.result.values)
                notify(e, "Applied", "the optimized geometry is now the tool's rough placement", ok=True)

        @self.save_lib.on_click
        def _(e) -> None:
            try:
                path = self.save_to_library(self.save_name.value)
            except (OSError, ValueError) as err:
                return notify(e, "Save failed", str(err))
            notify(e, "Saved", str(path), ok=True)

    def save_to_library(self, name: str):
        """The current tool with the optimized values as its nominal, saved as a new
        library entry (tools/<name>.yaml); the tool in use is not changed."""
        from core.tool import save_tool

        r = self.result
        if r is None or not r.feasible or self.s.tool is None:
            raise ValueError("run Find shortest tool first")
        name = name.strip() or f"{self.s.tool.name}_opt"
        x = self.s.tool.extension.model_copy(update={"nominal": r.values})
        ee = self.s.tool.model_copy(update={"name": name, "extension": x})
        return save_tool(ee, name)

    def update_hint(self) -> None:
        """Show the pose-limits hint while the tool is a legacy (L / theta / d) tool."""
        tool = self.s.tool
        self.legacy_hint.visible = tool is not None and tool.extension.type != "pose"

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
        self.save_lib.disabled = True
        self._stamp = stamp_for(s, "tool")  # the setup this search ran on
        self._pose = s.tool.extension.type == "pose"

        def say(msg: str, f: float) -> None:
            self.info.content = f"_{msg}_"
            self.progress.value = 100.0 * f

        def work() -> None:
            try:
                self.result = optimize_tool(
                    *args, st=st, progress=say, cancelled=self._cancel.is_set, sites=sites
                )
                self._show()
            except CancelledError:  # Cancel while a reachability map was being built
                self.info.content = "**Cancelled** (while building the reachability map)."
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
        pose = self._pose
        rows = "\n".join(
            f"| {i + 1} | {x.L * 1000:.0f} | {'✔' if x.feasible else '✘'} | {x.reached}/{x.total} | "
            f"{x.stations} | {x.cost:.2f} |"
            if pose
            else f"| {x.L * 1000:.1f} | {'✔' if x.feasible else '✘'} | {x.reached}/{x.total} | {x.stations} |"
            for i, x in enumerate(r.samples)
        )
        head = (
            "| # | connector (mm) | full | reached | stations | cost |\n|---|---|---|---|---|---|\n"
            if pose
            else "| L (mm) | full | reached | stations |\n|---|---|---|---|\n"
        )
        table = head + rows
        if not r.feasible:
            self.info.content = f"⚠ {r.message}\n\n{table}"
            return
        v = r.values
        shape = "" if pose else f"θ {v.theta_deg:.1f}°, d {v.d * 1000:.0f} mm, roll {v.roll_deg:.1f}° · "
        self.info.content = (
            f"**{r.message}**  \n{shape}cost {r.solution.cost:.2f} · {len(r.solution.stations)} station(s)"
            "  \nNext: **Apply to tool**, then **Results → Paths & simulation → Build paths**."
            f"\n\n{table}"
        )
        self.apply.disabled = self.save_lib.disabled = False
        self.s.stamps[r.solution.label] = self._stamp
        self.s.result_points[r.solution.label] = r.point_results
        self.s.results = sorted([*self.s.results, r.solution], key=lambda x: x.cost)
        self.on_solution()
