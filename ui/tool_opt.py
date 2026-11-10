"""Tool-geometry panel (Optimize tab, M8): shortest tool length, then bend/offset/roll,
with the coverage-vs-length samples and an "apply to tool" action."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import viser

from core.feasibility import CancelledError
from core.ik import make_ik
from core.robot import Robot
from core.schema import ExtensionValues, Optimizer
from opt.tool_geometry import GeometryResult, GeometrySettings, optimize_tool
from ui.log_panel import log
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
        self.on_start: Callable[[], None] | None = None  # set by app.py: e.g. clear old paths
        gui = server.gui
        with gui.add_folder("Tool geometry"):
            opt = session.extra.get("optimizer") or Optimizer()
            self.max_stations = gui.add_number("Max stations", opt.max_stations, min=1, max=40, step=1)
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
            from ui.costs_ui import CostPanel

            self.costs = CostPanel(server, session, field="tool_weights")  # its own costs (v6)
            self.run = gui.add_button("Optimize tool", color="green")
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Pose tools: the head pose (x y z rx ry rz within its ranges) with the best overall cost. "
                "Straight / bent tools: the shortest L, then θ / d / roll._"
            )
            self.legacy_hint = gui.add_markdown(
                "_This tool is the legacy type: the optimizer only changes L / θ / d / roll. For x y z / "
                "W P R limits set **Tool → Extension → Type = pose**._",
                visible=False,
            )
            self.design = gui.add_dropdown(
                "Design",
                ("—",),
                visible=False,
                hint="the best designs found (also ones that miss points); each is listed in Results",
            )
            self.apply = gui.add_button("Apply to tool", disabled=True)
            self.save_name = gui.add_text("Save as tool", "", hint="library name for the optimized tool")
            self.save_lib = gui.add_button("Save optimized tool to library", disabled=True)
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

        @self.apply.on_click
        def _(e) -> None:
            d = self.chosen()
            if d is not None:
                self.apply_values(d.values)
                notify(
                    e,
                    "Applied",
                    f"design {d.solution.label.split(':')[0]} is now the tool's rough placement",
                    ok=True,
                )

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

        d = self.chosen()
        if d is None or self.s.tool is None:
            raise ValueError("run Optimize tool first")
        name = name.strip() or f"{self.s.tool.name}_opt"
        x = self.s.tool.extension.model_copy(update={"nominal": d.values})
        ee = self.s.tool.model_copy(update={"name": name, "extension": x})
        return save_tool(ee, name)

    def chosen(self):
        """The design picked in the dropdown (opt.tool_geometry.Design), else the result's
        best feasible geometry, else None."""
        from opt.tool_geometry import Design

        r = self.result
        if r is None:
            return None
        ds = r.designs
        if ds:
            opts = list(self.design.options)
            k = opts.index(self.design.value) if self.design.value in opts else 0
            return ds[min(k, len(ds) - 1)]
        return Design(r.values, r.solution, r.point_results, True) if r.feasible else None

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
        if s.robot is None:
            return notify(event, "No robot", "Choose a robot first (tab 1 Robot & cell)")
        if not getattr(make_ik(s.robot), "batch", False):
            return notify(
                event,
                "Not supported",
                "Tool optimization needs an analytic IK solver (spec.yaml: ik ur_analytic or opw). "
                "See the user guide, Adding a robot.",
            )
        opts = self.planner_options() if self.planner_options else {"zoning": False}
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
            self.costs.commit(),  # the tool optimizer's own cost weights
            int(self.max_stations.value),
        )
        st = GeometrySettings(L_tol=self.tol.value / 1000, shape_evals=int(self.shape_evals.value))
        req = self.costs.requirements()
        st.planner.zoning = opts["zoning"] or req["require_one_config"]
        st.planner.require_all = req["require_all"]
        st.planner.require_one_config = req["require_one_config"]
        self._require_all = req["require_all"]
        st.planner.areas = opts.get("areas", True)
        sites = s.sites()
        obstacles = s.obstacle_entries()  # stay-out zones
        if sites is None:
            args = (*args[:-1], 1)  # static base: one station
        if self.on_start is not None:
            self.on_start()
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
                    *args,
                    st=st,
                    progress=say,
                    cancelled=self._cancel.is_set,
                    sites=sites,
                    obstacles=obstacles,
                )
                self._show()
            except CancelledError:  # Cancel while a reachability map was being built
                self.info.content = "**Cancelled** (while building the reachability map)."
                log("Tool optimization cancelled (while building the reachability map).", "warn")
            except Exception as err:
                self.info.content = f"⚠ tool optimization failed: {err}"
                log(f"Tool optimization failed: {err}", "error")
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._t0 = time.monotonic()
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
        n = len(self.s.targets.points)
        ds = r.designs
        if not ds and r.feasible and r.solution is not None:  # older results: just the best
            from opt.tool_geometry import Design

            ds = r.designs = [Design(r.values, r.solution, r.point_results, True)]
        if getattr(self, "_require_all", False):  # must be met: only designs reaching every point
            ds = r.designs = [d for d in ds if d.feasible]
        names = [
            f"#{i + 1} {'✔' if d.feasible else '✘'} {n - len(d.solution.unreached)}/{n} pts · "
            f"cost {d.solution.cost:.2f} · {d.solution.label.split(':')[0]}"
            for i, d in enumerate(ds)
        ]
        self.design.options = tuple(names) if names else ("—",)
        self.design.value = self.design.options[0]
        self.design.visible = bool(names)
        self.apply.disabled = self.save_lib.disabled = not names
        listing = "  \n".join(names)
        if not r.feasible:
            pick = f"\n\n**Best designs (they miss points)**, pick one under *Design*:  \n{listing}"
            self.info.content = f"⚠ {r.message}" + (pick if names else "") + f"\n\n{table}"
            log(f"Tool optimization finished: {r.message}", "warn", getattr(self, "_t0", None))
        else:
            v = r.values
            shape = "" if pose else f"θ {v.theta_deg:.1f}°, d {v.d * 1000:.0f} mm, roll {v.roll_deg:.1f}° · "
            self.info.content = (
                f"**{r.message}**  \n{shape}cost {r.solution.cost:.2f} · "
                f"{len(r.solution.stations)} station(s)"
                "  \nNext: **Apply to tool**, then **Results → Paths & simulation → Build paths**."
                f"\n\n**Best designs:**  \n{listing}\n\n{table}"
            )
            log(f"Tool optimization finished: {r.message}.", "done", getattr(self, "_t0", None))
        for d in ds:  # every offered design is a result to inspect / build paths for
            self.s.stamps[d.solution.label] = self._stamp
            self.s.result_points[d.solution.label] = d.point_results
        if ds:
            labels = {d.solution.label for d in ds}
            keep = [x for x in self.s.results if x.label not in labels]
            self.s.results = sorted([*keep, *(d.solution for d in ds)], key=lambda x: x.cost)
            self.on_solution()
