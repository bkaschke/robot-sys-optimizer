"""Results -> Tolerance check (v8 T5): Monte Carlo of part / stop offsets for the selected plan
(opt.tolerance, run as a job: worker process + check pool like the planner)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import viser

from core.feasibility import CancelledError
from opt.tolerance import ToleranceSettings
from ui.log_panel import log
from ui.widgets import notify


class TolerancePanel:
    def __init__(self, server: viser.ViserServer, session, selected: Callable):
        self.s, self.selected = session, selected
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.on_done: Callable[[], None] | None = None  # e.g. recolour the part (T8)
        gui = server.gui
        with gui.add_folder("Tolerance check", expand_by_default=False) as self.folder:
            gui.add_markdown(
                "_Moves the part and the stops by random offsets and checks every planned point "
                "again from its stop: how often is each point still reachable?_"
            )
            st = ToleranceSettings()
            self.part_mm = gui.add_number("Part offset ± (mm)", st.part_mm, min=0.0, max=100.0, step=0.5)
            self.part_deg = gui.add_number("Part tilt up to (deg)", st.part_deg, min=0.0, max=10.0, step=0.1)
            self.station_mm = gui.add_number(
                "Stop offset ± (mm)",
                st.station_mm,
                min=0.0,
                max=100.0,
                step=0.5,
                hint="carrier positioning error",
            )
            self.samples = gui.add_number("Samples", st.samples, min=2, max=1000, step=1)
            self.seed = gui.add_number(
                "Seed", st.seed, min=0, max=10**6, step=1, hint="same seed = same samples"
            )
            self.run = gui.add_button("Run tolerance check", color="green")
            self.gate_md = gui.add_markdown("", visible=False)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("")
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())

    def settings(self) -> ToleranceSettings:
        return ToleranceSettings(
            float(self.part_mm.value),
            float(self.part_deg.value),
            float(self.station_mm.value),
            int(self.samples.value),
            int(self.seed.value),
        )

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "The tolerance check is already running")
        sol = self.selected()
        res = s.result_points.get(sol.label) if sol is not None else None
        if sol is None or not res:
            return notify(
                event, "No plan", "Select a plan in the Results list first (with its joint solutions)"
            )
        if s.robot is None or s.tool_model is None or s.part is None:
            return notify(event, "Cannot check", "Need a robot, a tool and a part")
        from ui import jobs
        from ui.plan_job import payload_for

        st = self.settings()
        spec = {"sol": sol, "results": res, "settings": st, "workers": jobs.CHECK_WORKERS}
        in_process = block or not jobs.USE_WORKER
        payload = payload_for(s, spec, live=in_process)
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        t0 = time.monotonic()

        def on_msg(kind: str, data) -> None:
            if kind == "progress":
                self.status.content = f"_{data[0]}_"
                self.progress.value = 100.0 * data[1]
            elif kind == "warn":
                log(f"Tolerance check: {data}.", "warn")

        def work() -> None:
            try:
                r = jobs.call("ui.plan_job:tolerance_job", payload, on_msg, self._cancel.is_set, in_process)
                s.tolerance = r
                self.status.content = summary(r)
                log(
                    f"Tolerance check of **{sol.label}**: every point reached in "
                    f"{100 * r.all_reached:.0f} % of {st.samples} samples.",
                    "done" if r.all_reached >= 0.999 else "warn",
                    t0,
                )
                if self.on_done is not None:
                    self.on_done()
            except (CancelledError, jobs.JobCancelledError):
                self.status.content = "**Cancelled.**"
                log("Tolerance check cancelled.", "warn", t0)
            except Exception as err:
                self.status.content = f"⚠ tolerance check failed: {err}"
                log(f"Tolerance check failed: {err}", "error")
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()


def summary(r) -> str:
    st = r.settings
    head = (
        f"**{100 * r.all_reached:.0f} %** of {st.samples} samples reach every planned point "
        f"(part ±{st.part_mm:g} mm, {st.part_deg:g}°, stops ±{st.station_mm:g} mm)."
    )
    worst = [p for p in r.worst(10) if p.rate < 1.0]
    if not worst:
        return head + "  \nEvery point was reached in every sample."
    rows = "\n".join(
        f"| {p.pid} | {p.station + 1} | {100 * p.rate:.0f} % | "
        f"{'—' if p.fail_mm is None else f'{p.fail_mm:.1f}'} | {p.clearance_mm:.0f} | {p.limit_deg:.0f} |"
        for p in worst
    )
    return (
        head
        + "\n\n**Least robust points** (fails from = the point's shift in the smallest failing sample):\n\n"
        + "| point | stop | reached | fails from (mm) | clearance (mm) | limit margin (°) |\n"
        + "|---|---|---|---|---|---|\n"
        + rows
    )
