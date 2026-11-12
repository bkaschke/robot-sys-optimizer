"""Optimize -> Gantry placement (v9, issue #9): place the whole rail / gantry frame relative to the
part, then plan its stops.

The job (``frame_job``, in the worker like the planner) scores every candidate frame
(opt.frame_place: standoff of the track line from the part, the gantry frame along / across the
part and shifted, the robot mount height in the z range) with the reach maps, then plans the best
few in full (the station planner with the current settings and costs). The table lists them;
**Use this frame** moves the project's carrier frame (and robot mount height) there and puts that
frame's plans in Results.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import viser

from core.feasibility import CancelledError
from opt.frame_place import FrameSettings, frame_text
from ui.log_panel import log
from ui.widgets import notify


def frame_job(payload: dict, emit, cancelled) -> list[dict]:
    """Score the candidate frames, plan the best ``top`` in full. Rows: frame, score, best plan."""
    from core import schema
    from core.robot import Robot
    from core.standoff import footprint
    from opt.frame_place import rank_frames
    from opt.reachability import map_for_tcps
    from ui.plan_job import session_from, station_job
    from ui.session import Session
    from ui.staleness import stamp_for
    from ui.stations import world_targets

    spec = payload["spec"]
    fs: FrameSettings = spec["frame"]
    s0 = session_from(payload)
    emit("progress", ("loading reachability map", 0.0))
    rmap = map_for_tcps(
        Robot(s0.robot.spec),
        s0.tool_model,
        [c.name for c in s0.tool.tcps],
        cancelled=cancelled,
        any_offset=s0.tool.extension.type in ("pose", "bent2"),
    )
    fp = footprint(s0.part_mesh(), s0.part.T_world_part)
    ranked = rank_frames(
        s0.base,
        rmap,
        world_targets(s0),
        s0.part_mesh(),
        s0.part.T_world_part,
        fp,
        fs,
        resolve=s0.resolve,
        obstacles=s0.obstacles(),
        progress=lambda msg, f: emit("progress", (f"scoring frames: {msg}", f)),
        cancelled=cancelled,
    )
    good = [f for f in ranked if not f.reason][: max(1, fs.top)]
    rows = []
    for n, f in enumerate(good):
        if cancelled():
            raise CancelledError()
        emit(
            "progress", (f"planning frame {n + 1} / {len(good)}: {frame_text(f)}", 0.6 + 0.4 * n / len(good))
        )
        s = Session()
        p = schema.Project.model_validate_json(payload["project"])
        s.apply_project(p, Path(payload["project_dir"]))
        s.T_world_base = payload["T_base"].copy()
        if payload["T_part"] is not None:
            s.part.T_world_part = payload["T_part"].copy()
        s.base = f.base
        sols: dict = {}

        def collect(kind, data, _sols=sols) -> None:
            if kind == "solution":
                _sols[data[0].label] = data

        sub = {k: spec[k] for k in ("settings", "weights", "workers")}
        out = station_job({"session": s, "spec": {**sub, "rules": None}}, collect, cancelled)
        best = out["best"]
        plans = sorted(sols.values(), key=lambda sr: (bool(sr[0].violations), sr[0].cost))
        stamp = stamp_for(s, "plan")
        rows.append(
            {
                "frame": frame_text(f),
                "base": f.base,
                "uncovered": f.uncovered,
                "stops_est": f.stops,
                "stations": len(best.stations) if best is not None else None,
                "reached": (len(best.assignment) if best is not None else 0),
                "cost": best.cost if best is not None else None,
                "cycle_s": best.cycle_s if best is not None else None,
                "plans": plans,
                "stamps": {sol.label: dict(stamp) for sol, _ in plans},
            }
        )
    rows.sort(key=lambda r: (r["cost"] is None, r["cost"] if r["cost"] is not None else 0.0))
    return rows


def table(rows: list[dict], n_points: int) -> str:
    def f(v, fmt):
        return "—" if v is None else format(v, fmt)

    lines = [
        f"| {k + 1} | {r['frame']} | {f(r.get('stations'), 'd')} | {r.get('reached', 0)}/{n_points} | "
        f"{f(r.get('cycle_s'), '.0f')} | {f(r.get('cost'), '.2f')} |"
        for k, r in enumerate(rows)
    ]
    return "| # | frame | stations | reached | cycle (s) | cost |\n|---|---|---|---|---|---|\n" + "\n".join(
        lines
    )


class FramePanel:
    def __init__(self, server: viser.ViserServer, session, stations, carrier, on_results: Callable[[], None]):
        self.s, self.stations, self.carrier, self.on_results = session, stations, carrier, on_results
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.rows: list[dict] = []
        gui = server.gui
        self.folder = gui.add_folder("Gantry placement", expand_by_default=True)
        with self.folder:
            self.note = gui.add_markdown("")
            self.standoff = gui.add_vector2(
                "Track standoff (mm)",
                (300.0, 1200.0),
                step=50.0,
                hint="rail: from the part outline to the track line",
            )
            self.step = gui.add_number("Standoff step (mm)", 150.0, min=25.0, max=1000.0, step=25.0)
            self.shift = gui.add_number(
                "Frame shift (mm)",
                600.0,
                min=0.0,
                max=5000.0,
                step=50.0,
                hint="gantry: off the part's centre",
            )
            self.z = gui.add_vector2(
                "Robot mount height (mm)",
                (0.0, 0.0),
                step=25.0,
                hint="T_mount z range; equal values = keep it",
            )
            self.z_steps = gui.add_number("Height steps", 3, min=1, max=9, step=1)
            self.top = gui.add_number("Plan the best", 3, min=1, max=8, step=1, hint="frames planned in full")
            self.run = gui.add_button("Place the frame", color="green")
            self.gate_md = gui.add_markdown("", visible=False)
            self.stop = gui.add_button("Cancel", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("")
            self.choice = gui.add_dropdown("Frame", ("—",), visible=False)
            self.use = gui.add_button("Use this frame", visible=False, hint="the carrier frame and its plans")
        self.run.on_click(lambda e: self.start(e))
        self.stop.on_click(lambda e: self._cancel.set())
        self.use.on_click(lambda e: self.use_frame(e))
        self.refresh()

    def refresh(self) -> None:
        """Fields for the current carrier (rail / gantry); the mount height range follows it."""
        b = self.s.base
        kind = b.carrier if b.mode == "movable" else None
        ok = kind in ("rail", "gantry")
        self.note.content = (
            f"_Places the **{kind}** frame relative to the part, then plans its stops (the station planner's "
            "settings and costs)._"
            if ok
            else "_For a rail or a gantry (External axis tab)._"
        )
        self.standoff.visible = self.step.visible = kind == "rail"
        self.shift.visible = kind == "gantry"
        for w in (self.z, self.z_steps, self.top, self.run):
            w.visible = ok
        if ok and self.z.value == (0.0, 0.0):
            z = b.T_mount.xyz[2] * 1000
            self.z.value = (z, z)

    def settings(self) -> FrameSettings:
        lo, hi = (v / 1000 for v in self.z.value)
        return FrameSettings(
            d_min=self.standoff.value[0] / 1000,
            d_max=self.standoff.value[1] / 1000,
            step=max(self.step.value, 25.0) / 1000,
            shift=self.shift.value / 1000,
            z_lo=lo,
            z_hi=hi,
            z_steps=int(self.z_steps.value),
            top=int(self.top.value),
        )

    def start(self, event=None, block: bool = False) -> None:
        s = self.s
        if self._thread is not None and self._thread.is_alive():
            return notify(event, "Busy", "Placing the frame already")
        if s.base.mode != "movable" or s.base.carrier not in ("rail", "gantry"):
            return notify(event, "Not a rail or gantry", "Set one up in the External axis tab")
        if s.tool_model is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot place", "Need a tool, a part and targets")
        if s.view_stash:
            return notify(
                event, "Viewing a compared robot", "Results → Robot: go back to the project's robot first"
            )
        from opt.placement import PlannerSettings
        from ui import jobs
        from ui.plan_job import payload_for

        st, o = self.stations, self.stations.options()
        planner = PlannerSettings(
            max_stations=int(st.max_stations.value),
            candidate_step=st.step.value / 1000,
            refine=st.refine.key != "none",
            refine_method=st.refine.key,
            zoning=o["zoning"],
            areas=o["areas"],
            require_all=o["require_all"],
            require_one_config=o["require_one_config"],
            cycle=getattr(s, "cycle", None),
        )
        spec = {
            "frame": self.settings(),
            "settings": planner,
            "weights": st.costs.commit(),
            "workers": jobs.CHECK_WORKERS,
        }
        payload = payload_for(s, spec, live=False)
        in_process = block or not jobs.USE_WORKER
        self._cancel.clear()
        self.run.visible, self.stop.visible, self.progress.visible = False, True, True
        t0 = time.monotonic()
        n = len(s.targets.points)

        def on_msg(kind: str, data) -> None:
            if kind == "progress":
                self.status.content = f"_{data[0]}_"
                self.progress.value = 100.0 * data[1]

        def work() -> None:
            try:
                self.rows = jobs.call(
                    "ui.frame_ui:frame_job", payload, on_msg, self._cancel.is_set, in_process
                )
                self.status.content = (
                    table(self.rows, n) if self.rows else "**No frame works** (standoff, keep-outs)."
                )
                labels = tuple(f"#{k + 1} {r['frame']}" for k, r in enumerate(self.rows))
                self.choice.options = labels or ("—",)
                self.choice.visible = self.use.visible = bool(labels)
                log(f"Gantry placement finished: {len(self.rows)} frame(s) planned.", "done", t0)
            except (CancelledError, jobs.JobCancelledError):
                self.status.content = "**Cancelled.**"
                log("Gantry placement cancelled.", "warn", t0)
            except Exception as err:
                self.status.content = f"⚠ placement failed: {err}"
                log(f"Gantry placement failed: {err}", "error")
                raise
            finally:
                self.run.visible, self.stop.visible, self.progress.visible = True, False, False

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        if block:
            self._thread.join()

    def use_frame(self, event=None, index: int | None = None) -> bool:
        """The chosen frame becomes the project's carrier frame; its plans go to Results."""
        if not self.rows:
            return False
        if index is None:
            labels = list(self.choice.options)
            index = labels.index(self.choice.value) if self.choice.value in labels else 0
        r = self.rows[index]
        s = self.s
        if self.carrier is not None:
            self.carrier.load(r["base"])
            if not self.carrier.commit(event):
                return False
        else:
            s.base = r["base"]
        s.results = [sol for sol, _ in r["plans"]]
        s.result_points = {sol.label: dict(res) for sol, res in r["plans"]}
        s.stamps = {**{k: v for k, v in s.stamps.items() if k == "feasibility"}, **r["stamps"]}
        self.on_results()
        log(f"Carrier frame placed: {r['frame']} ({len(s.results)} plan(s) in Results).", "done")
        notify(event, "Frame placed", r["frame"], ok=True)
        return True
