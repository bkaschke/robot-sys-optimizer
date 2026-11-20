"""One part of a batch run (v13 #5): the part package in the project's cell, planned, its paths
built, saved as a project and reported, by the app's own panels.

``run_part`` builds the app without a browser (``app.build_app`` on a viser server nobody opens),
so a batch plans exactly as the GUI does: the project's planner settings, costs and collision
settings (its saved panel settings, ``WorkState.ui``), the same path build and the same report.
It runs in a worker process of ``ui.batch_run`` (one part per call) or in the calling process.

Output, ``<out>/<part>/``: ``project.json`` + ``assets/`` (opens with Project -> Open, plans
included), ``export/report.html`` and ``report.pdf`` (when Edge or Chrome is found).
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from pathlib import Path

from core import part_package as pp
from core import schema

STAGES = ("loading", "frame search", "planning", "paths", "saving", "report")


def _stage(emit, name: str, stage: str) -> None:
    if emit is not None:
        emit(name, stage, STAGES.index(stage) / len(STAGES))


def prepare(project_dir: Path, pkg_dir: Path, out: Path) -> Path:
    """``out`` as a project folder: the project's ``assets/`` (its tool, carrier and stay-out files)
    and its project.json with the package's part, pose and targets. Returns ``out``."""
    project_dir, out = Path(project_dir), Path(out)
    pkg = pp.load_part_package(pkg_dir)
    project = pp.apply_to_project(schema.load_project(project_dir / "project.json"), pkg, pkg_dir)
    if out.exists():
        shutil.rmtree(out)  # a re-run replaces the part's old results
    out.mkdir(parents=True)
    if (project_dir / "assets").is_dir():
        shutil.copytree(project_dir / "assets", out / "assets", ignore=shutil.ignore_patterns("_builder"))
    schema.save_project(project, out / "project.json")
    return out


def peak_memory() -> int | None:
    """Peak working set of this process in bytes (Windows; None elsewhere)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t)
            for n in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    try:
        ok = fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    except (OSError, ctypes.ArgumentError):
        return None
    return int(c.PeakWorkingSetSize) if ok else None


def _last_problem(since: float) -> str:
    """The newest error / warning the app logged since ``since`` (the planner's reason), as
    ": <message>"; "" when there is none."""
    from ui.log_panel import entries

    msgs = [m for t, level, m in entries() if t >= since and level in ("error", "warn")]
    return f": {msgs[-1].replace('**', '')}" if msgs else ""


def row_for(name: str, s, sol, cycle: float | None) -> dict:
    n = len(s.targets.points)
    reached = n - len(sol.unreached) - len(sol.unfit)
    return {
        "part": name,
        "points": n,
        "reached": reached,
        "unreached": len(sol.unreached) + len(sol.unfit),
        "stops": len(sol.stations),
        "plan": sol.label,
        "cost": round(float(sol.cost), 3),
        "cycle_s": None if cycle is None else round(float(cycle), 1),
    }


def run_part(job: dict, emit=None, cancelled=lambda: False) -> dict:
    """``job``: name, project_dir, pkg_dir, out_dir, frame_search (bool), check_workers (int),
    port (first viser port to try), quick (bool: the quick station plan). Returns the summary row;
    ``status`` is "done" or "failed" with ``reason`` (a failure never raises)."""
    os.environ.setdefault("RSO_TEXTURE_PX", "64")  # nobody looks at the robot here
    name = job["name"]
    out = Path(job["out_dir"]) / pp.safe(name)
    t0 = time.monotonic()
    row = {"part": name, "status": "failed", "reason": "", "folder": str(out), "report": ""}
    server = app = None
    try:
        import viser

        from app import build_app
        from ui import jobs
        from ui.staleness import STEP_COMPONENTS, confirm_step

        jobs.USE_WORKER = False  # this process is already the worker
        jobs.CHECK_WORKERS = int(job.get("check_workers", 1))
        _stage(emit, name, "loading")
        prepare(job["project_dir"], job["pkg_dir"], out)
        server = viser.ViserServer(host="127.0.0.1", port=int(job.get("port", 9200)), verbose=False)
        app = build_app(server)
        if not app["project"]._load(None, out):
            raise ValueError("the part's project could not be opened (see the log)")
        s = app["session"]
        if not s.targets.points:
            raise ValueError("the part package has no targets")
        for step in STEP_COMPONENTS:  # running the batch confirms the setup it plans with
            confirm_step(s, step)
        carrier = s.base.carrier if s.base.mode == "movable" else None
        t_plan = time.time()
        if cancelled():
            raise InterruptedError("cancelled")
        if job.get("frame_search") and carrier in ("rail", "gantry"):
            _stage(emit, name, "frame search")
            frame = app["frame"]
            frame.start(block=True)
            if not frame.rows or not frame.use_frame(index=0):
                raise ValueError("the axis frame search found no frame")
        else:
            _stage(emit, name, "planning")
            app["stations"].start(block=True, quick=bool(job.get("quick")))
        if cancelled():
            raise InterruptedError("cancelled")
        results = app["results"]
        results.refresh()
        sol = results.selected()
        if sol is None:
            raise ValueError("the planner found no plan" + _last_problem(t_plan))
        _stage(emit, name, "paths")
        sim = app["sim"]
        sim.build(block=True)
        cycle = (getattr(sim, "cycle", None) or {}).get("total") if sim.plan is not None else None
        _stage(emit, name, "saving")
        app["project"].save_now(out)
        _stage(emit, name, "report")
        html, _ = results.export.export_report(sol)
        row["report"] = str(html)
        from core.report import html_to_pdf

        pdf = out / "report.pdf"
        if html_to_pdf(html, pdf):
            row["report"] = str(pdf)
        else:
            row["reason"] = "no Edge / Chrome to print the PDF: the HTML report was written"
        row.update(row_for(name, s, sol, cycle if cycle is not None else sol.cycle_s), status="done")
    except InterruptedError:
        row.update(status="cancelled", reason="cancelled")
    except Exception as err:  # a failed part is reported, the batch goes on
        row["reason"] = f"{type(err).__name__}: {err}"
        row["trace"] = traceback.format_exc(limit=6)
    finally:
        if app is not None:
            app["builder_window"].stop()
        if server is not None:
            server.stop()
    row["run_s"] = round(time.monotonic() - t0, 1)
    peak = peak_memory()
    if peak:
        row["peak_mb"] = round(peak / 2**20)
    if emit is not None:
        emit(name, row["status"], 1.0)
    return row
