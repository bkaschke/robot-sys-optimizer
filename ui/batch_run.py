"""Batch runs (v13 #5): the station optimizer on many part packages with one project's cell.

Every part runs in its own fresh worker process (``ui.batch_part.run_part``; spawn, one part
per process, so a part's memory is freed when it ends and no module state leaks into the GUI).
Several parts run at once when the PC has room for them (``plan_workers``): then each part
checks serially, else the one part uses the app's check workers (Settings). Progress messages
come back through a queue; Cancel stops the parts not started, asks the running ones to stop
at their next stage and ends them after a grace period. The summary (``ui.batch_summary``) is
written after every part.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from core import schema
from ui import batch_summary

MEM_PER_PART = 1.5 * 2**30  # bytes a part's process needs at most (v13 #5 Status: measured)
CANCEL_GRACE_S = 5.0
PORT0 = 9300  # first viser port of the workers' (unopened) apps


def free_memory() -> int | None:
    """Bytes of physical memory free now (None when unknown)."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
        return None
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def plan_workers(n_parts: int, cores: int | None = None, free: int | None = None) -> tuple[int, str]:
    """How many parts run at once, and why: half the logical cores (a part's planner, collision
    checks and app threads keep more than one busy) and 80 % of the free memory; one at a time
    when that leaves fewer than two."""
    cores = cores if cores is not None else os.cpu_count() or 1
    free = free if free is not None else free_memory()
    by_cores = max(1, cores // 2)
    by_mem = int(0.8 * free // MEM_PER_PART) if free else 1
    w = min(n_parts, by_cores, by_mem)
    mem = f"{free / 2**30:.0f} GB free" if free else "free memory unknown"
    if n_parts <= 1:
        return 1, "one part"
    if w < 2:
        return 1, f"one part at a time: {cores} cores, {mem}"
    return w, f"{w} parts at once: {cores} cores, {mem}"


def _child(job: dict) -> dict:
    """In the worker process: one part, with its messages put on the shared queue."""
    from ui.batch_part import run_part

    q, stop = job.pop("queue"), job.pop("stop")

    def emit(name, stage, frac) -> None:
        try:
            q.put((name, stage, frac))
        except Exception:
            pass

    return run_part(job, emit, stop.is_set)


def meta_for(project_dir: Path, parts_folder: str, workers: int, why: str) -> dict:
    meta = {"project": str(project_dir), "parts_folder": str(parts_folder), "workers": why}
    try:
        p = schema.load_project(Path(project_dir) / "project.json")
        meta["robot"] = (p.robot.name or p.robot.urdf or "-") if p.robot else "-"
        meta["tool"] = p.end_effector.name if p.end_effector else "-"
        meta["axis"] = p.base.carrier if p.base.mode == "movable" else "static"
    except (OSError, ValueError):
        pass
    meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return meta


def run_batch(
    project_dir: Path,
    parts: list[tuple],
    out_dir: Path,
    frame_search: bool = False,
    workers: int | None = None,
    check_workers: int = 1,
    on_progress=None,
    on_row=None,
    cancelled=lambda: False,
    parts_folder: str = "",
) -> list[dict]:
    """Plan every ``(name, package folder)`` of ``parts`` with the project at ``project_dir``; the
    results go to ``out_dir``. An entry may carry a third item, a dict merged into that part's job
    (e.g. ``{"quick": True}``: the quick station plan). ``on_progress(name, stage, fraction)`` is
    called from a helper thread, ``on_row(row)`` from this one as each part ends. Returns the
    summary rows (in the order of ``parts``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [(p[0], Path(p[1]), dict(p[2]) if len(p) > 2 else {}) for p in parts]
    why = ""
    if workers is None:
        workers, why = plan_workers(len(parts))
    workers = max(1, min(int(workers), len(parts) or 1))
    why = why or f"{workers} part(s) at once (set)"
    meta = meta_for(project_dir, parts_folder, workers, why)
    rows: dict[str, dict] = {}

    def summary(pdf: bool = False) -> None:
        batch_summary.write(out_dir, meta, [rows[n] for n, *_ in parts if n in rows], pdf=pdf)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q, stop = manager.Queue(), manager.Event()
    done = threading.Event()

    def pump() -> None:
        while not done.is_set():
            try:
                msg = q.get(timeout=0.2)
            except (queue.Empty, EOFError, OSError):
                continue
            if on_progress is not None:
                on_progress(*msg)

    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()
    jobs = [
        {
            "name": name,
            "project_dir": str(project_dir),
            "pkg_dir": str(d),
            "out_dir": str(out_dir),
            "frame_search": frame_search,
            "check_workers": 1 if workers > 1 else max(1, int(check_workers)),
            "port": PORT0 + 7 * i,
            "queue": q,
            "stop": stop,
            **extra,
        }
        for i, (name, d, extra) in enumerate(parts)
    ]
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1)
    try:
        futures = {ex.submit(_child, job): job["name"] for job in jobs}
        pending = set(futures)
        cancel_at = None
        while pending:
            finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in finished:
                name = futures[f]
                try:
                    rows[name] = f.result()
                except Exception as err:  # the worker died (killed, out of memory, ...)
                    rows[name] = {"part": name, "status": "failed", "reason": f"worker ended: {err}"}
                summary()
                if on_row is not None:
                    on_row(rows[name])
            if cancelled() and cancel_at is None:
                cancel_at = time.monotonic()
                stop.set()
                for f in pending:
                    f.cancel()
            if cancel_at is not None and time.monotonic() - cancel_at > CANCEL_GRACE_S:
                for p in list(getattr(ex, "_processes", {}).values()):
                    p.terminate()
                break
        for name in futures.values():
            if name not in rows:
                rows[name] = {"part": name, "status": "cancelled", "reason": "cancelled"}
    finally:
        ex.shutdown(wait=cancelled() is False, cancel_futures=True)
        done.set()
        pumper.join(timeout=2)
        manager.shutdown()
    summary(pdf=True)
    return [rows[n] for n, *_ in parts]
