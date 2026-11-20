"""Batch queue runner: plan every pending job of a batch folder with v13's batch runner.

``drive`` first turns each pending job into a part package (``ui.batch_materialize.make_package``:
targets generated on the job's selection or on the detected tool surface, the review pictures,
the setup warnings), then hands the packages to ``ui.batch_run.run_batch``: one fresh process per
job that plans in a browserless app exactly as the GUI does (stations, optional axis frame search,
paths, cycle time, saved project, report.pdf), several jobs at once when the PC has room.

- **Statuses** are written to ``batch.json`` one job at a time (``core.batch.update_job``):
  running -> done / failed (one-line ``error``, traceback in ``<job id>/error.txt``) /
  needs_review (the detector is not sure, nothing planned).
- **Cancel:** jobs not finished go back to ``pending`` and run again on the next run.
- **Resume:** only pending jobs run (an interrupted ``running`` job is pending again).
- Output: ``<batch>/results/`` (v13: a project folder per job, ``summary.json`` for Results ->
  Batch results), ``summary.csv`` / ``summary.html`` / ``review.html`` in the batch folder.

Headless: ``python -m ui.batch_job <batch_dir>``. This module does not import viser.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from pathlib import Path

from core import batch as bt
from ui.batch_run import plan_workers, run_batch

TERMINAL = ("done", "failed", "cancelled")
REVIEW_KEYS = ("review", "review_note")  # job.summary keys the pictures own


def one_line(err: BaseException) -> str:
    summary = getattr(err, "summary", None)
    if summary:
        return str(summary).splitlines()[0]
    lines = str(err).strip().splitlines()
    msg = lines[0] if lines else ""
    if isinstance(err, ValueError) and msg:
        return msg  # our own checks: already a sentence
    return f"{type(err).__name__}: {msg}" if msg else type(err).__name__


def _write_error(job_dir: Path, text: str) -> None:
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "error.txt").write_text(text, encoding="utf-8")
    except OSError:
        pass  # the one-line error in the manifest still says what happened


def _pictures(job, batch_dir: Path) -> None:
    """Render the review pictures (never raises)."""
    from ui.batch_materialize import job_pictures

    try:
        job_pictures(job, batch_dir)
    except Exception:  # noqa: BLE001 - job_pictures already catches; belt and braces
        pass


def _rel(p: str, batch_dir: Path) -> str:
    try:
        return Path(p).resolve().relative_to(Path(batch_dir).resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


def run_name(job: bt.BatchJob, taken: set[str]) -> str:
    """The job's name in the runner (its results folder): the job's name, made unique."""
    from core.part_package import safe

    name = safe(job.name or job.id)
    if name in taken:
        name = f"{name}_{job.id}"
    taken.add(name)
    return name


def apply_row(job: bt.BatchJob, row: dict, batch_dir: Path) -> None:
    """Put a runner summary row (``ui.batch_part.run_part``) on the job."""
    keep = {k: v for k, v in job.summary.items() if k in REVIEW_KEYS}
    job.run_s = float(row.get("run_s") or 0.0)
    st = row.get("status")
    if st == "done":
        job.status, job.error = "done", str(row.get("reason") or "")
        job.out_dir = _rel(row.get("folder", ""), batch_dir) if row.get("folder") else ""
        job.summary = {
            "label": row.get("plan", ""),
            "stations": row.get("stops"),
            "reached": row.get("reached"),
            "total": row.get("points"),
            "cost": row.get("cost"),
            "cycle_s": row.get("cycle_s"),
            "report": _rel(row["report"], batch_dir) if row.get("report") else "",
            **({"peak_mb": row["peak_mb"]} if row.get("peak_mb") else {}),
            **keep,
        }
    elif st == "cancelled":
        job.status, job.error, job.summary = "pending", "cancelled: runs again on the next Run batch", keep
    else:
        job.status, job.summary = "failed", keep
        job.error = str(row.get("reason") or "failed").splitlines()[0]
        _write_error(Path(batch_dir) / job.id, f"{job.error}\n\n{row.get('trace', '')}")


def job_row(job: bt.BatchJob, index: int = 0, n: int = 0) -> dict:
    """The ``job`` message: what the batch panel lists per finished job."""
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "summary": dict(job.summary),
        "error": job.error,
        "run_s": job.run_s,
        "index": index,
        "n": n,
    }


def prepare(m: bt.BatchManifest, todo: list, bdir: Path, on_msg, cancelled) -> list[tuple]:
    """Make the part package of every job of ``todo``; returns the runner's part entries
    ``(name, package folder, {"quick": ...}, job)``. Jobs that cannot run are marked here."""
    from ui.batch_materialize import make_package

    path = bdir / bt.MANIFEST_NAME
    parts, taken = [], set()
    for i, job in enumerate(todo):
        if cancelled():
            break
        who = job.name or job.id
        on_msg("progress", (f"preparing {who} ({i + 1}/{len(todo)})", 0.0))
        (bdir / job.id / "error.txt").unlink(missing_ok=True)
        job.status, job.error, job.run_s = "running", "", 0.0
        job.summary = {}
        bt.update_job(path, job, fallback=m)
        t0 = time.monotonic()
        try:
            pkg = make_package(m, job, bdir)
        except bt.NeedsReview as err:  # not an error: the detector is not sure, a person must look
            job.status, job.error = "needs_review", one_line(err)
        except Exception as err:  # this job fails, the batch goes on
            job.status, job.error = "failed", one_line(err)
            tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            _write_error(bdir / job.id, f"{job.error}\n\n{tb}")
        else:
            parts.append((run_name(job, taken), pkg, {"quick": job.planner.get("mode") == "quick"}, job))
        _pictures(job, bdir)
        if job.status != "running":
            job.run_s = round(time.monotonic() - t0, 1)
        bt.update_job(path, job, fallback=m)
        if job.status != "running":
            on_msg("job", job_row(job, i, len(todo)))
    return parts


def drive(
    batch_dir, on_msg, cancelled=lambda: False, check_workers: int = 1, workers: int | None = None
) -> dict:
    """Run every pending job (blocking). Messages: ``progress`` (text, 0..1 over the batch),
    ``warn``, ``job`` (job_row). Returns {"status": done | cancelled, "jobs": [job_row ...]}."""
    from ui.batch_materialize import setup_dir

    bdir = Path(batch_dir)
    path = bdir / bt.MANIFEST_NAME
    m = bt.load_manifest(path)
    todo = bt.pending_jobs(m)
    bt.save_manifest(m, path)  # interrupted 'running' jobs are pending again
    setup = setup_dir(m, bdir)
    if todo and not (setup / "project.json").is_file():
        raise ValueError(f"the batch has no cell project ({bt.SETUP}/ or {m.cell_project or 'cell_project'})")
    rows: list[dict] = []
    lock = threading.Lock()

    def finish(job, row) -> None:
        with lock:
            if job.status != "running":
                return  # already applied
            apply_row(job, row, bdir)
            bt.update_job(path, job, fallback=m)
            rows.append(job_row(job, len(rows), len(todo)))
            on_msg("job", rows[-1])

    def prep_msg(kind: str, data) -> None:
        if kind == "job":
            rows.append(data)
        on_msg(kind, data)

    try:
        parts = prepare(m, todo, bdir, prep_msg, cancelled)
        by_name = {p[0]: p[3] for p in parts}
        if parts and not cancelled():
            if workers is None:
                workers, why = plan_workers(len(parts))
            else:
                why = f"{workers} at once (set)"
            on_msg("progress", (f"{len(parts)} job(s) to plan: {why}", 0.0))
            times = [j.run_s for j in m.jobs if j.status == "done" and j.run_s]
            stages: dict[str, tuple[str, float]] = {}
            started: dict[str, float] = {}
            out = bdir / bt.RESULTS

            def progress(name: str, stage: str, frac: float) -> None:
                stages[name] = (stage, frac)
                started.setdefault(name, time.monotonic())
                if stage in TERMINAL:
                    times.append(time.monotonic() - started[name])
                ended = sum(st in TERMINAL for st, _ in stages.values())
                left = len(parts) - ended
                eta = bt.eta_text(times, math.ceil(left / max(1, workers)))
                now = ", ".join(f"{n}: {st}" for n, (st, _) in stages.items() if st not in TERMINAL)
                text = (
                    f"{ended}/{len(parts)} planned"
                    + (f"; {now}" if now else "")
                    + (f" - {eta}" if eta else "")
                )
                on_msg("progress", (text, sum(f for _, f in stages.values()) / len(parts)))

            got = run_batch(
                setup,
                [p[:3] for p in parts],
                out,
                frame_search=m.frame_search,
                workers=workers,
                check_workers=check_workers,
                on_progress=progress,
                on_row=lambda row: finish(by_name[row["part"]], row),
                cancelled=cancelled,
                parts_folder=str(bdir),
            )
            for row in got:
                finish(by_name[row["part"]], row)
        for job in todo:  # cancelled before its package was made
            if job.status == "running":
                finish(job, {"status": "cancelled"})
    finally:
        try:
            bt.write_summary(bt.load_manifest(path), bdir)
        except Exception as err:  # noqa: BLE001
            on_msg("warn", f"batch summary not written ({one_line(err)})")
    return {"status": "cancelled" if cancelled() else "done", "jobs": rows}


def run_batch_thread(batch_dir, on_msg, cancelled=lambda: False, check_workers: int = 1) -> threading.Thread:
    """Start ``drive`` in a daemon thread. It ends with a ``finished`` message:
    {"status": done | cancelled | failed, "jobs": [...], "error": one line}."""

    def work() -> None:
        try:
            out = drive(batch_dir, on_msg, cancelled, check_workers)
            on_msg("finished", {**out, "error": ""})
        except Exception as err:  # e.g. an unreadable batch.json: say so
            on_msg("finished", {"status": "failed", "jobs": [], "error": one_line(err)})

    t = threading.Thread(target=work, daemon=True, name="batch-runner")
    t.start()
    return t


# --- CLI ------------------------------------------------------------------------------------


def _line(row: dict) -> str:
    s = row["summary"]
    what = (
        f"{s['stations']} station(s), {s['reached']}/{s['total']} reached, cost {s['cost']:.2f}"
        if s.get("stations")
        else ""
    )
    tail = f"  {row['error']}" if row["error"] else ""
    return f"{row['id']:<10} {row['status']:<12} {what} ({row['run_s']:.0f} s){tail}"


def _review_only(batch_dir: Path) -> int:
    from ui.batch_materialize import build_review

    try:
        print(f"review page: {build_review(batch_dir, lambda t: print(t, flush=True))}", flush=True)
        return 0
    except Exception as err:  # noqa: BLE001
        print(f"review page failed: {one_line(err)}", flush=True)
        return 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the pending jobs of a batch folder (batch.json).")
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--review-only", action="store_true", help="only (re)build review.html, run no jobs")
    ap.add_argument("--frame-search", action="store_true", help="rail / gantry: search the frame per job")
    a = ap.parse_args(argv)
    if a.review_only:
        return _review_only(a.batch_dir)
    if a.frame_search:
        p = Path(a.batch_dir) / bt.MANIFEST_NAME
        try:
            m = bt.load_manifest(p)
            m.frame_search = True
            bt.save_manifest(m, p)
        except ValueError as err:
            print(f"batch failed: {one_line(err)}", flush=True)
            return 1
    cancel = threading.Event()
    done: dict = {}
    t0 = time.monotonic()

    def on_msg(kind: str, data) -> None:
        if kind == "job":
            print(_line(data), flush=True)
        elif kind == "progress":
            print(f"[{time.monotonic() - t0:7.1f} s] {data[0]}", flush=True)
        elif kind == "warn":
            print(f"warning: {data}", flush=True)
        elif kind == "finished":
            done.update(data)

    t = run_batch_thread(a.batch_dir, on_msg, cancel.is_set)
    while t.is_alive():
        try:
            t.join(0.5)
        except KeyboardInterrupt:
            print("cancelling: unfinished jobs stay pending", flush=True)
            cancel.set()
    if done.get("error"):
        print(f"batch failed: {done['error']}", flush=True)
    rows = done.get("jobs", [])
    bad = [r for r in rows if r["status"] != "done"]
    print(f"batch {done.get('status', 'failed')}: {len(rows) - len(bad)}/{len(rows)} job(s) done", flush=True)
    for name in ("review.html", f"{bt.RESULTS}/summary.pdf"):
        if (Path(a.batch_dir) / name).is_file():
            print(f"{name}: {Path(a.batch_dir) / name}", flush=True)
    return 0 if done.get("status") == "done" and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
