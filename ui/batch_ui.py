"""Optimize -> *Several parts (batch)*: v13's batch runs and our batch queue in one panel.

Jobs are queued into ``<workspace>/batch_<time>/batch.json`` (``core.batch``; the queue half is
``ui.batch_queue_ui.QueueEdits``): the current part with its face selection, or a folder of part
packages and / or part files. *Run batch* copies the open project to ``_setup/`` (no need to save
it first; a batch made in another project keeps the cell it was made with) and runs the pending
jobs (``ui.batch_job``): each becomes a part package, v13's runner plans them in fresh processes,
several at once when the PC has room (stations, axis frame search, paths, a saved project and a
PDF report per job). The table follows each job; when it ends, Results -> *Batch results* opens
``<batch>/results`` (``on_done``). *Open review page* writes ``review.html`` (pictures per job).
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import viser

from core import batch as bt
from core import data_paths, schema
from ui import batch_prefs
from ui.batch_queue_ui import MODES, QueueEdits, job_table
from ui.log_panel import log
from ui.widgets import notify


def write_setup(session, batch_dir: Path) -> Path:
    """The open project as it is now, as the project folder ``<batch>/_setup`` for the runner."""
    setup = Path(batch_dir) / bt.SETUP
    if setup.exists():
        shutil.rmtree(setup)
    setup.mkdir(parents=True)
    src = Path(session.project_dir) / "assets"
    if src.is_dir():
        shutil.copytree(src, setup / "assets", ignore=shutil.ignore_patterns("_builder"))
    project = session.snapshot().model_copy(update={"results": [], "compared": []})
    schema.save_project(project, setup / "project.json")
    return setup


class BatchPanel(QueueEdits):
    def __init__(self, server: viser.ViserServer, session, project, targets, stations, on_done=None):
        self.s, self.project, self.targets, self.stations = session, project, targets, stations
        self.on_done = on_done
        self.batch_dir: Path | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._review: threading.Thread | None = None
        gui = server.gui
        self.folder = gui.add_folder("Several parts (batch)", expand_by_default=True)
        with self.folder:
            gui.add_markdown(
                "_Queue parts, then **Run batch**: each is planned with this project's robot, end "
                "effector, external axis, stay-outs and settings (stations, paths, a saved project and a "
                "PDF report per part; Results -> Batch results)._"
            )
            self.existing = gui.add_dropdown("Batch", ("—",), hint="batches in the workspace (batch_*)")
            self.open_batch_btn = gui.add_button("Open batch")
            self.new_batch_btn = gui.add_button("New batch", hint="the next job starts a new batch")
            self.batch_md = gui.add_markdown("_No batch yet: it is created with the first job._")
            with gui.add_folder("Add jobs", expand_by_default=True):
                self.mode = gui.add_dropdown("Mode", tuple(MODES), hint="the planner used for new jobs")
                self.add = gui.add_button(
                    "Add current part", color="green", hint="the part with its face selection (Targets tab)"
                )
                self.parts_folder = gui.add_text(
                    "Parts folder",
                    batch_prefs.get("parts_folder"),
                    hint="part packages (package builder) and / or part files (STL, OBJ, PLY, STEP)",
                )
                self.min_conf = gui.add_number(
                    "Min confidence",
                    0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    hint="part files: below this the detected tool surface waits for review",
                )
                self.folder_btn = gui.add_button("Add folder")
            with gui.add_folder("Queue", expand_by_default=True):
                self.table = gui.add_markdown("")
                self.pick = gui.add_dropdown("Job", ("—",))
                self.remove_btn = gui.add_button("Remove")
                self.up_btn = gui.add_button("Move up")
                self.down_btn = gui.add_button("Move down")
                self.reset_btn = gui.add_button("Reset to pending", hint="run a failed or reviewed job again")
                self.review_btn = gui.add_button(
                    "Open for review", hint="load the part of a needs_review job and run Detect tool surface"
                )
                self.review_page_btn = gui.add_button("Open review page", hint="write review.html")
            self.frame_search = gui.add_checkbox(
                "Search the axis frame per part",
                False,
                hint="Rail / gantry: place the frame for each part first (Gantry placement); off = the "
                "project's frame",
            )
            self.workers_md = gui.add_markdown("")
            self.reason_md = gui.add_markdown("")
            self.run_btn = gui.add_button("Run batch", color="green")
            self.stop = gui.add_button("Cancel", color="red", visible=False)
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.status = gui.add_markdown("")
        g = self._guard
        self.add.on_click(g(self.add_to_queue))
        self.open_batch_btn.on_click(g(lambda e: self.open_batch(self.existing.value, e)))
        self.new_batch_btn.on_click(g(lambda e: self.new_batch()))
        self.folder_btn.on_click(g(lambda e: self.add_folder(self.parts_folder.value, e)))
        self.parts_folder.on_update(
            lambda e: batch_prefs.put("parts_folder", self.parts_folder.value.strip())
        )
        self.remove_btn.on_click(g(lambda e: self.remove(e)))
        self.up_btn.on_click(g(lambda e: self.move(-1, e)))
        self.down_btn.on_click(g(lambda e: self.move(1, e)))
        self.reset_btn.on_click(g(lambda e: self.reset(e)))
        self.review_btn.on_click(g(lambda e: self.open_for_review(e)))
        self.review_page_btn.on_click(g(lambda e: self.open_review(e)))
        self.run_btn.on_click(g(lambda e: self.run(e)))
        self.stop.on_click(lambda e: self._cancel.set())
        self._lock_stations()
        self.list_batches()

    # --- helpers ------------------------------------------------------------------------------

    @staticmethod
    def _guard(fn):
        def inner(event=None):
            try:
                fn(event)
            except Exception as err:  # never die silently in a callback
                notify(event, "Batch", f"{type(err).__name__}: {err}")

        return inner

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _lock_stations(self) -> None:
        """While a batch runs, the Station planner says so instead of queuing behind it."""
        self.stations.blocked_reason = lambda: (
            "A batch is running: wait for it or cancel it first" if self.running else None
        )

    def list_batches(self) -> list[str]:
        ws = data_paths.workspace_dir()
        names = sorted((p.name for p in ws.glob("batch_*") if (p / bt.MANIFEST_NAME).is_file()), reverse=True)
        self.existing.options = tuple(names) or ("—",)
        return names

    def why_not(self) -> str:
        s = self.s
        if s.robot is None:
            return "Choose a robot first (tab 1)."
        if s.tool_model is None:
            return "Set up the end effector first (tab 3)."
        if self.batch_dir is None:
            return "Add jobs or open a batch first."
        try:
            m = self.manifest()
        except ValueError as err:
            return str(err)
        if not bt.pending_jobs(m):
            return "No pending jobs: add some or reset a failed one."
        return ""

    def refresh(self) -> None:
        from ui.batch_run import plan_workers

        reason = self.why_not()
        self.reason_md.content = f"_{reason}_" if reason else ""
        busy = self.running and self._thread is not threading.current_thread()  # not its own last refresh
        self.run_btn.disabled = bool(reason) or busy
        if self.batch_dir is None:
            self.table.content = ""
            return
        try:
            m = self.manifest()
        except ValueError as err:
            self.table.content = f"⚠ {err}"
            return
        self.batch_md.content = f"Batch `{self.batch_dir.name}` ({self.batch_dir})"
        self.table.content = job_table(m)
        n = sum(j.status in ("pending", "running") for j in m.jobs)
        self.workers_md.content = f"_{n} pending: {plan_workers(n)[1]}._" if n else ""
        keep = self.pick.value.split()[0]
        opts = tuple(f"{j.id}  {j.name}" if j.name != j.id else j.id for j in m.jobs)
        self.pick.options = opts or ("—",)
        self.pick.value = next((o for o in opts if o.split()[0] == keep), opts[0]) if opts else "—"

    # --- batch folder -------------------------------------------------------------------------

    def open_batch(self, name: str, event=None) -> bool:
        d = data_paths.workspace_dir() / name
        if name == "—" or not (d / bt.MANIFEST_NAME).is_file():
            notify(event, "Open batch", "Pick a batch folder from the list (workspace/batch_*)")
            return False
        self.frame_search.value = bt.load_manifest(d / bt.MANIFEST_NAME).frame_search  # a bad file raises
        self.batch_dir = d
        self.status.content = ""
        self.refresh()
        return True

    def new_batch(self) -> None:
        if self.running:
            raise ValueError("a batch is running: wait for it or cancel it first")
        self.batch_dir = None
        self.batch_md.content = "_New batch: it is created with the next job._"
        self.status.content = ""
        self.refresh()

    def _setup(self, m: bt.BatchManifest) -> str:
        """Write ``_setup`` from the open project when the batch belongs to it (or has none yet);
        returns a note when the batch keeps the cell it was made with."""
        own = bt.cell_path(m, self.batch_dir) == Path(self.s.project_dir).resolve()
        if own or not (self.batch_dir / bt.SETUP / "project.json").is_file():
            write_setup(self.s, self.batch_dir)
            return ""
        return f"planned with the cell this batch was made with ({bt.SETUP}/), not the open project"

    # --- run ----------------------------------------------------------------------------------

    def run(self, event=None, block: bool = False) -> None:
        from ui import jobs
        from ui.batch_job import run_batch_thread

        if self.running:
            return notify(event, "Busy", "The batch is already running")
        if self.stations._thread is not None and self.stations._thread.is_alive():
            return notify(event, "Busy", "The Station planner is running: wait for it first")
        if reason := self.why_not():
            return notify(event, "Cannot run the batch", reason)
        m = self.manifest()
        if m.frame_search != bool(self.frame_search.value):
            m.frame_search = bool(self.frame_search.value)
            self._save(m)
        try:
            note = self._setup(m)
        except (OSError, ValueError) as err:
            return notify(event, "Cannot write the batch's cell", str(err))
        self._cancel.clear()
        self.run_btn.visible, self.stop.visible, self.progress.visible = False, True, True
        self.progress.value = 0.0
        self.status.content = f"_{note}_" if note else ""
        t0 = time.monotonic()
        warns: list[str] = [note] if note else []
        log(f"Batch started: `{self.batch_dir}`.", "info")

        def on_msg(kind: str, data) -> None:
            try:
                if kind == "progress":
                    self.status.content = f"_{data[0]}_"
                    self.progress.value = 100.0 * data[1]
                elif kind == "warn":
                    warns.append(str(data))
                    log(f"Batch: {data}.", "warn")
                elif kind == "job":
                    self.refresh()
                    if data["status"] == "failed":
                        log(f"Batch: **{data['name'] or data['id']}** failed: {data['error']}", "error")
                elif kind == "finished":
                    self._finished(data, warns, t0)
            except Exception as err:  # a display problem must not kill the runner thread
                log(f"Batch panel: {type(err).__name__}: {err}", "error")

        self._thread = run_batch_thread(self.batch_dir, on_msg, self._cancel.is_set, jobs.CHECK_WORKERS)
        if block:
            self._thread.join()

    def _finished(self, data: dict, warns: list[str], t0) -> None:
        try:
            rows = data.get("jobs", [])
            count = {k: sum(r["status"] == k for r in rows) for k in ("done", "failed", "needs_review")}
            ok = not data.get("error")
            head = f"**Batch {data['status']}**" if ok else f"⚠ **Batch failed**: {data['error']}"
            text = f"{head}: " + ", ".join(f"{n} {k}" for k, n in count.items() if n) + "."
            if data.get("status") == "cancelled":
                text += " Unfinished jobs are pending again."
            text += f"\n\nFolder: `{self.batch_dir}`"
            if warns:
                text += "\n\n" + "\n\n".join(f"⚠ {w}" for w in warns[:5])
            self.status.content = text
            log(f"Batch finished ({data['status']}).", "done" if data["status"] == "done" else "warn", t0)
            results = self.batch_dir / bt.RESULTS
            if self.on_done is not None and (results / "summary.json").is_file():
                self.on_done(results)
        finally:
            self.run_btn.visible, self.stop.visible, self.progress.visible = True, False, False
            self.refresh()

    def open_review(self, event=None, block: bool = False) -> None:
        """Write ``<batch>/review.html`` (rendering missing pictures first, in a thread); the status
        line shows the path."""
        from ui.batch_materialize import build_review

        if self.batch_dir is None or not (self.batch_dir / bt.MANIFEST_NAME).is_file():
            return notify(event, "Review page", "No batch yet: add a job first")
        if self._review is not None and self._review.is_alive():
            return notify(event, "Review page", "Already being written")

        def work() -> None:
            try:
                p = build_review(self.batch_dir, lambda t: setattr(self.status, "content", f"_{t}_"))
                self.status.content, ok = f"Review page: `{p}` (open this file in a browser)", "done"
            except Exception as err:  # say what happened
                self.status.content, ok = f"⚠ Review page failed: {type(err).__name__}: {err}", "error"
            log(self.status.content.replace("`", ""), ok)

        self._review = threading.Thread(target=work, daemon=True, name="batch-review")
        self._review.start()
        if block:
            self._review.join()
