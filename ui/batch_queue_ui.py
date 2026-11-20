"""The queue half of Optimize -> *Several parts (batch)* (``ui.batch_ui.BatchPanel``): add jobs,
edit the queue. The manifest on disk is the truth: every edit reloads it first, and the runner
updates single jobs (``bt.update_job``).

Jobs come from three places: *Add current part* (the part, its face selection and the
point-generator settings), and *Add folder*, which queues every part package in the folder
(v13, made in the package builder's Part tab: part, pose and targets) and every part file
(STL / OBJ / PLY / STEP) as an *auto: tool surface* job (the surface is detected when the batch
runs; below *Min confidence* the job waits for a person: needs_review).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from core import batch as bt
from core import data_paths
from core.batch import job_table  # noqa: F401 (the queue table; tests import it here)
from ui.widgets import notify

MODES = {"Plan stations": "cost", "Quick plan": "quick"}  # label -> BatchJob.planner["mode"]
REDO = ("failed", "partial", "cancelled", "needs_review")  # statuses that can be reset to pending


def _copy_part(src: Path, parts: Path) -> Path:
    """``src`` copied into the batch's ``parts/`` under a unique name (identical copies reused)."""
    parts.mkdir(exist_ok=True)
    dest, k = parts / src.name, 1
    while dest.exists() and bt.mesh_sha1(dest) != bt.mesh_sha1(src):
        dest, k = parts / f"{src.stem}_{k}{src.suffix}", k + 1
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest


class QueueEdits:
    """Mixed into BatchPanel: needs ``s``, ``project``, ``targets``, ``batch_dir``, ``running`` and
    the queue widgets (``mode``, ``pick``, ``min_conf``, ``status``, ``refresh``, ``list_batches``)."""

    def manifest(self) -> bt.BatchManifest:
        if self.batch_dir is None:
            raise ValueError("no batch is open: add a job or open a batch")
        return bt.load_manifest(self.batch_dir / bt.MANIFEST_NAME)

    def _save(self, m: bt.BatchManifest) -> None:
        bt.save_manifest(m, self.batch_dir / bt.MANIFEST_NAME)

    def _job_id(self) -> str:
        v = self.pick.value
        if v == "—":
            raise ValueError("the queue has no job selected")
        return v.split()[0]

    def _planner(self) -> dict:
        return {"mode": MODES[self.mode.value]}

    def _new_batch(self) -> None:
        """Create the batch folder and its manifest on the first job."""
        if self.batch_dir is None:
            self.batch_dir = data_paths.workspace_dir() / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            self._save(bt.BatchManifest(cell_project=bt.rel_cell(self.s.project_dir, self.batch_dir)))

    def _ensure_batch(self) -> None:
        from ui.workflow import setup_reason

        if reason := setup_reason(self.s):
            raise ValueError(reason)
        self._new_batch()

    # --- add ----------------------------------------------------------------------------------

    def add_reason(self) -> str:
        from ui.workflow import setup_reason

        s = self.s
        if s.part is None:
            return "Load a part first"
        if len(self.targets.selection) == 0:
            return "Select some faces of the part first (Targets tab)"
        return setup_reason(s)

    def add_to_queue(self, event=None) -> bool:
        """Queue the current part with its face selection and point-generator settings."""
        from core.target_gen import generate_targets

        if self.running:
            notify(event, "Batch running", "Wait for the batch to finish before adding jobs")
            return False
        if reason := self.add_reason():
            notify(event, "Cannot queue", reason)
            return False
        s, sel, gen = self.s, [int(f) for f in self.targets.selection], self.targets.gen_settings()
        mesh = self.targets.mesh()
        if mesh is None:
            notify(event, "Cannot queue", "The part mesh could not be read")
            return False
        P, _N, _g = generate_targets(mesh, sel, gen)
        if len(P) == 0:
            notify(
                event, "No targets", "These settings generate 0 targets on the selection; change them first"
            )
            return False
        self._new_batch()
        m = self.manifest()
        dest = _copy_part(s.resolve(s.part.mesh_path), self.batch_dir / "parts")
        job = bt.add_job(
            m,
            part=dest.relative_to(self.batch_dir).as_posix(),
            selection=sel,
            gen=gen,
            units=s.part.units,
            pose=[[float(x) for x in row] for row in s.part.T_world_part],
            mesh_sha1=bt.mesh_sha1(dest),
            planner=self._planner(),
            summary={"targets_preview": len(P)},
        )
        self._save(m)
        self.list_batches()
        self.refresh()
        notify(event, "Queued", f"{job.id}: {len(sel)} faces, {len(P)} targets", ok=True)
        return True

    def _add_package(self, m, d: Path) -> str:
        """Queue the part package folder ``d`` (copied into the batch); "" or why not."""
        from core import part_package as pp
        from core.transforms import pose_to_matrix

        try:
            pkg = pp.load_part_package(d)
        except ValueError as err:
            return str(err)
        if not pkg.targets.points:
            return "the package has no targets"
        job = bt.add_job(m, part="-", selection=[], name=pkg.name, source="package", planner=self._planner())
        rel = f"{bt.PACKAGES}/{pp.safe(job.id)}"
        dest = self.batch_dir / rel
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(d, dest)
        mesh = dest / pkg.part.mesh
        job.package, job.part, job.units = rel, mesh.relative_to(self.batch_dir).as_posix(), pkg.part.units
        job.pose, job.mesh_sha1 = pose_to_matrix(pkg.part.pose).tolist(), bt.mesh_sha1(mesh)
        job.summary = {"targets_preview": len(pkg.targets.points)}
        return ""

    def add_folder(self, folder: str, event=None) -> tuple[list[str], list[str]]:
        """Queue every part package (a folder with part.json) and every part file of ``folder``
        (an auto: tool surface job with the current generator settings and part pose). Returns
        (queued job ids, skipped "name: why")."""
        if self.running:
            notify(event, "Batch running", "Wait for the batch to finish before adding jobs")
            return [], []
        d = Path(folder.strip().strip('"'))
        if not folder.strip() or not d.is_dir():
            notify(event, "Add folder", f"'{folder}' is not a folder on this computer")
            return [], []
        try:
            self._ensure_batch()
        except ValueError as err:
            notify(event, "Cannot queue", str(err))
            return [], []
        s, gen = self.s, self.targets.gen_settings()
        units = s.part.units if s.part else "m"
        pose = (s.part.T_world_part if s.part else np.eye(4)).tolist()
        m = self.manifest()
        queued, skipped = [], []
        for src in sorted(d.iterdir(), key=lambda p: p.name.lower()):
            if src.is_dir():
                if (src / "part.json").is_file():
                    n = len(m.jobs)
                    why = self._add_package(m, src)
                    if why:
                        del m.jobs[n:]
                        skipped.append(f"{src.name}: {why}")
                    else:
                        queued.append(m.jobs[-1].id)
                continue  # other folders are not parts
            if src.suffix.lower() not in bt.PART_EXTS:
                skipped.append(f"{src.name}: unsupported file type {src.suffix or '(none)'}")
                continue
            dest = _copy_part(src, self.batch_dir / "parts")
            job = bt.add_job(
                m,
                part=dest.relative_to(self.batch_dir).as_posix(),
                selection=[],
                gen=gen,
                name=src.stem,
                units=units,
                pose=pose,
                mesh_sha1=bt.mesh_sha1(dest),
                planner=self._planner(),
                selection_rule="tool_surface",
                min_confidence=float(self.min_conf.value),
            )
            reason = bt.check_job(job, self.batch_dir)
            if reason:
                m.jobs.remove(job)
                skipped.append(f"{src.name}: {reason.split(': ', 1)[-1]}")
            else:
                queued.append(job.id)
        self._save(m)
        self.list_batches()
        self.refresh()
        text = f"{len(queued)} queued"
        if skipped:
            text += f", {len(skipped)} skipped: " + "; ".join(skipped[:5])
        self.status.content = text
        notify(event, "Add folder", text, ok=not skipped or bool(queued))
        return queued, skipped

    def open_for_review(self, event=None) -> bool:
        """Load the part of the picked needs_review job into the Targets tab and run Detect."""
        from ui.session import PartState as _PartState

        m, jid = self.manifest(), self._job_id()
        job = next((j for j in m.jobs if j.id == jid), None)
        if job is None or job.status != "needs_review":
            notify(event, "Open for review", "Pick a job whose status is needs_review")
            return False
        src = self.batch_dir / job.part
        dest = self.s.project_dir / "assets" / "parts" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        self.s.part = _PartState(self.s.relative(dest), job.units, np.array(job.pose, float))
        refresh = getattr(self.targets.tools, "on_mesh_replaced", None)
        if refresh:
            refresh()
        self.targets.on_part_changed()
        self.status.content = (
            f"_{job.id} loaded for review: accept or adjust the selection in the Targets tab, "
            "then **Add current part** (a faces job) and remove this one._"
        )
        return self.targets.detect.detect(event)

    # --- edit ---------------------------------------------------------------------------------

    def _editable(self, m: bt.BatchManifest, jid: str, event) -> bool:
        job = next((j for j in m.jobs if j.id == jid), None)
        if job is None:
            notify(event, "Batch", f"{jid} is no longer in the queue")
            return False
        if self.running and job.status != "pending":
            notify(
                event, "Batch running", f"{jid} is {job.status}; only pending jobs can change while it runs"
            )
            return False
        return True

    def remove(self, event=None) -> bool:
        m, jid = self.manifest(), self._job_id()
        if not self._editable(m, jid, event):
            return False
        m.jobs = [j for j in m.jobs if j.id != jid]
        self._save(m)
        self.refresh()
        return True

    def move(self, delta: int, event=None) -> bool:
        m, jid = self.manifest(), self._job_id()
        i = next((n for n, j in enumerate(m.jobs) if j.id == jid), -1)
        k = i + delta
        if i < 0 or not 0 <= k < len(m.jobs):
            return False
        if not (self._editable(m, jid, event) and self._editable(m, m.jobs[k].id, event)):
            return False
        m.jobs[i], m.jobs[k] = m.jobs[k], m.jobs[i]
        self._save(m)
        self.refresh()
        return True

    def reset(self, event=None) -> bool:
        m, jid = self.manifest(), self._job_id()
        job = next((j for j in m.jobs if j.id == jid), None)
        if job is None or job.status not in REDO:
            notify(
                event, "Reset", "Only failed, partial, cancelled or needs_review jobs can be reset to pending"
            )
            return False
        job.status, job.error = "pending", ""
        job.summary = {k: v for k, v in job.summary.items() if k == "targets_preview"}
        bt.update_job(self.batch_dir / bt.MANIFEST_NAME, job)
        self.refresh()
        return True
