"""Batch queue manifest ("select now, optimize later"): pure data, no viser, no ui.

``batch.json`` lists jobs that start from one cell (the project's robot, tool, external axis,
stay-outs and settings, copied to ``_setup/``): a part mesh with a face selection or the
tool-surface rule (``source="mesh"``), or a part package made in the package builder
(``source="package"``, v13). ``ui.batch_package`` turns each job into a part package and
``ui.batch_job`` plans the packages with v13's runner (``ui.batch_run``). Writes are atomic
(tmp file, then replace).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from core.target_gen import GenSettings

VERSION = 3  # 2: BatchManifest.pdf; 3: job source / package, frame_search (older files load unchanged)
MANIFEST_NAME = "batch.json"
Status = Literal["pending", "running", "done", "failed", "cancelled", "partial", "needs_review"]
Rule = Literal["faces", "tool_surface"]
Source = Literal["mesh", "package"]
SETUP = "_setup"  # the cell project the jobs run with (a copy of the project, ui.batch_run)
RESULTS = "results"  # the runner's output: a project folder per job + summary.json (v13)
PACKAGES = "packages"  # a part package per job (ui.batch_package)
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
from core.step_import import STEP_EXTS, StepImportError, read_step  # noqa: E402

MESH_EXTS = (".stl", ".obj", ".ply")
PART_EXTS = MESH_EXTS + STEP_EXTS  # a part may also be a STEP model


class NeedsReview(ValueError):  # noqa: N818
    """A tool_surface job whose detection is not confident enough: a person must look (not a failure)."""


def _identity() -> list[list[float]]:
    return np.eye(4).tolist()


class BatchJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    source: Source = "mesh"  # "package": a v13 part package (part, pose and targets) in ``package``
    package: str = ""  # package folder relative to the batch folder (source="package")
    part: str  # mesh path relative to the batch folder (copied there at queue time)
    units: str = "m"
    pose: list[list[float]] = Field(default_factory=_identity)  # 4x4 T_world_part
    mesh_sha1: str = ""
    selection: list[int] = Field(default_factory=list)  # face ids
    selection_rule: Rule = "faces"  # "tool_surface": detected at materialize time (core.tool_surface_pick)
    min_confidence: float = 0.7  # below this a tool_surface job is marked needs_review, not planned
    confidence: float | None = None  # detection confidence of the last tool_surface run
    detect_summary: str = ""  # one line from the detector (ASCII)
    warnings: list[str] = Field(default_factory=list)  # detector warnings of the last tool_surface run
    gen: GenSettings = Field(default_factory=GenSettings)
    planner: dict = Field(default_factory=dict)  # PlannerSettings fields + objective/weights/workers
    status: Status = "pending"
    error: str = ""  # one line
    run_s: float = 0.0
    out_dir: str = ""
    summary: dict = Field(default_factory=dict)  # stations, reached, total, cost
    auto_confirmed: list[str] = Field(default_factory=list)  # setup steps confirmed by the batch
    # detected / secondary faces of the last detection (not saved; the review pictures use them)
    _detected: list = PrivateAttr(default_factory=list)
    _secondary: list = PrivateAttr(default_factory=list)


class BatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = VERSION
    created: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    cell_project: str = ""  # cell study folder whose robot, tool, carrier, stay-outs, optimizer are reused
    pdf: bool = False  # unused since v13's runner (it prints report.pdf when Edge / Chrome is found)
    frame_search: bool = False  # rail / gantry: search the axis frame per job first (Gantry placement)
    jobs: list[BatchJob] = Field(default_factory=list)


def save_manifest(manifest: BatchManifest, path: Path) -> Path:
    """Write atomically: a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_manifest(path: Path) -> BatchManifest:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"cannot read the batch file {path}: {err}") from err
    if not isinstance(data, dict) or data.get("version", 0) > VERSION:
        raise ValueError(f"{path.name} was written by a newer version of the app; update the app to open it")
    return BatchManifest.model_validate(data)


def add_job(
    manifest: BatchManifest, *, part: str, selection, gen: GenSettings | None = None, **kw
) -> BatchJob:
    """Append a job with a unique id (``job_001``, ...). ``kw``: any other BatchJob field."""
    taken = {j.id for j in manifest.jobs}
    jid = kw.pop("id", None)
    n = len(manifest.jobs) + 1
    if jid is None:
        while f"job_{n:03d}" in taken:
            n += 1
        jid = f"job_{n:03d}"
    elif jid in taken:
        raise ValueError(f"job id '{jid}' already exists in this batch")
    kw.setdefault("name", jid)
    job = BatchJob(id=jid, part=part, selection=[int(f) for f in selection], gen=gen or GenSettings(), **kw)
    manifest.jobs.append(job)
    return job


def reset_interrupted(manifest: BatchManifest) -> int:
    """Resume: a job still 'running' was interrupted, so it is pending again."""
    n = 0
    for j in manifest.jobs:
        if j.status == "running":
            j.status, n = "pending", n + 1
    return n


def pending_jobs(manifest: BatchManifest) -> list[BatchJob]:
    """Jobs to run: pending, plus interrupted 'running' ones (reset to pending). Done jobs are
    skipped; failed, cancelled and partial jobs are not re-run unless the user resets them."""
    reset_interrupted(manifest)
    return [j for j in manifest.jobs if j.status == "pending"]


def eta_text(times: list[float], remaining: int) -> str:
    """ "about 12 min left" from the run times (s) of finished jobs and the number of jobs still
    to finish (the running one included); "" when there is nothing to go by. The median is
    used, so one very long or very short job does not swing the estimate."""
    times = sorted(t for t in times if t and t > 0)
    if not times or remaining <= 0:
        return ""
    mid = len(times) // 2
    median = times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2
    left = median * remaining
    if left < 90:
        return "about 1 min left"
    if left < 5400:
        return f"about {left / 60:.0f} min left"
    return f"about {left / 3600:.1f} h left"


def mesh_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_job(job: BatchJob, batch_dir: Path) -> str | None:
    """A one-line reason why the job cannot run, or None."""
    import trimesh

    who = job.name or job.id
    p = Path(batch_dir) / job.part
    if not p.is_file():
        return f"{who}: mesh {job.part} is missing from the batch folder"
    if job.mesh_sha1 and mesh_sha1(p) != job.mesh_sha1:
        return f"{who}: mesh {job.part} changed since it was queued, so the selected faces no longer match"
    if job.units not in UNIT_SCALE:
        return f"{who}: unknown units '{job.units}'"
    if job.source == "package":  # part, pose and targets are in the package
        from core.part_package import load_part_package

        try:
            pkg = load_part_package(Path(batch_dir) / job.package)
        except ValueError as err:
            return f"{who}: part package {job.package}: {err}"
        return None if pkg.targets.points else f"{who}: part package {job.package} has no targets"
    if p.suffix.lower() not in PART_EXTS:
        return f"{who}: unsupported mesh type {p.suffix}"
    try:
        if p.suffix.lower() in STEP_EXTS:  # face ids index the tessellated triangles
            nf = len(read_step(p, "m").mesh.faces)
        else:
            nf = len(trimesh.load(p, force="mesh").faces)
    except StepImportError as err:
        return f"{who}: cannot read part {job.part}: {err}"
    except Exception as err:  # noqa: BLE001 - any loader failure is a reason
        return f"{who}: cannot read mesh {job.part}: {err}"
    if nf == 0:
        return f"{who}: mesh {job.part} contains no triangles"
    if job.selection_rule != "faces":  # faces come from the detector at materialize time
        return None
    if not job.selection:
        return f"{who}: no faces selected"
    bad = [f for f in job.selection if f < 0 or f >= nf]
    if bad:
        return f"{who}: {len(bad)} selected face id(s) out of range (mesh has {nf} faces, e.g. {bad[0]})"
    return None


# --- B4: paths, per-job updates and the batch summary ------------------------------------------


def rel_cell(cell: Path | str, batch_dir: Path) -> str:
    """``cell_project`` as stored: relative to the batch folder when possible (both on one
    drive, e.g. ``../study_x``), so moving the workspace keeps the link; else absolute."""
    cell, batch_dir = Path(cell).resolve(), Path(batch_dir).resolve()
    try:
        return Path(os.path.relpath(cell, batch_dir)).as_posix()
    except ValueError:  # another drive (Windows)
        return str(cell)


def cell_path(manifest: BatchManifest, batch_dir: Path) -> Path:
    """The cell study folder: ``cell_project`` resolved against the batch folder."""
    p = Path(manifest.cell_project)
    return p if p.is_absolute() else (Path(batch_dir) / p).resolve()


def update_job(path: Path, job: BatchJob, fallback: BatchManifest | None = None) -> bool:
    """Write one job's entry into the manifest on disk, keeping every other job as the file has
    it (the GUI may queue or reorder jobs while one runs). A job no longer in the file (removed
    meanwhile) is not written back: False. ``fallback`` is saved whole when the file is
    missing or unreadable."""
    path = Path(path)
    try:
        current = load_manifest(path)
    except ValueError:
        if fallback is None:
            raise
        save_manifest(fallback, path)
        return True
    for i, j in enumerate(current.jobs):
        if j.id == job.id:
            current.jobs[i] = job.model_copy(deep=True)
            save_manifest(current, path)
            return True
    return False


def _cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")[:120]


def job_table(m: BatchManifest) -> str:
    if not m.jobs:
        return "_The queue is empty._"
    lines = []
    for j in m.jobs:
        s = j.summary or {}
        n = s.get("total", s.get("targets_preview", ""))
        auto = j.selection_rule == "tool_surface"
        conf = f" ({j.confidence:.2f})" if auto and j.confidence is not None else ""
        faces = f"auto{conf}" if auto and not j.selection else str(len(j.selection))
        if j.source == "package":
            faces = "package"
        got = f"{s.get('reached', '')}/{s.get('total', '')}, {s['cost']:.2f}" if s.get("stations") else ""
        if s.get("cycle_s"):
            got += f", {s['cycle_s']:.0f} s"
        mode = "quick" if j.planner.get("mode") == "quick" else "plan"
        name = f" {_cell(j.name)}" if j.name != j.id else ""
        lines.append(
            f"| {j.id}{name} | {_cell(Path(j.part).name)} | {faces} | {n} | {mode} | "
            f"{j.status} | {got} | {_cell(j.error)} |"
        )
    return (
        "| job | part | faces | targets | mode | status | reached, cost, cycle | error |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(lines)
    )


def set_pdf(path: Path, on: bool) -> None:
    """Switch the per-job PDF report of the batch on disk (the jobs are kept as the file has them)."""
    m = load_manifest(path)
    if m.pdf != bool(on):
        m.pdf = bool(on)
        save_manifest(m, path)


SUMMARY_COLUMNS = (
    "job",
    "name",
    "part",
    "status",
    "confidence",
    "stations",
    "reached",
    "total",
    "cost",
    "run_s",
)
SUMMARY_COLUMNS += ("cycle_s", "error", "report")


def summary_rows(manifest: BatchManifest) -> list[dict]:
    """One row per job (SUMMARY_COLUMNS); empty cells for what a job does not have."""
    rows = []
    for j in manifest.jobs:
        s = j.summary or {}
        cost = s.get("cost")
        rows.append(
            {
                "job": j.id,
                "name": j.name,
                "part": j.part,
                "status": j.status,
                "confidence": "" if j.confidence is None else f"{j.confidence:.2f}",
                "stations": s.get("stations", ""),
                "reached": s.get("reached", ""),
                "total": s.get("total", ""),
                "cost": "" if cost is None else f"{cost:.3f}",
                "run_s": f"{j.run_s:.1f}" if j.run_s else "",
                "cycle_s": "" if s.get("cycle_s") is None else s["cycle_s"],
                "error": j.error,
                "report": s.get("report", ""),
            }
        )
    return rows


def summary_csv(manifest: BatchManifest) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, SUMMARY_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(summary_rows(manifest))
    return buf.getvalue()


def summary_html(manifest: BatchManifest) -> str:
    """A self-contained page (no scripts, no external links except the per-job reports)."""
    import html

    def cell(k: str, v) -> str:
        v = str(v)
        if k == "report" and v:
            return f'<td><a href="{html.escape(v, quote=True)}">report</a></td>'
        return f"<td>{html.escape(v)}</td>"

    head = "".join(f"<th>{c}</th>" for c in SUMMARY_COLUMNS)
    body = "\n".join(
        f'<tr class="{html.escape(r["status"])}">' + "".join(cell(k, r[k]) for k in SUMMARY_COLUMNS) + "</tr>"
        for r in summary_rows(manifest)
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Batch summary</title><style>'
        "body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}"
        "td,th{border:1px solid #999;padding:3px 8px}.failed td{color:#c62828}"
        ".partial td,.cancelled td,.needs_review td{color:#b26a00}"
        "@media(prefers-color-scheme:dark){body{background:#1e1e1e;"
        "color:#ddd}.failed td{color:#ef5350}}</style></head><body>"
        f"<h1>Batch summary</h1><p>Cell study: {html.escape(manifest.cell_project)}; created "
        f"{html.escape(manifest.created)}.</p><table><tr>{head}</tr>\n{body}</table></body></html>"
    )


def write_summary(manifest: BatchManifest, batch_dir: Path) -> tuple[Path, Path]:
    """``summary.csv`` and ``summary.html`` in the batch folder, plus ``review.html`` (an extra
    that never fails the summary)."""
    batch_dir = Path(batch_dir)
    try:
        from core.batch_review import write_review

        write_review(manifest, batch_dir)
    except Exception:  # noqa: BLE001
        pass
    out = []
    for name, text in (("summary.csv", summary_csv(manifest)), ("summary.html", summary_html(manifest))):
        p = batch_dir / name
        tmp = p.with_name(name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
        out.append(p)
    return out[0], out[1]
