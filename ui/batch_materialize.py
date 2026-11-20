"""Turn a batch job into a part package (the job's part, pose and generated targets) for v13's
batch runner, and render the review pictures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import batch as bt
from core import preflight as pf
from core import targets as tg
from core.schema import TargetGroup, Tolerances
from core.target_gen import generate_targets
from ui.session import Session, load_mesh, load_project


def ensure_group(session: Session, name: str) -> None:
    """Add target group ``name`` with the tool's tolerances (as the Targets tab does)."""
    if name in {g.name for g in session.targets.groups}:
        return
    tol = session.tool.tolerances if session.tool else Tolerances()
    g = TargetGroup(name=name, spin_deg=tol.spin_deg, tilt_deg=tol.tilt_deg, standoff_m=tol.standoff_m)
    session.targets = session.targets.model_copy(update={"groups": [*session.targets.groups, g]})


def up_in_part(T_world_part) -> np.ndarray:
    """World +Z expressed in the part frame (the part pose may be rotated)."""
    return np.asarray(T_world_part, float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def detect_for_pose(mesh, T_world_part, pad_radius_m=None):
    """Run the tool-surface detector on a part mesh placed with ``T_world_part``. STEP parts
    pass their CAD face ids (``mesh.metadata["cad_face_ids"]``)."""
    from core.tool_surface_pick import detect_tool_surface

    return detect_tool_surface(
        mesh,
        up=up_in_part(T_world_part),
        cad_face_ids=mesh.metadata.get("cad_face_ids"),
        pad_radius_m=pad_radius_m,
    )


def cell_pad_radius(manifest: bt.BatchManifest, batch_dir: Path) -> float | None:
    """The cell tool's pad radius (m) from its project.json, or None (unknown / unreadable)."""
    try:
        data = json.loads((bt.cell_path(manifest, batch_dir) / "project.json").read_text("utf-8"))
        r = (data.get("end_effector") or {}).get("pad_radius_m")
        return float(r) if r else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _remember(job: bt.BatchJob, g) -> None:
    """Keep the detected and secondary faces on the job (not saved) for the review pictures."""
    job._detected = [int(f) for f in g.faces]
    sec = [np.asarray(c.faces, int) for c in g.secondary]
    job._secondary = [int(f) for f in np.unique(np.concatenate(sec))] if sec else []


def apply_rule(job: bt.BatchJob, mesh, who: str, pad_radius_m: float | None = None) -> None:
    """tool_surface job: detect, store confidence + summary, set ``job.selection``. Raises
    ``bt.NeedsReview`` (one line) when the confidence is below ``job.min_confidence``."""
    g = detect_for_pose(mesh, np.array(job.pose, float), pad_radius_m)
    job.confidence = float(g.confidence)
    job.warnings = [str(w) for w in g.warnings]
    _remember(job, g)
    job.detect_summary = str(g.summary).replace("²", "2")
    if g.confidence < job.min_confidence or len(g.faces) == 0:
        raise bt.NeedsReview(
            f"{who}: tool surface confidence {g.confidence:.2f} is below {job.min_confidence:.2f}; "
            f"review it ({job.detect_summary})"
        )
    job.selection = [int(f) for f in g.faces]


def setup_dir(manifest: bt.BatchManifest, batch_dir: Path) -> Path:
    """The cell the jobs run with: ``<batch>/_setup`` (the open project, written at Run batch),
    else the ``cell_project`` folder (a batch from before v13's runner, or the CLI)."""
    setup = Path(batch_dir) / bt.SETUP
    return setup if (setup / "project.json").is_file() else bt.cell_path(manifest, batch_dir)


def make_package(manifest: bt.BatchManifest, job: bt.BatchJob, batch_dir: Path) -> Path:
    """The job as a part package (v13 ``core.part_package``) for the batch runner, and return its
    folder. A package job is already one (checked); a mesh job gets its targets generated on its
    selection (or on the detected tool surface) into ``<batch>/packages/<job id>/``. ValueError
    (one line) if the job cannot run or generates no targets; ``bt.NeedsReview`` for a doubtful
    detection."""
    from core import part_package as pp
    from core.schema import Part
    from core.transforms import matrix_to_pose

    batch_dir = Path(batch_dir)
    who = job.name or job.id
    reason = bt.check_job(job, batch_dir)
    if reason:
        raise ValueError(reason)
    if job.source == "package":
        return batch_dir / job.package
    if job.selection_rule == "tool_surface":  # before the cell is loaded: a doubtful job stays cheap
        apply_rule(job, load_mesh(batch_dir / job.part, job.units), who, cell_pad_radius(manifest, batch_dir))
    cell = setup_dir(manifest, batch_dir)
    if not (cell / "project.json").is_file():
        raise ValueError(f"{who}: the cell project '{cell}' has no project.json")
    s = Session()
    load_project(s, cell)  # the tool's tolerances for the new target group
    src = (batch_dir / job.part).resolve()
    mesh = load_mesh(src, job.units)
    # a wrong unit usually ends in "no targets generated": say why, next to that error
    job.warnings = [w for w in job.warnings if not w.startswith(pf.BATCH_PREFIX)] + [
        f"{pf.BATCH_PREFIX}{f.text}" for f in pf.check_part(mesh, np.array(job.pose, float))
    ]
    try:
        P, N, _ = generate_targets(mesh, job.selection, job.gen)
    except (ValueError, IndexError) as err:
        raise ValueError(f"{who}: target generation failed: {err}") from err
    if len(P) == 0:
        raise ValueError(f"{who}: no targets generated; check the selection and spacing")
    s.targets = s.targets.model_copy(update={"points": []})
    ensure_group(s, job.gen.group)
    s.targets = tg.append_points(s.targets, P, N, job.gen.group)
    part = Part(mesh=src.as_posix(), units=job.units, pose=matrix_to_pose(np.array(job.pose, float)))
    folder = batch_dir / bt.PACKAGES  # one folder per job id (the name may repeat)
    return pp.save_part_package(job.id, part, s.targets, Path, folder, overwrite=True, description=who)


# --- review pictures ------------------------------------------------------------------------

VIEWS = (("iso", "Iso view (offline render)"), ("top", "Top view (offline render)"))


def job_pictures(job: bt.BatchJob, batch_dir: Path) -> tuple[list[tuple[str, bytes]], str]:
    """Render iso + top pictures of the job's part with its selection highlighted (for a
    tool_surface job without a selection: the detected faces, secondary candidates in a second
    colour), save ``<job id>/review_<view>.png`` and record the relative paths in
    ``job.summary["review"]``. Returns ([(caption, png)], note). Never raises: a failure gives
    ([], one-line note) and ``job.summary["review_note"]``."""
    from core import soft_render

    batch_dir = Path(batch_dir)
    try:
        mesh = load_mesh(batch_dir / job.part, job.units)
        pose = np.array(job.pose, float)
        faces, second = list(job.selection), []
        if job.selection_rule == "tool_surface" and not faces:
            if not job._detected:  # a batch run before this feature: detect again
                _remember(job, detect_for_pose(mesh, pose))
            faces, second = list(job._detected), list(job._secondary)
        up = up_in_part(pose)
        out, review = [], {}
        (batch_dir / job.id).mkdir(parents=True, exist_ok=True)
        for view, caption in VIEWS:
            png = soft_render.render(mesh, highlight=faces, secondary=second or None, view=view, up=up)
            rel = f"{job.id}/review_{view}.png"
            (batch_dir / rel).write_bytes(png)
            review[view] = rel
            out.append((caption, png))
        job.summary = {k: v for k, v in job.summary.items() if k != "review_note"} | {"review": review}
        return out, ""
    except Exception as err:  # noqa: BLE001 - pictures never fail a job
        first = str(err).splitlines()[0] if str(err) else ""
        note = f"pictures not rendered: {type(err).__name__}: {first}"
        job.summary = {k: v for k, v in job.summary.items() if k != "review"} | {"review_note": note}
        return [], note


def build_review(batch_dir: Path, on_note=None, force: bool = False) -> Path:
    """(Re)write ``review.html``. Jobs whose pictures are missing (a batch run before this
    feature, or a deleted file) are rendered now when their mesh is available (``force``: all
    jobs); the manifest on disk gets the picture paths. ``on_note(text)`` reports progress."""
    from core.batch_review import write_review

    batch_dir = Path(batch_dir)
    path = batch_dir / bt.MANIFEST_NAME
    m = bt.load_manifest(path)
    todo = [j for j in m.jobs if force or not _has_pictures(j, batch_dir)]
    for i, job in enumerate(todo):
        if on_note:
            on_note(f"rendering pictures {i + 1}/{len(todo)} ({job.name or job.id})")
        if bt.check_job(job, batch_dir) and job.selection_rule == "faces":
            job.summary = {**job.summary, "review_note": "the part mesh is missing or changed"}
            continue
        job_pictures(job, batch_dir)
        bt.update_job(path, job)
    return write_review(bt.load_manifest(path), batch_dir)


def _has_pictures(job: bt.BatchJob, batch_dir: Path) -> bool:
    rev = (job.summary or {}).get("review") or {}
    return bool(rev) and all((batch_dir / rel).is_file() for rel in rev.values())
