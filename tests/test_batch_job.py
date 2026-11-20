"""Batch queue runner (ui.batch_job): jobs -> part packages -> v13's runner (faked here), statuses
on disk after every step, cancel / resume, package jobs, the CLI."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from core import batch as bt
from core import part_package as pp
from core.golden_parts import flat_plate
from core.target_gen import GenSettings
from ui import batch_job as bj
from ui import batch_summary
from ui.session import empty_cell, save_project

ROOT = Path(__file__).resolve().parents[1]


def _planner(mode: str = "cost") -> dict:
    return {"mode": mode}


def _make_batch(tmp_path, n=3, cell=None):
    cell_dir = tmp_path / "cell"
    save_project(cell or empty_cell(), cell_dir)
    bdir = tmp_path / "batch"
    (bdir / "parts").mkdir(parents=True)
    mesh = flat_plate()
    mesh.export(bdir / "parts" / "plate.stl")
    m = bt.BatchManifest(cell_project=bt.rel_cell(cell_dir, bdir))
    top = [i for i, nrm in enumerate(mesh.face_normals) if nrm[2] > 0.9]
    sha = bt.mesh_sha1(bdir / "parts" / "plate.stl")
    for _ in range(n):
        bt.add_job(
            m,
            part="parts/plate.stl",
            selection=top,
            gen=GenSettings(row_m=0.2, col_m=0.2),
            mesh_sha1=sha,
            planner=_planner(),
        )
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    return m, bdir


def _disk(bdir) -> dict:
    return {j.id: j.status for j in bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs}


@pytest.fixture
def fake(monkeypatch):
    """ui.batch_run.run_batch without processes: every part is "planned" (1 stop, all targets
    reached), its results folder and the runner's summary.json written. Records per part the job
    id (its package folder), the job's status on disk, its targets and its extras."""
    state = {"seen": [], "bdir": None, "fail": set(), "cancel_after": None, "calls": []}

    def run_batch(project_dir, parts, out_dir, frame_search=False, workers=None, check_workers=1,
                  on_progress=None, on_row=None, cancelled=lambda: False, parts_folder=""):  # fmt: skip
        state["calls"].append({"setup": Path(project_dir), "frame_search": frame_search, "workers": workers})
        bdir = Path(out_dir).parent
        rows = []
        for name, d, *extra in parts:
            job_id = Path(d).name
            if cancelled():
                rows.append({"part": name, "status": "cancelled", "reason": "cancelled"})
                continue
            pkg = pp.load_part_package(d)
            state["seen"].append((job_id, _disk(bdir).get(job_id), len(pkg.targets.points), *extra))
            on_progress(name, "planning", 0.5)
            folder = Path(out_dir) / pp.safe(name)
            folder.mkdir(parents=True, exist_ok=True)
            if name in state["fail"]:
                row = {
                    "part": name,
                    "status": "failed",
                    "reason": "ValueError: no plan",
                    "trace": "Traceback ...",
                }
            else:
                (folder / "report.html").write_text("<html></html>", encoding="utf-8")
                n = len(pkg.targets.points)
                row = {"part": name, "status": "done", "reason": "", "folder": str(folder),
                       "report": str(folder / "report.html"), "points": n, "reached": n, "unreached": 0,
                       "stops": 1, "plan": "1 station", "cost": 2.5, "cycle_s": 42.0}  # fmt: skip
            row["run_s"] = 3.0
            rows.append(row)
            batch_summary.write(out_dir, {}, rows)
            on_progress(name, row["status"], 1.0)
            on_row(row)
            if state["cancel_after"] is not None and len(rows) == state["cancel_after"][0]:
                state["cancel_after"][1].set()
        return rows

    monkeypatch.setattr(bj, "run_batch", run_batch)
    return state


def test_statuses_packages_and_summary(tmp_path, fake):
    m, bdir = _make_batch(tmp_path)
    fake["bdir"] = bdir
    j2 = m.jobs[1]
    j2.mesh_sha1 = "0" * 40  # the mesh "changed" since it was queued
    m.jobs[2].planner = _planner("quick")
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    msgs, after_each = [], []

    def on_msg(kind, data):
        msgs.append((kind, data))
        if kind == "job":
            after_each.append(_disk(bdir))

    out = bj.drive(bdir, on_msg)
    ids = [j.id for j in m.jobs]
    assert out["status"] == "done" and len(out["jobs"]) == 3
    assert after_each[0] == {ids[0]: "running", ids[1]: "failed", ids[2]: "pending"}  # failed while preparing
    assert _disk(bdir) == {ids[0]: "done", ids[1]: "failed", ids[2]: "done"}
    assert [(s[0], s[1], s[3]) for s in fake["seen"]] == [
        (ids[0], "running", {"quick": False}),
        (ids[2], "running", {"quick": True}),
    ]
    assert fake["seen"][0][2] > 0  # the package has the generated targets
    assert fake["calls"][0]["setup"] == (tmp_path / "cell").resolve()  # no _setup: the cell project
    d1, d2, _d3 = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs
    assert "changed since it was queued" in d2.error and "\n" not in d2.error
    assert (bdir / ids[1] / "error.txt").read_text(encoding="utf-8").startswith(d2.error)
    s = d1.summary
    assert s["stations"] == 1 and s["cost"] == 2.5 and s["cycle_s"] == 42.0 and s["reached"] == s["total"] > 0
    assert s["report"] == f"results/{ids[0]}/report.html" and (bdir / s["report"]).is_file()
    assert d1.out_dir == f"results/{ids[0]}" and d1.run_s == 3.0 and not d1.error
    assert set(s["review"]) == {"iso", "top"}  # the pictures are kept with the results
    prog = [d for k, d in msgs if k == "progress"]
    assert prog[0][0].startswith("preparing") and all(0 <= f <= 1 for _, f in prog)
    assert any(t.startswith("2/2 planned") for t, _ in prog)
    csv = (bdir / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert csv[0].startswith("job,name,part,status") and len(csv) == 4 and ",failed," in csv[2]
    assert (bdir / "review.html").is_file()
    # resume: done and failed jobs are skipped; a reset job runs again
    fake["seen"].clear()
    assert bj.drive(bdir, lambda *a: None)["jobs"] == [] and fake["seen"] == []
    d2.status, d2.mesh_sha1 = "pending", m.jobs[0].mesh_sha1
    bt.update_job(bdir / bt.MANIFEST_NAME, d2)
    rows = bj.drive(bdir, lambda *a: None)["jobs"]
    assert [r["id"] for r in rows] == [ids[1]] and rows[0]["status"] == "done"


def test_runner_failure_and_duplicate_names(tmp_path, fake):
    m, bdir = _make_batch(tmp_path, n=2)
    for j in m.jobs:
        j.name = "plate"  # the same name twice: two results folders all the same
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    fake["fail"] = {"plate"}
    bj.drive(bdir, lambda *a: None)
    a, b = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs
    assert a.status == "failed" and a.error == "ValueError: no plan"
    assert "Traceback ..." in (bdir / a.id / "error.txt").read_text(encoding="utf-8")
    assert b.status == "done" and b.out_dir == f"results/plate_{b.id}"


def test_cancel_sends_unfinished_jobs_back_to_pending(tmp_path, fake):
    m, bdir = _make_batch(tmp_path)
    ev = threading.Event()
    fake["cancel_after"] = (1, ev)
    msgs, done = [], threading.Event()
    t = bj.run_batch_thread(
        bdir, lambda k, d: (msgs.append((k, d)), k == "finished" and done.set()), ev.is_set
    )
    t.join(60)
    assert done.is_set()
    fin = next(d for k, d in msgs if k == "finished")
    ids = [j.id for j in m.jobs]
    assert fin["status"] == "cancelled" and _disk(bdir) == {
        ids[0]: "done",
        ids[1]: "pending",
        ids[2]: "pending",
    }
    j = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs[1]
    assert j.error.startswith("cancelled: runs again")
    ev.clear()
    fake["cancel_after"] = None
    assert [r["id"] for r in bj.drive(bdir, lambda *a: None)["jobs"]] == ids[1:]  # resume


def test_package_job_and_setup_folder(tmp_path, fake):
    """A v13 part package queued as a job: no target generation; the batch's _setup/ is the cell."""
    from core.schema import load_project as read_project

    m, bdir = _make_batch(tmp_path, n=0)
    rail = ROOT / "examples" / "rail_long_part"
    proj = read_project(rail / "project.json")
    pkg = pp.save_part_package("rail", proj.part, proj.targets, lambda r: ROOT / r, bdir / bt.PACKAGES)
    pkg_mesh = (pkg / pp.load_part_package(pkg).part.mesh).relative_to(bdir).as_posix()
    bt.add_job(m, part=pkg_mesh, selection=[], source="package", package=f"{bt.PACKAGES}/rail")
    bt.add_job(m, part=pkg_mesh, selection=[], source="package", package=f"{bt.PACKAGES}/gone")
    m.frame_search = True
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    save_project(empty_cell(), bdir / bt.SETUP)
    bj.drive(bdir, lambda *a: None, workers=1)
    a, b = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs
    assert a.status == "done" and a.summary["total"] == len(proj.targets.points)
    assert b.status == "failed" and "part package packages/gone" in b.error
    call = fake["calls"][0]
    assert call["setup"] == bdir / bt.SETUP and call["frame_search"] is True and call["workers"] == 1
    assert "package" in bt.job_table(bt.load_manifest(bdir / bt.MANIFEST_NAME))


def test_no_cell_project_fails_the_batch(tmp_path, fake):
    m, bdir = _make_batch(tmp_path, n=1)
    m.cell_project = "../nowhere"
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    msgs = []
    bj.run_batch_thread(bdir, lambda k, d: msgs.append((k, d))).join(30)
    fin = msgs[-1][1]
    assert fin["status"] == "failed" and "no cell project" in fin["error"]


def test_cli(tmp_path, fake, capsys):
    _m, bdir = _make_batch(tmp_path, n=1)
    assert bj.main([str(bdir), "--frame-search"]) == 0
    assert bt.load_manifest(bdir / bt.MANIFEST_NAME).frame_search is True
    out = capsys.readouterr().out
    assert "1/1 job(s) done" in out and "review.html" in out
    assert bj.main([str(Path(tmp_path) / "nowhere"), "--frame-search"]) == 1
    assert "batch failed" in capsys.readouterr().out


def test_cell_project_is_relative(tmp_path):
    cell, bdir = tmp_path / "ws" / "study", tmp_path / "ws" / "batch_1"
    rel = bt.rel_cell(cell, bdir)
    assert rel == "../study"
    m = bt.BatchManifest(cell_project=rel)
    assert bt.cell_path(m, bdir) == cell.resolve()


def test_import_is_viser_free():
    code = "import sys, ui.batch_job, ui.report_make, ui.report_text; print('viser' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


@pytest.mark.slow
def test_real_batch_of_a_mesh_job_and_a_package(tmp_path):
    """The rail example's cell; a queued mesh job (its part's top) and a v13 package, planned by the
    real runner (a process per job), each with a saved project and a report."""
    import numpy as np

    from core.schema import load_project as read_project
    from ui.session import Session, load_mesh, load_project

    rail = ROOT / "examples" / "rail_long_part"
    proj = read_project(rail / "project.json")
    s = Session()
    load_project(s, rail)
    bdir = tmp_path / "batch"
    save_project(s, bdir / bt.SETUP)
    m = bt.BatchManifest()
    src = s.resolve(s.part.mesh_path)
    (bdir / "parts").mkdir(parents=True)
    dest = bdir / "parts" / src.name
    dest.write_bytes(src.read_bytes())
    mesh = load_mesh(dest, s.part.units)
    top = np.flatnonzero(mesh.face_normals[:, 2] > 0.9)[:40]
    bt.add_job(m, part=f"parts/{src.name}", selection=top, units=s.part.units,
               pose=s.part.T_world_part.tolist(), gen=GenSettings(row_m=0.3, col_m=0.3),
               mesh_sha1=bt.mesh_sha1(dest), planner=_planner("quick"), name="top")  # fmt: skip
    pkg = pp.save_part_package("rail", proj.part, proj.targets, lambda r: ROOT / r, bdir / bt.PACKAGES)
    mesh_rel = (pkg / pp.load_part_package(pkg).part.mesh).relative_to(bdir).as_posix()
    bt.add_job(m, part=mesh_rel, selection=[], source="package", package=f"{bt.PACKAGES}/rail",
               planner=_planner("quick"), name="rail")  # fmt: skip
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    out = bj.drive(bdir, lambda *a: None, workers=1)
    jobs = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs
    assert out["status"] == "done" and [j.status for j in jobs] == ["done", "done"], [j.error for j in jobs]
    for j in jobs:
        p = read_project(bdir / j.out_dir / "project.json")
        assert p.results and j.summary["stations"] >= 1 and j.summary["report"]
        assert p.results[0].label.startswith("Quick")
    assert batch_summary.read(bdir / bt.RESULTS)["rows"][0]["part"] == "top"
