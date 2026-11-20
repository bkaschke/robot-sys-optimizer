"""Batch: the manifest's old pdf flag, the time-left estimate and the setup warnings."""

import json

import pytest

from core import batch as bt
from tests.test_batch_job import _make_batch, fake  # noqa: F401 (fixture)
from ui import batch_job as bj


def test_eta_text():
    assert bt.eta_text([], 3) == ""  # nothing finished yet: no guess
    assert bt.eta_text([60.0], 0) == ""
    assert bt.eta_text([0.0, 0.0], 2) == ""  # jobs without a run time do not count
    assert bt.eta_text([20.0], 2) == "about 1 min left"
    assert bt.eta_text([60.0, 600.0, 120.0], 4) == "about 8 min left"  # the median (120 s), not the mean
    assert bt.eta_text([100.0, 200.0], 2) == "about 5 min left"
    assert bt.eta_text([3600.0], 3) == "about 3.0 h left"


def test_manifest_pdf_flag_and_old_files(tmp_path):
    path = tmp_path / bt.MANIFEST_NAME
    path.write_text(json.dumps({"version": 1, "cell_project": "../cell", "jobs": []}), encoding="utf-8")
    assert bt.load_manifest(path).pdf is False  # a version 1 file loads
    bt.set_pdf(path, True)
    m = bt.load_manifest(path)
    assert m.pdf is True and m.cell_project == "../cell"
    stamp = path.stat().st_mtime_ns
    bt.set_pdf(path, True)  # unchanged: not rewritten
    assert path.stat().st_mtime_ns == stamp
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = bt.VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="newer version"):
        bt.load_manifest(path)


def test_progress_carries_the_time_left_after_the_first_job(tmp_path, fake, monkeypatch):  # noqa: F811
    _m, bdir = _make_batch(tmp_path, n=3)
    ticks = iter(range(0, 100000, 150))  # every clock read is 150 s later: jobs "take" minutes
    monkeypatch.setattr(bj.time, "monotonic", lambda: float(next(ticks)))
    prog = []
    bj.drive(bdir, lambda k, d: prog.append(d[0]) if k == "progress" else None, workers=1)
    first = [p for p in prog if p.startswith("0/3 planned")]
    later = [p for p in prog if p.startswith("1/3 planned") or p.startswith("2/3 planned")]
    assert first and not any("left" in p for p in first)  # nothing to go by yet
    assert later and all(" - about " in p and p.endswith(" left") for p in later)


def test_setup_findings_become_job_warnings(tmp_path, fake):  # noqa: F811
    """A plate queued with the wrong units (read as mm: 1 mm across) is planned, but the job says so."""
    from ui.session import empty_cell

    cell = empty_cell()
    cell.load_tool_from_library("orbital_sander")
    m, bdir = _make_batch(tmp_path, n=2, cell=cell)
    m.jobs[1].units = "mm"
    m.jobs[1].warnings = ["setup: an old line", "a detector warning"]
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    bj.drive(bdir, lambda *a: None)
    good, small = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs
    assert good.warnings == [] and good.status == "done"
    assert small.status == "failed" and "smaller than one grid cell" in small.error  # a 1 mm plate
    assert small.warnings[0] == "a detector warning"  # kept; the old setup line is replaced
    assert any(w.startswith("setup: The part is only") for w in small.warnings)
    assert "setup: an old line" not in small.warnings
    from core.batch_review import review_html

    assert "The part is only" in review_html(bt.load_manifest(bdir / bt.MANIFEST_NAME), bdir)


def test_queue_table_still_importable_from_the_panel_module():
    from ui.batch_queue_ui import job_table

    assert job_table(bt.BatchManifest()) == "_The queue is empty._" and job_table is bt.job_table
