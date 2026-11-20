"""Batch through the GUI path with the real runner (in process): a golden plate and a STEP part."""

import csv

import numpy as np
import pytest
import viser

from core import batch as bt
from tests.helpers import confirm_all

cq = pytest.importorskip("cadquery")


def _step_box(path):
    body = cq.Workplane("XY").box(200, 150, 40).edges("|Z").fillet(10)
    cq.exporters.export(body, str(path))
    return path


def test_check_job_accepts_step_and_range_checks_triangles(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    p = _step_box(parts / "box.step")
    job = bt.BatchJob(id="job_001", part="parts/box.step", selection=[0, 1], mesh_sha1=bt.mesh_sha1(p))
    assert bt.check_job(job, tmp_path) is None
    job.selection = [0, 10_000_000]
    msg = bt.check_job(job, tmp_path)
    assert msg and "out of range" in msg and "faces" in msg
    job.part = "parts/box.step"
    job.mesh_sha1 = ""
    p.write_text("not a step file")
    job.selection = [0]
    assert "cannot read part" in bt.check_job(job, tmp_path)


@pytest.mark.slow
def test_two_jobs_run_in_process_and_open_result(tmp_path, monkeypatch):
    from app import build_app
    from ui import jobs

    monkeypatch.setattr(jobs, "USE_WORKER", False)
    server = viser.ViserServer(host="127.0.0.1", port=8872, verbose=False)
    try:
        st = build_app(server)
        assert st["project"].load_example("rail_long_part")
        panel, tab, cell, s = st["batch"], st["targets"], st["cell"], st["session"]
        confirm_all(s)
        panel.mode.value = "Quick plan"

        def select_top():
            m = tab.mesh()
            tab.selection = np.flatnonzero(m.face_normals[:, 2] > 0.9)[:30]
            tab.gen_spacing = None

        cell.part_choice.value = "flat_plate"
        select_top()
        confirm_all(s)
        assert panel.add_to_queue(), panel.add_reason()
        step = _step_box(tmp_path / "box.step")
        cell.upload_units.value = "m"
        cell.upload_part("box.step", step.read_bytes())
        assert s.part.mesh_path.endswith(".step")
        select_top()
        assert len(tab.selection) > 0
        confirm_all(s)
        assert panel.add_to_queue(), panel.add_reason()
        assert [j.part.rsplit(".", 1)[-1] for j in panel.manifest().jobs] == ["stl", "step"]
        panel.run(block=True)
        m = panel.manifest()
        assert [j.status for j in m.jobs] == ["done", "done"], [(j.status, j.error) for j in m.jobs]
        for j in m.jobs:
            assert (panel.batch_dir / j.out_dir / "project.json").is_file()
        rows = list(csv.DictReader(open(panel.batch_dir / "summary.csv", encoding="utf-8")))
        assert len(rows) == 2
        view = st["batch_view"]  # Results -> Batch results opened the run
        assert view.out == panel.batch_dir / bt.RESULTS and len(view.done_parts()) == 2
        assert view.open_part(name=view.done_parts()[1]["part"])
        assert s.part is not None and s.part.mesh_path.endswith(".step") and s.results
    finally:
        server.stop()
