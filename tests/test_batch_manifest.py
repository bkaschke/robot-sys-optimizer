"""Batch manifest (core.batch) and a job as a part package (ui.batch_materialize)."""

import shutil

import pytest

from core import batch as bt
from core import part_package as pp
from core.golden_parts import flat_plate
from core.target_gen import GenSettings
from ui.batch_materialize import make_package
from ui.session import empty_cell, save_project


@pytest.fixture
def batch(tmp_path):
    save_project(empty_cell(), tmp_path / "cell")
    bdir = tmp_path / "batch"
    (bdir / "parts").mkdir(parents=True)
    mesh = flat_plate()
    mesh.export(bdir / "parts" / "plate.stl")
    m = bt.BatchManifest(cell_project=str(tmp_path / "cell"))
    top = [i for i, n in enumerate(mesh.face_normals) if n[2] > 0.9]
    job = bt.add_job(
        m,
        part="parts/plate.stl",
        selection=top,
        gen=GenSettings(row_m=0.1, col_m=0.1),
        mesh_sha1=bt.mesh_sha1(bdir / "parts" / "plate.stl"),
    )
    return m, job, bdir, mesh


def test_roundtrip_and_atomic(batch):
    m, job, bdir, _ = batch
    p = bdir / bt.MANIFEST_NAME
    bt.save_manifest(m, p)
    assert not [f for f in bdir.iterdir() if f.name.endswith(".tmp")]
    back = bt.load_manifest(p)
    assert back == m and back.version == bt.VERSION and back.jobs[0].gen.row_m == 0.1
    j2 = bt.add_job(m, part="parts/plate.stl", selection=[0])
    assert j2.id != job.id and len({j.id for j in m.jobs}) == 2


def test_pending_and_resume(batch):
    m, job, _, _ = batch
    j2 = bt.add_job(m, part="x", selection=[0])
    j3 = bt.add_job(m, part="x", selection=[0])
    job.status, j2.status = "done", "running"
    assert [j.id for j in bt.pending_jobs(m)] == [j2.id, j3.id]
    assert j2.status == "pending" and job.status == "done"


def test_check_job(batch):
    m, job, bdir, mesh = batch
    assert bt.check_job(job, bdir) is None
    bad = job.model_copy(update={"selection": [len(mesh.faces) + 5]})
    assert "out of range" in bt.check_job(bad, bdir)
    assert "missing" in bt.check_job(job.model_copy(update={"part": "nope.stl"}), bdir)
    mesh.apply_scale(2.0)
    mesh.export(bdir / "parts" / "plate.stl")
    assert "changed" in bt.check_job(job, bdir)


def test_make_package(batch):
    m, job, bdir, _ = batch
    pkg = make_package(m, job, bdir)
    assert pkg == bdir / bt.PACKAGES / job.id and (pkg / "plate.stl").is_file()
    p = pp.load_part_package(pkg)
    assert p.name == job.id and p.part.mesh == "plate.stl" and len(p.targets.points) > 10
    assert {g.name for g in p.targets.groups} == {"default"}
    assert make_package(m, job, bdir) == pkg  # made again: replaced


def test_materialize_errors(batch):
    m, job, bdir, _ = batch
    shutil.rmtree(bdir / "parts")
    with pytest.raises(ValueError, match="missing"):
        make_package(m, job, bdir)
