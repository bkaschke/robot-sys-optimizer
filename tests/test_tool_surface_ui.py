"""Detect tool surface (Targets tab) and the batch rule "auto: tool surface" (plan Part 4, T4)."""

import json

import numpy as np
import pytest
import trimesh
import viser

from core import batch as bt
from core.mandrel_cases import SYNTHETIC_CASES, build_case
from core.target_gen import GenSettings
from tests.helpers import confirm_all
from tests.test_batch_job import _disk, _planner, fake  # noqa: F401 (fixture)
from ui import batch_job as bj
from ui.session import PartState, empty_cell, save_project


def case_mesh(name: str) -> trimesh.Trimesh:
    c = next(c for c in SYNTHETIC_CASES if c.name == name)
    return build_case(c)[0]


def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    return T


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8873, verbose=False)
    state = build_app(server)
    assert state["project"].load_example("rail_long_part")
    yield state
    server.stop()


def load_part(app, mesh, name, T=None):
    s = app["session"]
    path = s.project_dir / "assets" / "parts" / f"{name}.stl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    s.part = PartState(s.relative(path), "m", np.eye(4) if T is None else T)
    app["targets"].on_part_changed()
    return app["targets"]


def test_detect_accept_secondary_discard(app):
    tab = load_part(app, case_mesh("female_coarse"), "mandrel_fc")
    p = tab.detect
    assert p.detect(block=True)
    assert p.guess is not None and len(p.guess.faces) > 0 and len(p.guess.secondary) >= 1
    assert len(tab.selection) == 0  # a preview, not the selection
    assert p.replace_btn.visible and "Confidence" in p.info.content
    assert p.low.visible == (p.guess.confidence < 0.7)
    base = np.unique(p.guess.faces)
    # Add a secondary: the preview grows, still not the selection
    assert p.add_secondary(0)
    assert set(base) < set(p.preview_faces) and len(tab.selection) == 0
    assert p.accept("set")
    assert np.array_equal(tab.selection, p.layer.drawn) or len(tab.selection) > len(base)
    assert set(base) <= set(tab.selection) and p.guess is None
    # discard clears; accept with nothing is refused
    assert p.detect(block=True)
    p.discard()
    assert p.guess is None and not p.replace_btn.visible
    assert p.accept("add") is False
    # "add" keeps the existing selection
    tab.select(np.array([0]), "set")
    assert p.detect(block=True) and p.accept("add")
    assert 0 in tab.selection and set(np.unique(p.guess.faces if p.guess else base)) <= set(
        tab.selection
    ) | set(base)


def test_low_confidence_is_prominent(app):
    tab = load_part(app, case_mesh("male_ambiguous_flange"), "mandrel_amb")
    assert tab.detect.detect(block=True)
    assert tab.detect.low.visible and "Low confidence" in tab.detect.low.content


def test_rotated_part_still_detects_top(app):
    m = case_mesh("plate_coarse")
    ref = load_part(app, m, "plate_ref")
    ref.detect.detect(block=True)
    want = np.unique(ref.detect.guess.faces)
    R = rot_x(90.0)
    mr = m.copy()
    mr.apply_transform(R)  # the top now faces -y in the part frame
    tab = load_part(app, mr, "plate_rot", T=np.linalg.inv(R))  # the pose puts the top back up
    tab.detect.detect(block=True)
    got = np.unique(tab.detect.guess.faces)
    assert np.array_equal(got, want)
    # without the pose the same mesh gives another surface: the up vector is really transformed
    tab2 = load_part(app, mr, "plate_rot2", T=np.eye(4))
    tab2.detect.detect(block=True)
    assert not np.array_equal(np.unique(tab2.detect.guess.faces), want)


def test_detect_error_is_reported(app, monkeypatch):
    tab = load_part(app, case_mesh("plate_coarse"), "plate_err")

    def boom(*a, **k):
        raise RuntimeError("no luck")

    monkeypatch.setattr("ui.tool_surface_ui.detect_for_pose", boom)
    tab.detect.detect(block=True)
    assert "Detection failed" in tab.detect.info.content and tab.detect.guess is None
    assert not tab.detect.detect_btn.disabled


# --- batch ------------------------------------------------------------------------------------


def make_batch(tmp_path, names, min_conf=0.7):
    cell = tmp_path / "cell"
    save_project(empty_cell(), cell)
    bdir = tmp_path / "batch"
    (bdir / "parts").mkdir(parents=True)
    m = bt.BatchManifest(cell_project=bt.rel_cell(cell, bdir))
    for n in names:
        f = bdir / "parts" / f"{n}.stl"
        case_mesh(n).export(f)
        bt.add_job(
            m,
            part=f"parts/{n}.stl",
            selection=[],
            gen=GenSettings(row_m=0.2, col_m=0.2),
            name=n,
            mesh_sha1=bt.mesh_sha1(f),
            planner=_planner(),
            selection_rule="tool_surface",
            min_confidence=min_conf,
        )
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    return m, bdir


def test_rule_job_packages_with_targets(tmp_path):
    from core.part_package import load_part_package
    from ui.batch_materialize import make_package

    m, bdir = make_batch(tmp_path, ["plate_coarse"])
    job = m.jobs[0]
    assert bt.check_job(job, bdir) is None  # no face-range check for rule jobs
    pkg = load_part_package(make_package(m, job, bdir))
    assert len(pkg.targets.points) > 0 and job.selection and job.confidence >= 0.7
    assert "²" not in job.detect_summary


def test_ambiguous_becomes_needs_review_not_planned(tmp_path, fake):  # noqa: F811
    m, bdir = make_batch(tmp_path, ["male_ambiguous_flange", "plate_coarse"])
    fake["bdir"] = bdir
    out = bj.drive(bdir, lambda k, d: None)
    assert out["status"] == "done"
    assert _disk(bdir) == {"job_001": "needs_review", "job_002": "done"}
    j1 = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs[0]
    assert j1.confidence < 0.7 and "confidence" in j1.error and "\n" not in j1.error
    assert [s[:2] for s in fake["seen"]] == [("job_002", "running")]  # job_001 was never planned
    assert not (bdir / bt.PACKAGES / "job_001").exists()  # only the review pictures
    csv = (bdir / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert "confidence" in csv[0] and any("needs_review" in r for r in csv)
    # needs_review is not pending; after Reset it is
    mm = bt.load_manifest(bdir / bt.MANIFEST_NAME)
    assert [j.id for j in bt.pending_jobs(mm)] == []
    mm.jobs[0].status = "pending"
    assert [j.id for j in bt.pending_jobs(mm)] == ["job_001"]
    assert bj._line(bj.job_row(j1)).startswith("job_001    needs_review")


def test_old_manifest_loads(tmp_path):
    m, bdir = make_batch(tmp_path, ["plate_coarse"])
    data = json.loads((bdir / bt.MANIFEST_NAME).read_text(encoding="utf-8"))
    for j in data["jobs"]:
        for k in ("selection_rule", "min_confidence", "confidence", "detect_summary"):
            j.pop(k)
    (bdir / bt.MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    j = bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs[0]
    assert j.selection_rule == "faces" and j.min_confidence == 0.7 and j.confidence is None


def test_add_folder_queues_and_skips(app, tmp_path):
    panel = app["batch"]
    panel.batch_dir = None
    load_part(app, case_mesh("plate_coarse"), "plate_f")
    confirm_all(app["session"])
    folder = tmp_path / "mandrels"
    folder.mkdir()
    case_mesh("plate_coarse").export(folder / "a.stl")
    case_mesh("female_coarse").export(folder / "b.ply")
    case_mesh("plate_coarse").export(folder / "c.obj")
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    (folder / "broken.stl").write_text("not a mesh", encoding="utf-8")
    queued, skipped = panel.add_folder(str(folder))
    assert len(queued) == 3
    assert len(skipped) == 2 and any("notes.txt" in s and "unsupported" in s for s in skipped)
    assert any("broken.stl" in s for s in skipped)
    jobs = panel.manifest().jobs
    assert [j.name for j in jobs] == ["a", "b", "c"]
    assert all(j.selection_rule == "tool_surface" and not j.selection for j in jobs)
    assert "queued" in panel.status.content and "skipped" in panel.status.content
    assert panel.add_folder(str(tmp_path / "nope")) == ([], [])
