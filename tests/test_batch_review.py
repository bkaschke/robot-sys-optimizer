"""core.batch_review + the review pictures written by the batch runner (ui.batch_job)."""

import json
import re
from pathlib import Path

import pytest

from core import batch as bt
from core import batch_review as br
from core.mandrel_cases import build_case, iter_cases
from core.target_gen import GenSettings
from tests.test_batch_job import _disk, _make_batch, _planner, fake  # noqa: F401
from ui import batch_job as bj
from ui.session import empty_cell, save_project


def _png(path: Path, color=(200, 30, 30)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)


def _manifest(tmp_path):
    m = bt.BatchManifest(cell_project="cell")
    specs = [
        ("a_done_hi", "done", 0.95),
        ("b_failed", "failed", None),
        ("c_review", "needs_review", 0.4),
        ("d_done_lo", "done", 0.75),
        ("e_pending", "pending", None),
        ("f_manual", "done", None),
    ]
    for i, (name, status, conf) in enumerate(specs):
        j = bt.add_job(
            m,
            part=f"parts/{name}.stl",
            selection=[] if conf is not None else [1, 2],
            selection_rule="tool_surface" if conf is not None else "faces",
            name=name,
            status=status,
            confidence=conf,
            error="boom: unreachable" if status == "failed" else "",
            warnings=["steep area <b>20%</b>"] if status == "needs_review" else [],
            summary={
                "stations": 2,
                "reached": 9,
                "total": 10,
                "cost": 3.5,
                "report": f"job_{i + 1:03d}/export/report.html",
            }
            if status == "done"
            else {},
        )
        if i != 3:  # d_done_lo has no pictures: placeholder
            _png(tmp_path / j.id / "review_iso.png")
            _png(tmp_path / j.id / "review_top.png")
            j.summary["review"] = {"iso": f"{j.id}/review_iso.png", "top": f"{j.id}/review_top.png"}
        else:
            j.summary["review_note"] = "pictures not rendered: ValueError: nope"
    return m


def test_review_html_content_and_order(tmp_path):
    m = _manifest(tmp_path)
    html = br.review_html(m, tmp_path)
    for j in m.jobs:
        assert j.name in html
    assert html.count("data:image/png;base64,") == 10  # 5 jobs x 2 pictures
    assert "pictures not rendered: ValueError: nope" in html  # placeholder with the reason
    order = re.findall(r'<article class="card" data-status="(\w+)" id="(job_\d+)"', html)
    names = [m.jobs[int(i[-3:]) - 1].name for _, i in order]
    assert names == ["c_review", "b_failed", "d_done_lo", "a_done_hi", "f_manual", "e_pending"]
    assert "confidence <b>95%</b>" in html and "manual selection" in html
    assert "steep area &lt;b&gt;20%&lt;/b&gt;" in html  # escaped
    assert "boom: unreachable" in html and "9 / 10" in html
    assert 'href="job_001/export/report.html"' in html
    assert "prefers-color-scheme:dark" in html and "break-inside:avoid" in html
    assert "<title>Batch review" in html


def test_no_external_urls(tmp_path):
    html = br.review_html(_manifest(tmp_path), tmp_path)
    assert "http://" not in html and "https://" not in html
    assert not re.search(r"<link|@import|url\(|src=\"(?!data:)", html)


def test_write_review_and_empty_batch(tmp_path):
    p = br.write_review(_manifest(tmp_path), tmp_path)
    assert p == tmp_path / "review.html" and p.stat().st_size > 1000
    assert "no jobs" in br.review_html(bt.BatchManifest(), tmp_path)


def test_missing_picture_file_gives_placeholder(tmp_path):
    m = _manifest(tmp_path)
    (tmp_path / "job_001" / "review_iso.png").unlink()
    html = br.review_html(m, tmp_path)
    assert "review_iso.png is missing" in html


def test_old_manifest_loads(tmp_path):
    m = bt.BatchManifest(cell_project="c")
    bt.add_job(m, part="p.stl", selection=[1])
    d = json.loads(m.model_dump_json())
    del d["jobs"][0]["warnings"]  # a manifest written before this field
    p = tmp_path / bt.MANIFEST_NAME
    p.write_text(json.dumps(d), encoding="utf-8")
    old = bt.load_manifest(p)
    assert old.jobs[0].warnings == []
    assert "p.stl" in br.review_html(old, tmp_path)


# --- run_job writes the pictures ---------------------------------------------------------------


def _tool_surface_batch(tmp_path, case_name, min_conf):
    case = next(c for c, *_ in iter_cases([case_name]))
    mesh, _, _ = build_case(case)
    cell_dir = tmp_path / "cell"
    save_project(empty_cell(), cell_dir)
    bdir = tmp_path / "batch"
    (bdir / "parts").mkdir(parents=True)
    mesh.export(bdir / "parts" / "m.stl")
    m = bt.BatchManifest(cell_project=bt.rel_cell(cell_dir, bdir))
    bt.add_job(
        m,
        part="parts/m.stl",
        selection=[],
        selection_rule="tool_surface",
        min_confidence=min_conf,
        gen=GenSettings(row_m=0.2, col_m=0.2),
        mesh_sha1=bt.mesh_sha1(bdir / "parts" / "m.stl"),
        planner=_planner(),
    )
    bt.save_manifest(m, bdir / bt.MANIFEST_NAME)
    return m, bdir


def _run(m, bdir):
    bj.drive(bdir, lambda *a: None)
    return bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs[0]


def test_needs_review_job_gets_pictures_and_warnings(tmp_path, fake):  # noqa: F811
    m, bdir = _tool_surface_batch(tmp_path, "male_ambiguous_flange", 1.01)
    j = _run(m, bdir)
    assert j.status == "needs_review"
    for v in ("iso", "top"):
        f = bdir / j.id / f"review_{v}.png"
        assert f.is_file() and f.read_bytes()[:4] == b"\x89PNG"
        assert j.summary["review"][v] == f"{j.id}/review_{v}.png"
    assert j.confidence is not None and j.detect_summary
    assert "review_note" not in j.summary
    page = (bdir / "review.html").read_text(encoding="utf-8")  # refreshed after the job
    assert "needs_review" in page and "data:image/png;base64," in page


def test_done_job_pictures(tmp_path, fake):  # noqa: F811
    m, bdir = _tool_surface_batch(tmp_path, "plate_a", 0.0)
    fake["bdir"] = bdir
    j = _run(m, bdir)
    assert j.status == "done", j.error
    assert (bdir / j.id / "review_iso.png").is_file() and (bdir / j.id / "review_top.png").is_file()
    assert j.summary["review"]["top"] == f"{j.id}/review_top.png" and j.selection
    page = (bdir / "review.html").read_text(encoding="utf-8")
    assert "job_001" in page and f"results/{j.id}/" in page  # the project folder link


def test_render_failure_never_fails_the_job(tmp_path, fake, monkeypatch):  # noqa: F811
    from core import soft_render

    def boom(*a, **k):
        raise RuntimeError("no memory for the picture")

    monkeypatch.setattr(soft_render, "render", boom)
    m, bdir = _tool_surface_batch(tmp_path, "plate_a", 0.0)
    fake["bdir"] = bdir
    j = _run(m, bdir)
    assert j.status == "done" and "review" not in j.summary
    assert "pictures not rendered: RuntimeError: no memory" in j.summary["review_note"]
    assert "pictures not rendered" in (bdir / "review.html").read_text(encoding="utf-8")
    assert _disk(bdir)[j.id] == "done"


def test_build_review_renders_missing_pictures(tmp_path, fake):  # noqa: F811
    from ui.batch_materialize import build_review

    m, bdir = _make_batch(tmp_path, n=2)  # faces jobs, never run: no pictures yet
    notes = []
    p = build_review(bdir, notes.append)
    assert p == bdir / "review.html" and len(notes) == 2
    for j in bt.load_manifest(bdir / bt.MANIFEST_NAME).jobs:
        assert (bdir / j.id / "review_iso.png").is_file()
    assert build_review(bdir, notes.append) == p and len(notes) == 2  # nothing left to render


def test_cli_review_only(tmp_path, fake, capsys):  # noqa: F811
    m, bdir = _make_batch(tmp_path, n=1)
    assert bj.main([str(bdir), "--review-only"]) == 0
    assert "review page:" in capsys.readouterr().out and (bdir / "review.html").is_file()


@pytest.fixture(scope="module")
def _planned(tmp_path_factory):
    import numpy as np
    import viser

    from app import build_app
    from core.schema import TargetGroup
    from core.targets import append_points
    from tests.helpers import confirm_all

    server = viser.ViserServer(host="127.0.0.1", port=8781, verbose=False)
    try:
        state = build_app(server)
        s = state["session"]
        s.project_dir = tmp_path_factory.mktemp("proj")
        state["cell"].part_pose.xyz.value = (900.0, 0.0, 300.0)
        pts = np.array([(x, 0.0, 0.02) for x in (-0.2, 0.0, 0.2)])
        s.targets = s.targets.model_copy(update={"groups": [TargetGroup(name="top", tilt_deg=10.0)]})
        s.targets = append_points(s.targets, pts, np.tile([0, 0, 1.0], (3, 1)), "top")
        state["targets"].refresh()
        confirm_all(state["stations"])
        state["stations"].start(block=True)
        yield s, state["results"].selected()
    finally:
        server.stop()


def test_report_make_images(_planned, tmp_path):
    from core import soft_render
    from ui import report_make as rm

    s, sol = _planned
    png = soft_render.render(__import__("core.golden_parts", fromlist=["x"]).flat_plate())
    out = rm.export_all(s, sol, tmp_path / "a", images=[("Iso view (offline render)", png)])
    html = out["report.html"].read_text(encoding="utf-8")
    assert "Iso view (offline render)" in html and "data:image/png;base64," in html
    assert rm.NO_VIEWS_BATCH not in html
    plain = rm.export_all(s, sol, tmp_path / "b")["report.html"].read_text(encoding="utf-8")
    assert "3D views not included" in plain  # no pictures: the old note stays
