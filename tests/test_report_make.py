"""ui.report_make: the batch runner's report / CSV / userframes equal the Export panel's."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.report import PdfError
from core.schema import TargetGroup
from core.targets import append_points
from tests.helpers import confirm_all

ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str) -> str:
    """Drop timestamps (dates / clock times) so two renders of one result compare equal."""
    return re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "<time>", text)


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8779, verbose=False)
    try:
        state = build_app(server)
        s, cell, tab = state["session"], state["cell"], state["targets"]
        s.project_dir = tmp_path_factory.mktemp("proj")
        cell.part_pose.xyz.value = (900.0, 0.0, 300.0)
        pts = np.array([(x, 0.0, 0.02) for x in (-0.2, 0.0, 0.2)])
        s.targets = s.targets.model_copy(update={"groups": [TargetGroup(name="top", tilt_deg=10.0)]})
        s.targets = append_points(s.targets, pts, np.tile([0, 0, 1.0], (3, 1)), "top")
        tab.refresh()
        confirm_all(state["stations"])
        state["stations"].start(block=True)
        sol = state["results"].selected()
        assert sol is not None
        yield s, state["results"].export, sol
    finally:
        server.stop()


def test_html_matches_export_panel(planned):
    s, panel, sol = planned
    from ui import report_make as rm

    _, data = panel.export_report(sol)
    assert _norm(data.decode()) == _norm(rm.make_report(s, sol))


def test_stale_banner_is_the_same(planned):
    s, panel, sol = planned
    from ui import report_make as rm

    panel.stale = ["Part"]
    try:
        _, data = panel.export_report(sol)
    finally:
        panel.stale = []
    html = rm.make_report(s, sol, stale=["Part"])
    assert "OUT OF DATE: Part" in html and _norm(data.decode()) == _norm(html)


def test_csv_and_userframes_match_export_panel(planned):
    s, panel, sol = planned
    from ui import report_make as rm

    _, csv_data = panel.export_csv(sol)
    assert csv_data.decode() == rm.make_csv(s, sol)
    _, uf = panel.export_userframes(sol)
    assert uf.decode() == rm.make_userframes(s, sol)


def test_export_all_writes_the_panel_files(planned, tmp_path):
    s, panel, sol = planned
    from ui import report_make as rm

    out = rm.export_all(s, sol, tmp_path / "export")
    assert set(out) == {"tcp_poses.csv", "userframes.csv", "report.html"}
    for name, path in out.items():
        assert isinstance(path, Path) and path.name == name and path.stat().st_size > 100
    assert out["tcp_poses.csv"].read_bytes() == panel.export_csv(sol)[1]
    assert out["userframes.csv"].read_bytes() == panel.export_userframes(sol)[1]


def test_export_all_pdf_failure_is_recorded(planned, tmp_path, monkeypatch):
    s, _, sol = planned
    from ui import report_make as rm

    def boom(html, pdf, timeout=90.0):
        raise PdfError("no browser found")

    monkeypatch.setattr("core.report.html_to_pdf", boom)
    out = rm.export_all(s, sol, tmp_path, pdf=True)
    assert isinstance(out["report.html"], Path)
    assert isinstance(out["report.pdf"], str) and "no browser found" in out["report.pdf"]


def test_one_failing_file_does_not_stop_the_others(planned, tmp_path, monkeypatch):
    s, _, sol = planned
    from ui import report_make as rm

    monkeypatch.setattr(rm, "make_userframes", lambda *a: (_ for _ in ()).throw(ValueError("needs a part")))
    out = rm.export_all(s, sol, tmp_path)
    assert "needs a part" in out["userframes.csv"] and isinstance(out["report.html"], Path)


def test_module_import_does_not_pull_in_viser():
    code = "import sys, ui.report_make; sys.exit(1 if 'viser' in sys.modules else 0)"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
