"""Review fixes J: report 'Not checked', CSV comments + empty clearance, stale banner, tolerance
identity, export error reporting, PDF messages, UR notice only for UR robots."""

import csv
import html
import io
import pathlib
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core import export as ex
from core import report as rp
from core.robot import load_spec
from core.schema import Solution, Station
from core.targets import parse_csv


def _sol(label="p1"):
    return Solution(label=label, stations=[Station(xyz=(0, 0, 0))], assignment={1: 0})


def _input(**kw):
    base = dict(
        title="t",
        robot="ur20",
        tool="sander",
        process="sanding",
        base_mode="fixed",
        solution=_sol(),
        rows=[{"status": "reachable", "point_id": 1}],
        station_poses=[np.eye(4)],
    )
    base.update(kw)
    return rp.ReportInput(**base)


# --- 1. Not checked --------------------------------------------------------------------------------


def test_report_has_not_checked_section_and_deviation_line():
    page = rp.build_report(
        _input(
            geometry=[{"name": "part", "mode": "reduce", "kind": "approximate", "deviation_m": 0.004}],
            geometry_warning="Simplified part collision mesh deviates up to 4.0 mm",
        )
    )
    assert "<h2>Not checked</h2>" in page
    for word in ("payload", "torque", "Transit between stops", "sampled"):
        assert word.lower() in page.lower()
    assert "approximate" in page and "4.0 mm" in page
    assert "deviates up to 4.0 mm" in page


def test_csv_comment_rows():
    text = ex.rows_to_csv([{"point_id": 1}], comments=["not checked: payload"])
    assert text.startswith("# not checked: payload\n")


def test_inf_clearance_is_an_empty_cell():
    assert ex.clearance_cell(float("inf")) == ""
    assert ex.clearance_cell(99.999) == ""
    assert ex.clearance_cell(0.0123) == "12.3"


def test_exported_csv_still_parses():
    text = ex.rows_to_csv([{"point_id": 1}], comments=ex.csv_comments(["Part"], None))
    lines = text.splitlines()
    assert any(x.startswith("# not checked") for x in lines) and any("OUT OF DATE: Part" in x for x in lines)
    body = "\n".join(x for x in lines if not x.startswith("#"))
    assert "point_id" in next(csv.DictReader(io.StringIO(body)))
    assert len(parse_csv("# a\n# b\n1,2,3,0,0,1\n")) == 1


# --- 3. stale banner --------------------------------------------------------------------------------


def test_stale_banner_in_report():
    page = rp.build_report(_input(stale=["Part", "Targets"]))
    assert "OUT OF DATE: Part, Targets" in page
    assert "OUT OF DATE" not in rp.build_report(_input())


def _panel(tolerance=None):
    from ui.export_ui import ExportPanel

    p = object.__new__(ExportPanel)
    p.s = SimpleNamespace(tolerance=tolerance, stamps={}, extra={})
    p.selected = lambda: _sol()
    p.info = SimpleNamespace(content="")
    return p


def test_run_notifies_on_unexpected_exception(monkeypatch):
    import ui.export_ui as eu

    msgs = []
    monkeypatch.setattr(eu, "notify", lambda ev, title, body="", ok=False: msgs.append((title, body)))
    monkeypatch.setattr("ui.staleness.solution_changes", lambda s, sol: [])

    def boom(sol, client=None):
        raise RuntimeError("kaput")

    eu.ExportPanel._run(_panel(), SimpleNamespace(client=None), boom)
    assert msgs and msgs[0][0] == "Export failed" and "kaput" in msgs[0][1]


def test_export_anyway_passes_stale_to_writer(monkeypatch):
    import ui.export_ui as eu

    seen = []
    monkeypatch.setattr(eu, "notify", lambda *a, **k: None)
    monkeypatch.setattr("ui.staleness.solution_changes", lambda s, sol: ["Part"])
    p = _panel()

    def fn(sol, client=None):
        seen.append(list(p.stale))
        return pathlib.Path("x.csv"), b""

    eu.ExportPanel._run(p, SimpleNamespace(client=None), fn, confirmed=True)
    assert seen == [["Part"]]


# --- 4. tolerance identity --------------------------------------------------------------------------


def test_tolerance_with_other_stamp_not_shown():
    from ui.export_ui import tolerance_for

    s = SimpleNamespace(stamps={"p1": {"part": "new"}})

    def mk(**k):
        return SimpleNamespace(label="p1", **k)

    assert tolerance_for(s, _sol(), mk(plan_stamp={"part": "new"})) is not None
    assert tolerance_for(s, _sol(), mk(plan_stamp={"part": "old"})) is None
    assert tolerance_for(s, _sol(), mk()) is None  # no stamp recorded: unknown
    assert tolerance_for(s, _sol(), None) is None


def test_heatmap_ignores_stale_tolerance():
    from ui.part_heatmap import point_scores

    tol = SimpleNamespace(label="p1", plan_stamp={"part": "old"}, points={1: SimpleNamespace(rate=0.5)})
    s = SimpleNamespace(tolerance=tol, stamps={"p1": {"part": "new"}}, feasibility={}, max_manip=None)
    sel = {1: SimpleNamespace(status="reachable", q=[0], clearance=1, limit_margin_deg=1, manipulability=1)}
    assert point_scores(s, "Tolerance reach rate", sel, "p1") == {}
    tol.plan_stamp = {"part": "new"}
    assert point_scores(s, "Tolerance reach rate", sel, "p1") == {1: 50.0}


# --- 5. PDF -----------------------------------------------------------------------------------------


def test_pdf_distinguishes_not_found_from_failed(tmp_path, monkeypatch):
    html = tmp_path / "r.html"
    html.write_text("<p>x</p>")
    pdf = tmp_path / "r.pdf"
    real = pathlib.Path.is_file
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: False if self.suffix == ".exe" else real(self))
    with pytest.raises(rp.PdfError, match="[Nn]o Edge"):
        rp.html_to_pdf(html, pdf)

    monkeypatch.setattr("shutil.which", lambda n: "fake.exe" if n == "msedge" else None)
    monkeypatch.setattr(
        pathlib.Path,
        "is_file",
        lambda self: self.name == "fake.exe" or (self.suffix != ".exe" and real(self)),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 3, stdout="", stderr="sandbox failure\nmore"),
    )
    with pytest.raises(rp.PdfError, match="exit code 3.*sandbox failure"):
        rp.html_to_pdf(html, pdf)

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(rp.PdfError, match="timed out"):
        rp.html_to_pdf(html, pdf)


# --- 6. UR notice -----------------------------------------------------------------------------------


def test_ur_notice_only_for_ur_robots():
    for name in ("ur10e", "ur20", "ur30"):
        assert load_spec(name).mesh_license == "ur"
    assert load_spec("gofa10").mesh_license is None
    assert html.escape(rp.UR_NOTICE) in rp.build_report(_input(mesh_license="ur"))
    assert rp.UR_NOTICE not in rp.build_report(_input(mesh_license=None))
