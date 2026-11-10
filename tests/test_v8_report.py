"""v8 T19: customer report (sections, PDF) and the URSim check script (dry run)."""

import sys
from pathlib import Path

import pytest
import viser

from tests.e2e import cdp
from tests.helpers import confirm_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8877, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    confirm_all(st["stations"])
    st["stations"].start(block=True, quick=True)
    st["tolerance"].samples.value = 3
    st["tolerance"].start(block=True)
    yield st
    server.stop()


def test_report_has_the_customer_sections(app):
    ex = app["results"].export
    sol = app["results"].selected()
    path, data = ex.export_report(sol)
    page = data.decode()
    for h in ("Summary", "Cycle time", "Tolerance check", "Unreached points", "Stations", "Points"):
        assert f"<h2>{h}</h2>" in page, h
    assert "<script" not in page and "Estimated cycle time" in page


@pytest.mark.skipif(cdp.edge() is None, reason="needs Edge to print the PDF")
def test_report_prints_to_pdf(app):
    ex = app["results"].export
    path, data = ex.export_pdf(app["results"].selected())
    assert path.suffix == ".pdf" and data.startswith(b"%PDF") and len(data) > 5000


def test_ursim_script_dry_run(app, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import ursim_check

    ex = app["results"].export
    csv_path, _ = ex.export_csv(app["results"].selected())
    assert ursim_check.main([str(csv_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    scripts = sorted((csv_path.parent / "ursim").glob("stop_*.script"))
    assert scripts and "points ->" in out
    text = scripts[0].read_text()
    assert text.startswith("def stop_1():") and "movej([d2r(" in text and text.rstrip().endswith("end")
