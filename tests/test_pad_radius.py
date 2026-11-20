"""Tool pad radius: schema, Tool tab field, Detect warning, batch, and staleness."""

from __future__ import annotations

import json

import numpy as np
import pytest
import viser

from core.schema import EndEffector
from core.tool import load_tool
from tests.test_tool_surface_pick import _fillet_trough
from ui import staleness
from ui.batch_materialize import cell_pad_radius
from ui.session import PartState
from ui.tool_surface_ui import pad_radius_m


def _ee(**kw) -> EndEffector:
    base = load_tool("orbital_sander").model_dump(mode="python")
    base.pop("pad_radius_m", None)
    return EndEffector.model_validate({**base, **kw})


def test_schema_round_trip_and_old_files_load():
    ee = _ee(pad_radius_m=0.025)
    assert EndEffector.model_validate_json(ee.model_dump_json()).pad_radius_m == 0.025
    old = json.loads(_ee().model_dump_json())
    old.pop("pad_radius_m")
    assert EndEffector.model_validate(old).pad_radius_m is None
    with pytest.raises(ValueError):
        _ee(pad_radius_m=-0.01)


def test_pad_radius_m_helper():
    class S:
        tool = None

    s = S()
    assert pad_radius_m(s) is None
    s.tool = _ee()
    assert pad_radius_m(s) is None
    s.tool = _ee(pad_radius_m=0.025)
    assert pad_radius_m(s) == 0.025


def test_cell_pad_radius_reads_project_json(tmp_path):
    from core import batch as bt

    m = bt.BatchManifest(cell_project="cell")
    (tmp_path / "cell").mkdir()
    assert cell_pad_radius(m, tmp_path) is None  # no file
    (tmp_path / "cell" / "project.json").write_text(json.dumps({"end_effector": {"pad_radius_m": 0.03}}))
    assert cell_pad_radius(m, tmp_path) == 0.03
    (tmp_path / "cell" / "project.json").write_text("{}")
    assert cell_pad_radius(m, tmp_path) is None


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8875, verbose=False)
    state = build_app(server)
    yield state
    server.stop()


def test_gui_field_writes_value_and_does_not_stale(app):
    tool, s = app["tool"], app["session"]
    s.tool = _ee()
    tool.refresh()
    before = staleness.components(s)["tool"]
    tool.pad_radius.value = 25.0
    assert s.tool.pad_radius_m == pytest.approx(0.025)
    assert staleness.components(s)["tool"] == before  # does not mark plans out of date
    tool.pad_radius.value = 0.0
    assert s.tool.pad_radius_m is None
    tool.pad_radius.value = 25.0
    tool.refresh()
    assert tool.pad_radius.value == pytest.approx(25.0)


def _detect_warnings(app, mesh):
    s = app["session"]
    path = s.project_dir / "assets" / "parts" / "fillet.stl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    s.part = PartState(s.relative(path), "m", np.eye(4))
    app["targets"].on_part_changed()
    p = app["targets"].detect
    assert p.detect(block=True)
    return p.warn.content if p.warn.visible else ""


def test_detect_warns_only_with_pad_radius(app):
    tool, s = app["tool"], app["session"]
    s.tool = _ee()
    tool.refresh()
    mesh = _fillet_trough(radius=0.005)
    assert "pad radius" not in _detect_warnings(app, mesh)
    tool.pad_radius.value = 25.0
    assert "smaller than the pad radius 25 mm" in _detect_warnings(app, mesh)
