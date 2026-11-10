"""v8 T17: standard tubes, bend angle steps (inside the search) and STEP export."""

import importlib.util

import numpy as np
import pytest
import viser

from core.schema import EndEffector, Extension, ExtensionValues
from core.step_export import snap_values
from tests.helpers import confirm_all

HAS_CQ = importlib.util.find_spec("cadquery") is not None


def test_snap_values_rounds_inside_the_ranges():
    v = ExtensionValues(theta_deg=37.0, roll_deg=-8.0, theta2_deg=52.4)
    s = snap_values(v, 15.0)
    assert (s.theta_deg, s.roll_deg, s.theta2_deg) == (30.0, -15.0, 45.0)
    s = snap_values(v, 15.0, {"theta_deg": (31.0, 40.0), "roll_deg": (-5.0, 5.0), "theta2_deg": (-90, 90)})
    assert s.theta_deg == 37.0  # no multiple of 15 inside 31..40: kept
    assert s.roll_deg == 0.0 and s.theta2_deg == 45.0
    assert snap_values(v, 0.0) is v


@pytest.mark.skipif(not HAS_CQ, reason="needs CadQuery")
def test_step_round_trip_keeps_the_tube_length(tmp_path):
    import cadquery as cq

    from core.step_export import tool_step

    ee = EndEffector(
        name="probe",
        extension=Extension(type="straight", L=(0.1, 0.5), radius=0.02, nominal=ExtensionValues(L=0.2763)),
        tcps=[{"name": "tip"}],
    )
    p = tool_step(ee, None, tmp_path / "probe.step", tube_od_mm=20.0, wall_mm=2.0)
    bb = cq.importers.importStep(str(p)).val().BoundingBox()
    assert bb.zmax == pytest.approx(276.3, abs=0.1)  # the tube's end, mm
    assert bb.zmin == pytest.approx(-8.0, abs=0.1) and bb.xmax == pytest.approx(31.5, abs=0.1)  # flange plate


@pytest.mark.slow
def test_optimizer_uses_only_bend_steps_and_tube(tmp_path):
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8875, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        tool = app["tool_opt"]
        tool.shape_evals.value = 6
        tool.bend_step.value = "15°"
        tool.tube.value = "30"
        assert tool.tube_od() == 30.0
        confirm_all(tool)
        tool.start(block=True)
        for d in tool.result.designs:
            for k in ("theta_deg", "roll_deg"):
                a = getattr(d.values, k)
                assert np.isclose(a / 15.0, round(a / 15.0)), (k, a)
        if HAS_CQ:
            ex = app["results"].export
            sol = app["results"].selected()
            path, data = ex.export_step(sol)
            assert path.exists() and data.startswith(b"ISO-10303-21")
        tool.bend_step.value = "free"
        tool.tube.value = "tool's own radius"
    finally:
        server.stop()
