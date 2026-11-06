"""v3 feedback round (docs/V3_PLAN.md), quick items: paths clear, mm uploads, double-sided
meshes, gizmo size / always on top."""

import pytest
import viser

from ui.widgets import GizmoStyle


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8797, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_paths_clear_on_new_runs_and_target_changes(app):
    s, sim = app["session"], app["sim"]
    assert app["project"].load_example("rail_long_part")
    app["stations"].start(block=True)
    sim.build(block=True)
    assert sim.plan is not None and sim._handles
    sim.check_targets()
    assert sim.plan is not None  # unchanged targets: kept
    s.targets = s.targets.model_copy(update={"points": s.targets.points[:-1]})
    sim.check_targets()
    assert sim.plan is None and not sim._handles and "targets changed" in sim.info.content
    sim.build(block=True)
    assert sim.plan is not None
    app["stations"].start(block=True)  # a new plan: old paths go
    assert sim.plan is None and "station plan" in sim.info.content
    sim.build(block=True)
    app["tool_opt"].shape_evals.value = 0
    app["tool_opt"].start(block=True)
    assert sim.plan is None and "tool optimizer" in sim.info.content


def test_uploads_default_to_mm(app):
    cell = app["cell"]
    assert cell.upload_units.value == "mm"


def test_meshes_are_double_sided(app):
    cell = app["cell"]
    assert app["project"].load_example("mobile_amr")
    assert cell.part_handle.side == "double"
    assert all(h.side == "double" for h in app["carrier"]._body_handles.values())


def test_gizmo_size_and_on_top(app):
    ob, cell, tool = app["objects"], app["cell"], app["tool"]
    assert GizmoStyle.on_top and cell.base.gizmo.depth_test is False  # default: grabbable through bodies
    ob.gizmo_size.value = 2.0
    assert cell.base.gizmo.scale == pytest.approx(0.6) and tool.mount.gizmo.scale == pytest.approx(0.24)
    ob.gizmo_top.value = False
    assert cell.part_pose.gizmo.depth_test is True
    ob.gizmo_size.value = 1.0
    ob.gizmo_top.value = True
    assert cell.base.gizmo.scale == pytest.approx(0.3) and cell.base.gizmo.depth_test is False
