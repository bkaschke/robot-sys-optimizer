"""Check all targets after planning (user report, 2026-11-06): the check runs each point from
its stop in the selected plan by default (it used to check from wherever the robot stood,
reporting half the planned points as unreachable), includes the carrier bodies in the
collision check, and Move robot to this solution moves the carrier to that stop."""

import numpy as np
import pytest
import viser

from tests.helpers import confirm_all
from ui.optimize import CHECK_FROM, INSPECT_SOURCES


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8796, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


@pytest.mark.slow
def test_check_uses_the_plan_stops(app):
    s, opt = app["session"], app["optimize"]
    assert app["project"].load_example("mobile_amr")
    confirm_all(app["stations"])
    app["stations"].start(block=True)
    sol = app["results"].selected()
    assert sol is not None and not sol.unreached
    assert opt.check_from.value == CHECK_FROM[0]
    opt.start(block=True)
    res = list(s.feasibility.values())
    assert len(res) == len(s.targets.points)
    assert all(r.status != "unreachable" for r in res), opt.summary.content
    assert opt.station_of == sol.assignment and "from its stop" in opt.summary.content
    # inspect a point of the last stop: Move robot to this solution brings the carrier there
    k = len(sol.stations) - 1
    pid = next(p for p, st in sol.assignment.items() if st == k)
    opt.source.value = INSPECT_SOURCES[0]
    ids = [r.id for r in opt._results()]
    opt.pick.value = ids.index(pid)
    assert f"from stop **S{k + 1}**" in opt.detail.content
    opt.goto._impl_click = None
    r = s.feasibility[pid]
    opt.go_station(opt.checked_plan, k)
    opt.show_q(r.q)
    c = s.carrier()
    v = app["carrier"].current(c)
    assert all(v[d] == pytest.approx(sol.stations[k].carrier[d], abs=1e-6) for d in ("x", "y", "yaw"))
    assert np.allclose(s.q, r.q)


def test_arrows_step_and_move_the_robot(app):
    s, opt = app["session"], app["optimize"]
    sol = app["results"].selected()
    opt.source.value = INSPECT_SOURCES[1]  # the selected plan
    opt.pick.value = 0
    for _ in range(len(sol.assignment) + 2):  # past the end: stays on the last point
        opt.step(+1)
        r = opt._selected()
        k = sol.assignment[r.id]
        assert np.allclose(s.q, r.q, atol=1e-9)  # the robot is at the point at once
        c = s.carrier()
        v = app["carrier"].current(c)
        assert v["x"] == pytest.approx(sol.stations[k].carrier["x"], abs=1e-6)
    assert int(opt.pick.value) == len(opt._results()) - 1
    opt.step(-1)
    assert int(opt.pick.value) == len(opt._results()) - 2


@pytest.mark.slow
def test_switch_between_plans(app):
    s, res, sim = app["session"], app["results"], app["sim"]
    assert len(s.results) >= 2, "the mobile example offers plans for several stop counts"
    res.choice.value = res.choice.options[0]
    sim.build(block=True)
    assert sim.plan is not None
    first = res.selected()
    res.switch(+1)
    sol = res.selected()
    assert sol is not first and sol is s.results[1]
    assert "**▶**2" in res.compare.content and res.compare.content.count("| # |") == 1
    # the robot moved to the new plan's first stop
    c = s.carrier()
    v = app["carrier"].current(c)
    assert all(v[d] == pytest.approx(sol.stations[0].carrier[d], abs=1e-6) for d in ("x", "y", "yaw"))
    # paths built for the other plan are cleared, not shown for this one
    assert sim.plan is None and not sim._handles and "another plan" in sim.info.content
    # the check follows the selected plan
    app["optimize"].start(block=True)
    assert app["optimize"].checked_plan is sol
    res.switch(-1)
    assert res.selected() is first
    res.switch(-1)
    assert res.selected() is first  # stays on the first one


def test_switching_a_legacy_tool_to_pose_keeps_its_shape(app):
    s, tool, panel = app["session"], app["tool"], app["tool_opt"]
    assert app["project"].load_example("rail_long_part")
    T0 = s.tool_model.T_flange_tcp(s.tool.tcps[0].name).copy()
    assert tool.legacy_note.visible
    panel.update_hint()
    assert panel.legacy_hint.visible  # the optimizer panel points to the pose limits
    tool.ext_type.value = "pose"
    x = s.tool.extension
    assert x.type == "pose" and not tool.legacy_note.visible and tool.ext_ranges["rx_deg"].visible
    assert np.allclose(s.tool_model.T_flange_tcp(s.tool.tcps[0].name), T0, atol=1e-9)  # same shape
    for key in ("x", "y", "z", "rx_deg", "ry_deg", "rz_deg"):
        lo, hi = getattr(x, key)
        assert lo == pytest.approx(hi)  # fixed until widened
    tool.ext_ranges["z"].value = (100.0, 600.0)
    assert s.tool.extension.z == pytest.approx((0.1, 0.6)) and "z" in tool._nominal
    assert "W" in tool.ext_ranges["rx_deg"].label and "R" in tool.ext_ranges["rz_deg"].label
    panel.update_hint()
    assert not panel.legacy_hint.visible


def test_paths_panel_is_open_above_export(app):
    sim, res = app["sim"], app["results"]
    assert sim.folder.order < res.export.folder.order  # shown above Export, not below it


def test_check_from_current_position_is_still_available(app):
    opt = app["optimize"]
    opt.check_from.value = CHECK_FROM[1]
    opt.start(block=True)
    assert "current robot position" in opt.summary.content
    assert opt.checked_plan is None and opt.station_of == {}
    opt.check_from.value = CHECK_FROM[0]


def test_check_without_a_plan_falls_back_to_current_position(app):
    opt = app["optimize"]
    assert app["project"].load_example("rail_long_part")  # no results yet
    opt.start(block=True)
    assert "current robot position" in opt.summary.content and "no plan selected" in opt.summary.content
