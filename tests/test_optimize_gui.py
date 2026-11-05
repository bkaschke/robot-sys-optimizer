"""M5 through the GUI: run the feasibility check from the Optimize tab, colour points,
and jog the robot to a reported solution."""

from dataclasses import dataclass

import numpy as np
import pytest
import viser

from ui.targets_ui import STATUS_COLORS


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8768, verbose=False)
    state = build_app(server)
    cell, tab = state["cell"], state["targets"]
    cell.part_pose.xyz.value = (900.0, 0.0, 300.0)  # flat plate, as in the golden scene
    tab.on_part_click(FakeClick((0.9, 0.0, 2.0), (0, 0, -1)))  # select the plate top
    tab.spacing.value = 150.0
    tab.run_generate()
    state["optimize"].start(block=True)
    yield state
    server.stop()


def test_every_point_checked_and_coloured(app):
    s, tab = app["session"], app["targets"]
    ids = [p.id for p in s.targets.points]
    assert sorted(s.feasibility) == sorted(ids) and len(ids) > 10
    assert tab.color_by.value == "feasibility"
    colors = np.asarray(tab._handles["points"].colors)
    expected = np.array([STATUS_COLORS[s.feasibility[i].status] for i in ids])
    assert np.array_equal(colors, expected)
    assert "reachable" in app["optimize"].summary.content


def test_goto_solution_puts_tcp_on_target(app):
    s, opt = app["session"], app["optimize"]
    results = list(s.feasibility.values())
    k = next(i for i, r in enumerate(results) if r.status != "unreachable")
    opt.pick.value = k
    r = opt._selected()
    opt.show_q(r.q)
    pt = next(p for p in s.targets.points if p.id == r.id)
    T_world_tcp = s.T_world_flange() @ s.tool_model.T_flange_tcp(r.tcp)
    p_world = s.part.T_world_part[:3, :3] @ np.array(pt.p) + s.part.T_world_part[:3, 3]
    assert np.allclose(T_world_tcp[:3, 3], p_world, atol=1e-3)  # sliders round to 0.1 deg
    assert "clearance" in opt.detail.content


def test_missing_inputs_reported(app):
    s, opt = app["session"], app["optimize"]
    saved = s.targets
    s.targets = s.targets.model_copy(update={"points": []})
    try:
        before = dict(s.feasibility)
        opt.start(block=True)  # no targets: refuses without crashing or clearing results
        assert s.feasibility == before
    finally:
        s.targets = saved


def test_heatmap_and_move_base(app):
    s, pl = app["session"], app["placement"]
    pl.compute()  # uses the cached map for the sander TCP
    v = pl.votes
    n = len(s.targets.points)
    assert v.coverage.max() == pytest.approx(n)
    assert len(pl._handles) == 2  # heatmap image + best-base marker
    x, y = v.best()
    T = s.T_world_base.copy()
    T[:2, 3] = (x, y)
    app["cell"].set_base(T)
    assert np.allclose(s.T_world_base[:2, 3], (x, y))
    assert np.allclose(app["cell"].base.gizmo.position[:2], (x, y))
