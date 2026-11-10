"""v4 plan item 5: up to 40 stops with early stopping, editable cost weights (separate costs per
stop, per group and per wrist flip) and must-be-met requirements."""

import time

import pytest
import viser

from core.schema import Optimizer, Weights
from opt.cost import cost_breakdown
from opt.zoning import violations
from tests.helpers import confirm_all


def test_legacy_config_weight_becomes_group_and_flip():
    w = Weights.model_validate({"config": 2.5, "stations": 4.0})
    assert (w.group, w.wrist_flip, w.stations) == (2.5, 2.5, 4.0)
    assert Weights.model_validate({"config": 2.5, "group": 0.5}).group == 0.5
    assert "config" not in Weights().model_dump()
    with pytest.raises(ValueError):
        Optimizer(max_stations=41)


def test_count_costs_are_separate():
    w = Weights(group=2.0, wrist_flip=3.0, stations=5.0)
    cost, terms = cost_breakdown([], 0, [[0, 0], [1, 0], [2, 0]], w, groups=2, wrist_flips=1)
    assert terms["group"] == 4.0 and terms["wrist_flip"] == 3.0 and terms["stations"] == 10.0


def test_violations():
    assert violations(0, [], False, True, False) == []
    assert violations(3, [], False, True, False) == ["3 point(s) unreached"]
    assert violations(3, [], False, False, False) == []
    assert "zoning off" in violations(0, [], False, False, True)[0]


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8806, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


@pytest.mark.slow
def test_thirty_stops_stop_early(app):
    s, stp = app["session"], app["stations"]
    assert app["project"].load_example("rail_long_part")
    assert stp.max_stations.max == 40
    stp.max_stations.value = 30
    t0 = time.time()
    confirm_all(stp)
    stp.start(block=True)
    counts = sorted(len(r.stations) for r in s.results)
    assert counts[0] == 2 and len(counts) <= 1 + stp.costs.fields["stations"].value and max(counts) <= 4
    assert time.time() - t0 < 120  # not 29 full plans
    assert all(not r.violations for r in s.results)


@pytest.mark.slow
def test_costs_panel_feeds_the_plan_and_the_project(app):
    s, stp = app["session"], app["stations"]
    stp.max_stations.value = 3
    stp.costs.fields["stations"].value = 0.0
    stp.costs.fields["group"].value = 2.0
    assert s.extra["optimizer"].weights.stations == 0.0 and s.extra["optimizer"].weights.group == 2.0
    confirm_all(stp)
    stp.start(block=True)
    assert all(r.breakdown["stations"] == 0.0 for r in s.results)
    stp.costs.set_weights(Weights())
    assert stp.costs.fields["stations"].value == 5.0 and s.extra["optimizer"].weights == Weights()
    stp.costs.require_config.value = True
    assert stp.options()["zoning"] is True and app["tool_opt"].planner_options()["require_one_config"]
    stp.costs.require_config.value = False


@pytest.mark.slow
def test_requirement_rejects_plans_that_break_it(app):
    s, stp, r = app["session"], app["stations"], app["results"]
    stp.max_stations.value = 1  # the rail needs two stops: one stop cannot reach everything
    stp.costs.require_all.value = True
    try:
        confirm_all(stp)
        stp.start(block=True)
    finally:
        stp.costs.require_all.value = False
    sol = s.results[0]
    assert sol.unreached and sol.violations == [f"{len(sol.unreached)} point(s) unreached"]
    assert r.choice.options[0].startswith("#1 ✗")
    r.draw()
    assert "must-be-met" in r.summary.content
    stp.max_stations.value = 3
    stp.costs.require_all.value = True
    try:
        confirm_all(stp)
        stp.start(block=True)
    finally:
        stp.costs.require_all.value = False
    assert not s.results[0].violations and not s.results[0].unreached  # ranked first: valid plans
