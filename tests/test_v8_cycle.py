"""v8 T6: cycle time (core.cycle_time, planner estimate, built paths)."""

import math

import numpy as np
import pytest
import viser

from core.cycle_time import CycleSettings, joint_move_time, trapezoid
from tests.helpers import confirm_all


def test_trapezoid_by_hand():
    assert trapezoid(0.0, 1.0, 1.0) == 0.0
    # 2 m at 1 m/s, 1 m/s^2: 1 s up (0.5 m), 1 s cruise (1 m), 1 s down (0.5 m) = 3 s
    assert trapezoid(2.0, 1.0, 1.0) == pytest.approx(3.0)
    # 0.5 m never reaches 1 m/s: triangle, 2 * sqrt(0.5 / 1)
    assert trapezoid(0.5, 1.0, 1.0) == pytest.approx(2 * math.sqrt(0.5))
    assert trapezoid(-2.0, 1.0, 1.0) == trapezoid(2.0, 1.0, 1.0)


def test_joint_move_is_the_slowest_joint_capped_by_its_limit():
    cs = CycleSettings(joint_speed=2.0, joint_acc=1.0)
    dq = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 4.0])
    # joint 6 limited to 1 rad/s: 4 rad -> 4 / 1 + 1 / 1 = 5 s; joint 2 at 2 rad/s: 2 * sqrt(2) s
    t = joint_move_time(dq, cs, v_limits=[2, 2, 2, 2, 2, 1.0])
    assert t == pytest.approx(5.0)
    assert joint_move_time(dq[:2], cs) == pytest.approx(2 * math.sqrt(2.0))


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8855, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


@pytest.mark.slow
def test_plan_estimate_cost_term_and_path_cycle_time(app):
    s, stp, sim = app["session"], app["stations"], app["sim"]
    assert app["project"].load_example("rail_long_part")
    confirm_all(stp)
    stp.start(block=True, quick=True)
    sol = s.results[0]
    assert sol.cycle_s and sol.cycle_s > 0
    assert sol.breakdown["cycle"] == 0.0  # weight 0 by default: plans cost what they did before v8
    assert "Cycle time" in app["results"].card.content and "estimate" in app["results"].card.content
    # a slower carrier makes the estimate longer
    app["cycle"].carrier_speed.value = 50.0
    stp.start(block=True, quick=True)
    slow = next(r for r in s.results if len(r.stations) == len(sol.stations))
    if len(sol.stations) > 1:
        assert slow.cycle_s > sol.cycle_s
    app["cycle"].carrier_speed.value = 500.0
    # the cost term: weight per minute
    stp.costs.fields["cycle"].value = 2.0
    stp.start(block=True, quick=True)
    r = s.results[0]
    assert r.breakdown["cycle"] == pytest.approx(2.0 * r.cycle_s / 60, abs=0.05)
    stp.costs.fields["cycle"].value = 0.0
    sim.build(block=True)
    c = sim.cycle
    assert c["total"] == pytest.approx(sum(v for k, v in c.items() if k != "total"))
    assert c["process"] > 0 and c["settle"] == pytest.approx(len(sim.plan.stations))
    assert "Cycle time:" in sim.info.content
