"""Mobile contour with a standoff range (External axis -> *Standoff from* / *Standoff up to*):
the planner picks the gap per stop (DOF ``gap``) within the range."""

import numpy as np
import pytest
import viser

from core.schema import Base
from opt.contour import ContourSites
from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8824, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def _ranged(s, lo=0.4, hi=0.7):
    s.base = s.base.model_copy(update={"facing": "front", "fixed_standoff_m": lo, "max_standoff_m": hi})
    sites = s.sites()
    assert isinstance(sites, ContourSites)
    return sites


def test_largest_standoff_must_not_be_below_the_smallest(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    d = s.base.model_dump()
    d.update(fixed_standoff_m=0.5, max_standoff_m=0.3)
    with pytest.raises(ValueError):
        Base.model_validate(d)


def test_gap_dof_moves_the_base_out_and_keeps_facing(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    fixed = s.sites()
    assert fixed.dofs == ["t"]  # the shipped example: one fixed standoff
    sites = _ranged(s)
    assert sites.dofs == ["t", "gap"] and np.allclose(sites.bounds[1], (0.4, 0.7))
    part = sites.zones.part
    for t in np.linspace(0, sites.bounds[0, 1], 25, endpoint=False):
        for g in (0.4, 0.55, 0.7):
            v = sites.full({"t": float(t), "gap": g})
            assert v["gap"] == g
            assert sites.c.footprint(v).distance(part) == pytest.approx(g, abs=0.01)
            assert v["yaw"] == pytest.approx(sites.full({"t": float(t)})["yaw"])
        v = sites.full({"t": float(t), "gap": 0.7})
        free = sites.table[sites._row(float(t)), 4] > 0.5 and sites.zones.check(sites.c.footprint(v)).ok
        assert sites.accept({"t": float(t), "gap": 0.7})[0] == (free and sites.clear_of_obstacles(v))
    far = sites.full({"t": 0.0, "gap": 0.7})  # 0.1 m beyond the range: refused
    o = sites.table[0, 6:8]
    far.update(x=far["x"] + 0.1 * o[0], y=far["y"] + 0.1 * o[1])
    assert not sites.accept(far)[0]
    cands = sites.candidates(np.zeros((1, 3)), 0.5)
    assert {round(c["gap"], 3) for c in cands} >= {0.4, 0.7}


@pytest.mark.slow
def test_plan_with_a_standoff_range(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    _ranged(s)
    st = app["stations"]
    st.zoning.value = "best per point"
    confirm_all(st)
    st.start(block=True)
    best = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not best.unreached
    sites = s.sites()
    for stn in best.stations:
        assert 0.4 - 1e-9 <= stn.carrier["gap"] <= 0.7 + 1e-9
        gap = sites.c.footprint(stn.carrier).distance(sites.zones.part)
        assert 0.398 <= gap <= 0.702


def test_gui_range_fields(app):
    car, s = app["carrier"], app["session"]
    assert app["project"].load_example("mobile_amr")
    assert car.max_so.visible and car.max_so.value == car.fixed_so.value == 400.0
    car.max_so.value = 700.0
    assert car.commit() and s.base.max_standoff_m == pytest.approx(0.7)
    assert s.sites().dofs == ["t", "gap"]
    car.fixed_so.value, car.max_so.value = 500.0, 300.0  # up to below from: one fixed standoff
    assert car.commit() and s.base.max_standoff_m is None and s.base.fixed_standoff_m == pytest.approx(0.5)
