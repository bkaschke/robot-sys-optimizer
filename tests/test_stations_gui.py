"""M7 (+ M9 base mode) through the GUI: plan mobile-base stations for the long rail from the
Optimize tab, then inspect and apply a station from the Results tab; results survive a
project save/load."""

import numpy as np
import pytest
import viser

from ui.session import Session, load_project, save_project


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8769, verbose=False)
    state = build_app(server)
    cell, tab = state["cell"], state["targets"]
    cell.part_choice.value = "long_rail"
    cell.part_pose.xyz.value = (0.0, 800.0, 500.0)
    tab.normal_dir.value = (0.0, 0.0, 1.0)
    tab.normal_angle.value = 5.0
    from core import targets as tg

    tab.select(tg.select_by_normal(tab.mesh(), (0, 0, 1), 5.0), "set")
    tab.method.value = "Projected grid"
    tab.spacing.value = 250.0
    tab.run_generate()
    # several stations need a movable base (static mode = one base): plan-default mobile base
    from ui.carrier_ui import DEFAULTS

    state["carrier"].load(DEFAULTS["mobile"])
    assert state["carrier"].commit()
    st = state["stations"]
    st.max_stations.value = 4
    st.start(block=True)
    yield state
    server.stop()


def test_streamed_solutions_cover_the_rail(app):
    s = app["session"]
    assert s.results, app["stations"].status.content
    best = s.results[0]
    n = len(s.targets.points)
    assert n >= 20
    assert best.unreached == [] and len(best.assignment) == n
    assert 2 <= len(best.stations) <= 4
    assert [r.cost for r in s.results] == sorted(r.cost for r in s.results)


def test_results_tab_shows_and_applies_station(app):
    s, res = app["session"], app["results"]
    assert res.choice.value.startswith("#1")
    assert "station" in res.summary.content and "Cost breakdown" in res.breakdown.content
    k = len(s.results[0].stations)
    assert len(res.station.options) == k
    assert sum(1 for h in res._ghosts if hasattr(h, "update_cfg")) == k  # one ghost robot per station
    res.station.value = res.station.options[-1]
    st = s.results[0].stations[-1]
    from ui.results import station_T

    app["cell"].set_base(station_T(st))
    assert np.allclose(s.T_world_base[:3, 3], st.xyz)


def test_results_saved_with_project(app, tmp_path):
    s = app["session"]
    save_project(s, tmp_path)
    s2 = Session()
    load_project(s2, tmp_path)
    assert [r.label for r in s2.results] == [r.label for r in s.results]
    assert s2.results[0].assignment == s.results[0].assignment


def test_stations_carry_carrier_values_and_respect_standoff(app):
    from core.carrier import Carrier
    from core.standoff import footprint

    s = app["session"]
    c = Carrier(s.base)
    part_fp = footprint(s.part_mesh(), s.part.T_world_part)
    for st in s.results[0].stations:
        assert set(st.carrier) >= {"x", "y", "yaw"}
        assert c.footprint(st.carrier).distance(part_fp) >= s.base.standoff_m.carrier_part_min - 1e-6
