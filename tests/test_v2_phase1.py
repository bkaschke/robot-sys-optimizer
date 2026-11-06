"""v2 phase 1 (docs/V2_PLAN.md): fixes from the user's v1 feedback.

Jog smoothing, carrier jog re-poses instead of re-sending meshes, Move robot to station
moves the carrier, the inspector follows the selected plan, the standoff zones follow a
moved part, examples are read-only templates, length tolerance has no upper limit."""

import shutil
import threading
import time

import numpy as np
import pytest
import viser

from core.transforms import make_T
from ui.widgets import Latest


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8790, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


@pytest.fixture(scope="module")
def planned(app):
    """The rail example, planned once with the panel defaults."""
    assert app["project"].load_example("rail_long_part")
    app["stations"].start(block=True)
    s = app["session"]
    assert s.results and not s.results[0].unreached, app["stations"].status.content
    return app


# --- jog smoothing ------------------------------------------------------------------------------


def test_latest_runs_only_the_newest_pending_call():
    seen = []
    gate = threading.Event()

    def slow(x):
        seen.append(x)
        if x == 0:
            gate.wait(2)

    f = Latest(slow)
    t = threading.Thread(target=f, args=(0,))
    t.start()
    time.sleep(0.05)
    for x in range(1, 50):  # a drag: many events while the first one is still running
        f(x)
    gate.set()
    t.join()
    assert seen == [0, 49]  # intermediate events dropped, the newest one applied
    f(50)  # idle again: runs synchronously
    assert seen[-1] == 50


def test_carrier_jog_moves_meshes_without_re_adding(app):
    assert app["project"].load_example("mobile_amr")
    car, s = app["carrier"], app["session"]
    handles = dict(car._body_handles)
    assert handles
    sl = car._dof["x"]
    sl.value = sl.value + 300.0
    assert car._body_handles == handles  # same scene nodes, only re-posed
    c = s.carrier()
    v = c.clip(car.values())
    assert np.allclose(s.T_world_base, c.T_world_base(v), atol=1e-9)
    body = c.moving_bodies(v)[0]
    assert np.allclose(handles[body.name].position, body.T[:3, 3], atol=1e-9)


def test_set_values_moves_carrier_and_robot_together(app):
    assert app["project"].load_example("mobile_amr")
    car, s = app["carrier"], app["session"]
    car.set_values({"x": 1.2, "y": -0.7, "yaw": np.radians(30)})
    c = s.carrier()
    v = c.clip(car.values())
    assert v["x"] == pytest.approx(1.2) and v["y"] == pytest.approx(-0.7)
    assert np.degrees(v["yaw"]) == pytest.approx(30, abs=1e-6)
    assert np.allclose(s.T_world_base, c.T_world_base(v), atol=1e-9)


def test_zones_follow_a_moved_part(app):
    assert app["project"].load_example("mobile_amr")
    car, cell, s = app["carrier"], app["cell"], app["session"]
    before = car._zones.part.bounds
    T = s.part.T_world_part.copy()
    shift = np.array([0.4, -0.3, 0.0])
    cell.part_pose._apply(make_T(T[:3, :3], T[:3, 3] + shift), source="fields")
    after = car._zones.part.bounds  # the zones were rebuilt around the moved part
    assert np.allclose(np.subtract(after, before), [*shift[:2], *shift[:2]], atol=1e-9)
    # and the readout distance is measured to the moved part
    c = s.carrier()
    from core.standoff import Zones, footprint

    z = Zones(s.base, footprint(s.part_mesh(), s.part.T_world_part))
    d = z.check(c.footprint(c.clip(car.values()))).part_distance
    assert f"**{d * 1000:.0f} mm**" in car.info.content


# --- results -> robot, inspector ----------------------------------------------------------------


def test_move_robot_to_station_moves_carrier(planned):
    s, res, car = planned["session"], planned["results"], planned["carrier"]
    sol = s.results[0]
    k = len(sol.stations) - 1
    res.go_to_station(sol, k)
    c = s.carrier()
    v = c.clip(car.values())
    for d in c.dofs:
        assert v[d] == pytest.approx(sol.stations[k].carrier[d], abs=1e-6)
    assert np.allclose(s.T_world_base, c.T_world_base(v), atol=1e-9)
    # the arm shows the solution of a point that station serves
    rp = s.result_points[sol.label]
    served = [p for p, st in sol.assignment.items() if st == k]
    assert any(np.allclose(s.q, rp[p].q, atol=1e-6) for p in served)


def test_inspector_follows_the_selected_plan(planned):
    opt, s = planned["optimize"], planned["session"]
    from ui.optimize import INSPECT_SOURCES

    assert opt.source.value == INSPECT_SOURCES[1]  # switched by the new plan
    assert len(opt._results()) == len(s.targets.points)
    opt.pick.value = 0
    assert "station **S" in opt.detail.content
    sol = s.results[0]
    r = opt._selected()
    opt.goto._impl_click = None
    opt.move_to_point(sol, r.id)
    assert np.allclose(s.q, r.q, atol=1e-6)


# --- examples are templates -------------------------------------------------------------------


def test_examples_are_read_only_templates(app, tmp_path):
    from ui.project_panel import EXAMPLES, WORKSPACE

    pp, s = app["project"], app["session"]
    assert pp.load_example("rail_long_part")
    first = s.project_dir
    assert pp.load_example("rail_long_part")
    assert s.project_dir == first  # an unchanged copy is reused, not duplicated
    # change the copy: the next open must not replace it
    s.targets = s.targets.model_copy(update={"points": s.targets.points[:3]})
    pp.save_now()
    assert pp.load_example("rail_long_part")
    second = s.project_dir
    assert second != first and len(s.targets.points) > 3
    saved_copy = (first / "project.json").read_text(encoding="utf-8")
    assert '"group"' in saved_copy  # the edited copy is still there
    # saving into examples/ is refused; loading from it opens a working copy instead
    with pytest.raises(ValueError, match="read-only"):
        pp.save_now(EXAMPLES / "rail_long_part")
    assert pp._load(None, EXAMPLES / "rail_long_part")
    assert "workspace" in str(s.project_dir)
    # Save as example writes examples/<name> and lists it
    name = "zz_test_example"
    try:
        path = pp.save_as_example(name)
        assert path.is_file() and name in pp.example.options
        with pytest.raises(ValueError, match="exists"):
            pp.save_as_example(name)
        pp.save_as_example(name, overwrite=True)
        with pytest.raises(ValueError):
            pp.save_as_example("../evil")
    finally:
        shutil.rmtree(EXAMPLES / name, ignore_errors=True)
        pp.example.options = tuple(o for o in pp.example.options if o != name)
        for d in (first, second):
            if d.is_relative_to(WORKSPACE / "examples") and d.name != "rail_long_part":
                shutil.rmtree(d, ignore_errors=True)
        if (first / "project.json").exists():  # restore the unchanged copy for later tests
            shutil.copy(EXAMPLES / "rail_long_part" / "project.json", first / "project.json")


def test_save_as_copies_project_assets(app, tmp_path):
    from ui.session import Session, save_project

    s = Session()
    s.load_robot_from_registry("ur20")
    s.project_dir = tmp_path / "a"
    (s.project_dir / "assets" / "parts").mkdir(parents=True)
    (s.project_dir / "assets" / "parts" / "p.stl").write_text("solid x\nendsolid x\n")
    save_project(s, tmp_path / "b")
    assert (tmp_path / "b" / "assets" / "parts" / "p.stl").is_file()


def test_length_tolerance_has_no_upper_limit(app):
    tol = app["tool_opt"].tol
    tol.value = 250.0
    assert tol.value == 250.0 and tol.max is None
