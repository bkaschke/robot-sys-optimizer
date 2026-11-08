"""v5 testing feedback: the mobile base drives around stay-out zones between stops, keeping
its standoff from them (opt/navigate.py, Sites.route)."""

import numpy as np
import pytest
import viser
from shapely.geometry import Point

from core.schema import Pose, Primitive
from opt.navigate import resample, route


def test_route_goes_around_a_disc():
    disc = Point(0, 0).buffer(0.5)

    def ok(x, y, yaw):  # a 0.2 m square base, 0.1 m clearance from the disc
        sq = Point(x, y).buffer(0.1, cap_style=3)
        return sq.distance(disc) >= 0.1

    path = route(ok, (-1.5, 0.0, 0.0), (1.5, 0.0, np.pi / 2), (-2, -2, 2, 2), radius=0.15)
    assert path is not None and len(path) > 2  # not the straight line through the disc
    pts = resample(path, 0.02, 0.15)
    assert all(ok(*p) for p in pts)
    assert np.allclose(pts[-1], (1.5, 0.0, np.pi / 2))
    length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:], strict=False))
    assert length < 3.0 + 1.5  # close to the shortest way round (~3.6 m)
    blocked = lambda x, y, yaw: x < 0.9 or x > 1.1  # noqa: E731  wall across the floor
    assert route(blocked, (-1, 0, 0), (1.5, 0, 0), (-2, -2, 2, 2), radius=0.15) is None


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8811, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_mobile_base_drives_around_a_stayout_zone(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    app["stations"].start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert len(sol.stations) >= 2
    a, b = sol.stations[0].carrier, sol.stations[1].carrier
    direct = s.sites().transit(a, b, 0.05)
    mid = direct[len(direct) // 2][0]
    s.stayouts = [
        Primitive(
            shape="box",
            name="pillar",
            attach="fixed",
            size=(0.3, 0.3, 1.5),
            pose=Pose(xyz=(mid["x"], mid["y"], 0.0)),
        )
    ]
    try:
        sites = s.sites()
        assert not all(sites.transit_ok(v, ()) for v, _ in direct)  # the old move hits it
        moves = sites.transit(a, b, 0.05)
        assert moves and all(ok for _, ok in moves)
        end = sites.full(b)
        assert np.allclose([moves[-1][0][k] for k in ("x", "y")], [end["x"], end["y"]], atol=1e-6)
        nav = sites.nav_clearance
        assert nav == pytest.approx(s.base.fixed_standoff_m)  # the contour standoff
        zone = sites.obstacles[0].footprint
        A = sites.full(a)
        for v, _ in moves:
            near = min(np.hypot(v["x"] - e["x"], v["y"] - e["y"]) for e in (A, end)) < nav
            if not near:
                assert sites.c.footprint(sites.full(v)).distance(zone) >= nav - 1e-6
        # the simulation's carrier path uses the same route
        path = app["sim"]._carrier_path(sol, 0.05)(0, 1)
        assert len(path) == len(moves) and all(ok for *_, ok in path)
    finally:
        s.stayouts = []


# --- home position (v5 testing feedback) ------------------------------------------------------


def test_home_from_jog_typed_values_and_saved_with_the_project(app, tmp_path):
    from ui.session import Session, load_project, save_project

    s, cell = app["session"], app["cell"]
    hp = cell.home_panel
    assert app["project"].load_example("rail_long_part")
    assert s.home_deg is None and np.allclose(s.home_q(), s.robot.home_q())  # spec default
    q = np.radians([10, -80, 70, -100, -90, 5])
    cell.set_q(q)  # jog
    assert hp.take_current()  # "Set home to current pose"
    assert np.allclose(s.home_q(), q, atol=1e-6) and "saved in this project" in hp.info.content
    hp.joints.value = "0, -90, 90, -90, -90, 0"
    assert hp.apply_text() and np.allclose(s.home_deg, [0, -90, 90, -90, -90, 0])
    hp.joints.value = "0, -90, oops"
    assert not hp.apply_text() and hp.joints.value.startswith("0, -90, 90")  # reset, unchanged
    hp.joints.value = "0, -90, 90, -90, -90, 9999"
    assert hp.apply_text() and s.home_deg[5] == pytest.approx(np.degrees(s.robot.upper[5]))  # clipped
    hp.set_home([0, -90, 90, -90, -90, 0])
    cell.set_q(np.zeros(6))
    cell.go_home()
    assert np.allclose(s.q, np.radians([0, -90, 90, -90, -90, 0]))
    d = save_project(s, tmp_path / "p")
    s2 = Session()
    load_project(s2, d.parent if d.is_file() else d)
    assert s2.home_deg == [0, -90, 90, -90, -90, 0]
    hp.set_home(None)
    assert s.home_deg is None and "robot default" in hp.info.content
