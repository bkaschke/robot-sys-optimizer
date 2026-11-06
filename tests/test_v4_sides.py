"""v3 feedback items 2 and 3: robot sides that may face the part (several), part sides the
mobile base may work from, and a rail / gantry track kept along the part's long axis."""

import numpy as np
import pytest
import viser

from core.standoff import footprint
from core.transforms import make_T, rpy_to_matrix
from opt.contour import ContourSites


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8800, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_several_robot_sides_on_one_contour(app):
    s, car = app["session"], app["carrier"]
    assert app["project"].load_example("mobile_amr")
    for k in car.side_boxes:
        car.side_boxes[k].value = k in ("left", "right")  # side-on: parallel to the part
    assert car.commit() and s.base.robot_sides() == ["left", "right"]
    sites = s.sites()
    assert isinstance(sites, ContourSites) and sites.sides == ["left", "right"]
    t_all = np.linspace(0, sites.bounds[0, 1], 60, endpoint=False)
    assert {sites.side_at({"t": t}) for t in t_all} == {"left", "right"}
    for t in t_all:  # every sample keeps the gap, whichever side faces the part
        v = sites.full({"t": float(t)})
        assert sites.c.footprint(v).distance(sites.zones.part) == pytest.approx(0.4, abs=0.002)
    app["stations"].start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not sol.unreached
    assert {sites.side_at(st.carrier) for st in sol.stations} <= {"left", "right"}


def test_part_sides_restrict_the_stops(app):
    s, car = app["session"], app["carrier"]
    assert app["project"].load_example("mobile_amr")
    for k, box in car.part_boxes.items():
        box.value = k == "-y"
    assert car.commit() and s.base.part_sides == ["-y"]
    sites = s.sites()
    ok = [
        t for t in np.linspace(0, sites.bounds[0, 1], 200, endpoint=False) if sites.accept({"t": float(t)})[0]
    ]
    assert ok and len(ok) < 200  # some of the contour is excluded
    R = s.part.T_world_part[:3, :3]
    part = sites.zones.part
    from shapely.geometry import Point

    for t in ok:
        v = sites.full({"t": float(t)})
        c = np.array([v["x"], v["y"]])
        near = part.exterior.interpolate(part.exterior.project(Point(c)))
        out = c - np.array([near.x, near.y])
        n = R.T @ np.array([out[0], out[1], 0.0])
        assert abs(n[1]) >= abs(n[0]) and n[1] < 0  # on the part's -Y side


def test_rail_track_follows_the_part(app):
    s, car, cell = app["session"], app["carrier"], app["cell"]
    assert app["project"].load_example("rail_long_part")
    T = s.part.T_world_part.copy()
    cell.part_pose._apply(make_T(rpy_to_matrix((0, 0, np.radians(30))), T[:3, 3]), source="fields")
    car.align.value = True
    assert car.commit() and s.base.align_to_part

    def axis_and_centre():
        rect = footprint(s.part_mesh(), s.part.T_world_part).minimum_rotated_rectangle
        c4 = np.asarray(rect.exterior.coords)[:4]
        e = np.roll(c4, -1, axis=0) - c4
        u = e[int(np.argmax(np.linalg.norm(e, axis=1)))]
        return u / np.linalg.norm(u), np.asarray(rect.centroid.coords[0])

    u, centre = axis_and_centre()
    yaw = s.base.pose.rpy[2]
    assert abs(np.cos(yaw) * u[0] + np.sin(yaw) * u[1]) == pytest.approx(1.0, abs=1e-9)  # along the part
    o = np.array(s.base.pose.xyz[:2])
    assert (o - centre) @ u == pytest.approx(0.0, abs=1e-9)  # centred along the part
    # the part moves: the track follows
    T = s.part.T_world_part.copy()
    cell.part_pose._apply(
        make_T(rpy_to_matrix((0, 0, np.radians(-20))), T[:3, 3] + [0.5, 0.2, 0]), source="fields"
    )
    car.realign()  # (the GUI calls this from the part-moved listener)
    u, centre = axis_and_centre()
    yaw = s.base.pose.rpy[2]
    assert abs(np.cos(yaw) * u[0] + np.sin(yaw) * u[1]) == pytest.approx(1.0, abs=1e-9)
    assert (np.array(s.base.pose.xyz[:2]) - centre) @ u == pytest.approx(0.0, abs=1e-9)
