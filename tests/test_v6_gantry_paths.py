"""v6 feedback: a floor X-Y gantry with a bridge (attach "x_axis"), and paths that sand each
point once (moves that would run over another point lift off)."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from core.carrier import Carrier
from core.patterns import pass_over_lifts, point_spacing
from core.schema import Base, CarrierJoints, Pose, Primitive
from ui.project_panel import EXAMPLES


def _gantry(carrier="gantry") -> Base:
    joints = CarrierJoints(x=(-1.0, 1.0), y=(-0.5, 0.5), yaw_deg=(-180, 180) if carrier == "mobile" else None)
    return Base(
        mode="movable",
        carrier=carrier,
        pose=Pose(xyz=(0.0, -1.0, 0.0)),
        bodies=[
            Primitive(name="bridge", shape="box", size=(0.3, 1.5, 0.1), attach="x_axis"),
            Primitive(name="carriage", shape="box", size=(0.4, 0.4, 0.1), pose=Pose(xyz=(0, 0, 0.1))),
        ],
        T_mount=Pose(xyz=(0, 0, 0.2)),
        joints=joints,
    )


def test_bridge_moves_in_x_only_and_its_urdf_link_follows():
    c = Carrier(_gantry())
    for v in ({"x": -0.5, "y": -0.4}, {"x": 0.7, "y": 0.4}):
        T = {b.name: b.T for b in c.moving_bodies(v)}
        assert np.allclose(T["bridge"][:3, 3], (v["x"], -1.0, 0.0))  # stays centred in y
        assert np.allclose(T["carriage"][:3, 3], (v["x"], -1.0 + v["y"], 0.1))
        fp = c.footprint(v)
        assert fp.contains(fp.__class__([(v["x"] - 0.1, -1.7), (v["x"] + 0.1, -1.7), (v["x"], -1.6)]))
        model = pin.buildModelFromXML(c.urdf())
        data = model.createData()
        pin.framesForwardKinematics(model, data, np.array([v["x"], v["y"]]))
        bridge = data.oMf[model.getFrameId("body_bridge")].homogeneous
        assert np.allclose(bridge, T["bridge"], atol=1e-9)
    for kind in ("rail", "mobile"):  # one axis / a mobile base: the bridge rides on the carriage
        c = Carrier(_gantry(kind))
        assert all(p.attach != "x_axis" for p in c.specs)


def test_bridge_collision_body_follows_the_base():
    from core.collision import CollisionScene
    from core.robot import Robot
    from ui.session import Session, load_project

    s = Session()
    load_project(s, EXAMPLES / "floor_xy_rails")
    assert any(p.attach == "x_axis" for p in s.base.bodies)
    sites = s.sites()
    bb, wb = sites.scene_bodies()
    v0 = sites.full({"x": 0.0, "y": 0.0})
    scene = CollisionScene(
        Robot(s.robot.spec), None, None, sites.T(v0), None, base_bodies=bb, world_bodies=wb
    )
    for v in ({"x": -2.0, "y": -0.3}, {"x": 1.5, "y": 0.35}):
        T_wb = sites.T(v)
        scene.set_base(T_wb)
        want = {b.name: b.T for b in sites.c.moving_bodies(v)}
        gid = scene.gm.getGeometryId("carrier_bridge")
        got = np.linalg.inv(scene.T_univ_base) @ scene.gm.geometryObjects[gid].placement.homogeneous
        assert np.allclose(T_wb @ got, want["bridge"], atol=1e-9)


def test_moves_over_another_point_lift():
    P = np.array([[x, y, 0.0] for y in (0.0, 0.1) for x in (0.0, 0.1, 0.2, 0.3)])
    assert point_spacing(P) == pytest.approx(0.1)
    neighbours = P[[0, 1, 2, 3, 7, 6, 5, 4]]  # zigzag: each move to a neighbour
    assert not pass_over_lifts(neighbours, P, 0.1).any()
    jump = P[[0, 3, 1, 2]]  # 0 -> 3 runs over 1 and 2, 3 -> 1 over 2, 1 -> 2 is a neighbour
    assert pass_over_lifts(jump, P, 0.1).tolist() == [True, True, False]
    assert not pass_over_lifts(P[[0, 6]], P, 0.1).any()  # 0.045 m from (0.1, 0): beyond 0.4 x spacing
    diag = P[[0, 7]]  # (0,0) -> (0.3,0.1) passes 0.032 m from (0.1, 0.0)
    assert pass_over_lifts(diag, P, 0.1).tolist() == [True]


def test_ordered_sequences_lift_over_sanded_points():
    from types import SimpleNamespace

    from core.paths import ordered_sequences
    from core.patterns import PatternSettings

    # an L-shaped set: a zigzag's row link from the long row's end runs back over that row
    pts = [(x, 0.0) for x in np.arange(0, 0.5, 0.1)] + [(0.0, 0.1), (0.1, 0.1)]
    points = {i: (np.array([x, y, 0.0]), np.array([0, 0, 1.0])) for i, (x, y) in enumerate(pts)}
    res = {i: SimpleNamespace(q=np.zeros(6), status="ok") for i in points}
    sol = SimpleNamespace(
        stations=[SimpleNamespace()], assignment={i: 0 for i in points}, zones=[], split_stops=None
    )
    for crossing in (False, True):
        seqs = ordered_sequences(sol, res, points, PatternSettings("Zigzag", no_crossing=crossing))
        pids, lifts = seqs[0][0].pids, seqs[0][0].lifts
        P = np.array([points[i][0] for i in pids])
        over = pass_over_lifts(P, np.array([v[0] for v in points.values()]), 0.1)
        assert over.any()  # this order runs over a point at least once
        if crossing:
            assert all(lifts[k] for k in np.flatnonzero(over))
        else:
            assert not any(lifts)
