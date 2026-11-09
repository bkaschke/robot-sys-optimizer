"""v6: a carrier lift is planned per stop; carriage bodies stay put while the lift raises the
robot (and the lift-stage bodies rise with it), in the carrier model and the collision scene."""

import json

import numpy as np
import pytest

from core.carrier import Carrier, lift_levels
from core.golden_parts import tall_wall
from core.schema import Project
from ui.project_panel import EXAMPLES


def _base(name: str):
    return Project.model_validate(json.loads((EXAMPLES / name / "project.json").read_text())).base


def test_lift_levels():
    assert np.allclose(lift_levels(0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0])
    assert len(lift_levels(0.0, 0.1)) == 2 and len(lift_levels(0, 5.0)) == 6
    assert np.allclose(lift_levels(0.3, 0.3), [0.3])


def test_lift_moves_the_robot_and_the_stage_not_the_carriage():
    c = Carrier(_base("mobile_lift"))
    assert c.dofs == ["x", "y", "yaw", "lift"]
    v0 = {"x": 0.5, "y": -1.0, "yaw": 0.3, "lift": 0.0}
    v1 = {**v0, "lift": 0.8}
    dz = c.T_world_base(v1)[:3, 3] - c.T_world_base(v0)[:3, 3]
    assert np.allclose(dz, (0, 0, 0.8))
    b0 = {b.name: b.T for b in c.moving_bodies(v0)}
    b1 = {b.name: b.T for b in c.moving_bodies(v1)}
    assert np.allclose(b0["chassis"], b1["chassis"]) and np.allclose(b0["mast"], b1["mast"])
    assert np.allclose(b1["lift_stage"][:3, 3] - b0["lift_stage"][:3, 3], (0, 0, 0.8))
    assert c.footprint(v0).equals(c.footprint(v1))  # the outline does not depend on the lift
    assert c.travel(v0, v1) == pytest.approx(0.8)  # lift-only move
    lifts = {round(s["lift"], 3) for s in c.samples(0.5)}
    assert lifts == {0.0, 0.25, 0.5, 0.75, 1.0}
    urdf = c.urdf()  # the stage hangs on the lift joint, the chassis on the carriage
    assert '<joint name="body_lift_stage_joint" type="fixed"><parent link="lift_link"/>' in urdf
    assert '<joint name="body_chassis_joint" type="fixed"><parent link="carriage"/>' in urdf


def test_sites_plan_the_lift_and_the_scene_keeps_the_chassis_down():
    from core.collision import CollisionScene
    from core.robot import Robot
    from opt.contour import ContourSites

    base = _base("mobile_lift")
    c = Carrier(base)
    sites = ContourSites(c, tall_wall(), np.eye(4))
    assert sites.dofs == ["t", "lift"] and np.allclose(sites.bounds[1], (0, 1))
    cand = sites.candidates(None, 0.5)
    assert {round(v["lift"], 2) for v in cand} == {0.0, 0.25, 0.5, 0.75, 1.0}
    f = sites.full({"t": 1.0})
    assert f["lift"] == pytest.approx(0.5)  # no lift given: mid height
    assert sites.full({"t": 1.0, "lift": 0.9})["lift"] == pytest.approx(0.9)
    assert sites.start(np.eye(4))["lift"] == pytest.approx(0.5)

    robot = Robot.from_registry("ur20")
    bb, wb = sites.scene_bodies()
    follow = {e[0]: e[3] for e in bb if len(e) == 4}
    assert set(follow) >= {"chassis", "mast"} and "lift_stage" not in follow
    scene = CollisionScene(robot, None, None, sites.T(f), None, base_bodies=bb, world_bodies=wb)
    for lift in (0.0, 1.0):
        v = sites.full({"t": 1.0, "lift": lift})
        T_wb = sites.T(v)
        scene.set_base(T_wb)
        want = {b.name: b.T for b in c.moving_bodies(v)}
        for name in ("chassis", "lift_stage"):
            gid = scene.gm.getGeometryId(f"carrier_{name}")
            T_ub = scene.T_univ_base
            got = np.linalg.inv(T_ub) @ scene.gm.geometryObjects[gid].placement.homogeneous
            assert np.allclose(T_wb @ got, want[name], atol=1e-9), (name, lift)


def test_transit_moves_the_lift_with_the_carrier():
    from opt.contour import ContourSites

    c = Carrier(_base("mobile_lift"))
    sites = ContourSites(c, tall_wall(), np.eye(4))
    a, b = {"t": 1.0, "lift": 0.0}, {"t": 1.6, "lift": 1.0}
    moves = sites.transit(a, b, 0.05)
    lifts = [v["lift"] for v, _ in moves]
    assert lifts[-1] == pytest.approx(1.0) and np.all(np.diff(lifts) > 0)
    assert all(ok for _, ok in moves)
    T = sites.T(moves[-1][0])
    assert np.allclose(T, sites.T(b))


def test_gantry_examples_have_two_axes():
    c = Carrier(_base("gantry_overhead"))
    assert c.dofs == ["x", "y"]
    T = c.T_world_base({"x": 0.0, "y": 0.0})
    assert T[2, 2] == pytest.approx(-1.0) and 3.0 < T[2, 3] < 4.4  # upside down, above the wall
    floor = Carrier(_base("floor_xy_rails"))
    assert floor.dofs == ["x", "y"]
    assert floor.T_world_base({"x": 0.0, "y": 0.0})[2, 2] == pytest.approx(1.0)  # upright
    for g in (c, floor):  # each has a bridge on the X axis: it moves in x, stays centred in y
        T0 = {b.name: b.T for b in g.moving_bodies({"x": 0.0, "y": 0.0})}["bridge"]
        T1 = {b.name: b.T for b in g.moving_bodies({"x": 1.0, "y": 0.3})}["bridge"]
        assert np.allclose(T1[:3, 3] - T0[:3, 3], g.T_frame[:3, 0])
