"""v8 T7: collision checks between path samples (core.motion_check)."""

import numpy as np
import pytest
import trimesh
import viser

from core.collision import CollisionScene
from core.motion_check import check_moves
from core.paths import PathPlan, PathSample, StationPath
from core.robot import Robot
from tests.helpers import confirm_all


class _Checker:
    """Just what check_moves uses: scene, base pose, set_base."""

    def __init__(self, scene):
        self.scene, self.T_world_base = scene, np.eye(4)

    def set_base(self, T):
        self.T_world_base = np.asarray(T, float)
        self.scene.set_base(T)


def _sample(q, a, b):
    return PathSample(0, None, a, b, 0.0, np.eye(4), "tcp", np.asarray(q, float), kind="home")


def test_a_sweep_through_a_wall_between_two_clear_poses_is_flagged():
    robot = Robot.from_registry("ur20")
    wall = trimesh.creation.box((0.6, 0.05, 1.0))  # thin wall on the +y side, within reach
    T_wall = np.eye(4)
    T_wall[:3, 3] = (0.0, 0.9, 0.6)
    scene = CollisionScene(robot, None, wall, np.eye(4), T_wall, floor=False)
    arm = [-1.2, 1.0, -1.4, -1.57, 0.0]  # reaching out sideways at ~0.6 m height
    qa, qb = np.array([0.0, *arm]), np.array([np.pi, *arm])  # base turns half a turn
    assert not scene.collides(qa) and not scene.collides(qb)
    mid = [qa + (qb - qa) * f for f in np.linspace(0, 1, 50)]
    assert any(scene.collides(q) for q in mid)  # the way goes through the wall
    plan = PathPlan([StationPath(0, np.eye(4), {}, [_sample(qa, 1, 1), _sample(qb, 1, 2)])], 0.01)
    assert check_moves(_Checker(scene), plan) == [1]
    assert plan.samples[1].issue == "motion" and plan.counts()["motion"] == 1
    # the other way round (base turning back through -y) is clear
    plan2 = PathPlan(
        [StationPath(0, np.eye(4), {}, [_sample(qa, 1, 1), _sample(qa - (qb - qa), 1, 2)])], 0.01
    )
    assert check_moves(_Checker(scene), plan2) == []


@pytest.mark.slow
def test_example_paths_pass_the_motion_check():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8856, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        confirm_all(app["stations"])
        app["stations"].start(block=True, quick=True)
        sim = app["sim"]
        assert sim.motion.value
        sim.build(block=True)
        assert sim.plan is not None and sim.plan.counts()["motion"] == 0
    finally:
        server.stop()
