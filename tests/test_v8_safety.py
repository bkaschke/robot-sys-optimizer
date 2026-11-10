"""v8 T18: cable wrap cost, hose pack clearance, reduced-speed zones, reach envelope / fence."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import viser

from core import collision
from core.collision import CollisionScene
from core.cycle_time import CycleSettings, path_time
from core.paths import PathPlan, PathSample, StationPath
from core.robot import Robot
from core.schema import Primitive
from core.stayout import obstacles
from opt.cost import wrap_excess
from tests.helpers import confirm_all


def test_wrap_excess_counts_joint6_turn_beyond_the_limit():
    def r(q6):
        return SimpleNamespace(q=np.array([0, 0, 0, 0, 0, np.radians(q6)]))

    results = {1: r(-200), 2: r(0), 3: r(250), 4: r(10)}
    seqs = {0: [[1, 2, 3]], 1: [[4]]}
    assert wrap_excess(seqs, results, 360.0) == pytest.approx((450 - 360) / 90)
    assert wrap_excess(seqs, results, 720.0) == 0.0


def test_hose_pack_clearance_makes_a_close_arm_collide():
    robot = Robot.from_registry("ur20")
    q = np.radians([0, -90, 0, -90, 0, 0])  # upper arm straight up
    mid = (robot.frame_pose(q, "upper_arm_link")[:3, 3] + robot.frame_pose(q, "forearm_link")[:3, 3]) / 2
    box = trimesh.creation.box((0.1, 0.1, 0.1))

    def scene_at(d):
        T = np.eye(4)
        T[:3, 3] = mid + np.array([0.0, d, 0.0])  # beside the upper arm (its mesh sits at +y)
        return CollisionScene(robot, None, box, np.eye(4), T, floor=False)

    ds = np.linspace(0.2, 0.9, 71)
    hits = [scene_at(d).collides(q) for d in ds]
    assert hits[0] and not hits[-1]
    d_free = ds[hits.index(False)]  # the first clear distance without a hose pack
    old = collision.DRESS_M
    try:
        collision.DRESS_M = 0.1
        assert scene_at(d_free).collides(q)  # with 100 mm of hose pack it is too close
    finally:
        collision.DRESS_M = old
    assert not scene_at(d_free).collides(q)


def test_reduced_speed_zones_slow_the_cycle_and_do_not_block():
    zone = Primitive(shape="box", size=(1, 1, 1), name="slow", speed_pct=25.0)
    hard = Primitive(shape="box", size=(1, 1, 1), name="hard")
    assert [o.name for o in obstacles([zone, hard])] == ["hard"]

    def s(x):
        T = np.eye(4)
        T[0, 3] = x
        return PathSample(0, 0, 1, 2, 0.5, T, "tip", np.zeros(6))

    plan = PathPlan([StationPath(0, np.eye(4), {}, [s(0.0), s(0.1), s(0.2)])], 0.1)
    cs = CycleSettings(settle_s=0.0)
    fast = path_time(plan, cs)
    slow = path_time(plan, cs, slow=lambda smp: 0.25 if smp.T_part_tcp[0, 3] > 0.15 else 1.0)
    assert "reduced" not in fast and slow["reduced"] == pytest.approx(0.1 / cs.process_speed * 3)
    assert slow["total"] == pytest.approx(fast["total"] + slow["reduced"])


@pytest.mark.slow
def test_safety_view_and_wrap_term_on_the_rail():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8876, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        confirm_all(app["stations"])
        app["stations"].start(block=True, quick=True)
        sol = app["session"].results[0]
        assert "wrap" in sol.breakdown
        safety = app["safety"]
        safety.show.value = True
        assert safety._handles and "Suggested fence" in safety.info.content
        safety.show.value = False
        assert not safety._handles
        app["stayouts"].add_shape("box")
        sp = app["stayouts"]
        sp.speed.value = 30.0
        assert app["session"].stayouts[-1].speed_pct == 30.0 and not app["session"].obstacles()
    finally:
        server.stop()
