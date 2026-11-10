"""v8 T14: least-joint-motion point order (core.path_order_gtsp)."""

import numpy as np
import pytest
import viser

from core.cycle_time import CycleSettings
from core.path_order_gtsp import _cost_matrix, joint_order, path_cost
from tests.helpers import confirm_all


def test_joint_order_beats_a_zigzag_and_keeps_the_start():
    # points along a line in joint 1, visited in a zigzag: the best open path is monotone
    q1 = np.array([0.0, 1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5])
    Q = np.zeros((len(q1), 6))
    Q[:, 0] = q1
    o = joint_order(Q)
    assert o[0] == 0 and sorted(o) == list(range(len(q1)))
    C = _cost_matrix(Q, CycleSettings(), None)
    assert path_cost(o, C) < path_cost(list(range(len(q1))), C)
    assert np.all(np.diff(q1[o]) > 0)  # 0, 0.1, ..., 1.0


def test_never_worse_than_the_given_order():
    rng = np.random.default_rng(3)
    for _ in range(5):
        Q = rng.uniform(-1, 1, (12, 6))
        C = _cost_matrix(Q, CycleSettings(), None)
        assert path_cost(joint_order(Q), C) <= path_cost(list(range(12)), C) + 1e-9


@pytest.mark.slow
def test_examples_order_with_less_joint_motion_and_clean_paths():
    from app import build_app
    from core.configs import config_label
    from core.ik import make_ik
    from core.path_order import ordered_sequences
    from core.path_order_gtsp import reorder

    server = viser.ViserServer(host="127.0.0.1", port=8872, verbose=False)
    try:
        app = build_app(server)
        for ex in ("rail_long_part", "mobile_amr", "floor_xy_rails"):
            assert app["project"].load_example(ex)
            s, sim = app["session"], app["sim"]
            confirm_all(app["stations"])
            app["stations"].start(block=True, quick=True)
            sol = s.results[0]
            res = s.result_points[sol.label]
            pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
            ik = make_ik(s.robot)
            label = lambda r, ik=ik: config_label(ik, np.asarray(r.q, float))  # noqa: E731
            seqs = ordered_sequences(sol, res, pts, sim.pattern_settings(), label)
            new = reorder(seqs, res, pts)

            def motion(ss, res=res):
                tot = 0.0
                for k in ss:
                    for q in ss[k]:
                        Q = np.array([res[p].q for p in q.pids])
                        tot += path_cost(list(range(len(Q))), _cost_matrix(Q, CycleSettings(), None))
                return tot

            assert motion(new) <= motion(seqs) + 1e-9, ex
        sim.order.value = "Least joint motion (GTSP)"
        sim.build(block=True)  # the last example's paths with the new order
        assert sim.plan is not None and sim.plan.counts()["motion"] == 0
        sim.order.value = "Pattern order"
    finally:
        server.stop()


def test_gtsp_picks_the_branch_that_saves_motion():
    from core.path_order_branches import solve_gtsp

    # three points; the planned solutions (option 0) alternate between two far-apart arm poses,
    # each point also has a solution near the first one: the route should use those
    a, b = np.zeros(6), np.full(6, 2.0)
    clusters = [[a], [b, a + 0.05], [b, a + 0.1]]
    labels = [["A"], ["B", "A"], ["B", "A"]]
    route, cost = solve_gtsp(clusters, labels, CycleSettings())
    assert route[0] == (0, 0) and sorted(c for c, _ in route) == [0, 1, 2]
    assert all(o == 1 for c, o in route if c > 0)
    _, given = solve_gtsp(clusters[:1], labels[:1], CycleSettings())
    assert cost < 1.0  # tiny moves, no reconfiguration


def test_branch_options_are_the_same_tcp_pose():
    from pathlib import Path

    from core import golden_parts as gp
    from core.collision import CollisionScene
    from core.configs import config_label
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik
    from core.path_order_branches import branch_options
    from core.robot import Robot
    from core.schema import TargetGroup
    from core.tool import ToolModel, load_tool
    from core.transforms import make_T

    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, Path(__file__).resolve().parents[1] / ee.urdf)
    part, T_part = gp.flat_plate(), make_T(p=(0.8, 0.0, 0.4))
    scene = CollisionScene(robot, tm, part, np.eye(4), T_part)
    ch = FeasibilityChecker(robot, make_ik(robot), tm, scene, np.eye(4), T_part)
    r = ch.check_point(0, np.array([0.0, 0.0, 0.02]), np.array([0.0, 0.0, 1.0]), TargetGroup(name="g"))
    assert r.q is not None
    opts = branch_options(ch, r, lambda q: config_label(ch.ik, q))
    assert len(opts) >= 2 and np.allclose(opts[0][1], r.q)
    assert len({lab for lab, _ in opts}) == len(opts)
    T = robot.fk(np.asarray(r.q, float))
    for _, q in opts:
        assert np.allclose(robot.fk(q), T, atol=1e-6)
        assert np.all(q >= ch.lo - 1e-9) and np.all(q <= ch.hi + 1e-9)
        assert not scene.collides(q, r.tcp)
