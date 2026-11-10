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
