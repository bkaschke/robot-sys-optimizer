"""FANUC CRX-10iA / CRX-20iA/L / CRX-30iA and ABB GoFa 10 / GoFa 12 / PoWa 30 in the registry,
with the offset-wrist IK (core.ik.offset_wrist: q6 sweep, up to 16 solutions)."""

import numpy as np
import pytest

from core.ik import angle_diff, make_ik
from core.robot import Robot, list_robots

COBOTS = ("crx10ia", "crx20ia_l", "crx30ia", "gofa10", "gofa12", "powa30")


def _has(sols, q) -> bool:
    return any(np.abs(angle_diff(s, q)).max() < 1e-5 for s in sols)


@pytest.mark.parametrize("name", COBOTS)
def test_cobot_ik_round_trip(name):
    r = Robot.from_registry(name)
    ik = make_ik(r)
    assert name in list_robots() and r.spec.ik == "offset_wrist" and ik.batch
    rng = np.random.default_rng(3)
    Qs = np.array([r.random_q(rng) for _ in range(300)])
    T = np.array([r.fk(q) for q in Qs])
    Q, V = ik.solve_batch(T)
    assert Q.shape == (300, 16, 6) and V.any(axis=1).all()
    for i in range(len(T)):  # every solution reproduces the pose
        for s in Q[i][V[i]]:
            assert np.abs(r.fk(s) - T[i]).max() < 1e-6
    mu = ik.manipulability(Qs)
    found = np.array([_has(Q[i][V[i]], q) for i, q in enumerate(Qs)])
    assert found[mu > 0.2 * np.median(mu)].all()  # only near-singular poses may be missed
    assert found.mean() > 0.98
    assert np.allclose(mu[:5], [r.manipulability(q, np.zeros(3)) for q in Qs[:5]], rtol=1e-6)


def test_offset_wrist_matches_on_the_ur20_and_the_numpy_fallback():
    """The UR20 meets the structure too (J4 meets J5): same solutions as its closed form; the
    numpy fallback agrees with the compiled kernel."""
    r = Robot.from_registry("ur20")
    ur, ow = make_ik(r), make_ik(r, "offset_wrist")
    q = np.radians([20, -80, 70, -60, 80, 30])
    T = r.fk(q)
    assert _has(ow.solve(T), q)
    assert all(_has(ow.solve(T), s) for s in ur.solve(T))
    gofa = Robot.from_registry("gofa12")
    ik = make_ik(gofa)
    rng = np.random.default_rng(0)
    T = np.array([gofa.fk(gofa.random_q(rng)) for _ in range(40)])
    Qc, Vc = ik.solve_batch(T, compiled=True)
    Qn, Vn = ik.solve_batch(T, compiled=False)
    same = [all(_has(Qc[i][Vc[i]], s) for s in Qn[i][Vn[i]]) for i in range(len(T))]
    assert np.mean(same) > 0.9


def test_structure_is_checked():
    """An arm without intersecting J4 / J5 axes is refused with a clear message."""
    r = Robot.from_registry("generic_6r")  # spherical wrist: J4 meets J5 -> accepted
    make_ik(r, "offset_wrist")
    from core.ik.offset_wrist import OffsetWristIK

    ik = OffsetWristIK(r)
    ik.p = ik.p.copy()
    ik.p[4] = ik.p[4] + 0.05 * np.cross(ik.a[3], ik.a[4]) / np.linalg.norm(np.cross(ik.a[3], ik.a[4]))
    with pytest.raises(ValueError, match="joint 4 and joint 5"):
        ik._check_structure()


def test_fanuc_base_joint_counts_as_a_full_turn_for_maps():
    from opt.reachability import map_limits

    r = Robot.from_registry("crx20ia_l")  # J1 +-179.9 deg
    lo, hi = map_limits(r)
    assert lo[0] <= -np.pi and hi[0] >= np.pi
    assert r.lower[0] > -np.pi  # the exact checks keep the real limit


def test_cobot_specs_and_home_poses():
    from core.collision import CollisionScene

    for name in COBOTS:
        r = Robot.from_registry(name)
        assert r.spec.reach_m and r.within_limits(r.home_q())
        T = r.fk(r.home_q())
        assert T[2, 2] < -0.99  # home: tool axis pointing down
        assert not CollisionScene(r, None, None, np.eye(4), None).collides(r.home_q())
    assert Robot.from_registry("crx10ia").spec.frame_convention == "fanuc"
