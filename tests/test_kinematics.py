"""M1 checks: registry, FK, Jacobian, analytic + numeric IK.

Done-when: IK round trip (random q -> FK -> IK -> FK) matches within 1e-6 m on
10,000 samples, and all solution branches are found.
"""

import numpy as np
import pytest

from core.ik import angle_diff, expand_to_limits, make_ik, wrap_to_pi
from core.ik.numeric import NumericIK
from core.ik.ur import URAnalyticIK
from core.robot import Robot, list_robots, load_spec

N_ROUND_TRIP = 10_000


@pytest.fixture(scope="module")
def ur20() -> Robot:
    return Robot.from_registry("ur20")


@pytest.fixture(scope="module")
def ur_ik(ur20) -> URAnalyticIK:
    return URAnalyticIK(ur20)


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(translation error m, rotation error ~rad) between two 4x4 poses.

    ||Ra^T Rb - I||_F / sqrt(2) equals the rotation angle to first order and, unlike
    arccos of the trace, keeps full precision near zero.
    """
    dp = np.linalg.norm(a[:3, 3] - b[:3, 3])
    dr = np.linalg.norm(a[:3, :3].T @ b[:3, :3] - np.eye(3)) / np.sqrt(2)
    return float(dp), float(dr)


# --- registry ------------------------------------------------------------------


def test_registry_lists_ur20():
    assert "ur20" in list_robots()
    spec = load_spec("ur20")
    assert spec.ik == "ur_analytic" and spec.flange_link == "tool0"


def test_unknown_robot_rejected():
    with pytest.raises(FileNotFoundError):
        load_spec("not_a_robot")


def test_model_shape(ur20):
    assert ur20.nq == 6
    assert ur20.joint_names[0] == "shoulder_pan_joint"
    assert np.allclose(ur20.upper, [2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
    assert np.allclose(ur20.lower, -ur20.upper)


# --- FK and Jacobian -------------------------------------------------------------


def test_fk_home_pose(ur20):
    # UR zero pose, DH frame: flange at (a2 + a3, -(d4 + d6), d1 - d5). base_link is the DH
    # frame rotated pi about z, so compare the frame-independent horizontal distance and height.
    s = ur20.spec.dh
    T = ur20.fk(np.zeros(6))
    assert np.isclose(np.hypot(T[0, 3], T[1, 3]), np.hypot(abs(s.a2 + s.a3), s.d4 + s.d6), atol=1e-9)
    assert np.isclose(T[2, 3], s.d1 - s.d5, atol=1e-9)


def test_jacobian_matches_finite_difference(ur20):
    rng = np.random.default_rng(1)
    offset = np.array([0.05, -0.02, 0.3])  # a TCP-like point in the flange frame
    for _ in range(20):
        q = ur20.random_q(rng)
        J = ur20.jacobian(q, offset)
        eps = 1e-7
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            Tp, Tm = ur20.fk(q + dq), ur20.fk(q - dq)
            pp = Tp[:3, :3] @ offset + Tp[:3, 3]
            pm = Tm[:3, :3] @ offset + Tm[:3, 3]
            assert np.allclose((pp - pm) / (2 * eps), J[:3, i], atol=1e-6)
            dR = Tp[:3, :3] @ Tm[:3, :3].T  # small rotation ~ I + [w]x * 2eps
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
            assert np.allclose(w / (2 * eps), J[3:, i], atol=1e-6)


# --- IK helpers ------------------------------------------------------------------


def test_wrap_and_expand():
    assert np.isclose(wrap_to_pi(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(angle_diff(np.pi - 0.01, -np.pi + 0.01), -0.02)
    sols = np.full((1, 6), 0.5)
    hi = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
    out = expand_to_limits(sols, -hi, hi)
    # 0.5 and 0.5 - 2pi fit in +-2pi (5 joints); only 0.5 fits the +-pi elbow
    assert out.shape == (2**5, 6)
    assert np.all(np.isclose(out[:, 2], 0.5))
    out4 = expand_to_limits(np.full((1, 6), 4.0), -hi, hi)  # elbow 4.0 only fits as 4.0 - 2pi
    assert np.allclose(out4[:, 2], 4.0 - 2 * np.pi)
    narrow = expand_to_limits(np.full((1, 6), 3.0), -np.ones(6), np.ones(6))
    assert narrow.shape == (0, 6)  # no 2pi image of 3.0 fits +-1 rad


def test_make_ik_dispatch(ur20):
    assert isinstance(make_ik(ur20), URAnalyticIK)
    assert isinstance(make_ik(ur20, "numeric"), NumericIK)


# --- analytic IK -----------------------------------------------------------------


def test_dh_model_matches_urdf(ur20, ur_ik):
    rng = np.random.default_rng(3)
    for _ in range(200):
        q = ur20.random_q(rng)
        dp, dr = _pose_err(ur_ik.fk(q), ur20.fk(q))
        assert dp < 1e-9 and dr < 1e-9


def test_ur_ik_round_trip_10k(ur20, ur_ik):
    rng = np.random.default_rng(4)
    worst_p = worst_r = 0.0
    found = checked = 0
    for _ in range(N_ROUND_TRIP):
        q = ur20.random_q(rng)
        T = ur20.fk(q)
        sols = ur_ik.solve(T)
        assert len(sols) >= 1
        for s in sols:
            dp, dr = _pose_err(ur20.fk(s), T)
            worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
        if ur_ik.singularity_margin(q) > 1e-3:
            checked += 1
            # 1e-4 rad, not 1e-6: the URDF stores pi/2 as 1.570796327, so it differs from the
            # exact DH model by ~1e-10, which near-singular poses amplify to ~1e-6 rad in joint
            # space (pose error stays < 1e-9 m). Distinct branches are >> 1e-4 apart.
            if np.min(np.abs(angle_diff(sols, q)).max(axis=1)) < 1e-4:
                found += 1
    assert worst_p < 1e-6, f"worst position error {worst_p}"
    assert worst_r < 1e-6, f"worst rotation error {worst_r}"
    assert checked > 0.97 * N_ROUND_TRIP
    assert found == checked, f"original branch missing in {checked - found} of {checked}"


def test_ur_ik_finds_all_branches(ur20, ur_ik):
    """Every solution a many-seed numeric IK can find is in the analytic set."""
    rng = np.random.default_rng(5)
    num = NumericIK(ur20, n_starts=48)
    for _ in range(25):
        q = ur20.random_q(rng)
        T = ur20.fk(q)
        analytic = ur_ik.solve(T)
        numeric = num.solve(T, rng=rng)
        for s in numeric:
            assert np.min(np.abs(angle_diff(analytic, s)).max(axis=1)) < 1e-4
        assert len(analytic) <= 8


def test_ur_ik_generic_pose_has_8_solutions(ur20, ur_ik):
    # a comfortable pose in front of the robot: all 8 branches exist
    q = np.array([0.4, -1.3, 1.2, -1.4, -1.2, 0.3])
    sols = ur_ik.solve(ur20.fk(q))
    assert len(sols) == 8
    assert len({tuple(np.round(wrap_to_pi(s), 6)) for s in sols}) == 8


def test_ur_ik_unreachable_returns_empty(ur20, ur_ik):
    T = np.eye(4)
    T[:3, 3] = [5.0, 0, 0]
    assert ur_ik.solve(T).shape == (0, 6)


# --- numeric IK ------------------------------------------------------------------


def test_numeric_ik_converges(ur20):
    rng = np.random.default_rng(6)
    num = NumericIK(ur20, n_starts=8)
    for _ in range(30):
        q = ur20.random_q(rng)
        T = ur20.fk(q)
        sols = num.solve(T, seed=q + rng.normal(0, 0.2, 6), rng=rng)
        assert len(sols) >= 1
        dp, dr = _pose_err(ur20.fk(sols[0]), T)
        assert dp < 1e-6 and dr < 1e-6
        assert np.all(sols >= ur20.lower - 1e-9) and np.all(sols <= ur20.upper + 1e-9)
