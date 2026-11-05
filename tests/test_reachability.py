"""M6 checks. Done-when: the heatmap peak lies within 50 mm of the exact best base found by
brute-force grid search on the flat-plate scene.

The map for UR20 + sander (default settings) takes ~30 s to build the first time and is
then loaded from cache/reachability/.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from core.feasibility import candidate_poses
from core.ik import angle_diff, expand_to_limits
from core.ik.ur import URAnalyticIK, solve_batch, ur_manipulability
from core.robot import Robot
from core.schema import TargetGroup
from core.tool import ToolModel, load_tool
from core.transforms import make_T, rpy_to_matrix
from opt.reachability import MapSettings, ReachMap, vote_bases, within_limits

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def robot():
    return Robot.from_registry("ur20")


@pytest.fixture(scope="module")
def ik(robot):
    return URAnalyticIK(robot)


@pytest.fixture(scope="module")
def T_ft():
    ee = load_tool("orbital_sander")
    return ToolModel(ee, REPO / ee.urdf).T_flange_tcp("pad")


@pytest.fixture(scope="module")
def rmap(robot, T_ft):
    return ReachMap.load_or_build(robot, T_ft)


# --- vectorized kinematics ----------------------------------------------------------------


def test_solve_batch_matches_scalar(robot, ik):
    rng = np.random.default_rng(11)
    qs = np.array([robot.random_q(rng) for _ in range(500)])
    Ts = np.array([robot.fk(q) for q in qs])
    Q, V = solve_batch(ik, Ts)
    for k in range(len(Ts)):
        ref = ik.solve(Ts[k])
        got = Q[k][V[k]]
        assert len(got) == len(ref)
        for s in got:
            assert np.min(np.abs(angle_diff(ref, s)).max(axis=1)) < 1e-9
            assert np.allclose(robot.fk(s), Ts[k], atol=1e-8)


def test_closed_form_manipulability(robot, ik):
    rng = np.random.default_rng(12)
    for _ in range(50):
        q = robot.random_q(rng)
        assert np.isclose(ur_manipulability(ik, q), robot.manipulability(q, [0.02, -0.03, 0.3]), atol=1e-9)


def test_within_limits_matches_expand(robot):
    rng = np.random.default_rng(13)
    lo, hi = robot.lower + 0.1, robot.upper - 0.1
    q = rng.uniform(-np.pi, np.pi, (300, 6))
    expected = np.array([len(expand_to_limits(x[None], lo, hi)) > 0 for x in q])
    assert np.array_equal(within_limits(q, lo, hi), expected)


# --- map -------------------------------------------------------------------------------------


def test_map_agrees_with_fk_samples(robot, rmap, T_ft):
    """Poses produced by FK from comfortable joint values must be found reachable."""
    rng = np.random.default_rng(14)
    hits = 0
    n = 300
    for _ in range(n):
        q = rng.uniform(robot.lower + 0.3, robot.upper - 0.3)
        q[2] = rng.uniform(-2.5, 2.5)
        T = robot.fk(q) @ T_ft
        score, _ = rmap.lookup(T[None, :3, 3], T[None, :3, 2], 10.0)
        hits += score[0] > 0
    assert hits >= 0.97 * n  # misses only at the edge of the workspace (cell discretization)


def test_map_rejects_out_of_reach(rmap):
    far = np.array([[3.0, 0.0, 0.5], [0.0, 0.0, 3.5]])
    score, mu = rmap.lookup(far, np.array([[0, 0, -1.0]] * 2), 10.0)
    assert np.all(score == 0) and np.all(mu == 0)


def test_map_is_yaw_symmetric(rmap):
    rng = np.random.default_rng(15)
    p = rng.uniform([-1.2, -1.2, 0.0], [1.2, 1.2, 1.2], (200, 3))
    a = rng.normal(size=(200, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R = rpy_to_matrix((0, 0, 0.9))
    s1, m1 = rmap.lookup(p, a, 10.0)
    s2, m2 = rmap.lookup(p @ R.T, a @ R.T, 10.0)
    assert np.allclose(s1, s2, atol=1e-9) and np.allclose(m1, m2, atol=1e-6)


def test_cache_round_trip(robot, T_ft, tmp_path):
    small = MapSettings(res=0.2, n_dirs=20, n_spins=2)
    t = time.time()
    a = ReachMap.load_or_build(robot, T_ft, small, cache_dir=tmp_path)
    build = time.time() - t
    t = time.time()
    b = ReachMap.load_or_build(robot, T_ft, small, cache_dir=tmp_path)
    assert time.time() - t < build
    assert np.array_equal(a.reach, b.reach) and np.allclose(a.manip, b.manip)
    other = T_ft.copy()
    other[2, 3] += 0.1  # a longer tool is a different map
    assert ReachMap.cache_key(robot, other, small) != ReachMap.cache_key(robot, T_ft, small)


# --- done-when: heatmap peak vs brute force ----------------------------------------------------


def test_heatmap_peak_matches_brute_force(robot, ik, rmap, T_ft):
    # flat plate tilted 20 deg about x: the scene has no mirror symmetry, so the optimum
    # is unique (an untilted plate has two equal optima, one on each side)
    T_part = make_T(rpy_to_matrix((np.radians(20), 0, 0)), (1.3, 0.4, 0.5))
    pts = np.array([(x, y, 0.02) for x in (-0.4, 0, 0.4) for y in (-0.25, 0, 0.25)])
    P = pts @ T_part[:3, :3].T + T_part[:3, 3]
    N = np.tile(T_part[:3, 2], (len(P), 1))
    group = TargetGroup(name="g", tilt_deg=10.0)

    xs, ys = np.arange(-1.0, 3.0, 0.025), np.arange(-1.6, 2.4, 0.025)
    votes = vote_bases(rmap, P, N, 10.0, xs, ys)
    map_best = np.array(votes.best())
    assert votes.coverage.max() == pytest.approx(9.0)

    # exact score: same definition (coverage + mean normalized best manipulability), but from
    # IK over the feasibility checker's tilt/spin samples instead of the map
    lo, hi = robot.lower + np.radians(5), robot.upper - np.radians(5)
    T_tcp_f = np.linalg.inv(T_ft)
    n_part = np.array([0, 0, 1.0])
    poses = [np.array([T for *_, T in candidate_poses(p, n_part, group, ["pad"])]) for p in pts]

    def exact(x: float, y: float) -> float:
        cov = qual = 0.0
        T_base_part = make_T(p=(-x, -y, 0.0)) @ T_part
        for poses_k in poses:  # poses are in the part frame
            q, v = solve_batch(ik, T_base_part @ poses_k @ T_tcp_f)
            v &= within_limits(q, lo, hi)
            if v.any():
                cov += 1
                qual += np.where(v, ur_manipulability(ik, q), 0.0).max() / rmap.max_manip
        return cov + qual / len(pts)

    def argmax(xg, yg):
        S = np.array([[exact(x, y) for x in xg] for y in yg])
        k = np.unravel_index(S.argmax(), S.shape)
        return np.array([xg[k[1]], yg[k[0]]])

    coarse = argmax(np.arange(-1.0, 3.0, 0.15), np.arange(-1.6, 2.4, 0.15))
    fine = argmax(
        np.arange(coarse[0] - 0.2, coarse[0] + 0.201, 0.025),
        np.arange(coarse[1] - 0.2, coarse[1] + 0.201, 0.025),
    )
    # the mirror-image optimum must be clearly worse, or the comparison would be a coin toss
    mirror = np.array([fine[0], 2 * 0.4 - fine[1]])
    assert exact(*mirror) < exact(*fine) - 0.05
    dist = np.linalg.norm(map_best - fine)
    assert dist <= 0.05, f"map peak {map_best} vs brute force {fine}: {dist * 1000:.0f} mm"
