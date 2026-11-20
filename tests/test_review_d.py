"""Review v8 M1: coverage lookups must stay within a memory budget and give the same answers.

The reference functions below are verbatim copies of the pre-fix implementation (all targets
per 1024-base chunk, full-size temporaries); the fixed code must match them exactly.
"""

import tracemalloc

import numpy as np
import pytest

import opt.reachability as reach
from opt.reachability import (
    PoseOffsetMap,
    ReachMap,
    coverage_matrix,
    fibonacci_sphere,
    vote_bases,
)

# --- reference (pre-fix) implementation -------------------------------------------------------


def ref_lookup(self, p_base, axis_base, cone_deg=0.0):
    p = np.asarray(p_base, float)
    a = np.asarray(axis_base, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    r = np.hypot(p[:, 0], p[:, 1])
    phi = np.arctan2(p[:, 1], p[:, 0])
    c, s = np.cos(-phi), np.sin(-phi)
    a_loc = np.column_stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1], a[:, 2]])
    cos = a_loc @ self.dirs.T  # (M, D)
    in_cone = cos >= np.cos(np.radians(cone_deg))
    in_cone[np.arange(len(p)), np.argmax(cos, axis=1)] = True
    fi = (r - self.r0) / self.res
    fj = (p[:, 2] - self.z0) / self.res
    nr, nz = self.shape
    inside = (fi >= 0) & (fi <= nr - 1) & (fj >= 0) & (fj <= nz - 1)
    fi, fj = np.clip(fi, 0, nr - 1 - 1e-9), np.clip(fj, 0, nz - 1 - 1e-9)
    i0, j0 = fi.astype(int), fj.astype(int)
    ti, tj = fi - i0, fj - j0
    score = np.zeros(len(p))
    mu = np.zeros(len(p))
    for di, dj, w in (
        (0, 0, (1 - ti) * (1 - tj)),
        (1, 0, ti * (1 - tj)),
        (0, 1, (1 - ti) * tj),
        (1, 1, ti * tj),
    ):
        score += w * (self.reach[i0 + di, j0 + dj] & in_cone).any(axis=1)
        mu += w * np.where(in_cone, self.manip[i0 + di, j0 + dj], 0.0).max(axis=1)
    return np.where(inside, score, 0.0), np.where(inside, mu, 0.0)


class RefMap:
    """The old ReachMap.lookup on the same arrays."""

    def __init__(self, m):
        self.m, self.max_manip = m, m.max_manip

    def lookup(self, p, a, cone=0.0):
        return ref_lookup(self.m, p, a, cone)


class RefPoseMap:
    """The old PoseOffsetMap.lookup (on the old ReachMap.lookup)."""

    def __init__(self, m, T_flange_tcp, spins=12):
        self.base = RefMap(m)
        self.max_manip = m.max_manip
        T_tf = np.linalg.inv(np.asarray(T_flange_tcp, float))
        self.t_tf, self.z_tf = T_tf[:3, 3], T_tf[:3, 2]
        self.spins = np.arange(spins) * 2 * np.pi / spins

    def lookup(self, p_base, axis_base, cone_deg=0.0):
        p = np.asarray(p_base, float)
        a = np.asarray(axis_base, float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        ref = np.where(np.abs(a[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        x = ref - (ref * a).sum(1, keepdims=True) * a
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = np.cross(a, x)
        score = np.zeros(len(p))
        mu = np.zeros(len(p))
        for s in self.spins:
            xs, ys = np.cos(s) * x + np.sin(s) * y, -np.sin(s) * x + np.cos(s) * y
            R = np.stack([xs, ys, a], axis=2)
            pf = p + R @ self.t_tf
            af = R @ self.z_tf
            sc, m = self.base.lookup(pf, af, cone_deg)
            score, mu = np.maximum(score, sc), np.maximum(mu, m)
        return score, mu


def ref_coverage_matrix(rmap, P_world, N_world, cone_deg, bases, R_world_base=None):
    R = np.eye(3) if R_world_base is None else np.asarray(R_world_base, float)
    P = np.asarray(P_world, float)
    A = -np.asarray(N_world, float) @ R
    cones = np.broadcast_to(np.asarray(cone_deg, float), (len(P),))
    B, T = len(bases), len(P)
    score = np.zeros((B, T))
    mu = np.zeros((B, T))
    for cone in np.unique(cones):
        cols = np.flatnonzero(cones == cone)
        for s in range(0, B, 1024):
            b = bases[s : s + 1024]
            pb = ((P[cols][None] - b[:, None]) @ R).reshape(-1, 3)
            ab = np.broadcast_to(A[cols][None], (len(b), len(cols), 3)).reshape(-1, 3)
            sc, m = rmap.lookup(pb, ab, cone)
            score[s : s + len(b), cols] = sc.reshape(len(b), -1)
            mu[s : s + len(b), cols] = m.reshape(len(b), -1)
    return score, mu


def ref_vote_bases(rmap, P_world, N_world, cone_deg, xs, ys, base_z=0.0, R_world_base=None, quality="min"):
    chunk = 1024
    R = np.eye(3) if R_world_base is None else np.asarray(R_world_base, float)
    P = np.asarray(P_world, float)
    A = -np.asarray(N_world, float) @ R
    cones = np.broadcast_to(np.asarray(cone_deg, float), (len(P),))
    gx, gy = np.meshgrid(xs, ys)
    bases = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, base_z)])
    cov = np.zeros(len(bases))
    qual = np.zeros(len(bases)) if quality == "mean" else np.full(len(bases), np.inf)
    for cone in np.unique(cones):
        sel = cones == cone
        Pt, At = P[sel], A[sel]
        for s in range(0, len(bases), chunk):
            b = bases[s : s + chunk]
            pb = ((Pt[None, :, :] - b[:, None, :]) @ R).reshape(-1, 3)
            ab = np.broadcast_to(At[None], (len(b), len(At), 3)).reshape(-1, 3)
            sc, mu = rmap.lookup(pb, ab, cone)
            cov[s : s + chunk] += sc.reshape(len(b), -1).sum(1)
            mu_n = (mu / rmap.max_manip).reshape(len(b), -1)
            if quality == "mean":
                qual[s : s + chunk] += mu_n.sum(1)
            else:
                qual[s : s + chunk] = np.minimum(qual[s : s + chunk], mu_n.min(1))
    qual = qual / max(1, len(P)) if quality == "mean" else np.where(np.isfinite(qual), qual, 0.0)
    return cov.reshape(gx.shape), qual.reshape(gx.shape)


# --- fixtures ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rmap():
    """A map of the real shape (45 x 90 x 600, as for the UR20 flange map)."""
    rng = np.random.default_rng(0)
    reach_arr = rng.random((45, 90, 600)) > 0.6
    manip = (rng.random((45, 90, 600)) * reach_arr).astype(np.float32)
    return ReachMap(0.0, -1.5, 0.05, fibonacci_sphere(600), reach_arr, manip, float(manip.max()))


def _targets(rng, T):
    P = rng.uniform(-1.2, 1.2, (T, 3))
    N = rng.normal(size=(T, 3))
    return P, N / np.linalg.norm(N, axis=1, keepdims=True)


def _boundary_queries(m, cone_deg, n=200):
    """Queries in the x-z half plane (phi = 0, so the radial frame is the base frame) whose axis
    sits exactly ``cone_deg`` from a map direction: the borderline cone test."""
    rng = np.random.default_rng(1)
    k = rng.integers(0, len(m.dirs), n)
    d = m.dirs[k]
    perp = np.cross(d, rng.normal(size=(n, 3)))
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    t = np.radians(cone_deg)
    a = np.cos(t) * d + np.sin(t) * perp
    p = np.column_stack([rng.uniform(0.05, 2.0, n), np.zeros(n), rng.uniform(-1.4, 2.8, n)])
    return p, a


# --- equivalence ------------------------------------------------------------------------------


@pytest.mark.parametrize("cone", [0.0, 5.0, 10.0, 25.0])
def test_lookup_identical_to_reference(rmap, cone, monkeypatch):
    rng = np.random.default_rng(2)
    p = rng.uniform(-2.5, 2.5, (5000, 3))
    a = rng.normal(size=(5000, 3))
    pb, ab = _boundary_queries(rmap, cone)
    p, a = np.vstack([p, pb]), np.vstack([a, ab])
    want = ref_lookup(rmap, p, a, cone)
    monkeypatch.setattr(reach, "LOOKUP_ROWS", 777)  # several chunks with a ragged end
    got = rmap.lookup(p, a, cone)
    assert np.array_equal(got[0], want[0])
    assert np.array_equal(got[1], want[1])


def test_pose_offset_map_identical_to_reference(rmap, monkeypatch):
    rng = np.random.default_rng(3)
    T_ft = np.eye(4)
    T_ft[:3, 3] = (0.05, -0.03, 0.25)
    c, s = np.cos(0.4), np.sin(0.4)
    T_ft[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    p = rng.uniform(-2.0, 2.0, (3000, 3))
    a = rng.normal(size=(3000, 3))
    want = RefPoseMap(rmap, T_ft).lookup(p, a, 10.0)
    monkeypatch.setattr(reach, "LOOKUP_ROWS", 1000)
    got = PoseOffsetMap(rmap, T_ft).lookup(p, a, 10.0)
    assert np.array_equal(got[0], want[0])
    assert np.array_equal(got[1], want[1])


def test_coverage_matrix_identical_to_reference(rmap, monkeypatch):
    rng = np.random.default_rng(4)
    P, N = _targets(rng, 120)
    cones = rng.choice([5.0, 10.0, 15.0], 120)
    bases = np.column_stack([rng.uniform(-2, 2, (1500, 2)), rng.uniform(-0.2, 0.5, 1500)])
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    want = ref_coverage_matrix(RefMap(rmap), P, N, cones, bases, R)
    got = coverage_matrix(rmap, P, N, cones, bases, R)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    # small budgets force target chunks (T > budget) and ragged base chunks
    monkeypatch.setattr(reach, "PAIR_BUDGET", 50)
    monkeypatch.setattr(reach, "LOOKUP_ROWS", 33)
    got = coverage_matrix(rmap, P[:40], N[:40], cones[:40], bases[:97], R)
    want = ref_coverage_matrix(RefMap(rmap), P[:40], N[:40], cones[:40], bases[:97], R)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


@pytest.mark.parametrize("quality", ["min", "mean"])
def test_vote_bases_identical_to_reference(rmap, quality, monkeypatch):
    rng = np.random.default_rng(5)
    P, N = _targets(rng, 60)
    xs, ys = np.linspace(-2, 2, 41), np.linspace(-1.5, 1.5, 31)
    want = ref_vote_bases(RefMap(rmap), P, N, 10.0, xs, ys, 0.1, quality=quality)
    got = vote_bases(rmap, P, N, 10.0, xs, ys, 0.1, quality=quality)
    assert np.array_equal(got.coverage, want[0]) and np.array_equal(got.quality, want[1])
    # a budget below T splits the targets: sums may then differ by rounding only
    monkeypatch.setattr(reach, "PAIR_BUDGET", 25)
    got = vote_bases(rmap, P, N, 10.0, xs, ys, 0.1, quality=quality)
    np.testing.assert_allclose(got.coverage, want[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.quality, want[1], rtol=0, atol=1e-12)


# --- memory -----------------------------------------------------------------------------------


def _peak(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_coverage_matrix_memory_bounded(rmap):
    """1024 bases x 200 targets took about 2.1 GB before the fix."""
    rng = np.random.default_rng(6)
    P, N = _targets(rng, 200)
    bases = np.column_stack([rng.uniform(-2, 2, (1024, 2)), np.zeros(1024)])
    peak = _peak(lambda: coverage_matrix(rmap, P, N, 10.0, bases))
    assert peak < 300e6, f"peak {peak / 1e6:.0f} MB"


def test_pose_offset_and_vote_memory_bounded(rmap):
    rng = np.random.default_rng(7)
    P, N = _targets(rng, 200)
    T_ft = np.eye(4)
    T_ft[:3, 3] = (0.05, 0.0, 0.2)
    pm = PoseOffsetMap(rmap, T_ft)
    bases = np.column_stack([rng.uniform(-2, 2, (300, 2)), np.zeros(300)])
    peak = _peak(lambda: coverage_matrix(pm, P, N, 10.0, bases))
    assert peak < 300e6, f"pose map peak {peak / 1e6:.0f} MB"
    xs, ys = np.linspace(-2, 2, 32), np.linspace(-2, 2, 32)  # 1024 bases
    peak = _peak(lambda: vote_bases(rmap, P, N, 10.0, xs, ys))
    assert peak < 300e6, f"vote peak {peak / 1e6:.0f} MB"
