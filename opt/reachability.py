"""Reachability map, inverse reachability and base voting (plan § Solver strategy 1-2).

The map answers "can the TCP be at position p (base frame) with tool axis d?" using IK +
joint limits only (no collision, no torque: the exact check in core/feasibility does that).

Symmetry: for arms whose first joint is a full-turn rotation about base z (all UR arms),
reachability is invariant under rotation about that axis. The map is therefore stored on a
2D (r, z) grid, with tool directions expressed in the *radial frame* (x = radial
direction of p, z = base z). Consequence for placement: votes do not depend on base yaw.

Map arrays: reach[nr, nz, D] (bool, any spin) and manip[nr, nz, D] (best |det J| over
spins and IK branches). Built once per (robot, TCP offset, settings) and cached on disk.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.ik.ur import URAnalyticIK, solve_batch, ur_manipulability
from core.robot import Robot

MAP_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "reachability"


@dataclass(frozen=True)
class MapSettings:
    res: float = 0.05  # m, (r, z) cell size
    n_dirs: int = 600  # tool-axis directions on the sphere (~8 deg apart: a 10 deg cone holds several)
    n_spins: int = 4  # rotations about the tool axis
    joint_margin_deg: float = 5.0


DEFAULT_SETTINGS = MapSettings()


def fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5**0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def _frames_for_dirs(dirs: np.ndarray) -> np.ndarray:
    """Rotation matrices (D,3,3) whose z column is each direction (x chosen stably)."""
    z = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    a = np.where(np.abs(z[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    x = a - (a * z).sum(1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=2)


def within_limits(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where some 2*pi copy of each joint value lies in [lo, hi] (vectorized, all joints)."""
    k_lo = np.ceil((lo - q) / (2 * np.pi))
    k_hi = np.floor((hi - q) / (2 * np.pi))
    return np.all(k_lo <= k_hi, axis=-1)


@dataclass
class ReachMap:
    r0: float
    z0: float
    res: float
    dirs: np.ndarray  # (D, 3) in the radial frame
    reach: np.ndarray  # (nr, nz, D) bool
    manip: np.ndarray  # (nr, nz, D) float32
    max_manip: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.reach.shape[0], self.reach.shape[1]

    # --- build ----------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        robot: Robot,
        T_flange_tcp: np.ndarray,
        settings: MapSettings = DEFAULT_SETTINGS,
        progress: Callable[[float], None] | None = None,
    ) -> ReachMap:
        if robot.spec.ik != "ur_analytic":
            raise NotImplementedError("reachability maps need the UR analytic solver (v1)")
        if robot.upper[0] - robot.lower[0] < 2 * np.pi - 1e-6:
            raise NotImplementedError("map symmetry needs a full-turn first joint")
        ik = URAnalyticIK(robot)
        tcp_len = float(np.linalg.norm(T_flange_tcp[:3, 3]))
        reach = (robot.spec.reach_m or 2.0) + tcp_len + settings.res
        d1 = ik.d1
        r_vals = np.arange(0.0, reach + 1e-9, settings.res)
        z_vals = np.arange(d1 - reach, d1 + reach + 1e-9, settings.res)
        dirs = fibonacci_sphere(settings.n_dirs)
        R_dirs = _frames_for_dirs(dirs)
        spins = np.arange(settings.n_spins) * 2 * np.pi / settings.n_spins
        Rz = np.array([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in spins])
        R_all = np.einsum("dij,sjk->dsik", R_dirs, Rz).reshape(-1, 3, 3)  # (D*S,3,3)
        T_tcp_flange = np.linalg.inv(T_flange_tcp)
        m = np.radians(settings.joint_margin_deg)
        lo, hi = robot.lower + m, robot.upper - m
        nr, nz, D, S = len(r_vals), len(z_vals), len(dirs), len(spins)
        reach_arr = np.zeros((nr, nz, D), bool)
        manip_arr = np.zeros((nr, nz, D), np.float32)
        T = np.zeros((D * S, 4, 4))
        T[:, :3, :3] = R_all
        T[:, 3, 3] = 1.0
        for i, r in enumerate(r_vals):
            for j, z in enumerate(z_vals):
                T[:, :3, 3] = (r, 0.0, z)
                q, valid = solve_batch(ik, T @ T_tcp_flange)
                valid &= within_limits(q, lo, hi)
                mu = np.where(valid, ur_manipulability(ik, q), 0.0).max(axis=1).reshape(D, S)
                ok = valid.any(axis=1).reshape(D, S)
                reach_arr[i, j] = ok.any(axis=1)
                manip_arr[i, j] = mu.max(axis=1)
            if progress is not None:
                progress((i + 1) / nr)
        return cls(0.0, float(z_vals[0]), settings.res, dirs, reach_arr, manip_arr, float(manip_arr.max()))

    # --- cache ------------------------------------------------------------------------------

    @staticmethod
    def cache_key(robot: Robot, T_flange_tcp: np.ndarray, settings: MapSettings) -> str:
        urdf = (robot.spec.folder / robot.spec.urdf).read_bytes()
        payload = {
            "v": MAP_VERSION,
            "robot": robot.spec.name,
            "urdf": hashlib.sha256(urdf).hexdigest(),
            "dh": robot.spec.dh.model_dump() if robot.spec.dh else None,
            "limits": np.round(np.concatenate([robot.lower, robot.upper]), 9).tolist(),
            "tcp": np.round(T_flange_tcp, 6).tolist(),
            "settings": settings.__dict__,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    @classmethod
    def load_or_build(
        cls,
        robot: Robot,
        T_flange_tcp: np.ndarray,
        settings: MapSettings = DEFAULT_SETTINGS,
        progress: Callable[[float], None] | None = None,
        cache_dir: Path = CACHE_DIR,
    ) -> ReachMap:
        path = cache_dir / f"{cls.cache_key(robot, T_flange_tcp, settings)}.npz"
        if path.is_file():
            d = np.load(path)
            return cls(
                float(d["r0"]),
                float(d["z0"]),
                float(d["res"]),
                d["dirs"],
                d["reach"],
                d["manip"],
                float(d["max_manip"]),
            )
        m = cls.build(robot, T_flange_tcp, settings, progress)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, r0=m.r0, z0=m.z0, res=m.res, dirs=m.dirs, reach=m.reach, manip=m.manip, max_manip=m.max_manip
        )
        tmp.replace(path)
        return m

    # --- queries -----------------------------------------------------------------------------

    def lookup(self, p_base: np.ndarray, axis_base: np.ndarray, cone_deg: float = 0.0):
        """Reach score in [0, 1] and best manipulability for TCP positions and tool axes
        (base frame), allowing any tool axis within ``cone_deg`` of the given one.

        The cone is applied around the *actual* axis (map directions inside it, or the
        nearest direction if none is), so the answer is not biased by direction binning.
        Bilinear in (r, z); points off the grid score 0.
        """
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


# --- base voting ------------------------------------------------------------------------------


@dataclass
class VoteGrid:
    xs: np.ndarray
    ys: np.ndarray
    coverage: np.ndarray  # (ny, nx) expected number of reachable targets
    quality: np.ndarray  # (ny, nx) normalized manipulability: worst target (min) or mean
    score: np.ndarray  # coverage + quality (coverage dominates)

    def best(self) -> tuple[float, float]:
        k = np.unravel_index(np.argmax(self.score), self.score.shape)
        return float(self.xs[k[1]]), float(self.ys[k[0]])


def vote_bases(
    rmap: ReachMap,
    P_world: np.ndarray,
    N_world: np.ndarray,
    cone_deg: np.ndarray | float,
    xs: np.ndarray,
    ys: np.ndarray,
    base_z: float = 0.0,
    R_world_base: np.ndarray | None = None,
    quality: str = "min",
    chunk: int = 1024,
) -> VoteGrid:
    """Score every base position (x, y, base_z) by how many targets it can reach.

    Targets: positions P and outward normals N (world); the tool axis is -N. ``cone_deg``
    is the tilt tolerance (scalar or per target). Base orientation is R_world_base
    (upright by default); base yaw does not matter (see module docstring).

    quality "min" (default) scores the worst target's manipulability: it keeps every
    point away from singularities and gives a sharp optimum. "mean" averages, which is
    nearly flat over a wide area so its peak position is poorly defined.
    """
    R = np.eye(3) if R_world_base is None else np.asarray(R_world_base, float)
    P = np.asarray(P_world, float)
    A = -np.asarray(N_world, float) @ R  # tool axes in the base frame (rows: R^T a)
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
    shape = gx.shape
    return VoteGrid(xs, ys, cov.reshape(shape), qual.reshape(shape), (cov + qual).reshape(shape))


def coverage_matrix(
    rmap: ReachMap,
    P_world: np.ndarray,
    N_world: np.ndarray,
    cone_deg: np.ndarray | float,
    bases: np.ndarray,
    R_world_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-base, per-target (score, manipulability), each (B, T). Bases are (B, 3) positions."""
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
