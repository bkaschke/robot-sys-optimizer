"""Inverse kinematics: shared helpers and solver dispatch.

Every solver has the same interface (v7: the planners, maps and paths only use this):

    solve(T_base_flange) -> (k, nq)      joint solutions; analytic ones once per branch,
                                         wrapped to (-pi, pi] (``expand_to_limits`` adds
                                         the 2*pi copies a joint range allows)
    solve_batch(T (N, 4, 4), return_singular=False)
                                     -> q (N, B, nq), valid (N, B)[, singular (N,)]
    manipulability(q (..., nq))      -> |det J|-type measure (any TCP offset)
    config_label(q)                  -> joint-configuration label, e.g. "S+E-W+"
    n_branches                       most configurations a pose can have
    batch                            True when solve_batch is fast (analytic): the
                                     reachability maps and the station planner need it
    base_height                      shoulder height above the base frame (maps)

Solvers: ``ur_analytic`` (core.ik.ur, UR family), ``opw`` (core.ik.opw, industrial arms with
a spherical wrist: FANUC, ABB, KUKA, Yaskawa ...), ``offset_wrist`` (core.ik.offset_wrist,
cobots with an offset wrist: FANUC CRX, ABB GoFa / PoWa; q6 sweep), ``numeric`` (core.ik.numeric, any chain;
slow, no maps).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.robot import Robot


def wrap_to_pi(a):
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2 * np.pi)


def angle_diff(a, b):
    """Shortest signed angle a - b, elementwise, broadcastable."""
    return wrap_to_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def expand_to_limits(sols: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """All 2*pi-shifted copies of each solution that lie within [lower, upper]."""
    sols = np.atleast_2d(np.asarray(sols, dtype=float))
    n = len(lower)
    out = []
    for s in sols:
        options = []
        for j in range(n):
            base = wrap_to_pi(s[j])
            ks = range(
                int(np.floor((lower[j] - base) / (2 * np.pi))) - 1,
                int(np.ceil((upper[j] - base) / (2 * np.pi))) + 2,
            )
            vals = [base + 2 * np.pi * k for k in ks]
            options.append([v for v in vals if lower[j] - tol <= v <= upper[j] + tol])
        out.extend(itertools.product(*options))
    return np.array(out, dtype=float).reshape(-1, n)


def best_copies(
    sols: np.ndarray, lo: np.ndarray, hi: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``expand_to_limits`` + "largest limit margin" pick, per solution row.

    For each row, the 2*pi copy of every joint inside [lo, hi] that is furthest from the
    hard limits [lower, upper]. The margin min_j(min(q_j - lower_j, upper_j - q_j)) is a
    minimum of per-joint terms, so choosing each joint's best copy independently maximizes
    it. Returns (q (m, n), ok (m,)): ok is False where some joint has no copy in range."""
    s = wrap_to_pi(np.atleast_2d(np.asarray(sols, float)))
    mid = (lower + upper) / 2
    kc = np.round((mid - s) / (2 * np.pi))
    cand = s[..., None] + 2 * np.pi * (kc[..., None] + np.array([-1.0, 0.0, 1.0]))  # (m, n, 3)
    inside = (cand >= lo[:, None] - tol) & (cand <= hi[:, None] + tol)
    margin = np.minimum(cand - lower[:, None], upper[:, None] - cand)
    margin = np.where(inside, margin, -np.inf)
    k = np.argmax(margin, axis=-1)
    q = np.take_along_axis(cand, k[..., None], axis=-1)[..., 0]
    ok = np.all(np.take_along_axis(inside, k[..., None], axis=-1)[..., 0], axis=-1)
    return q, ok


def make_ik(robot: Robot, kind: str | None = None):
    """Solver for ``robot`` named by its spec (or ``kind`` to override)."""
    kind = kind or robot.spec.ik
    if kind == "ur_analytic":
        from core.ik.ur import URAnalyticIK

        return URAnalyticIK(robot)
    if kind == "opw":
        from core.ik.opw import OPWIK

        return OPWIK(robot)
    if kind == "offset_wrist":
        from core.ik.offset_wrist import OffsetWristIK

        return OffsetWristIK(robot)
    if kind == "numeric":
        from core.ik.numeric import NumericIK

        return NumericIK(robot)
    raise NotImplementedError(f"IK solver '{kind}' not implemented yet")
