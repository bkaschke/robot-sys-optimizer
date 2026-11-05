"""Multi-start damped-least-squares IK on Pinocchio. Works for any URDF chain.

Slower and less complete than analytic IK; use it as a fallback (e.g. rail/gantry +
arm as one chain) or to cross-check analytic solvers.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from core.ik import angle_diff
from core.robot import Robot


class NumericIK:
    def __init__(
        self,
        robot: Robot,
        n_starts: int = 16,
        max_iter: int = 200,
        tol: float = 1e-10,
        damping: float = 1e-6,
    ):
        self.robot = robot
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping

    def _descend(self, target: pin.SE3, q: np.ndarray) -> np.ndarray | None:
        r = self.robot
        lo, hi = r.lower, r.upper
        for _ in range(self.max_iter):
            T = r.fk(q)
            cur = pin.SE3(T[:3, :3], T[:3, 3])
            fMd = cur.actInv(target)  # desired pose seen from the current flange
            err = pin.log6(fMd).vector
            if err @ err < self.tol**2:
                return q
            J = pin.computeFrameJacobian(r.model, r.data, q, r.flange_frame_id, pin.ReferenceFrame.LOCAL)
            J = pin.Jlog6(fMd.inverse()) @ J
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), err)
            step = np.linalg.norm(dq)
            if step > 0.5:  # trust region keeps big early steps from jumping branches
                dq *= 0.5 / step
            q = np.clip(q + dq, lo, hi)
        return None

    def solve(
        self, T_base_flange: np.ndarray, seed=None, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Unique converged solutions within joint limits, nearest to ``seed`` first."""
        rng = rng or np.random.default_rng()
        T = np.asarray(T_base_flange, float)
        target = pin.SE3(T[:3, :3], T[:3, 3])
        starts = [self.robot.random_q(rng) for _ in range(self.n_starts)]
        if seed is not None:
            starts.insert(0, np.clip(np.asarray(seed, float), self.robot.lower, self.robot.upper))
        found: list[np.ndarray] = []
        for q0 in starts:
            q = self._descend(target, q0)
            if q is None:
                continue
            if not any(np.max(np.abs(angle_diff(q, f))) < 1e-6 for f in found):
                found.append(q)
        if not found:
            return np.zeros((0, self.robot.nq))
        sols = np.array(found)
        if seed is not None:
            sols = sols[np.argsort(np.abs(angle_diff(sols, seed)).sum(axis=1))]
        return sols
