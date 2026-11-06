"""Closed-form IK for the UR family (standard DH, 6R, offset wrist). Up to 8 solutions.

DH chain (theta, d, a, alpha):
    1: q1, d1, 0,  pi/2      4: q4, d4, 0,  pi/2
    2: q2, 0,  a2, 0         5: q5, d5, 0, -pi/2
    3: q3, 0,  a3, 0         6: q6, d6, 0,  0

Derivation used here (frame 1 = after joint 1):
- Frame-5 origin p5 = p - d6*z6 has a constant component d4 along z1:
  p5x*sin(q1) - p5y*cos(q1) = d4  ->  two q1.
- R16 = Rz(phi) Ry(-q5) Rz(q6), phi = q2+q3+q4  ->  cos(q5) = R16[2,2] (two q5), then
  q6 and phi from row/column 2 of R16.
- Frame-4 origin in frame 1 is (a2 c2 + a3 c23, a2 s2 + a3 s23, d4): planar 2R -> two q3.

The constant transforms between URDF frames and DH frames 0/6 are measured from the URDF at
construction, so URDF conventions (base rotated pi, tool0 vs flange) never get hard-coded.
"""

from __future__ import annotations

import numpy as np

from core.ik import wrap_to_pi
from core.robot import Robot

_EPS = 1e-10


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array(
        [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1.0]]
    )


def _inv(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ p
    return Ti


class URAnalyticIK:
    def __init__(self, robot: Robot):
        if robot.spec.dh is None or robot.nq != 6:
            raise ValueError(f"{robot.spec.name}: UR analytic IK needs 6 joints and DH params")
        self.robot = robot
        p = robot.spec.dh
        self.d1, self.a2, self.a3, self.d4, self.d5, self.d6 = p.d1, p.a2, p.a3, p.d4, p.d5, p.d6
        self._alpha = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
        self._d = (self.d1, 0.0, 0.0, self.d4, self.d5, self.d6)
        self._a = (0.0, self.a2, self.a3, 0.0, 0.0, 0.0)
        # T_base_link_dh0 from the URDF; T_dh6_flange closes the loop at q = 0
        q0 = np.zeros(6)
        self.T_b_0 = robot.frame_pose(q0, p.frame)
        self.T_6_f = _inv(self.T_b_0 @ self._dh_fk(q0)) @ robot.fk(q0)
        self.T_0_b = _inv(self.T_b_0)
        self.T_f_6 = _inv(self.T_6_f)

    def _A(self, i: int, q: float) -> np.ndarray:
        return _dh(q, self._d[i], self._a[i], self._alpha[i])

    def _dh_fk(self, q) -> np.ndarray:
        T = np.eye(4)
        for i in range(6):
            T = T @ self._A(i, q[i])
        return T

    def fk(self, q) -> np.ndarray:
        """T_base_flange from the DH model (independent of Pinocchio; used in tests)."""
        return self.T_b_0 @ self._dh_fk(q) @ self.T_6_f

    def singularity_margin(self, q) -> float:
        """0 at a wrist, elbow or shoulder singularity, ~1 far from all of them."""
        q = np.asarray(q, float)
        T06 = self._dh_fk(q)
        p5 = T06[:3, 3] - self.d6 * T06[:3, 2]
        r = np.hypot(p5[0], p5[1])
        shoulder = np.sqrt(max(0.0, 1 - (self.d4 / r) ** 2)) if r > 0 else 0.0
        return float(min(abs(np.sin(q[4])), abs(np.sin(q[2])), shoulder))

    def solve(self, T_base_flange: np.ndarray) -> np.ndarray:
        T = self.T_0_b @ np.asarray(T_base_flange, float) @ self.T_f_6  # T_dh0_dh6
        p, z6 = T[:3, 3], T[:3, 2]
        p5 = p - self.d6 * z6
        r = np.hypot(p5[0], p5[1])
        if r < abs(self.d4) - _EPS:
            return np.zeros((0, 6))
        psi = np.arctan2(p5[1], p5[0])
        asn = np.arcsin(np.clip(self.d4 / r, -1.0, 1.0))
        sols = []
        for q1 in (psi + asn, psi + np.pi - asn):
            T16 = _inv(self._A(0, q1)) @ T
            R = T16[:3, :3]
            c5 = np.clip(R[2, 2], -1.0, 1.0)
            for q5 in (np.arccos(c5), -np.arccos(c5)):
                s5 = np.sin(q5)
                if abs(s5) < _EPS:
                    # wrist singular: only phi -/+ q6 is determined; pick q6 = 0.
                    # c5 = +1: R16 = Rz(phi + q6);  c5 = -1: R16 = Rz(phi - q6) diag(-1, 1, -1)
                    q6 = 0.0
                    if c5 > 0:
                        phi = np.arctan2(R[1, 0], R[0, 0])
                    else:
                        phi = np.arctan2(-R[1, 0], -R[0, 0])
                else:
                    q6 = np.arctan2(-R[2, 1] / s5, R[2, 0] / s5)
                    phi = np.arctan2(-R[1, 2] / s5, -R[0, 2] / s5)
                T14 = T16 @ _inv(self._A(4, q5) @ self._A(5, q6))
                x, y = T14[0, 3], T14[1, 3]
                c3 = (x * x + y * y - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
                if abs(c3) > 1 + 1e-9:
                    continue
                c3 = np.clip(c3, -1.0, 1.0)
                for q3 in (np.arccos(c3), -np.arccos(c3)):
                    q2 = np.arctan2(y, x) - np.arctan2(self.a3 * np.sin(q3), self.a2 + self.a3 * np.cos(q3))
                    q4 = phi - q2 - q3
                    sols.append(wrap_to_pi([q1, q2, q3, q4, q5, q6]))
        if not sols:
            return np.zeros((0, 6))
        return _dedupe(np.array(sols))


def _dedupe(sols: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    keep: list[np.ndarray] = []
    for s in sols:
        if not any(np.max(np.abs(wrap_to_pi(s - k))) < tol for k in keep):
            keep.append(s)
    return np.array(keep)


# --- vectorized solver (reachability maps) ---------------------------------------------------


def _inv_batch(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    Ti = np.zeros_like(T)
    Ti[..., :3, :3] = np.swapaxes(R, -1, -2)
    Ti[..., :3, 3] = -np.einsum("...ji,...j->...i", R, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def _dh_batch(theta: np.ndarray, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2], T[..., 0, 3] = ct, -st * ca, st * sa, a * ct
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st, ct * ca, -ct * sa, a * st
    T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = sa, ca, d
    T[..., 3, 3] = 1.0
    return T


def solve_batch(ik: URAnalyticIK, T_base_flange: np.ndarray, return_singular: bool = False):
    """All 8 branches for N poses at once.

    Returns ``q`` (N, 8, 6) wrapped to (-pi, pi] and ``valid`` (N, 8). Branch index
    b = 4*i1 + 2*i5 + i3 (shoulder, wrist, elbow choice). Wrist-singular poses (|s5| tiny)
    are marked invalid here; use ``ik.solve`` for those few. ``return_singular`` adds a
    (N,) mask of the poses where a branch was dropped only for being wrist-singular.
    """
    T = ik.T_0_b @ np.asarray(T_base_flange, float) @ ik.T_f_6  # (N,4,4) in DH frames
    n = T.shape[0]
    p5 = T[:, :3, 3] - ik.d6 * T[:, :3, 2]
    r = np.hypot(p5[:, 0], p5[:, 1])
    ok_r = r >= abs(ik.d4) + 1e-12
    psi = np.arctan2(p5[:, 1], p5[:, 0])
    asn = np.arcsin(np.clip(ik.d4 / np.where(ok_r, r, 1.0), -1, 1))
    q = np.zeros((n, 8, 6))
    valid = np.zeros((n, 8), bool)
    singular = np.zeros(n, bool)
    for i1, q1 in enumerate((psi + asn, psi + np.pi - asn)):
        T16 = _inv_batch(_dh_batch(q1, ik.d1, 0.0, np.pi / 2)) @ T
        R = T16[:, :3, :3]
        c5 = np.clip(R[:, 2, 2], -1, 1)
        for i5, sgn in enumerate((1.0, -1.0)):
            q5 = sgn * np.arccos(c5)
            s5 = np.sin(q5)
            ok5 = np.abs(s5) > 1e-9
            s5s = np.where(ok5, s5, 1.0)
            q6 = np.arctan2(-R[:, 2, 1] / s5s, R[:, 2, 0] / s5s)
            phi = np.arctan2(-R[:, 1, 2] / s5s, -R[:, 0, 2] / s5s)
            A56 = _dh_batch(q5, ik.d5, 0.0, -np.pi / 2) @ _dh_batch(q6, ik.d6, 0.0, 0.0)
            T14 = T16 @ _inv_batch(A56)
            x, y = T14[:, 0, 3], T14[:, 1, 3]
            c3 = (x * x + y * y - ik.a2**2 - ik.a3**2) / (2 * ik.a2 * ik.a3)
            ok3 = np.abs(c3) <= 1 + 1e-9
            c3 = np.clip(c3, -1, 1)
            for i3, sg3 in enumerate((1.0, -1.0)):
                q3 = sg3 * np.arccos(c3)
                q2 = np.arctan2(y, x) - np.arctan2(ik.a3 * np.sin(q3), ik.a2 + ik.a3 * np.cos(q3))
                q4 = phi - q2 - q3
                b = 4 * i1 + 2 * i5 + i3
                q[:, b] = wrap_to_pi(np.stack([q1, q2, q3, q4, q5, q6], axis=1))
                valid[:, b] = ok_r & ok5 & ok3
            singular |= ok_r & ~ok5
    if return_singular:
        return q, valid, singular
    return q, valid


def ur_manipulability(ik: URAnalyticIK, q: np.ndarray) -> np.ndarray:
    """Yoshikawa measure |det J| of a UR arm, vectorized over (..., 6).

    For a square Jacobian this equals sqrt(det(J J^T)) and is independent of the
    reference point and frame, so it holds for any TCP offset.
    """
    q = np.asarray(q, float)
    q2, q3, q4, q5 = q[..., 1], q[..., 2], q[..., 3], q[..., 4]
    return np.abs(
        ik.a2
        * ik.a3
        * np.sin(q3)
        * np.sin(q5)
        * (ik.a2 * np.cos(q2) + ik.a3 * np.cos(q2 + q3) + ik.d5 * np.sin(q2 + q3 + q4))
    )
