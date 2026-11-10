"""Closed-form IK for industrial 6R arms with an ortho-parallel base and a spherical wrist
(OPW: FANUC, ABB, KUKA, Yaskawa, Staubli and most other 6-axis arms). Up to 8 solutions.

Geometry (Brandstoetter, Angerer, Hofbaur 2014), in the solver's "theta" angles:
    shoulder at (a1, b, c1) after Rz(theta1); upper arm c2 along z after Ry(theta2);
    the wrist centre sits at (a2, 0, c3) after Ry(theta2 + theta3); the wrist is
    Rz(theta4) Ry(theta5) Rz(theta6) and the flange is c4 further along the last z.
Joint values: q_i = (theta_i + offset_i) * sign_i (``spec.yaml`` -> ``opw``), the same
convention as ROS-Industrial's opw_kinematics parameter files.

The URDF's ``base_link`` is the OPW base frame; the constant flange offset is measured from
the URDF at construction, and the parameters are checked against the URDF on random poses
(a wrong parameter file fails loudly instead of giving wrong joint values).

Configuration labels and manipulability use the three factors of |det J|: the wrist
centre's horizontal reach x1 (shoulder), sin(theta3 + psi3) (elbow), sin(theta5) (wrist),
as for UR arms (core.ik.ur), so labels never change without passing a singularity.
"""

from __future__ import annotations

import numpy as np

from core.ik import wrap_to_pi
from core.robot import Robot

_EPS = 1e-9


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


class OPWIK:
    n_branches = 8
    batch = True

    def __init__(self, robot: Robot, check: bool = True):
        p = robot.spec.opw
        if p is None or robot.nq != 6:
            raise ValueError(f"{robot.spec.name}: OPW IK needs 6 joints and 'opw' parameters in spec.yaml")
        self.robot = robot
        self.a1, self.a2, self.b = p.a1, p.a2, p.b
        self.c1, self.c2, self.c3, self.c4 = p.c1, p.c2, p.c3, p.c4
        self.offsets = np.asarray(p.offsets, float)
        self.signs = np.asarray(p.sign_corrections, float)
        self.k = float(np.hypot(self.a2, self.c3))
        self.psi3 = float(np.arctan2(self.a2, self.c3))
        self.base_height = self.c1  # shoulder height above base_link (reachability maps)
        self.T_6_f = np.linalg.inv(self._fk_opw(np.zeros(6))) @ robot.fk(np.zeros(6))
        self.T_f_6 = np.linalg.inv(self.T_6_f)
        if check:
            self.check()

    # --- forward -------------------------------------------------------------------------

    def theta(self, q) -> np.ndarray:
        return np.asarray(q, float) * self.signs - self.offsets

    def joints(self, th) -> np.ndarray:
        return (np.asarray(th, float) + self.offsets) * self.signs

    def _fk_opw(self, q) -> np.ndarray:
        """T_base_e (OPW flange frame) for joint values q (6,)."""
        t = self.theta(q)
        x1 = self.a1 + self.c2 * np.sin(t[1]) + self.k * np.sin(t[1] + t[2] + self.psi3)
        z1 = self.c2 * np.cos(t[1]) + self.k * np.cos(t[1] + t[2] + self.psi3)
        R1 = _rz(t[0])
        C = R1 @ np.array([x1, self.b, z1]) + np.array([0.0, 0.0, self.c1])
        R = R1 @ _ry(t[1] + t[2]) @ _rz(t[3]) @ _ry(t[4]) @ _rz(t[5])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, C + self.c4 * R[:, 2]
        return T

    def fk(self, q) -> np.ndarray:
        return self._fk_opw(q) @ self.T_6_f

    def check(self, n: int = 20, tol: float = 1e-6) -> None:
        """Raise ValueError when the parameters do not reproduce the URDF's forward kinematics."""
        rng = np.random.default_rng(0)
        for _ in range(n):
            q = self.robot.random_q(rng)
            err = np.abs(self.fk(q) - self.robot.fk(q)).max()
            if err > tol:
                raise ValueError(
                    f"{self.robot.spec.name}: the 'opw' parameters in spec.yaml do not match the URDF "
                    f"(flange pose differs by {err * 1000:.1f} mm at q = "
                    f"{np.degrees(q).round(1).tolist()} deg). "
                    "Check a1 a2 b c1..c4, the offsets and sign_corrections."
                )

    # --- inverse -------------------------------------------------------------------------

    def solve_batch(self, T_base_flange: np.ndarray, return_singular: bool = False):
        """All 8 branches for N poses: q (N, 8, 6) wrapped to (-pi, pi], valid (N, 8)
        [, singular (N,)]. Branch b = 4*i1 + 2*i3 + i5 (shoulder, elbow, wrist). Wrist-singular
        poses are invalid here and flagged; ``solve`` handles them."""
        T = np.asarray(T_base_flange, float) @ self.T_f_6
        N = T.shape[0]
        R = T[:, :3, :3]
        C = T[:, :3, 3] - self.c4 * R[:, :, 2]
        rho2 = C[:, 0] ** 2 + C[:, 1] ** 2
        ok_r = rho2 >= self.b**2
        xr = np.sqrt(np.maximum(rho2 - self.b**2, 0.0))
        q = np.zeros((N, 8, 6))
        valid = np.zeros((N, 8), bool)
        singular = np.zeros(N, bool)
        for i1, x1 in enumerate((xr, -xr)):
            t1 = np.arctan2(C[:, 1], C[:, 0]) - np.arctan2(self.b, x1)
            u, w = x1 - self.a1, C[:, 2] - self.c1
            cg = (u * u + w * w - self.c2**2 - self.k**2) / (2 * self.c2 * self.k)
            ok_g = np.abs(cg) <= 1 + 1e-9
            beta = np.arctan2(u, w)
            for i3, sg in enumerate((1.0, -1.0)):
                g = sg * np.arccos(np.clip(cg, -1, 1))  # theta3 + psi3
                t2 = beta - np.arctan2(self.k * np.sin(g), self.c2 + self.k * np.cos(g))
                t3 = g - self.psi3
                Rc = _rz(t1) @ _ry(t2 + t3)
                Rw = np.swapaxes(Rc, -1, -2) @ R  # Rz(t4) Ry(t5) Rz(t6)
                s5 = np.hypot(Rw[:, 0, 2], Rw[:, 1, 2])
                ok5 = s5 > 1e-9
                singular |= ok_r & ok_g & ~ok5
                for i5, sw in enumerate((1.0, -1.0)):
                    t5 = np.arctan2(sw * s5, Rw[:, 2, 2])
                    t4 = np.arctan2(sw * Rw[:, 1, 2], sw * Rw[:, 0, 2])
                    t6 = np.arctan2(sw * Rw[:, 2, 1], -sw * Rw[:, 2, 0])
                    th = np.stack([t1, t2, t3, t4, t5, t6], axis=1)
                    b = 4 * i1 + 2 * i3 + i5
                    q[:, b] = wrap_to_pi(self.joints(th))
                    valid[:, b] = ok_r & ok_g & ok5
        if return_singular:
            return q, valid, singular
        return q, valid

    def solve(self, T_base_flange: np.ndarray) -> np.ndarray:
        """Unique solutions (k, 6) in (-pi, pi], wrist-singular poses included (theta4 = 0)."""
        T = np.asarray(T_base_flange, float)[None]
        Q, V, sing = self.solve_batch(T[0][None], return_singular=True)
        sols = list(Q[0][V[0]])
        if sing[0]:
            sols += self._singular(T[0])
        out: list[np.ndarray] = []
        for s in sols:
            if not any(np.max(np.abs(wrap_to_pi(s - o))) < 1e-9 for o in out):
                out.append(s)
        return np.array(out).reshape(-1, 6)

    def _singular(self, T_base_flange: np.ndarray) -> list[np.ndarray]:
        """Wrist-singular solutions (theta5 = 0 or pi): only theta4 +- theta6 is defined;
        theta4 = 0 is chosen."""
        T = T_base_flange @ self.T_f_6
        R = T[:3, :3]
        C = T[:3, 3] - self.c4 * R[:, 2]
        rho2 = C[0] ** 2 + C[1] ** 2
        if rho2 < self.b**2:
            return []
        xr = np.sqrt(rho2 - self.b**2)
        out = []
        for x1 in (xr, -xr):
            t1 = np.arctan2(C[1], C[0]) - np.arctan2(self.b, x1)
            u, w = x1 - self.a1, C[2] - self.c1
            cg = (u * u + w * w - self.c2**2 - self.k**2) / (2 * self.c2 * self.k)
            if abs(cg) > 1 + 1e-9:
                continue
            beta = np.arctan2(u, w)
            for sg in (1.0, -1.0):
                g = sg * np.arccos(np.clip(cg, -1, 1))
                t2 = beta - np.arctan2(self.k * np.sin(g), self.c2 + self.k * np.cos(g))
                t3 = g - self.psi3
                Rw = (_rz(t1) @ _ry(t2 + t3)).T @ R
                t5 = 0.0 if Rw[2, 2] > 0 else np.pi
                M = _ry(t5).T @ Rw  # = Rz(t6)
                t6 = np.arctan2(M[1, 0], M[0, 0])
                out.append(wrap_to_pi(self.joints([t1, t2, t3, 0.0, t5, t6])))
        return out

    # --- quality ---------------------------------------------------------------------------

    def _factors(self, q):
        t = np.asarray(q, float) * self.signs - self.offsets
        x1 = self.a1 + self.c2 * np.sin(t[..., 1]) + self.k * np.sin(t[..., 1] + t[..., 2] + self.psi3)
        return x1, np.sin(t[..., 2] + self.psi3), np.sin(t[..., 4])

    def manipulability(self, q) -> np.ndarray:
        """|det J| (vectorized over (..., 6)); independent of the TCP offset."""
        x1, e, w = self._factors(q)
        return np.abs(self.c2 * self.k * x1 * e * w)

    def config_label(self, q) -> str:
        x1, e, w = self._factors(np.asarray(q, float))
        return "".join(f"{k}{'+' if v >= 0 else '-'}" for k, v in (("S", x1), ("E", e), ("W", w)))
