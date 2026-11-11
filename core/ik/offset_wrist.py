"""IK for 6R arms with an offset (non-spherical) wrist: FANUC CRX, ABB GoFa / PoWa, ...

Needed geometry (checked from the URDF at q = 0): joint 1 vertical, joint 2 horizontal,
joint 3 parallel to joint 2, and the joint 4 and joint 5 axes intersecting (point W). Any
shoulder / elbow offsets, any J5-J6 offset and any flange offset are allowed. Such arms have
up to 16 solutions and no closed form, so the solver sweeps q6:

- For a given q6 the point W is known (it is fixed in link 5, which turns about the J6 axis
  relative to the flange): W(q6) = B (c0 + c1 cos q6 + c2 sin q6), B = T_target M^-1.
- W depends on q1..q3 only (it lies on the J4 axis): two q1 (lateral offset), then a planar
  2R (two elbows). That is four closed-form branches.
- The J5 axis must keep its fixed angle to the J4 axis: f(q6) = a4(q1..q3) . a5(q6) - c45.
  Its roots are bracketed on a grid of q6 and refined by bisection (all vectorized).
- q4 and q5 then follow from two single-axis rotation subproblems.

Every solution is checked by forward kinematics (1e-6). Roots closer together than the grid
step (near-singular poses) can be missed; ``solve`` uses a finer grid than ``solve_batch``.
Everything is product of exponentials in the base frame, measured from the URDF, so no URDF
convention is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from core.ik import wrap_to_pi
from core.robot import Robot

TOL_GEOM = 1e-6  # m / rad: the URDF must meet the structure to this
TOL_FK = 1e-6  # m (and rad-ish on the rotation matrix): accepted solutions
B = 16  # most solutions a pose can have

try:  # the compiled kernel (about 50x faster); the numpy code below is the fallback
    from core.ik import offset_wrist_nb as _kernel
except ImportError:  # pragma: no cover - numba missing
    _kernel = None


def _rot(k: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Rotation matrices about unit axis k (3,) by angles a (...,) -> (..., 3, 3)."""
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.asarray(a, float)[..., None, None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def _angle_about(k: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle about unit axis k (3,) turning u onto v (..., 3) (components along k ignored)."""
    up = u - (u @ k)[..., None] * k if u.ndim > 1 else u - (u @ k) * k
    vp = v - (v @ k)[..., None] * k
    return np.arctan2(np.cross(up, vp) @ k, np.sum(up * vp, axis=-1))


class OffsetWristIK:
    n_branches = 8  # distinct labels S±E±W± (two solutions can share one)
    batch = True

    def __init__(self, robot: Robot, grid: int = 64, fine_grid: int = 720):
        self._packed = None
        if robot.nq != 6:
            raise ValueError(f"{robot.spec.name}: offset-wrist IK needs 6 joints")
        self.robot, self.grid, self.fine_grid = robot, grid, fine_grid
        m, d = robot.model, robot.data
        q0 = np.zeros(6)
        pin.forwardKinematics(m, d, q0)
        pin.updateFramePlacements(m, d)
        T_wb = d.oMf[robot.base_frame_id].homogeneous
        Rbw, pbw = T_wb[:3, :3].T, -T_wb[:3, :3].T @ T_wb[:3, 3]
        J = pin.computeJointJacobians(m, d, q0)  # world frame: columns (v, w), v = p x w
        if any(m.joints[j].nq != 1 for j in range(1, m.njoints)):
            raise ValueError(f"{robot.spec.name}: offset-wrist IK needs six revolute joints")
        self.a = np.zeros((6, 3))  # joint axes (base frame, q = 0)
        self.p = np.zeros((6, 3))  # a point on each axis (foot from the base origin)
        for i in range(6):
            col = J[:, m.joints[i + 1].idx_v]
            w = col[3:] / np.linalg.norm(col[3:])
            pw = np.cross(w, col[:3])  # point on the axis (world)
            self.a[i] = Rbw @ w
            self.p[i] = Rbw @ pw + pbw
        self.M = robot.fk(q0)  # T_base_flange at q = 0
        self.Minv = np.linalg.inv(self.M)
        self._check_structure()
        a, p = self.a, self.p
        z = np.array([0.0, 0.0, 1.0])
        self.s1 = float(np.sign(a[0] @ z))
        self.e = a[1] / np.linalg.norm(a[1])
        self.s3 = float(np.sign(a[2] @ self.e))
        self.r = np.cross(z, self.e)  # in-plane horizontal; rotation about e: (r, z) as complex
        self.W0 = self._intersection(p[3], a[3], p[4], a[4])
        self.d_lat = float(self.W0 @ self.e)
        self.p2 = p[1]
        self.base_height = float(p[1][2])
        L1 = self._c(p[2])
        self.L1, self.L2 = L1, self._c(self.W0) - L1
        self._aL = (abs(self.L1), abs(self.L2))
        self._gam = float(np.angle(self.L2) - np.angle(self.L1))
        self.c45 = float(a[3] @ a[4])
        self.a4c = complex(a[3] @ self.r, a[3] @ z)
        self.a4e = float(a[3] @ self.e)
        self.th_e = float(np.arctan2(self.e[1], self.e[0]))
        # W and a5 for q6 = t in the q = 0 frame: X(t) = c0 + c1 cos t + c2 sin t (rotation -t)
        k, v = a[5], self.W0 - p[5]
        self.cW = (p[5] + k * (k @ v), v - k * (k @ v), -np.cross(k, v))
        u = a[4]
        self.cA = (k * (k @ u), u - k * (k @ u), -np.cross(k, u))
        perp = a[3] - self.c45 * a[4]
        self.u5 = perp / np.linalg.norm(perp)  # a direction normal to the J5 axis
        self.fwd = self.r * (1.0 if self.W0 @ self.r >= 0 else -1.0)  # towards W at q = 0

    # --- setup -----------------------------------------------------------------------------

    def _check_structure(self) -> None:
        a, p, n = self.a, self.p, self.robot.spec.name
        z = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(a[0], z)) > TOL_GEOM or np.linalg.norm(p[0][:2]) > TOL_GEOM:
            raise ValueError(f"{n}: joint 1 must be the vertical axis through the base origin")
        if abs(a[1] @ z) > TOL_GEOM:
            raise ValueError(f"{n}: joint 2 must be horizontal")
        if np.linalg.norm(np.cross(a[1], a[2])) > TOL_GEOM:
            raise ValueError(f"{n}: joints 2 and 3 must be parallel")
        if np.linalg.norm(np.cross(a[3], a[4])) < 1e-3:
            raise ValueError(f"{n}: joints 4 and 5 must not be parallel")
        if self._line_distance(p[3], a[3], p[4], a[4]) > TOL_GEOM:
            raise ValueError(f"{n}: the joint 4 and joint 5 axes must intersect")

    @staticmethod
    def _line_distance(p1, a1, p2, a2) -> float:
        n = np.cross(a1, a2)
        return abs((p2 - p1) @ n) / np.linalg.norm(n)

    @staticmethod
    def _intersection(p1, a1, p2, a2) -> np.ndarray:
        """Closest point between two (intersecting) lines."""
        A = np.array([[a1 @ a1, -(a1 @ a2)], [a1 @ a2, -(a2 @ a2)]])
        s, t = np.linalg.solve(A, [(p2 - p1) @ a1, (p2 - p1) @ a2])
        return (p1 + s * a1 + p2 + t * a2) / 2

    def _c(self, X: np.ndarray) -> complex:
        d = X - self.p2
        return complex(d @ self.r, d[2])

    # --- the sweep -------------------------------------------------------------------------

    S1 = np.array([1.0, 1.0, -1.0, -1.0])  # branch b: shoulder sign S1[b], elbow sign S3[b]
    S3 = np.array([1.0, -1.0, 1.0, -1.0])

    def _branch(self, W: np.ndarray, A5: np.ndarray, sgn1, sgn3):
        """q1..q3 and f for wrist points W (..., 3), J5 axes A5 (..., 3) and branch signs
        (broadcastable to (...)) -> q (..., 3), f (...), ok (...). Past a branch's edge the
        acos arguments are clamped: f continues (the two shoulders / elbows meet there)."""
        rho = np.hypot(W[..., 0], W[..., 1])
        cl = np.clip(self.d_lat / np.maximum(rho, 1e-12), -1.0, 1.0)
        phi = np.arctan2(W[..., 1], W[..., 0]) - self.th_e - sgn1 * np.arccos(cl)  # W.Rz(phi)e = d
        # W turned back by phi, in the arm plane relative to J2: c = x + i z
        cph, sph = np.cos(phi), np.sin(phi)
        px, py, pz = (
            cph * W[..., 0] + sph * W[..., 1] - self.p2[0],
            -sph * W[..., 0] + cph * W[..., 1] - self.p2[1],
            W[..., 2] - self.p2[2],
        )
        cr, ci = px * self.r[0] + py * self.r[1], pz
        aL1, aL2 = self._aL
        cv = (cr * cr + ci * ci - aL1**2 - aL2**2) / (2 * aL1 * aL2)
        ok = (rho >= abs(self.d_lat) - 1e-12) & (np.abs(cv) <= 1.0)
        psi = sgn3 * np.arccos(np.clip(cv, -1.0, 1.0)) - self._gam
        u = self.L1 + self.L2 * np.exp(1j * psi)
        q2 = np.arctan2(ci, cr) - np.angle(u)
        ca = self.a4c * np.exp(1j * (q2 + psi))  # a4 turned about e by q2 + psi
        a4x = ca.real * self.r[0] + self.a4e * self.e[0]
        a4y = ca.real * self.r[1] + self.a4e * self.e[1]
        x, y = cph * a4x - sph * a4y, sph * a4x + cph * a4y
        f = x * A5[..., 0] + y * A5[..., 1] + ca.imag * A5[..., 2] - self.c45
        q = np.stack(np.broadcast_arrays(self.s1 * phi, q2, self.s3 * psi), axis=-1)
        return q, f, ok

    def _at(self, Bm: np.ndarray, t: np.ndarray):
        """W and a5 (..., 3) for poses Bm (..., 4, 4) and q6 = t (...)."""
        ct, st = np.cos(t)[..., None], np.sin(t)[..., None]
        X = self.cW[0] + ct * self.cW[1] + st * self.cW[2]
        A = self.cA[0] + ct * self.cA[1] + st * self.cA[2]
        R, p = Bm[..., :3, :3], Bm[..., :3, 3]
        return np.einsum("...ij,...j->...i", R, X) + p, np.einsum("...ij,...j->...i", R, A)

    def _roots(self, T: np.ndarray, grid: int):
        """(pose index, branch, q6) of every root of f for poses T (N, 4, 4)."""
        Bm = T @ self.Minv
        N = len(T)
        t = np.arange(grid) * (2 * np.pi / grid)
        W, A = self._at(Bm[:, None], np.broadcast_to(t, (N, grid)))
        _, f, ok = self._branch(
            W[None], A[None], self.S1[:, None, None], self.S3[:, None, None]
        )  # (4, N, grid)
        f1, ok1 = np.roll(f, -1, axis=-1), np.roll(ok, -1, axis=-1)
        step = 2 * np.pi / grid
        sign = (f * f1 <= 0) & ~((f1 == 0) & (np.abs(f) > 0))  # f1 = 0: the next interval
        b, i, k = np.nonzero(ok & ok1 & sign)
        lo, hi, flo, fhi = t[k], t[k] + step, f[b, i, k], f1[b, i, k]
        # intervals with a branch edge (a shoulder or elbow pair meets there; f has a square-root
        # kink): the edge is located by bisection on the valid side and the bracket runs from
        # the valid sample to it, so roots right at the edge are found exactly
        eb, ei, ek = np.nonzero(ok ^ ok1)
        if len(eb):
            fwd = ok[eb, ei, ek]  # valid at the interval start: the edge lies ahead
            tv = np.where(fwd, t[ek], t[ek] + step)  # valid end
            tx = np.where(fwd, t[ek] + step, t[ek])  # invalid end
            Be, s1e, s3e = Bm[ei], self.S1[eb], self.S3[eb]
            for _ in range(40):
                mid = 0.5 * (tv + tx)
                W, A = self._at(Be, mid)
                _, _, okm = self._branch(W, A, s1e, s3e)
                tv, tx = np.where(okm, mid, tv), np.where(okm, tx, mid)
            W, A = self._at(Be, tv)
            _, fe, _ = self._branch(W, A, s1e, s3e)
            fv = np.where(fwd, f[eb, ei, ek], f1[eb, ei, ek])
            ts = np.where(fwd, t[ek], t[ek] + step)
            hit = fv * fe <= 0
            b, i = np.concatenate([b, eb[hit]]), np.concatenate([i, ei[hit]])
            lo, hi = np.concatenate([lo, ts[hit]]), np.concatenate([hi, tv[hit]])
            flo, fhi = np.concatenate([flo, fv[hit]]), np.concatenate([fhi, fe[hit]])
        Bi = Bm[i]
        s1, s3 = self.S1[b], self.S3[b]
        side = np.zeros(len(b))
        for it in range(18):  # Illinois regula falsi; every third step bisects (kinks at edges)
            den = fhi - flo
            x = np.where(
                np.abs(den) > 1e-300, (lo * fhi - hi * flo) / np.where(den == 0, 1, den), 0.5 * (lo + hi)
            )
            if it % 3 == 2:
                x = 0.5 * (lo + hi)
            x = np.clip(x, np.minimum(lo, hi), np.maximum(lo, hi))
            W, A = self._at(Bi, x)
            _, fx, _ = self._branch(W, A, s1, s3)
            keep_lo = fx * fhi > 0  # root between lo and x
            hi, fhi = np.where(keep_lo, x, hi), np.where(keep_lo, fx, fhi)
            lo, flo = np.where(keep_lo, lo, x), np.where(keep_lo, flo, fx)
            flo = np.where(keep_lo & (side == 1), flo / 2, flo)
            fhi = np.where(~keep_lo & (side == -1), fhi / 2, fhi)
            side = np.where(keep_lo, 1, -1)
        den = fhi - flo
        t_root = np.where(
            np.abs(den) > 1e-300, (lo * fhi - hi * flo) / np.where(den == 0, 1, den), 0.5 * (lo + hi)
        )
        return i, b, np.clip(t_root, np.minimum(lo, hi), np.maximum(lo, hi)), Bi

    def _finish(self, T, i, b, t, Bi):
        """Full joint vectors for roots (pose i, branch b, q6 = t); FK-checked."""
        W, A = self._at(Bi, t)
        q13, _, _ = self._branch(W, A, self.S1[b], self.S3[b])
        R1 = _rot(self.a[0], q13[:, 0])
        R2 = _rot(self.a[1], q13[:, 1])
        R3 = _rot(self.a[2], q13[:, 2])
        R6 = _rot(self.a[5], t)
        R123 = R1 @ R2 @ R3
        RG = np.swapaxes(R123, -1, -2) @ T[i, :3, :3] @ self.Minv[:3, :3] @ np.swapaxes(R6, -1, -2)
        a4, a5 = self.a[3], self.a[4]
        q4 = _angle_about(a4, a5, RG @ a5)
        Rq = np.swapaxes(_rot(a4, q4), -1, -2) @ RG
        q5 = _angle_about(a5, self.u5, Rq @ self.u5)
        q = wrap_to_pi(np.column_stack([q13, q4, q5, t]))
        err = np.abs(self.fk(q) - T[i]).max(axis=(-2, -1))
        return q, err < TOL_FK

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Vectorized product-of-exponentials FK: T_base_flange (..., 4, 4)."""
        q = np.asarray(q, float)
        T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        for j in range(6):
            R = _rot(self.a[j], q[..., j])
            E = np.broadcast_to(np.eye(4), R.shape[:-2] + (4, 4)).copy()
            E[..., :3, :3] = R
            E[..., :3, 3] = self.p[j] - np.einsum("...ij,j->...i", R, self.p[j])
            T = T @ E
        return T @ self.M

    # --- solver interface ----------------------------------------------------------------

    def solve_batch(
        self, T_base_flange: np.ndarray, return_singular: bool = False, grid: int | None = None, compiled=None
    ):
        """Up to 16 solutions for N poses: q (N, 16, 6) in (-pi, pi], valid (N, 16)
        [, singular (N,) all False: roots closer than the grid resolves are missed]. The
        compiled kernel (core.ik.offset_wrist_nb) is used when numba is there."""
        T = np.ascontiguousarray(np.asarray(T_base_flange, float).reshape(-1, 4, 4))
        N = len(T)
        if compiled is None:
            compiled = _kernel is not None
        if compiled:
            if self._packed is None:
                self._packed = _kernel.pack(self)
            Q, V = _kernel.solve_kernel(T, *self._packed, grid or self.grid)
            return (Q, V, np.zeros(N, bool)) if return_singular else (Q, V)
        Q = np.zeros((N, B, 6))
        V = np.zeros((N, B), bool)
        for s in range(0, N, 512):  # bounded memory
            Ts = T[s : s + 512]
            i, b, t, Bi = self._roots(Ts, grid or self.grid)
            if len(i):
                q, ok = self._finish(Ts, i, b, t, Bi)
                i, q = i[ok], q[ok]
                for k in range(len(i)):
                    n = s + i[k]
                    row, used = Q[n], V[n]
                    if used.all() or (
                        used.any() and (np.abs(wrap_to_pi(row[used] - q[k])).max(1) < 1e-7).any()
                    ):
                        continue
                    j = int(np.argmin(used))
                    Q[n, j], V[n, j] = q[k], True
        if return_singular:
            return Q, V, np.zeros(N, bool)
        return Q, V

    def solve(self, T_base_flange: np.ndarray) -> np.ndarray:
        """Unique solutions (k, 6) in (-pi, pi] (fine q6 grid)."""
        Q, V = self.solve_batch(np.asarray(T_base_flange, float)[None], grid=self.fine_grid)
        return Q[0][V[0]]

    def manipulability(self, q) -> np.ndarray:
        """|det J| of the base-frame twist Jacobian (vectorized; independent of the TCP)."""
        q = np.asarray(q, float)
        sh = q.shape[:-1]
        q = q.reshape(-1, 6)
        if _kernel is not None:
            return _kernel.manip_kernel(
                np.ascontiguousarray(q), np.ascontiguousarray(self.a), np.ascontiguousarray(self.p)
            ).reshape(sh)
        cols = []
        R = np.broadcast_to(np.eye(3), (len(q), 3, 3))
        t = np.zeros((len(q), 3))
        for j in range(6):
            w = R @ self.a[j]
            pj = np.einsum("nij,j->ni", R, self.p[j]) + t
            cols.append(np.concatenate([np.cross(pj, w), w], axis=-1))
            Rj = _rot(self.a[j], q[:, j])
            t = t + np.einsum("nij,nj->ni", R, self.p[j] - Rj @ self.p[j])
            R = R @ Rj
        return np.abs(np.linalg.det(np.stack(cols, axis=-1))).reshape(sh)

    def config_label(self, q) -> str:
        """``S±E±W±``: wrist point in front of / behind J1, elbow sign, sign of sin q5."""
        q = np.asarray(q, float)
        W = self.W0
        for j in (2, 1, 0):  # W = E1 E2 E3 W0 (rotations about the axis lines)
            W = self.p[j] + _rot(self.a[j], q[j]) @ (W - self.p[j])
        s = W @ (_rot(self.a[0], q[0]) @ self.fwd)  # signed distance from J1, in the arm plane
        e = np.sin(self.s3 * q[2] + np.angle(self.L2) - np.angle(self.L1))
        return "".join(f"{k}{'+' if v >= 0 else '-'}" for k, v in (("S", s), ("E", e), ("W", np.sin(q[4]))))
