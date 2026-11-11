"""Compiled (numba) kernel of the offset-wrist IK (core.ik.offset_wrist): the same q6 sweep,
pose by pose in parallel, without allocations in the inner loops. The numpy version stays
as the fallback when numba is missing and as the reference in the tests.

Per pose: f(q6) is sampled on a grid for the four closed-form branches (shoulder x elbow);
sign changes are refined by Illinois regula falsi, branch edges (a shoulder or elbow pair
meets there; square-root kink) are located by bisection and bracketed up to the edge, and a
sampled |f| minimum that dips close to zero is resampled 16x finer (two roots in one grid
step: near-singular poses). Each root gives q4, q5 from two single-axis subproblems and is
kept when forward kinematics matches to 1e-6.

Constants (``pack``): K = cW0, cW1, cW2, cA0, cA1, cA2, p2, r, e (3 each), then th_e, d_lat,
L1 (re, im), L2 (re, im), |L1|, |L2|, gamma, a4c (re, im), a4e, c45, s1, s3.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

B = 16
TOL_FK = 1e-6


@njit(cache=True, inline="always")
def _rot(k, a, R):
    """Rotation about unit axis k by angle a, written into R (3, 3)."""
    c, s = math.cos(a), math.sin(a)
    v = 1.0 - c
    x, y, z = k[0], k[1], k[2]
    R[0, 0], R[0, 1], R[0, 2] = c + x * x * v, x * y * v - z * s, x * z * v + y * s
    R[1, 0], R[1, 1], R[1, 2] = y * x * v + z * s, c + y * y * v, y * z * v - x * s
    R[2, 0], R[2, 1], R[2, 2] = z * x * v - y * s, z * y * v + x * s, c + z * z * v


@njit(cache=True, inline="always")
def _mm(A, Bm, C):
    """C = A @ B (3 x 3; C must not alias A or B)."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * Bm[0, j] + A[i, 1] * Bm[1, j] + A[i, 2] * Bm[2, j]


@njit(cache=True, inline="always")
def _mtm(A, Bm, C):
    """C = A.T @ B (3 x 3)."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[0, i] * Bm[0, j] + A[1, i] * Bm[1, j] + A[2, i] * Bm[2, j]


@njit(cache=True, inline="always")
def _mmt(A, Bm, C):
    """C = A @ B.T (3 x 3)."""
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * Bm[j, 0] + A[i, 1] * Bm[j, 1] + A[i, 2] * Bm[j, 2]


@njit(cache=True, inline="always")
def _angle_about(k, u, R):
    """Angle about unit axis k turning u onto R @ u."""
    v0 = R[0, 0] * u[0] + R[0, 1] * u[1] + R[0, 2] * u[2]
    v1 = R[1, 0] * u[0] + R[1, 1] * u[1] + R[1, 2] * u[2]
    v2 = R[2, 0] * u[0] + R[2, 1] * u[1] + R[2, 2] * u[2]
    ku = k[0] * u[0] + k[1] * u[1] + k[2] * u[2]
    kv = k[0] * v0 + k[1] * v1 + k[2] * v2
    a0, a1, a2 = u[0] - ku * k[0], u[1] - ku * k[1], u[2] - ku * k[2]
    b0, b1, b2 = v0 - kv * k[0], v1 - kv * k[1], v2 - kv * k[2]
    c0, c1, c2 = a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0
    return math.atan2(c0 * k[0] + c1 * k[1] + c2 * k[2], a0 * b0 + a1 * b1 + a2 * b2)


@njit(cache=True, inline="always")
def _wrap(a):
    return math.pi - (math.pi - a) % (2.0 * math.pi)


@njit(cache=True)
def _branch(t, Bm, K, s1b, s3b, out):
    """q1..q3, f, ok for q6 = t and one branch: out = (q1, q2, q3, f, ok)."""
    ct, st = math.cos(t), math.sin(t)
    W0 = W1 = W2 = A0 = A1 = A2 = 0.0
    for j in range(3):
        x = K[j] + ct * K[3 + j] + st * K[6 + j]
        a = K[9 + j] + ct * K[12 + j] + st * K[15 + j]
        W0 += Bm[0, j] * x
        W1 += Bm[1, j] * x
        W2 += Bm[2, j] * x
        A0 += Bm[0, j] * a
        A1 += Bm[1, j] * a
        A2 += Bm[2, j] * a
    W0 += Bm[0, 3]
    W1 += Bm[1, 3]
    W2 += Bm[2, 3]
    th_e, d_lat, L1r, L1i, L2r, L2i, aL1, aL2 = K[27], K[28], K[29], K[30], K[31], K[32], K[33], K[34]
    gam, a4r, a4i, a4e, c45, s1, s3 = K[35], K[36], K[37], K[38], K[39], K[40], K[41]
    rho = math.hypot(W0, W1)
    cl = min(1.0, max(-1.0, d_lat / max(rho, 1e-12)))
    phi = math.atan2(W1, W0) - th_e - s1b * math.acos(cl)
    cph, sph = math.cos(phi), math.sin(phi)
    px = cph * W0 + sph * W1 - K[18]
    py = -sph * W0 + cph * W1 - K[19]
    cr = px * K[21] + py * K[22]
    ci = W2 - K[20]
    cv = (cr * cr + ci * ci - aL1 * aL1 - aL2 * aL2) / (2.0 * aL1 * aL2)
    ok = rho >= abs(d_lat) - 1e-12 and abs(cv) <= 1.0
    psi = s3b * math.acos(min(1.0, max(-1.0, cv))) - gam
    cps, sps = math.cos(psi), math.sin(psi)
    q2 = math.atan2(ci, cr) - math.atan2(L1i + L2r * sps + L2i * cps, L1r + L2r * cps - L2i * sps)
    ang = q2 + psi
    ca, sa = math.cos(ang), math.sin(ang)
    car, cai = a4r * ca - a4i * sa, a4r * sa + a4i * ca
    a4x = car * K[21] + a4e * K[24]
    a4y = car * K[22] + a4e * K[25]
    out[0], out[1], out[2] = s1 * phi, q2, s3 * psi
    out[3] = (cph * a4x - sph * a4y) * A0 + (sph * a4x + cph * a4y) * A1 + cai * A2 - c45
    out[4] = 1.0 if ok else 0.0


@njit(cache=True)
def _refine(lo, hi, flo, fhi, Bm, K, s1b, s3b, out):
    side = 0
    for it in range(60):
        if abs(hi - lo) < 1e-13:
            break
        den = fhi - flo
        x = 0.5 * (lo + hi) if (it % 3 == 2 or den == 0.0) else (lo * fhi - hi * flo) / den
        x = min(max(lo, hi), max(min(lo, hi), x))
        _branch(x, Bm, K, s1b, s3b, out)
        fx = out[3]
        if fx == 0.0:
            return x
        if fx * fhi > 0.0:
            hi, fhi = x, fx
            if side == 1:
                flo *= 0.5
            side = 1
        else:
            lo, flo = x, fx
            if side == -1:
                fhi *= 0.5
            side = -1
    den = fhi - flo
    return 0.5 * (lo + hi) if den == 0.0 else min(max(lo, hi), max(min(lo, hi), (lo * fhi - hi * flo) / den))


@njit(cache=True)
def _add(t, b, T, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, n, out, q, S):
    """Finish a root (q4, q5), check FK, store it unless already there; returns the count.
    S: scratch (8, 3, 3)."""
    _branch(t, Bm, K, S1[b], S3[b], out)
    q[0], q[1], q[2], q[5] = out[0], out[1], out[2], t
    R1, R2, R3, R6, X, Y, RG, R4 = S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7]
    _rot(ax[0], q[0], R1)
    _rot(ax[1], q[1], R2)
    _rot(ax[2], q[2], R3)
    _rot(ax[5], t, R6)
    _mm(R1, R2, X)
    _mm(X, R3, Y)  # R123
    _mtm(Y, T, X)  # R123^T R_T (T is 4 x 4; its top-left block is used)
    _mm(X, MR, Y)
    _mmt(Y, R6, RG)
    q[3] = _angle_about(ax[3], ax[4], RG)
    _rot(ax[3], q[3], R4)
    _mtm(R4, RG, X)
    q[4] = _angle_about(ax[4], u5, X)
    for j in range(6):
        q[j] = _wrap(q[j])
    # forward kinematics (product of exponentials) and the error against T
    R = Y
    for i in range(3):
        for j in range(3):
            R[i, j] = 1.0 if i == j else 0.0
    p0 = p1 = p2 = 0.0
    for j in range(6):
        _rot(ax[j], q[j], R1)
        d0 = pt[j, 0] - (R1[0, 0] * pt[j, 0] + R1[0, 1] * pt[j, 1] + R1[0, 2] * pt[j, 2])
        d1 = pt[j, 1] - (R1[1, 0] * pt[j, 0] + R1[1, 1] * pt[j, 1] + R1[1, 2] * pt[j, 2])
        d2 = pt[j, 2] - (R1[2, 0] * pt[j, 0] + R1[2, 1] * pt[j, 1] + R1[2, 2] * pt[j, 2])
        p0, p1, p2 = (
            p0 + R[0, 0] * d0 + R[0, 1] * d1 + R[0, 2] * d2,
            p1 + R[1, 0] * d0 + R[1, 1] * d1 + R[1, 2] * d2,
            p2 + R[2, 0] * d0 + R[2, 1] * d1 + R[2, 2] * d2,
        )
        _mm(R, R1, X)
        for i in range(3):
            for k in range(3):
                R[i, k] = X[i, k]
    err = 0.0
    for i in range(3):
        pi = (p0, p1, p2)[i] + R[i, 0] * M[0, 3] + R[i, 1] * M[1, 3] + R[i, 2] * M[2, 3]
        err = max(err, abs(pi - T[i, 3]))
        for j in range(3):
            err = max(err, abs(R[i, 0] * M[0, j] + R[i, 1] * M[1, j] + R[i, 2] * M[2, j] - T[i, j]))
    if err > TOL_FK:
        return n
    for s in range(n):
        d = 0.0
        for j in range(6):
            d = max(d, abs(_wrap(Qn[s, j] - q[j])))
        if d < 1e-7:
            return n
    if n < Qn.shape[0]:
        for j in range(6):
            Qn[n, j] = q[j]
        Vn[n] = True
        return n + 1
    return n


@njit(cache=True)
def _edge(tv, tx, fv, Bm, K, s1b, s3b, out):
    """Branch edge between valid tv and invalid tx: (t at the edge, f there)."""
    for _ in range(45):
        mid = 0.5 * (tv + tx)
        _branch(mid, Bm, K, s1b, s3b, out)
        if out[4] > 0.5:
            tv = mid
        else:
            tx = mid
    _branch(tv, Bm, K, s1b, s3b, out)
    return tv, out[3]


@njit(cache=True)
def _scan(t0, t1, n_sub, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S):
    """Every root of branch b in [t0, t1], sampled n_sub times (edges bracketed exactly)."""
    s1b, s3b = S1[b], S3[b]
    _branch(t0, Bm, K, s1b, s3b, out)
    ta, fa, oa = t0, out[3], out[4] > 0.5
    h = (t1 - t0) / n_sub
    for s in range(1, n_sub + 1):
        tb = t0 + s * h
        _branch(tb, Bm, K, s1b, s3b, out)
        fb, ob = out[3], out[4] > 0.5
        if oa and ob:
            if fa * fb <= 0.0 and not (fb == 0.0 and fa != 0.0):
                t = _refine(ta, tb, fa, fb, Bm, K, s1b, s3b, out)
                cnt = _add(t, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S)
        elif oa or ob:
            tv, tx, fv = (ta, tb, fa) if oa else (tb, ta, fb)
            te, fe = _edge(tv, tx, fv, Bm, K, s1b, s3b, out)
            if fv * fe <= 0.0:
                t = _refine(tv, te, fv, fe, Bm, K, s1b, s3b, out)
                cnt = _add(t, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S)
        ta, fa, oa = tb, fb, ob
    return cnt


@njit(cache=True, parallel=True)
def solve_kernel(T, Minv, K, S1, S3, ax, pt, M, u5, grid):
    """All solutions for poses T (N, 4, 4) -> Q (N, 16, 6), V (N, 16)."""
    N = T.shape[0]
    Q = np.zeros((N, B, 6))
    V = np.zeros((N, B), np.bool_)
    step = 2.0 * math.pi / grid
    MR = np.ascontiguousarray(Minv[:3, :3])
    for n in prange(N):
        Tn = T[n]
        Bm = np.empty((3, 4))
        for i in range(3):
            for j in range(4):
                Bm[i, j] = (
                    Tn[i, 0] * Minv[0, j]
                    + Tn[i, 1] * Minv[1, j]
                    + Tn[i, 2] * Minv[2, j]
                    + Tn[i, 3] * Minv[3, j]
                )
        out = np.empty(5)
        q = np.empty(6)
        S = np.empty((8, 3, 3))
        f = np.empty((4, grid))
        ok = np.empty((4, grid), np.bool_)
        Qn, Vn = Q[n], V[n]
        for b in range(4):
            for k in range(grid):
                _branch(k * step, Bm, K, S1[b], S3[b], out)
                f[b, k], ok[b, k] = out[3], out[4] > 0.5
        cnt = 0
        for b in range(4):
            for k in range(grid):
                km, k1, k2 = (k - 1) % grid, (k + 1) % grid, (k + 2) % grid
                t0 = k * step
                f0, f1, fm = f[b, k], f[b, k1], f[b, km]
                if not (ok[b, k] or ok[b, k1]):
                    continue
                # next to a branch edge (square-root kink, roots cluster there) or a dip of |f|
                # towards zero (two close roots: near-singular): sampled 16x finer
                near_edge = not (ok[b, km] and ok[b, k] and ok[b, k1] and ok[b, k2])
                dip = (
                    fm * f0 > 0.0
                    and f0 * f1 > 0.0
                    and abs(f0) <= abs(fm)
                    and abs(f0) <= abs(f1)
                    and abs(f0) < abs(fm) + abs(f1) - 2.0 * abs(f0)
                )
                if near_edge or dip:
                    cnt = _scan(
                        t0, t0 + step, 16, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S
                    )
                    if dip and not near_edge:
                        cnt = _scan(
                            t0 - step, t0, 16, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S
                        )
                elif f0 * f1 <= 0.0 and not (f1 == 0.0 and f0 != 0.0):
                    t = _refine(t0, t0 + step, f0, f1, Bm, K, S1[b], S3[b], out)
                    cnt = _add(t, b, Tn, Bm, K, S1, S3, ax, pt, M, MR, u5, Qn, Vn, cnt, out, q, S)
    return Q, V


def pack(ik) -> tuple:
    """The solver's constants as numba-friendly arrays."""
    K = np.concatenate(
        [
            *ik.cW,
            *ik.cA,
            ik.p2,
            ik.r,
            ik.e,
            [ik.th_e, ik.d_lat, ik.L1.real, ik.L1.imag, ik.L2.real, ik.L2.imag, ik._aL[0], ik._aL[1]],
            [ik._gam, ik.a4c.real, ik.a4c.imag, ik.a4e, ik.c45, ik.s1, ik.s3],
        ]
    ).astype(float)
    f = np.ascontiguousarray
    return (f(ik.Minv), K, f(ik.S1), f(ik.S3), f(ik.a), f(ik.p), f(ik.M), f(ik.u5, dtype=float))


@njit(cache=True, parallel=True)
def manip_kernel(Q, ax, pt):
    """|det J| (base-frame twist Jacobian) for joint vectors Q (n, 6)."""
    n = Q.shape[0]
    out = np.empty(n)
    for m in prange(n):
        J = np.empty((6, 6))
        R = np.eye(3)
        Rj = np.empty((3, 3))
        X = np.empty((3, 3))
        t0 = t1 = t2 = 0.0
        for j in range(6):
            w0 = R[0, 0] * ax[j, 0] + R[0, 1] * ax[j, 1] + R[0, 2] * ax[j, 2]
            w1 = R[1, 0] * ax[j, 0] + R[1, 1] * ax[j, 1] + R[1, 2] * ax[j, 2]
            w2 = R[2, 0] * ax[j, 0] + R[2, 1] * ax[j, 1] + R[2, 2] * ax[j, 2]
            p0 = R[0, 0] * pt[j, 0] + R[0, 1] * pt[j, 1] + R[0, 2] * pt[j, 2] + t0
            p1 = R[1, 0] * pt[j, 0] + R[1, 1] * pt[j, 1] + R[1, 2] * pt[j, 2] + t1
            p2 = R[2, 0] * pt[j, 0] + R[2, 1] * pt[j, 1] + R[2, 2] * pt[j, 2] + t2
            J[0, j], J[1, j], J[2, j] = p1 * w2 - p2 * w1, p2 * w0 - p0 * w2, p0 * w1 - p1 * w0
            J[3, j], J[4, j], J[5, j] = w0, w1, w2
            _rot(ax[j], Q[m, j], Rj)
            d0 = pt[j, 0] - (Rj[0, 0] * pt[j, 0] + Rj[0, 1] * pt[j, 1] + Rj[0, 2] * pt[j, 2])
            d1 = pt[j, 1] - (Rj[1, 0] * pt[j, 0] + Rj[1, 1] * pt[j, 1] + Rj[1, 2] * pt[j, 2])
            d2 = pt[j, 2] - (Rj[2, 0] * pt[j, 0] + Rj[2, 1] * pt[j, 1] + Rj[2, 2] * pt[j, 2])
            t0 += R[0, 0] * d0 + R[0, 1] * d1 + R[0, 2] * d2
            t1 += R[1, 0] * d0 + R[1, 1] * d1 + R[1, 2] * d2
            t2 += R[2, 0] * d0 + R[2, 1] * d1 + R[2, 2] * d2
            _mm(R, Rj, X)
            for a in range(3):
                for c in range(3):
                    R[a, c] = X[a, c]
        det = 1.0  # Gaussian elimination with partial pivoting
        for c in range(6):
            p = c
            for r in range(c + 1, 6):
                if abs(J[r, c]) > abs(J[p, c]):
                    p = r
            if J[p, c] == 0.0:
                det = 0.0
                break
            if p != c:
                for k in range(6):
                    J[c, k], J[p, k] = J[p, k], J[c, k]
                det = -det
            det *= J[c, c]
            for r in range(c + 1, 6):
                g = J[r, c] / J[c, c]
                for k in range(c, 6):
                    J[r, k] -= g * J[c, k]
        out[m] = abs(det)
    return out
