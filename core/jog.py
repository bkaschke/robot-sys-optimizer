"""Cartesian jogging (v6): move the TCP a step along / about the axes of a frame (world, robot
base, part, tool) and find the joints for it nearest the current ones, like a teach pendant.

Translations follow the frame's axes; rotations turn about the frame's axes **through the
TCP** (the TCP stays put while it turns). Pure numpy + the robot's IK: no viser."""

from __future__ import annotations

import numpy as np

AXES = ("X", "Y", "Z", "Rx", "Ry", "Rz")
MAX_STEP = np.radians(30.0)  # a jog needing a larger joint move is refused (flip / singularity)


def _rot(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def jog_pose(T_world_tcp: np.ndarray, R_world_frame: np.ndarray, axis: str, amount: float) -> np.ndarray:
    """TCP pose after a jog step: ``amount`` m (X/Y/Z) or rad (Rx/Ry/Rz) along / about the
    axis of the frame with world rotation ``R_world_frame`` (the tool frame = the TCP's own)."""
    k = AXES.index(axis)
    T = np.array(T_world_tcp, float)
    R_f = np.asarray(R_world_frame, float)
    if k < 3:
        T[:3, 3] += amount * R_f[:, k]
    else:
        T[:3, :3] = R_f @ _rot(k - 3, amount) @ R_f.T @ T[:3, :3]
    return T


def nearest_copy(q: np.ndarray, ref: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    """The 2*pi copy of q nearest ``ref`` per joint inside [lo, hi], or None."""
    c = q + 2 * np.pi * np.round((ref - q) / (2 * np.pi))
    c = np.where(c < lo - 1e-9, c + 2 * np.pi, c)
    c = np.where(c > hi + 1e-9, c - 2 * np.pi, c)
    return None if np.any(c < lo - 1e-9) or np.any(c > hi + 1e-9) else c


def nearest_ik(ik, robot, T_base_flange: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    """IK solution for ``T_base_flange`` nearest the current joints ``q`` (within limits)."""
    q = np.asarray(q, float)
    try:
        sols = ik.solve(T_base_flange, seed=q)
    except TypeError:  # analytic solvers take no seed
        sols = ik.solve(T_base_flange)
    best = None
    for s in np.atleast_2d(sols):
        c = nearest_copy(np.asarray(s, float), q, robot.lower, robot.upper)
        if c is not None and (best is None or np.max(np.abs(c - q)) < np.max(np.abs(best - q))):
            best = c
    return best
