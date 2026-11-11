"""A clear home pose (2026-11-11).

A robot's default home (``spec.yaml`` ``home_deg``) suits a floor-mounted arm in an open cell:
the FANUC CRX home (0, 0, 0, 0, -90, 0) stands the arm straight up, which on an inverted gantry
mount points it straight down into the part. The path builder then routes every move to and
from home through the part. ``clear_home`` looks for a folded pose that is clear everywhere it
is checked: shoulder / elbow (joints 2, 3) on a grid, wrist bend (joint 5) at the preferred
value or +-90 / 0, the other joints as preferred; candidates are tried compact first (flange
close to the base), then close to the preferred pose.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def clear_home(
    robot,
    collides: Callable[[np.ndarray], bool],
    q_pref: np.ndarray,
    step_deg: float = 15.0,
    margin_deg: float = 5.0,
    max_checks: int = 600,
) -> np.ndarray | None:
    """The first clear candidate (see the module docstring), or None."""
    q_pref = np.asarray(q_pref, float)
    n = len(q_pref)
    if n < 5:
        return None
    m = np.radians(margin_deg)
    lo, hi = robot.lower + m, robot.upper - m
    step = np.radians(step_deg)

    def axis(j: int) -> np.ndarray:
        if hi[j] <= lo[j]:
            return np.array([q_pref[j]])
        return np.unique(
            np.concatenate([np.arange(lo[j], hi[j] + 1e-9, step), [np.clip(q_pref[j], lo[j], hi[j])]])
        )

    wrist = {float(np.clip(v, lo[4], hi[4])) for v in (q_pref[4], np.pi / 2, -np.pi / 2, 0.0)}
    cands = []
    for a in axis(1):
        for b in axis(2):
            for w in wrist:
                q = q_pref.copy()
                q[1], q[2], q[4] = a, b, w
                cands.append(np.clip(q, lo, hi))
    Q = np.array(cands)
    reach = np.array([np.linalg.norm(robot.fk(q)[:3, 3]) for q in Q])
    score = reach + 0.05 * np.abs(Q - q_pref).sum(axis=1)
    for i in np.argsort(score, kind="stable")[:max_checks]:
        if not collides(Q[i]):
            return Q[i]
    return None


def blocked_stops(checker, q: np.ndarray | None, stops: list) -> list[int]:
    """Indices (``station`` of each stop) where the arm at ``q`` collides; stops are
    ``core.paths.StationPath`` (or anything with ``station`` and ``T_world_base``)."""
    if q is None:
        return []
    out = []
    for st in stops:
        checker.set_base(st.T_world_base)
        if checker.scene.collides(q):
            out.append(st.station)
    return out


def choose_home(checker, q_home: np.ndarray | None, base_poses: list) -> tuple[np.ndarray | None, str]:
    """``q_home`` when it is clear at every stop, else a clear folded pose with a note (the
    original when none is found: the paths flag it)."""
    if q_home is None:
        return None, ""

    def collides(q) -> bool:
        for T in base_poses:
            checker.set_base(T)
            if checker.scene.collides(q):
                return True
        return False

    if not collides(q_home):
        return q_home, ""
    q = clear_home(checker.robot, collides, q_home)
    if q is None:
        return q_home, ""
    deg = ", ".join(f"{v:.0f}" for v in np.degrees(q))
    return q, (
        f"The home pose collides at a stop, so these paths use a clear folded pose instead: **{deg}** deg "
        "(Robot & cell → Robot → *Home position*: type it and *Apply typed values* to keep it)."
    )
