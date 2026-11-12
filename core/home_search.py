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


def clear_homes(
    robot,
    collides: Callable[[np.ndarray], bool],
    q_pref: np.ndarray,
    step_deg: float = 15.0,
    margin_deg: float = 5.0,
    max_checks: int = 600,
    j2_sign: float | None = None,
):
    """The clear candidates in order (see the module docstring). ``j2_sign`` (+1 / -1):
    only shoulder angles of that sign (the side the plan's points are on, for an arm whose
    shoulder cannot turn a full circle: it cannot get to the other side), if any are clear."""
    q_pref = np.asarray(q_pref, float)
    n = len(q_pref)
    if n < 5:
        return
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
    if j2_sign:
        same = np.sign(Q[:, 1]) == np.sign(j2_sign)
        Q = Q[same] if same.any() else Q
    reach = np.array([np.linalg.norm(robot.fk(q)[:3, 3]) for q in Q])
    score = reach + 0.05 * np.abs(Q - q_pref).sum(axis=1)
    for i in np.argsort(score, kind="stable")[:max_checks]:
        if not collides(Q[i]):
            yield Q[i]


def clear_home(robot, collides, q_pref, **kw) -> np.ndarray | None:
    """The first clear candidate, or None."""
    return next(clear_homes(robot, collides, q_pref, **kw), None)


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


def plan_j2_sign(robot, qs) -> float | None:
    """The sign of the shoulder angle most planned poses use, for an arm whose shoulder cannot
    turn a full circle (opt.shoulder); None otherwise."""
    from opt.shoulder import shoulder_locked

    q2 = [float(np.asarray(q)[1]) for q in qs if q is not None]
    if not q2 or not shoulder_locked(robot):
        return None
    return float(np.sign(np.median(q2))) or None


def probes(checker, sol, res: dict, base_poses: list, per_stop: int = 3) -> list[tuple]:
    """A few planned poses per stop, backed off the surface (core.paths), for ``connects``:
    (T_world_base, tcp, joints)."""
    from core.paths import _Tracer

    tr = _Tracer(checker, None, 0.02, 30.0)
    out = []
    for k, T in enumerate(base_poses):
        rs = [
            r
            for p, r in res.items()
            if sol.assignment.get(p) == k and r.q is not None and r.status != "unreachable"
        ]
        if not rs:
            continue
        checker.set_base(T)
        for r in rs[:: max(1, int(np.ceil(len(rs) / per_stop)))][:per_stop]:
            out.append((T, r.tcp, (tr.retreat(r.q, r.tcp) or [np.asarray(r.q, float)])[-1]))
    return out


def connects(checker, q: np.ndarray, probe: list, time_limit: float = 0.7) -> bool:
    """A collision-free joint path from q to every probe pose (core.joint_plan)."""
    from core.joint_plan import plan_joint_path
    from core.paths import HOME_STEP

    for T, tcp, qb in probe:
        checker.set_base(T)

        def hit(x, tcp=tcp) -> bool:
            return checker.scene.collides(x, tcp)

        if plan_joint_path(hit, q, qb, checker.lo, checker.hi, HOME_STEP, time_limit=time_limit) is None:
            return False
    return True


def choose_home(
    checker,
    q_home: np.ndarray | None,
    base_poses: list,
    j2_sign: float | None = None,
    probe: list | None = None,
    max_tries: int = 15,
) -> tuple[np.ndarray | None, str]:
    """``q_home`` when it is clear at every stop (with ``j2_sign``: on that shoulder side or
    at zero; with ``probe``: joint paths from it reach those poses), else the first folded
    pose that is (if none of the ``max_tries`` first clear ones connects: the first clear one,
    unless the project's home itself is clear and on the right side) with a note; else the
    original (the paths flag what fails)."""
    if q_home is None:
        return None, ""

    def collides(q) -> bool:
        for T in base_poses:
            checker.set_base(T)
            if checker.scene.collides(q):
                return True
        return False

    wrong_side = bool(j2_sign) and np.sign(q_home[1]) == -np.sign(j2_sign) and abs(q_home[1]) > np.radians(5)
    hits = collides(q_home)
    if not hits and not wrong_side and (not probe or connects(checker, q_home, probe, 2.0)):
        return q_home, ""
    first = None
    q = None
    for k, c in enumerate(clear_homes(checker.robot, collides, q_home, j2_sign=j2_sign)):
        first = c if first is None else first
        if not probe or connects(checker, c, probe):
            q = c
            break
        if k + 1 >= max_tries:
            break
    if q is None and (hits or wrong_side):
        q = first  # the project's home cannot be used: a clear one is still better
    if q is None:
        return q_home, ""
    deg = ", ".join(f"{v:.0f}" for v in np.degrees(q))
    why = (
        "collides at a stop" if hits
        else "is on the other shoulder side from the planned poses" if wrong_side
        else "has no clear way to the planned poses"
    )  # fmt: skip
    return q, (
        f"The home pose {why}, so these paths use a clear folded pose instead: **{deg}** deg "
        "(Robot & cell → Robot → *Home position*: type it and *Apply typed values* to keep it)."
    )


def pick_home(checker, q_home, sol, res: dict, base_poses: list) -> tuple[np.ndarray | None, str]:
    """``choose_home`` for a plan's paths: the plan's shoulder side, probes at every stop."""
    live = [r.q for r in res.values() if r.status != "unreachable"]
    sign = plan_j2_sign(checker.robot, live)
    return choose_home(checker, q_home, base_poses, sign, probes(checker, sol, res, base_poses))
