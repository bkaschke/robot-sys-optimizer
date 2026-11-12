"""One shoulder side per stop for arms whose shoulder cannot turn a full circle (2026-11-11).

A UR arm's shoulder (joint 2) turns +-360 deg, so it can swing the arm over the top from one
shoulder side (configuration label ``S+``) to the other (``S-``) either way round. A FANUC
CRX or ABB GoFa / PoWa shoulder stops at +-180 deg: the only way between the sides is
through joint 2 = 0, the arm pointing straight along the base's z. On a floor mount that is
straight up and usually clear; hung from a gantry it points straight down into the part, and
the arm cannot change sides at that stop at all. A plan that mixes both sides in one stop
then has no collision-free path (the paths flag straight moves through the part).

``one_side`` runs after the exact verification of a plan (before zoning). For every stop
that uses both sides it tests whether the arm can swing between them (a joint path between
the two sides' best poses, backed off the surface, core.joint_plan). When it cannot, the
stop keeps the side that covers the most points (each point takes its best solution on that
side); the other points are moved to another stop that can do them on its side, or become
unreachable (reason ``shoulder side``). The planner's cost then prefers plans with more
stops when that reaches more points.
"""

from __future__ import annotations

import numpy as np

from core.configs import config_label
from core.feasibility import PointResult

FULL_TURN = 2 * np.pi + np.radians(20.0)  # shoulder range that can swing over the top
SWING_S = 1.5  # s per joint-path search between the two sides of a stop
REASON = "shoulder side"
NOTE = (
    "needs the other shoulder side at its stop, and the arm cannot swing between the sides "
    "there without hitting something (shoulder limited to less than a full turn)"
)


def side(label: str) -> str:
    """``S+`` / ``S-`` of a configuration label, "" without a shoulder sign."""
    return label[:2] if label.startswith("S") else ""


def shoulder_locked(robot) -> bool:
    """The shoulder (joint 2) cannot turn a full circle (with some room to spare)."""
    return bool(len(robot.lower) >= 2 and robot.upper[1] - robot.lower[1] < FULL_TURN)


def _quality(r: PointResult) -> float:
    return float(r.manipulability) + 0.01 * float(r.limit_margin_deg)


def can_swing(checker, ra: PointResult, rb: PointResult, time_limit: float = SWING_S) -> bool:
    """A collision-free joint path from ``ra``'s pose to ``rb``'s (both backed off the
    surface first) at the checker's current base."""
    from core.joint_plan import plan_joint_path
    from core.paths import HOME_STEP, _Tracer

    tr = _Tracer(checker, None, 0.02, 30.0)
    qa = (tr.retreat(ra.q, ra.tcp) or [np.asarray(ra.q, float)])[-1]
    qb = (tr.retreat(rb.q, rb.tcp) or [np.asarray(rb.q, float)])[-1]

    def hit(q) -> bool:
        return checker.scene.collides(q, ra.tcp)

    way = plan_joint_path(hit, qa, qb, checker.lo, checker.hi, HOME_STEP, time_limit=time_limit)
    return way is not None


def _options(checker, t, index, pid, max_poses, r0: PointResult | None = None) -> dict[str, PointResult]:
    """Every configuration of the point at the checker's base (``r0``, the verified solution
    at this stop, always among them)."""
    i = index[pid]
    opts = checker.check_point_configs(pid, t.P_part[i], t.N_part[i], t.groups[i], max_poses)
    opts = {lab: r for lab, r in opts.items() if r.status != "unreachable" and r.q is not None}
    if r0 is not None:
        opts.setdefault(config_label(checker.ik, r0.q), r0)
    return opts


def _best_on(opts: dict[str, PointResult], s: str) -> PointResult | None:
    on = [r for lab, r in opts.items() if side(lab) == s]
    return max(on, key=_quality) if on else None


def one_side(
    checker,
    t,
    stations: list[dict],
    sites,
    assignment: dict[int, int],
    results: dict[int, PointResult],
    max_poses: int = 40,
    time_limit: float = SWING_S,
) -> dict[int, str]:
    """Updates ``assignment`` / ``results`` in place (see the module docstring). Returns the
    shoulder side each stop must keep ({stop: "S+"}; stops where the arm can swing between the
    sides are not listed); {} when nothing had to change or the shoulder turns fully.

    The home pose is one pose for the whole plan and lies on one side, and the arm goes home
    between stops; so when any stop cannot swing between the sides, every such stop keeps the
    same side: the one the most points can be done on."""
    if not shoulder_locked(checker.robot) or not hasattr(checker.ik, "config_label"):
        return {}
    from opt.station_checks import _robust
    from opt.zoning import _prefetch_configs

    index = {int(pid): i for i, pid in enumerate(t.ids)}
    live = {pid: r for pid, r in results.items() if r.status != "unreachable" and r.q is not None}
    lab = {p: side(config_label(checker.ik, r.q)) for p, r in live.items()}
    stop_pids = {k: [p for p in live if assignment.get(p) == k] for k in range(len(stations))}
    free: set[int] = set()  # stops where the arm can swing between the sides
    blocked = False
    for k, pids in stop_pids.items():
        by: dict[str, list[int]] = {}
        for p in pids:
            by.setdefault(lab[p], []).append(p)
        if len(by) < 2:
            continue
        checker.set_base(sites.T(stations[k]))
        best = {s: max((live[p] for p in m), key=_quality) for s, m in by.items()}
        a, b = sorted(best)
        if can_swing(checker, best[a], best[b], time_limit):
            free.add(k)
        else:
            blocked = True
    if not blocked:
        return {}
    opts: dict[int, dict[str, PointResult]] = {}
    for k, pids in stop_pids.items():
        if k in free or not pids:
            continue
        T = sites.T(stations[k])
        _prefetch_configs(checker, t, index, T, pids, max_poses)
        checker.set_base(T)
        for p in pids:
            opts[p] = _options(checker, t, index, p, max_poses, live[p])
    score = {}
    for s in ("S+", "S-"):
        on = [_best_on(o, s) for o in opts.values()]
        score[s] = (sum(r is not None for r in on), sum(_quality(r) for r in on if r is not None))
    keep = max(sorted(score), key=lambda s: score[s])
    sides = {k: keep for k in range(len(stations)) if k not in free}
    orphans: list[tuple[int, int]] = []  # (pid, stop it came from)
    for p, o in opts.items():
        if lab[p] == keep:
            continue
        k = assignment[p]
        alt = _best_on(o, keep)
        checker.set_base(sites.T(stations[k]))
        if alt is not None and _robust(checker, t, index[p], alt, stations[k], sites)[0]:
            results[p] = alt
        else:
            orphans.append((p, k))
    for p, k0 in orphans:
        placed = False
        for k, v in enumerate(stations):
            if k == k0:
                continue
            checker.set_base(sites.T(v))
            cands = [
                r
                for c, r in _options(checker, t, index, p, max_poses).items()
                if k in free or side(c) == keep
            ]
            for r in sorted(cands, key=_quality, reverse=True):
                if _robust(checker, t, index[p], r, v, sites)[0]:
                    results[p], assignment[p] = r, k
                    placed = True
                    break
            if placed:
                break
        if not placed:
            results[p] = PointResult(p, "unreachable", REASON, notes=[NOTE])
    return sides
