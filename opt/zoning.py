"""Joint-configuration zoning of a station plan (runs after ``placement.verify``).

For every station:
  1. each assigned point is re-checked for EVERY joint configuration it can be done in
     (``FeasibilityChecker.check_point_configs``, one pass over the candidate poses);
  2. the fewest configurations that cover all points are chosen (``core.configs.min_zones``,
     exact), each point taking its best solution among them;
  3. the chosen solutions get the station's robustness checks (localization)
     again; a point that fails keeps its original verified solution (and configuration);
  4. each zone's points are put in serpentine path order and their joint values unwrapped
     (+-360 deg copies) for continuity; jumps of joint 4/6 above 180 deg that remain count.

Configuration changes = sum over stations of (zones - 1) + remaining wrist jumps. The cost
(``opt.cost``) charges ``Weights.config`` per change, so among the planner's alternatives
(different station counts) J can prefer one more stop over many reconfigurations.
"""

from __future__ import annotations

import numpy as np

from core.configs import config_label, min_zones, order_path, unwrap_path
from core.feasibility import FeasibilityChecker, PointResult
from core.schema import Zone


def _quality(r: PointResult) -> float:
    return float(r.manipulability) + 0.01 * float(r.limit_margin_deg)


def _prefetch_configs(checker, t, index: dict, T, pids: list[int], max_poses) -> None:
    """v8 T2: ``check_point_configs`` of these points at this stop in the checker's pool."""
    if getattr(checker, "pool", None) is None:
        return
    from opt.parallel import prefetch

    calls = [
        (T, "configs", pid, t.P_part[index[pid]], t.N_part[index[pid]], t.groups[index[pid]], max_poses, True)
        for pid in pids
    ]
    prefetch(checker, calls)


def zone_solution(
    checker: FeasibilityChecker,
    t,
    stations: list[dict],
    sites,
    assignment: dict[int, int],
    results: dict[int, PointResult],
    max_poses: int = 40,
) -> tuple[list[Zone], int]:
    """Zones for a verified plan; updates ``results`` in place (q, config, zone, seq).
    Returns (zones in station order, number of configuration changes)."""
    from opt.placement import _robust

    index = {int(pid): i for i, pid in enumerate(t.ids)}
    zones: list[Zone] = []
    changes = 0
    for k, v in enumerate(stations):
        pids = [pid for pid, s in assignment.items() if s == k and results[pid].status != "unreachable"]
        if not pids:
            continue
        checker.set_base(sites.T(v))
        for pid in pids:
            results[pid].config = config_label(checker.ik, results[pid].q)

        def search(pid: int) -> dict[str, PointResult]:
            i, r0 = index[pid], results[pid]
            opts = checker.check_point_configs(pid, t.P_part[i], t.N_part[i], t.groups[i], max_poses)
            opts.setdefault(r0.config, r0)  # the verified solution is always an option
            return opts

        # fast path: points already in the stop's most common configuration keep it; only the
        # others are searched. If they can all join it, one zone is optimal; else search all.
        labels = [results[p].config for p in pids]
        major = max(set(labels), key=labels.count)
        _prefetch_configs(
            checker, t, index, sites.T(v), [p for p in pids if results[p].config != major], max_poses
        )
        options: dict[int, dict[str, PointResult]] = {
            pid: {major: results[pid]} if results[pid].config == major else search(pid) for pid in pids
        }
        if not all(major in o for o in options.values()):
            again = [pid for pid, o in options.items() if len(o) == 1 and major in o]
            _prefetch_configs(checker, t, index, sites.T(v), again, max_poses)
            options = {pid: search(pid) if len(o) == 1 and major in o else o for pid, o in options.items()}
        chosen = min_zones({pid: {c: _quality(r) for c, r in o.items()} for pid, o in options.items()})
        final: dict[str, list[int]] = {}
        for config, members in chosen:
            for pid in members:
                r = options[pid][config]
                if r is not results[pid]:
                    ok, _ = _robust(checker, t, index[pid], r, v, sites)
                    if not ok:  # keep the verified solution (it passed the robustness checks)
                        r = results[pid]
                results[pid] = r
                final.setdefault(r.config, []).append(pid)
        for config, members in sorted(final.items(), key=lambda z: -len(z[1])):
            order = order_path(np.array([t.P_world[index[p]] for p in members]))
            members = [members[j] for j in order]
            path, jumps = unwrap_path([results[p].q for p in members], checker.lo, checker.hi)
            for seq, (pid, q) in enumerate(zip(members, path, strict=True)):
                r = results[pid]
                r.q, r.zone, r.seq = q, len(zones), seq
                lm = np.minimum(q - checker.robot.lower, checker.robot.upper - q).min()
                r.limit_margin_deg = float(np.degrees(lm))
            zones.append(Zone(station=k, config=config, points=members, wrap_jumps=jumps))
            changes += jumps
        changes += len(final) - 1
    return zones, changes


def zone_counts(zones) -> tuple[int, int]:
    """(extra groups beyond one per stop, wrist jumps inside groups)."""
    return len(zones) - len({z.station for z in zones}), sum(z.wrap_jumps for z in zones)


def violations(unreached: int, zones, zoning: bool, require_all: bool, require_one_config: bool) -> list[str]:
    """The "must be met" rules a plan breaks."""
    out = []
    if require_all and unreached:
        out.append(f"{unreached} point(s) unreached")
    if require_one_config:
        flips = zone_counts(zones)[1]
        if not zoning:
            out.append("configuration groups not computed (zoning off)")
        elif flips:
            out.append(f"{flips} wrist flip(s) inside a group")
    return out


MOTION_REF = np.pi / 2  # rad: a 90 deg joint move between neighbouring points costs 1


def stop_sequences(checker, t, assignment: dict[int, int], results: dict[int, PointResult], zones):
    """Per stop, its point sequences as the paths run them (core.paths.sequences): the zones,
    or without zones one group per joint configuration (largest first), each in serpentine
    order. Returns {stop: [[point ids], ...]}."""
    index = {int(pid): i for i, pid in enumerate(t.ids)}
    out: dict[int, list[list[int]]] = {}
    if zones:
        for z in zones:
            pts = [p for p in z.points if results.get(p) is not None and results[p].q is not None]
            if pts:
                out.setdefault(z.station, []).append(pts)
        return out
    by: dict[int, dict[str, list[int]]] = {}
    for pid, k in assignment.items():
        r = results.get(pid)
        if r is None or r.q is None or r.status == "unreachable":
            continue
        by.setdefault(k, {}).setdefault(config_label(checker.ik, np.asarray(r.q, float)), []).append(pid)
    for k, groups in by.items():
        for members in sorted(groups.values(), key=len, reverse=True):
            members = sorted(members)
            if len(members) > 2:
                order = order_path(np.array([t.P_world[index[p]] for p in members]))
                members = [members[j] for j in order]
            out.setdefault(k, []).append(members)
    return out


def config_changes(seqs: dict[int, list[list[int]]]) -> int:
    """Extra configuration groups beyond one per stop."""
    return sum(len(v) - 1 for v in seqs.values())


def arm_motion(seqs: dict[int, list[list[int]]], results: dict[int, PointResult]) -> float:
    """Mean over neighbouring points of a stop's sequences of the largest joint move
    (nearest 2*pi copy), in units of ``MOTION_REF``: large arm moves between points cost."""
    steps = []
    for groups in seqs.values():
        for members in groups:
            q = [np.asarray(results[p].q, float) for p in members]
            for a, b in zip(q[:-1], q[1:], strict=True):
                d = np.mod(b - a + np.pi, 2 * np.pi) - np.pi
                steps.append(float(np.max(np.abs(d))))
    return float(np.mean(steps)) / MOTION_REF if steps else 0.0
