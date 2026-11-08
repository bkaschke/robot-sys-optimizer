"""Joint-configuration zoning of a station plan (runs after ``placement.verify``).

For every station:
  1. each assigned point is re-checked for EVERY joint configuration it can be done in
     (``FeasibilityChecker.check_point_configs``, one pass over the candidate poses);
  2. the fewest configurations that cover all points are chosen (``core.configs.min_zones``,
     exact), each point taking its best solution among them;
  3. the chosen solutions get the station's robustness checks (tip-over, localization)
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
        options: dict[int, dict[str, PointResult]] = {}
        for pid in pids:
            i = index[pid]
            r0 = results[pid]
            r0.config = config_label(checker.ik, r0.q)
            opts = checker.check_point_configs(pid, t.P_part[i], t.N_part[i], t.groups[i], max_poses)
            opts.setdefault(r0.config, r0)  # the verified solution is always an option
            options[pid] = opts
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
