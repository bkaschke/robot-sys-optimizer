"""Point order of the paths (core.paths): per stop the planner's groups (zones, else one per
joint configuration and target group), each ordered by a path pattern (core.patterns), with the
moves that lift off the surface (crossings, passing over other target points)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.configs import order_path
from core.feasibility import PointResult
from core.patterns import (
    CORNERS,
    PatternSettings,
    crossing_lifts,
    order_points,
    pass_over_lifts,
    point_spacing,
)


def sequences(
    sol,
    results: dict[int, PointResult],
    P_part: dict[int, np.ndarray],
    label_of: Callable | None = None,
    group_of: dict[int, str] | None = None,
) -> dict[int, list]:
    """Per station: [(zone index | None, [point ids in path order])]. Without zones and with
    ``label_of(result)`` (joint configuration label), a stop's points are split by
    configuration, largest group first: one configuration change per extra configuration,
    not one per switch along the path. With ``group_of`` (point id -> target group), each
    sequence is also split by target group (v6): the tool is not dragged over an edge of the
    part from one face to the next; a planned joint move joins them."""
    out = _sequences(sol, results, P_part, label_of)
    if not group_of:
        return out
    split: dict[int, list] = {}
    for k, seqs in out.items():
        for zone, ids in seqs:
            parts: dict[str, list[int]] = {}
            for p in ids:  # keeps the path order inside each group
                parts.setdefault(group_of.get(p, ""), []).append(p)
            for g in parts.values():
                if len(parts) > 1 and len(g) > 2:  # re-order each face on its own
                    g = [g[i] for i in order_path(np.array([P_part[p] for p in g]))]
                split.setdefault(k, []).append((zone, g))
    return split


def _sequences(sol, results, P_part, label_of) -> dict[int, list]:
    out: dict[int, list] = {}

    def usable(p: int) -> bool:  # solved, and still a target (points deleted after planning are skipped)
        return p in P_part and results.get(p) is not None and results[p].q is not None

    if sol.zones:
        for zi, z in enumerate(sol.zones):
            pts = [p for p in z.points if usable(p)]
            if pts:
                out.setdefault(z.station, []).append((zi, pts))
        return out
    by: dict[int, list[int]] = {}
    for pid, k in sol.assignment.items():
        if usable(pid):
            by.setdefault(k, []).append(pid)
    for k, pids in by.items():
        split: dict[str, list[int]] = {}
        for p in sorted(pids):
            split.setdefault(label_of(results[p]) if label_of is not None else "", []).append(p)
        out[k] = []
        for ids in sorted(split.values(), key=len, reverse=True):
            order = order_path(np.array([P_part[p] for p in ids])) if len(ids) > 2 else np.arange(len(ids))
            out[k].append((None, [ids[i] for i in order]))
    return out


@dataclass
class Seq:
    zone: int | None
    pids: list[int]
    lifts: list[bool]  # per move pids[k] -> pids[k + 1]: lift off the surface


def ordered_sequences(
    sol,
    results,
    points: dict,
    pattern: PatternSettings | None = None,
    label_of: Callable | None = None,
    group_of: dict[int, str] | None = None,
) -> dict[int, list[Seq]]:
    """Per stop: its groups (zones) as Seq. Without a pattern, the planner's order. With one,
    each group is ordered by the pattern, groups follow each other nearest first, and a later
    group starts at the corner nearest where the previous one ended. With ``no_crossing`` a
    move lifts when it crosses an earlier move or would run over another target point."""
    P_part = {p: v[0] for p, v in points.items()}
    base = sequences(sol, results, P_part, label_of, group_of)
    if pattern is None:
        return {k: [Seq(z, ids, [False] * (len(ids) - 1)) for z, ids in v] for k, v in base.items()}
    out: dict[int, list[Seq]] = {}
    P_all = np.array([v[0] for v in points.values()], float).reshape(-1, 3)
    spacing = point_spacing(P_all) if pattern.no_crossing else 0.0
    for k, groups in base.items():
        cents = [np.mean([P_part[i] for i in ids], axis=0) for _, ids in groups]
        left, chain = list(range(1, len(groups))), [0]
        while left:
            here = groups[chain[-1]][0]  # finish this zone (configuration) before the next
            j = min(
                left,
                key=lambda i: (groups[i][0] != here, float(np.linalg.norm(cents[i] - cents[chain[-1]]))),
            )
            left.remove(j)
            chain.append(j)
        seqs, end = [], None
        for gi in chain:
            zone, ids = groups[gi]
            P = np.array([points[i][0] for i in ids])
            N = np.array([points[i][1] for i in ids])
            if end is None:
                o = order_points(P, N, pattern)
            else:  # start near the previous group's end
                o = min(
                    (order_points(P, N, pattern, c) for c in CORNERS),
                    key=lambda r: float(np.linalg.norm(P[r.order[0]] - end)),
                )
            pids = [ids[i] for i in o.order]
            lifts = np.zeros(len(pids) - 1, bool)
            if pattern.no_crossing:  # lift over crossings and over other points (sand each once)
                lifts = crossing_lifts(o.uv, o.order, o.passes) | pass_over_lifts(P[o.order], P_all, spacing)
            seqs.append(Seq(zone, pids, [bool(x) for x in lifts]))
            end = P[o.order[-1]]
        out[k] = seqs
    return out
