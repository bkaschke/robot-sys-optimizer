"""Placement cost (plan § Cost function); all terms roughly 0-1 except the counts.

J = w_L L/L_ref + w_U N_unreached + w_S (k-1) + w_T D_travel/D_ref + w_M mean(1/(1+mu))
  + w_C mean(max(0, 1 - c/c_ref)) + w_J mean(s_limit) + w_P max(tau_util)
  + w_standoff mean(carrier standoff penalty)   [M9: soft preferred-standoff term]
  + w_group N_groups + w_wrist_flip N_flips      [extra configuration groups per stop, wrist jumps]
  + w_motion mean(max joint move between neighbouring points) / 90 deg   [v6]

mu is |det J| normalized by the robot's best value (from the reachability map), c the
part clearance, travel the carrier's own metric (rail |ds|, gantry max axis, mobile
straight line), s_limit = max(0, 1 - joint-limit margin / 30 deg).
"""

from __future__ import annotations

import numpy as np

from core.schema import Weights

C_REF = 0.05  # m
D_REF = 1.0  # m
LIMIT_REF_DEG = 30.0


def path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def cycle_estimate(
    seqs, results, stations: list[dict], sites, robot, cs=None, zones=None, t=None
) -> dict[str, float]:
    """v8 T6: estimated cycle time of a plan (core.cycle_time.plan_estimate) with the carrier's
    travel metric between consecutive stops and the robot's joint speed limits. ``zones``
    (core.speed_zones) + ``t`` (opt.placement.Targets, world points): moves to points inside
    a reduced-speed zone are slower."""
    from core.cycle_time import CycleSettings, plan_estimate

    moves = [float(sites.travel(a, b)) for a, b in zip(stations[:-1], stations[1:], strict=True)]
    share = None
    if zones and t is not None:
        from core.speed_zones import share_at

        f = share_at(zones, t.P_world)
        share = {int(i): float(v) for i, v in zip(t.ids, f, strict=True) if v < 1.0}
    return plan_estimate(seqs, results, moves, cs or CycleSettings(), robot.model.velocityLimit, share)


def wrap_excess(seqs, results, limit_deg: float = 360.0, joint: int = 5) -> float:
    """v8 T18: per stop, how far joint 6 turns over the stop's points (max - min of its planned
    values, the cable / hose wraps that much) beyond ``limit_deg``; summed, in units of 90 deg."""
    total = 0.0
    for groups in seqs.values():
        q = [float(results[p].q[joint]) for members in groups for p in members if results[p].q is not None]
        if q:
            total += max(0.0, np.degrees(max(q) - min(q)) - limit_deg) / 90.0
    return total


def cost_breakdown(
    point_results: list,
    n_unreached: int,
    station_xy: np.ndarray,
    weights: Weights,
    tool_length: float = 0.0,
    L_ref: float = 1.0,
    max_manip: float = 1.0,
    travel: float | None = None,
    standoff_penalty: float = 0.0,
    groups: int = 0,
    wrist_flips: int = 0,
    motion: float = 0.0,
    cycle_s: float = 0.0,
    wrap: float = 0.0,
) -> tuple[float, dict[str, float]]:
    ok = [r for r in point_results if r.status != "unreachable"]
    mu = np.array([r.manipulability / max_manip for r in ok]) if ok else np.zeros(0)
    clear = np.array([min(r.clearance, 10.0) for r in ok]) if ok else np.zeros(0)
    lim = np.array([r.limit_margin_deg for r in ok]) if ok else np.zeros(0)
    terms = {
        "L": tool_length / L_ref if L_ref > 0 else 0.0,
        "unreached": float(n_unreached),
        "stations": float(len(station_xy) - 1),
        "travel": (path_length(station_xy) if travel is None else travel) / D_REF,
        "manip": float(np.mean(1 / (1 + mu))) if len(mu) else 1.0,
        "clearance": float(np.mean(np.maximum(0, 1 - clear / C_REF))) if len(clear) else 0.0,
        "limits": float(np.mean(np.maximum(0, 1 - lim / LIMIT_REF_DEG))) if len(lim) else 0.0,
        "standoff": float(standoff_penalty),
        "group": float(groups),
        "wrist_flip": float(wrist_flips),
        "motion": float(motion),
        "cycle": float(cycle_s) / 60.0,  # v8 T6: per minute
        "wrap": float(wrap),  # v8 T18: joint-6 turn beyond the limit, per 90 deg
    }
    w = weights.model_dump()
    weighted = {k: w[k] * v for k, v in terms.items()}
    return float(sum(weighted.values())), weighted
