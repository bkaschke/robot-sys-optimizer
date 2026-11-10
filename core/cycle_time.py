"""Cycle time (v8 T6): how long one part takes, from the planned motion.

Every move is a trapezoidal profile (accelerate, cruise, brake; a triangle when too short to
reach cruise speed). Joint moves are synchronized: all joints start and stop together, so the
move takes as long as its slowest joint. Speeds are the robot program's (``CycleSettings``),
capped by the robot's joint speed limits (URDF velocity limits).

- ``plan_estimate``: from a station plan without paths (the planner's cost term and the
  Results card): joint moves between neighbouring points of each stop's sequences, one joint
  move between sequences, carrier moves between stops (the carrier's travel metric at its
  speed) and a settle time per stop.
- ``path_time``: from built paths (core.paths): surface runs at the process speed, lifts /
  retreats at the approach speed, joint moves (home, reconfigure, planned moves) as joint
  runs, carrier runs per axis, plus the settle time per stop. Returns the parts too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CycleSettings:
    joint_speed: float = 1.05  # rad/s (UR movej default), capped per joint by the robot's limits
    joint_acc: float = 1.4  # rad/s^2 (UR movej default)
    process_speed: float = 0.1  # m/s along the surface
    approach_speed: float = 0.25  # m/s lifts, approach / retract (UR movel default)
    linear_acc: float = 1.2  # m/s^2
    carrier_speed: float = 0.5  # m/s (rail, gantry axes, mobile base); rad/s for a mobile yaw
    carrier_acc: float = 0.5  # m/s^2
    lift_speed: float = 0.1  # m/s
    settle_s: float = 1.0  # per stop: carrier settles / localizes before the arm starts


def trapezoid(d: float, v: float, a: float) -> float:
    """Time for distance ``d`` from rest to rest at max speed ``v`` and acceleration ``a``."""
    d = abs(float(d))
    if d <= 0 or v <= 0:
        return 0.0
    if a <= 0:
        return d / v
    if d <= v * v / a:  # never reaches v: accelerate half way, brake half way
        return 2.0 * np.sqrt(d / a)
    return d / v + v / a


def joint_move_time(dq, cs: CycleSettings, v_limits=None) -> float:
    """Synchronized joint move: the slowest joint's trapezoid (2*pi wraps not removed:
    pass the actual joint path)."""
    dq = np.abs(np.asarray(dq, float))
    v = np.full(len(dq), cs.joint_speed)
    if v_limits is not None:
        v = np.minimum(v, np.asarray(v_limits, float)[: len(dq)])
    return max((trapezoid(d, vi, cs.joint_acc) for d, vi in zip(dq, v, strict=True)), default=0.0)


def plan_estimate(
    seqs: dict[int, list[list[int]]],
    results: dict,
    stop_moves: list[float],
    cs: CycleSettings,
    v_limits=None,
) -> dict[str, float]:
    """Estimate for a plan: ``seqs`` = {stop: [[point ids in order], ...]} (opt.zoning
    .stop_sequences), ``stop_moves`` = carrier travel between consecutive stops (m, the
    carrier's metric). Returns {"joint", "carrier", "settle", "total"} in seconds."""
    joint = 0.0
    for groups in seqs.values():
        prev = None
        for members in groups:
            for p in members:
                q = np.asarray(results[p].q, float)
                if prev is not None:
                    joint += joint_move_time(np.mod(q - prev + np.pi, 2 * np.pi) - np.pi, cs, v_limits)
                prev = q
    carrier = sum(trapezoid(d, cs.carrier_speed, cs.carrier_acc) for d in stop_moves)
    settle = cs.settle_s * len(seqs)
    return {"joint": joint, "carrier": carrier, "settle": settle, "total": joint + carrier + settle}


def path_time(plan, cs: CycleSettings, v_limits=None) -> dict[str, float]:
    """Cycle time of built paths (core.paths.PathPlan). Returns seconds per kind of motion:
    {"process", "approach", "joint", "carrier", "settle", "total"}."""
    out = dict.fromkeys(("process", "approach", "joint", "carrier", "settle"), 0.0)
    for st in plan.stations:
        if st.samples:
            out["settle"] += cs.settle_s
    samples = plan.samples
    run_kind, run = None, None  # accumulated distance of the current run

    def close() -> None:
        if run_kind is None:
            return
        if run_kind == "process":
            out["process"] += trapezoid(run, cs.process_speed, cs.linear_acc)
        elif run_kind == "approach":
            out["approach"] += trapezoid(run, cs.approach_speed, cs.linear_acc)
        elif run_kind == "joint":
            out["joint"] += joint_move_time(run, cs, v_limits)
        elif run_kind == "carrier":
            out["carrier"] += max(
                (
                    trapezoid(d, cs.lift_speed if k == "lift" else cs.carrier_speed, cs.carrier_acc)
                    for k, d in run.items()
                ),
                default=0.0,
            )

    for a, b in zip(samples[:-1], samples[1:], strict=True):
        ca, cb = a.carrier, b.carrier
        if b.kind == "carrier" or (ca is not None and cb is not None and ca != cb) or a.station != b.station:
            kind = "carrier"
        elif b.kind in ("home",) or a.q is None or b.q is None or b.a == b.b and b.s == 0.0 and a.b != b.a:
            kind = "joint"
        elif b.kind in ("lift", "retreat") or a.kind in ("lift", "retreat"):
            kind = "approach"
        else:
            kind = "process"
        if kind != run_kind:
            close()
            run_kind, run = kind, ({} if kind == "carrier" else (np.zeros(6) if kind == "joint" else 0.0))
        if kind == "carrier":
            for k in set(ca or {}) | set(cb or {}):
                d = abs(float((cb or {}).get(k, 0.0)) - float((ca or {}).get(k, 0.0)))
                run[k] = run.get(k, 0.0) + d
        elif kind == "joint":
            if a.q is not None and b.q is not None:
                dq = np.abs(np.asarray(b.q, float) - np.asarray(a.q, float))
                run = run[: len(dq)] + dq if len(run) == len(dq) else dq
        else:
            run += float(np.linalg.norm(b.T_part_tcp[:3, 3] - a.T_part_tcp[:3, 3]))
    close()
    out["total"] = float(sum(out.values()))
    return out
