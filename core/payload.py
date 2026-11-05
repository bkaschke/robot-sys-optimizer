"""Static payload checks: joint torque utilization and rated payload.

tau = g_arm(q) + J_com^T(-m g) + J_tcp^T F, where F is the process force the tool exerts
on the part (along -target.z). Wrist-moment / payload-diagram checks need datasheet values
in the robot's spec.yaml; while those are missing the check is reported as skipped,
never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.robot import Robot


@dataclass
class PayloadResult:
    torque_util: np.ndarray  # |tau| / effort limit, per joint
    max_util: float
    ok: bool
    notes: list[str] = field(default_factory=list)


def torque_limits(robot: Robot) -> np.ndarray | None:
    """Per-joint limits: spec.yaml first, URDF effort limits as fallback."""
    spec = robot.spec.payload.joint_torque_Nm
    if spec and all(t is not None for t in spec) and len(spec) == robot.nq:
        return np.array(spec, float)
    eff = robot.effort
    return eff if np.all(eff > 0) else None


def static_check(
    robot: Robot,
    q: np.ndarray,
    mass_kg: float,
    com_flange: np.ndarray,
    force_base: np.ndarray | None = None,
    tcp_flange: np.ndarray | None = None,
    util_limit: float = 0.8,
) -> PayloadResult:
    notes: list[str] = []
    wrench = None
    if force_base is not None and np.any(force_base):
        wrench = np.concatenate([force_base, np.zeros(3)])
    tau = robot.static_torques(
        q,
        payload_kg=mass_kg,
        payload_com=com_flange,
        wrench=wrench,
        wrench_point=tcp_flange if tcp_flange is not None else np.zeros(3),
    )
    lim = torque_limits(robot)
    if lim is None:
        notes.append("joint torque limits unknown: torque check skipped")
        util = np.zeros(robot.nq)
    else:
        util = np.abs(tau) / lim
    ok = bool(util.max() <= util_limit)
    rated = robot.spec.payload.rated_kg
    if rated is not None and mass_kg > rated:
        ok = False
        notes.append(f"tool mass {mass_kg:.1f} kg exceeds rated payload {rated:g} kg")
    if not robot.spec.payload.is_complete():
        notes.append("payload diagram / wrist moments not in spec.yaml: check skipped")
    return PayloadResult(util, float(util.max()), ok, notes)
