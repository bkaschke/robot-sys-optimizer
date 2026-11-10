"""Hose / dress pack modelled as a shape on the arm (Optimize -> Cable & safety -> Hose pack =
*Hose along the arm*).

The hose is a chain of capsules (``Hose.radius``) from the upper arm to the wrist: one per arm
segment, from each joint's origin to the next one's, moved ``offset`` away from that line, in
the plane the joint turns in (perpendicular to both the joint axis and the segment), on the
``side`` (+1 / -1) the user picks. Each capsule is rigid on its joint's link, so it moves with
the arm exactly like the link meshes. The collision scene checks it against the part, the
stay-out zones, the floor and the carrier bodies (not against the arm or the tool it rides on).

Defined from the robot model alone (pinocchio joint placements and axes), so it works for any
6-axis arm in the registry; it is a layout model (a straight hose per segment), not a cable
simulation: the slack loop at the wrist is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hose:
    radius: float = 0.03  # m (hose diameter / 2)
    offset: float = 0.15  # m from the joint line to the hose axis
    side: float = 1.0  # +1 / -1: which side of the arm
    first_joint: int = 2  # pinocchio joint index the hose starts at (the upper arm)


def joint_axis(model, j: int) -> np.ndarray:
    """Axis of revolute joint ``j`` in its own frame."""
    name = model.joints[j].shortname()
    for k, ax in (("RX", (1.0, 0, 0)), ("RY", (0, 1.0, 0)), ("RZ", (0, 0, 1.0))):
        if name.endswith(k):
            return np.array(ax)
    try:
        return np.asarray(model.joints[j].extract().axis, float)
    except AttributeError:
        return np.array([0.0, 0.0, 1.0])


def _align_z(d: np.ndarray) -> np.ndarray:
    """Rotation taking +z to the unit vector ``d``."""
    z = d / np.linalg.norm(d)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def segments(robot, hose: Hose) -> list[tuple[int, np.ndarray, float]]:
    """(pinocchio joint, T_joint_capsule (capsule centre, z along it), length) per segment,
    from ``first_joint`` up to the joint carrying the flange."""
    m = robot.model
    last = m.frames[robot.flange_frame_id].parentJoint
    out = []
    for j in range(hose.first_joint, last):
        d = np.asarray(m.jointPlacements[j + 1].translation, float)
        L = float(np.linalg.norm(d))
        if L < 1e-3:
            continue
        u = d / L
        n = np.cross(joint_axis(m, j), u)
        if np.linalg.norm(n) < 1e-6:  # segment along the axis: any perpendicular
            n = np.cross(u, [0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0.0, 0.0])
        n = hose.side * n / np.linalg.norm(n)
        T = np.eye(4)
        T[:3, :3] = _align_z(u)
        T[:3, 3] = d / 2 + hose.offset * n
        out.append((j, T, L))
    return out
