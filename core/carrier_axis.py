"""Rail / gantry axis lengths in one setting (v7 round 2, T13).

``Base.axis_length_m`` holds the length of each linear axis ("x": the rail, or the gantry's
X rails; "y": the gantry bridge). Bodies marked ``Primitive.stretch`` follow it: a box gets
that size along the axis, a cylinder lying along the axis that height, each keeping its
centre. Unless ``Base.manual_travel``, the travel range follows too: the moving bodies stop
``end_stop_m`` before the ends of the stretched body,

    lo = c - L/2 + end_stop - a_min,   hi = c + L/2 - end_stop - a_max

with c the stretched body's centre along the axis and [a_min, a_max] the extent of the bodies
that move along that axis, relative to their moving frame (rail x and gantry y: the carriage
and lift-stage bodies; gantry x: those plus the bridge). For a symmetric carriage centred on
the track this is +-(L/2 - carriage half-extent - end stop).

Projects without stretched bodies are left exactly as they are (lengths not used, travel
stays manual).
"""

from __future__ import annotations

import numpy as np

from core.schema import Base, Pose, Primitive
from core.transforms import rpy_to_matrix

AXES = {"rail": ("x",), "gantry": ("x", "y")}
IDX = {"x": 0, "y": 1}


def axes(base: Base) -> tuple[str, ...]:
    """The linear axes of this carrier that can have a length (rail: x, gantry: x and y)."""
    return AXES.get(base.carrier or "", ()) if base.mode == "movable" else ()


def uses_lengths(base: Base) -> bool:
    """True when some body follows an axis length of this carrier."""
    ax = axes(base)
    return any(b.stretch in ax for b in base.bodies)


def stretch_body(b: Primitive, length: float) -> Primitive:
    """``b`` resized to ``length`` along its stretch axis (parent frame), centre kept."""
    i = IDX[b.stretch]
    R = rpy_to_matrix(b.pose.rpy)
    p = np.asarray(b.pose.xyz, float)
    if b.shape == "box":
        k = int(np.argmax(np.abs(R[i, :])))  # the box's own axis closest to the travel axis
        size = list(b.size)
        old = size[k]
        size[k] = float(length)
        upd: dict = {"size": tuple(size)}
        if k == 2:  # box frame at the bottom face: keep the centre where it was
            upd["pose"] = Pose(xyz=tuple(p + R[:, 2] * (old - length) / 2), rpy=b.pose.rpy)
        return b.model_copy(update=upd)
    if b.shape == "cylinder" and abs(R[i, 2]) > 0.9:  # lying along the axis
        new_p = p + R[:, 2] * (b.height - length) / 2  # frame at a cap: keep the centre
        return b.model_copy(update={"height": float(length), "pose": Pose(xyz=tuple(new_p), rpy=b.pose.rpy)})
    return b  # meshes and cylinders across the axis keep their size


def stretched(base: Base, bodies: list[Primitive]) -> list[Primitive]:
    """``bodies`` with every stretch body at its axis length (others unchanged)."""
    ax = axes(base)
    out = []
    for b in bodies:
        L = base.axis_length_m.get(b.stretch) if b.stretch in ax else None
        out.append(stretch_body(b, L) if L else b)
    return out


def _centre(b: Primitive, i: int) -> float:
    R = rpy_to_matrix(b.pose.rpy)
    p = np.asarray(b.pose.xyz, float)
    h = b.size[2] if b.shape == "box" else (b.height or 0.0) if b.shape == "cylinder" else 0.0
    return float((p + R[:, 2] * h / 2)[i])


def axis_travel(base: Base, resolve=None) -> dict[str, tuple[float, float]]:
    """Computed travel range per axis that has a length and a stretched body. Raises
    ValueError when the bodies leave no travel (axis too short)."""
    from core.carrier import primitive_mesh

    out: dict[str, tuple[float, float]] = {}
    bodies = stretched(base, base.bodies)
    for ax in axes(base):
        L = base.axis_length_m.get(ax)
        rails = [b for b in bodies if b.stretch == ax]
        if not L or not rails:
            continue
        i = IDX[ax]
        c = _centre(rails[0], i)
        attach = {"carriage", "lift"} | ({"x_axis"} if base.carrier == "gantry" and ax == "x" else set())
        movers = [b for b in bodies if b.attach in attach]
        if movers:
            pts = []
            for b in movers:
                m = primitive_mesh(b, resolve)
                T = rpy_to_matrix(b.pose.rpy)
                pts.append(m.vertices @ T.T + np.asarray(b.pose.xyz, float))
            coords = np.concatenate(pts)[:, i]
            a_min, a_max = float(coords.min()), float(coords.max())
        else:
            a_min = a_max = 0.0
        lo = c - L / 2 + base.end_stop_m - a_min
        hi = c + L / 2 - base.end_stop_m - a_max
        if lo > hi:
            raise ValueError(
                f"{ax.upper()} axis: {L * 1000:.0f} mm is too short for the moving bodies "
                f"({(a_max - a_min) * 1000:.0f} mm) and the end stops ({base.end_stop_m * 1000:.0f} mm each)"
            )
        out[ax] = (round(float(lo), 9), round(float(hi), 9))
    return out


def apply_axis_lengths(base: Base, resolve=None) -> Base:
    """``base`` with the stretch bodies at their lengths and, unless ``manual_travel``, the
    computed travel ranges in ``joints``. Unchanged when no body follows a length."""
    if not uses_lengths(base):
        return base
    upd: dict = {"bodies": stretched(base, base.bodies)}
    if not base.manual_travel:
        travel = axis_travel(base, resolve)
        if travel:
            upd["joints"] = base.joints.model_copy(update=travel)
    return base.model_copy(update=upd)
