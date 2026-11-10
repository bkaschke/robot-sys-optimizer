"""Collision checks between path samples (v8 T7).

The path builder checks every sample; a move between two clear samples can still clip the
part, a stay-out zone or the carrier (the arm sweeps through it). ``check_moves`` interpolates
each move in joint space, so that no joint moves more than ``max_step_deg`` between checks,
and flags the move's end sample with the issue ``motion`` when an intermediate pose collides.

Joint interpolation is exact for joint moves (home, reconfigure, planned moves). For surface,
lift and retreat moves the robot moves the TCP in a straight line instead; between samples a
few mm apart the two differ by far less than the clearance, so the check stands for both.
Carrier moves (the base moves) are checked when they are planned (core.paths), not here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

MAX_STEP_DEG = 2.0


def check_moves(
    checker,
    plan,
    max_step_deg: float = MAX_STEP_DEG,
    cancelled: Callable[[], bool] | None = None,
) -> list[int]:
    """Indices (into ``plan.samples``) of samples whose incoming move collides; marks them
    with ``issue = "motion"`` (samples already flagged keep their issue). Restores the
    checker's base pose."""
    samples = plan.samples
    base_of = {st.station: st.T_world_base for st in plan.stations}
    step = math.radians(max_step_deg)
    T0, cur = checker.T_world_base.copy(), None
    bad: list[int] = []
    try:
        for i in range(1, len(samples)):
            a, b = samples[i - 1], samples[i]
            if a.q is None or b.q is None or a.station != b.station or "carrier" in (a.kind, b.kind):
                continue
            Ta = a.T_world_base if a.T_world_base is not None else base_of[a.station]
            Tb = b.T_world_base if b.T_world_base is not None else base_of[b.station]
            if not np.allclose(Ta, Tb):
                continue  # the base moves (lift staging): checked with the carrier path
            qa, qb = np.asarray(a.q, float), np.asarray(b.q, float)
            n = int(math.ceil(float(np.max(np.abs(qb - qa))) / step))
            if n <= 1:
                continue  # no pose between the two samples to check
            if cancelled is not None and cancelled():
                break
            if cur is None or not np.allclose(cur, Tb):
                checker.set_base(Tb)
                cur = Tb
            for k in range(1, n):
                if checker.scene.collides(qa + (qb - qa) * (k / n), b.tcp):
                    bad.append(i)
                    if not b.issue:
                        b.issue = "motion"
                    break
    finally:
        checker.set_base(T0)
    return bad
