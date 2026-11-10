"""Check a robot registry entry (assets/robots/<name>/) before using it:

    python scripts/check_robot.py <name>

Loads spec.yaml and the URDF, builds the IK solver named by ``ik`` (for ``opw`` this also
checks the parameters against the URDF), runs an IK round trip on random poses, and checks
what the planners need: an analytic solver (station planner, tool optimizer, maps), a
full-turn first joint (reachability map symmetry), the reach and a home pose inside the
joint limits. Prints one line per check; exit code 1 if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ik import angle_diff, make_ik  # noqa: E402
from core.robot import Robot, list_robots  # noqa: E402


def check(name: str, n: int = 500) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []
    try:
        robot = Robot.from_registry(name)
    except Exception as err:  # report any load error as a failed check
        return [(False, f"load spec.yaml + URDF: {err}")]
    out.append(
        (True, f"loaded: {robot.nq} joints, base '{robot.spec.base_link}', flange '{robot.spec.flange_link}'")
    )
    try:
        ik = make_ik(robot)
    except Exception as err:
        return [*out, (False, f"IK solver '{robot.spec.ik}': {err}")]
    analytic = getattr(ik, "batch", False)
    out.append(
        (
            analytic,
            f"IK solver '{robot.spec.ik}'"
            + (
                ""
                if analytic
                else ": numeric only (feasibility check works; station planner, tool "
                "optimizer and reachability maps need ur_analytic or opw)"
            ),
        )
    )
    rng = np.random.default_rng(0)
    worst, missed = 0.0, 0
    for _ in range(n if analytic else 10):
        q = robot.random_q(rng)
        T = robot.fk(q)
        sols = ik.solve(T)
        worst = max([worst, *(float(np.abs(robot.fk(s) - T).max()) for s in sols)])
        missed += not any(np.abs(angle_diff(s, q)).max() < 1e-5 for s in sols)
    out.append(
        (
            worst < 1e-6 and (missed == 0 or not analytic),
            f"IK round trip: worst {worst:.1e} m, {missed} poses missed",
        )
    )
    turn = robot.urdf_upper[0] - robot.urdf_lower[0]
    out.append(
        (turn >= 2 * np.pi - 1e-6, f"joint 1 range {np.degrees(turn):.0f} deg (maps need a full turn)")
    )
    out.append((robot.spec.reach_m is not None, f"reach_m: {robot.spec.reach_m}"))
    h = robot.spec.home_deg
    if h is None:
        out.append((True, "home_deg not set: all joints 0 (clipped into the limits)"))
    else:
        q = np.radians(h)
        ok = len(h) == robot.nq and robot.within_limits(q)
        out.append((ok, f"home_deg {h} {'inside' if ok else 'OUTSIDE'} the joint limits"))
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: check_robot.py <name>   (registry: {', '.join(list_robots())})")
        return 2
    results = check(sys.argv[1])
    for ok, msg in results:
        print(("PASS " if ok else "FAIL ") + msg)
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
