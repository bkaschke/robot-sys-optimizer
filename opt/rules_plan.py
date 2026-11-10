"""Rules-only station planner (v7 round 2, T15): no costs, only hard rules.

Rules (all hard):
  1. every target reached;
  2. tool orientation within each group's spin / tilt tolerance (core.feasibility poses);
  3. joint limits with the margin (core.feasibility);
  4. no collision with the clearance (core.feasibility);
  5. one joint configuration per group, with no wrist jump > 180 deg (zoning on);
  6. manipulability >= limit x the robot's max (CheckSettings.min_manip; reason "singular").

Rules 2-4 hold for every point the checker reaches, so a plan breaks a rule when points stay
unreached (rule 1, or rule 6 when the reason is "singular") or a group's path has wrist flips
(rule 5).

Search (no CMA-ES: refinement optimizes a cost, which rules mode has none of). ``run_k(k,
exclude)`` is one quick plan (set cover + exact verification + zoning) with exactly k stops,
the stops in ``exclude`` left out of the set cover; None when k stops cannot cover the map
(``exclude=None``: plan with k stops even then, for the closest plan).
  * Fewest stations: k = 1, 2, ... Max stations; at the first k with a passing plan, try up
    to 3 alternative covers (leave one chosen stop out at a time) and keep the one with the
    fewest groups. Stop there.
  * Fewest groups: the same, then keep going while more stops give fewer groups (stop at Max
    stations, after ``patience`` counts without fewer groups, or when k stops can no longer
    beat the best: every stop has at least one group). Fewest groups, then fewer stations.
No plan passes: the closest one (fewest broken rules, then fewest broken points), marked with
its broken rules in ``Solution.violations``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.schema import Solution

RULES = {
    1: "every target reached",
    2: "tool orientation within each group's spin / tilt tolerance",
    3: "joint limits (with the margin)",
    4: "no collision (with the clearance)",
    5: "one joint configuration per group, no wrist jump > 180°",
    6: "manipulability at least the singularity limit",
}
OBJECTIVES = ("Fewest stations", "Fewest groups")
ALTERNATIVES = 3


def rule_counts(sol: Solution, results: dict) -> dict[int, int]:
    """Broken rules of a plan: rule number -> number of points (rule 5: wrist flips)."""
    singular = sum(1 for pid in sol.unreached if getattr(results.get(pid), "reason", "") == "singular")
    out = {}
    if len(sol.unreached) - singular:
        out[1] = len(sol.unreached) - singular
    if singular:
        out[6] = singular
    flips = sum(z.wrap_jumps for z in sol.zones)
    if flips:
        out[5] = flips
    return out


def rule_texts(counts: dict[int, int], k: int) -> list[str]:
    what = {
        1: "point(s) unreached",
        5: "wrist flip(s) inside a group",
        6: "point(s) only reachable close to a singularity",
    }
    plural = "s" if k != 1 else ""
    return [f"rule {r}: {n} {what[r]} at {k} station{plural}" for r, n in sorted(counts.items())]


def groups(sol: Solution) -> int:
    return len(sol.zones) or len(sol.stations)


@dataclass
class RulesResult:
    best: tuple | None  # (Solution, results) or None
    passed: bool
    tried: list = field(default_factory=list)  # every (Solution, results) planned


def plan_rules(
    run_k: Callable[[int, tuple], tuple | None],
    objective: str,
    max_k: int,
    patience: int = 2,
    cancelled: Callable[[], bool] | None = None,
    say: Callable[[str], None] | None = None,
) -> RulesResult:
    say = say or (lambda msg: None)
    tried: list = []

    def run(k: int, exclude: tuple = ()):
        if cancelled is not None and cancelled():
            return None
        out = run_k(k, exclude)
        if out is not None:
            tried.append(out)
        return out

    def passes(p) -> bool:
        return p is not None and not rule_counts(*p)

    def key(p) -> tuple:  # among passing plans
        return (
            (groups(p[0]), len(p[0].stations))
            if objective == OBJECTIVES[1]
            else (
                len(p[0].stations),
                groups(p[0]),
            )
        )

    def alternatives(p, k: int):
        best = p
        for j in range(min(ALTERNATIVES, len(p[0].stations))):
            say(f"rules: alternative cover {j + 1} at k={k}")
            alt = run(k, (dict(p[0].stations[j].carrier),))
            if passes(alt) and groups(alt[0]) < groups(best[0]):
                best = alt
        return best

    best, stale = None, 0
    for k in range(1, max_k + 1):
        if cancelled is not None and cancelled():
            break
        say(f"rules: k={k}")
        p = run(k)
        if p is None:
            continue
        if not passes(p):
            stale += best is not None
        else:
            if best is None:
                p = alternatives(p, k)
            if best is None or key(p) < key(best):
                best, stale = p, 0
            else:
                stale += 1
        if best is not None:
            if objective == OBJECTIVES[0]:
                break
            if stale >= patience or groups(best[0]) <= k + 1:  # more stops cannot give fewer groups
                break
    if best is not None:
        return RulesResult(best, True, tried)
    if not tried and not (cancelled is not None and cancelled()):
        # the map says Max stations stops cannot cover the targets: plan with them anyway, to
        # show the closest plan and which rules it breaks
        say(f"rules: k={max_k} (closest plan)")
        p = run_k(max_k, None)
        if p is not None:
            tried.append(p)
    if not tried:
        return RulesResult(None, False, tried)

    def broken(p) -> tuple:
        c = rule_counts(*p)
        return (len(c), sum(c.values()), len(p[0].stations))

    closest = min(tried, key=broken)
    return RulesResult(closest, False, tried)
