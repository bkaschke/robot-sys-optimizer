"""Results summary card (v7 round 2, T17): the selected plan at a glance, at the top of Results.

Stations, groups, points reached; worst clearance, closest joint limit, lowest manipulability
(% of the robot's max when a plan of this session knows it, else |det J|); the planner and its
mode, whether the rules are met, and the run time.
"""

from __future__ import annotations

import math

PLANNERS = {
    "cost": "Station planner (cost based)",
    "quick": "Quick station plan",
    "tool": "Tool optimizer",
    "rules: fewest stations": "Station planner (rules only, fewest stations)",
    "rules: fewest groups": "Station planner (rules only, fewest groups)",
}


def card(s, sol) -> str:
    """Markdown for ``sol`` (a schema.Solution) in session ``s``."""
    if sol is None:
        return ""
    n = len(s.targets.points)
    res = s.result_points.get(sol.label, {})
    ok = [r for r in res.values() if r.status != "unreachable" and r.q is not None]
    groups = str(len(sol.zones)) if sol.zones else "— (best per point)"
    clear = [r.clearance for r in ok if r.clearance is not None and math.isfinite(r.clearance)]
    worst_c = f"{min(clear) * 1000:.0f} mm" if clear else "—"
    lim = f"{min(r.limit_margin_deg for r in ok):.0f}°" if ok else "—"
    if ok:
        mu = min(r.manipulability for r in ok)
        top = getattr(s, "max_manip", None)  # set by a plan of this session (not saved)
        low_mu = f"{100 * mu / top:.0f} %" if top else f"{mu:.3f} (\\|det J\\|)"
    else:
        low_mu = "—"
    planner = PLANNERS.get(sol.planner, sol.planner or ("Tool optimizer" if sol.tool_values else "—"))
    rules = f"✘ {len(sol.violations)} broken" if sol.violations else "✔ met"
    run = f"{sol.run_s:.0f} s" if sol.run_s is not None else "—"
    cyc = f"{sol.cycle_s:.0f} s (estimate)" if sol.cycle_s is not None else "—"
    return (
        f"**Selected plan:** {sol.label}\n\n"
        "| | | | |\n|---|---|---|---|\n"
        f"| Stations | **{len(sol.stations)}** | Groups | **{groups}** |\n"
        f"| Points reached | **{n - len(sol.unreached)}/{n}** | Worst clearance | {worst_c} |\n"
        f"| Closest joint limit | {lim} | Lowest manipulability | {low_mu} |\n"
        f"| Planner | {planner} | Rules | {rules} |\n"
        f"| Run time | {run} | Cycle time | {cyc} |\n"
    )
