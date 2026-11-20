"""Viser-free texts shared by the GUI and the headless report (batch runner): why points are
unreachable (``REASONS``, ``unreached_report``) and the tolerance-check summary.

Moved out of ``ui.log_panel`` and ``ui.tolerance_ui`` (both re-export them under the old names)
so ``ui.report_make`` and ``ui.batch_job`` never import viser.
"""

from __future__ import annotations

from collections import Counter

# why a point is unreachable (core.feasibility stages, opt.sites checks), in plain words
REASONS = {
    "no_ik": "out of reach: no arm pose reaches it at any allowed tool angle",
    "joint_limits": "only reachable beyond the joint limits (incl. the margin)",
    "singular": "only reachable close to a singularity",
    "collision": "every arm pose collides (part, carrier, stay-out zone or floor)",
    "": "not covered by any stop",
}


def reason_text(code: str) -> str:
    return REASONS.get(code, code.replace("_", " "))


def _hit_objects(r) -> list[str]:
    """What the arm hits at an unreachable point ("part", "floor", ...), from its notes."""
    from ui.unreached_view import why

    w = why(r) if r is not None and getattr(r, "reason", "") == "collision" else ""
    return w.split(": ", 1)[1].split(", ") if ": " in w else []


def unreached_report(results: dict, ids, what: str) -> str | None:
    """One message on the points of ``ids`` that are unreachable in ``results`` (point id ->
    PointResult), grouped by reason with up to 12 point ids each; None when all are reached."""
    bad = [i for i in ids if (r := results.get(i)) is None or r.status == "unreachable"]
    if not bad:
        return None
    why = Counter((getattr(results.get(i), "reason", "") or "") for i in bad)
    parts = []
    for code, n in why.most_common():
        pts = [i for i in bad if (getattr(results.get(i), "reason", "") or "") == code]
        more = f" … +{len(pts) - 12}" if len(pts) > 12 else ""
        hits = Counter(o for i in pts for o in _hit_objects(results.get(i)))
        what_hit = f", hitting {', '.join(f'{o} ({c})' for o, c in hits.most_common())}" if hits else ""
        parts.append(f"{n} × {reason_text(code)}{what_hit} (points {', '.join(map(str, pts[:12]))}{more})")
    return f"{what}: **{len(bad)} point(s) unreachable**: " + "; ".join(parts)


def tolerance_summary(r) -> str:
    """The tolerance check result (opt.tolerance.ToleranceResult) as markdown."""
    st = r.settings
    head = (
        f"**{100 * r.all_reached:.0f} %** of {st.samples} samples reach every planned point "
        f"(part ±{st.part_mm:g} mm, {st.part_deg:g}°, stops ±{st.station_mm:g} mm)."
    )
    worst = [p for p in r.worst(10) if p.rate < 1.0]
    if not worst:
        return head + "  \nEvery point was reached in every sample."
    rows = "\n".join(
        f"| {p.pid} | {p.station + 1} | {100 * p.rate:.0f} % | "
        f"{'—' if p.fail_mm is None else f'{p.fail_mm:.1f}'} | {p.clearance_mm:.0f} | {p.limit_deg:.0f} |"
        for p in worst
    )
    return (
        head
        + "\n\n**Least robust points** (fails from = the point's shift in the smallest failing sample):\n\n"
        + "| point | stop | reached | fails from (mm) | clearance (mm) | limit margin (°) |\n"
        + "|---|---|---|---|---|---|\n"
        + rows
    )
