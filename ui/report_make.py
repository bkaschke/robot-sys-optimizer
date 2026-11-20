"""Report, CSV and userframe CSV for a solution, without a GUI (batch runner + Export panel).

Everything here takes a ``Session`` and a ``schema.Solution`` and returns text, so the Export
panel and a worker process produce identical files. Nothing here imports viser (the texts
come from the viser-free ``ui.report_text``).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from core import export as ex
from core.report import PdfError, ReportInput, build_report, md_html
from core.robot import Robot
from ui.part_heatmap import tolerance_for
from ui.plan_ids import key as plan_key

log = logging.getLogger(__name__)


NO_VIEWS_BATCH = (
    "3D views not included: they are rendered only from the GUI (the batch runner has no browser)."
)


def tool_of(session):
    tm = session.tool_model
    if tm is None:
        raise ValueError("no tool loaded")
    return tm


def rows(session, sol) -> list[dict]:
    s = session
    if s.robot is None:
        raise ValueError("no robot in this study (tab 1 Robot & cell)")
    return ex.tcp_rows(
        sol,
        s.result_points.get(plan_key(sol), {}),
        s.targets,
        Robot(s.robot.spec),
        tool_of(s),
        s.base,
        s.T_world_base,
    )


def margins_clearance(session) -> float:
    """The clearance the checks used: optimizer.margins, else the default."""
    extra = session.extra
    opt = extra.get("optimizer") if isinstance(extra, dict) else None
    m = getattr(opt, "margins", None)
    if m is not None and hasattr(m, "clearance_m"):
        return float(m.clearance_m)
    from core.schema import Margins

    return Margins().clearance_m


def geometry(session) -> tuple[list[dict], str | None]:
    """The part's collision copy as built from the current geometry settings (disk-cached),
    plus core.collision_mesh.deviation_warning for it."""
    s = session
    if s.part is None:
        return [], None
    try:
        from core.collision_config import current, part_low_info
        from core.collision_mesh import deviation_warning

        _, info = part_low_info(s.part_mesh(), current())
    except Exception as err:  # the report must still be written
        log.warning("collision geometry info unavailable: %s", err)
        return [], None
    row = {"name": "part", "component": "part", **info.as_dict()}
    return [row], deviation_warning({"part": row}, margins_clearance(s))


def _sim_of(sim, sol):
    """``sim`` (a SimPanel) when it holds this plan's built paths, else None."""
    if sim is None or sim.plan is None:
        return None
    own = getattr(sim, "key", None) or getattr(sim, "label", None)
    return sim if own == plan_key(sol) else None


def sections(session, sol, sim=None, frame=None) -> list[tuple[str, str]]:
    """The report's extra sections as HTML (from the GUI's own texts). ``sim``: the paths panel
    (its built paths count when they are this plan's); ``frame``: the Gantry placement panel."""
    from ui.report_text import unreached_report
    from ui.summary_card import card

    s, out = session, []
    out.append(("Summary", md_html(card(s, sol))))
    cyc = [f"Estimated cycle time: **{sol.cycle_s:.0f} s**" if sol.cycle_s is not None else ""]
    if _sim_of(sim, sol) is not None and getattr(sim, "cycle", None):
        c = sim.cycle
        parts = ", ".join(f"{k} {v:.0f} s" for k, v in c.items() if k != "total" and v > 0.05)
        cyc.append(f"Cycle time of the built paths: **{c['total']:.0f} s** ({parts}).")
    if any(cyc):
        out.append(("Cycle time", md_html("  \n".join(x for x in cyc if x))))
    if sim is not None:  # v10 T16 / T22: both carrier motions built for this plan
        from ui.sim_fields import motion_line

        line = motion_line(sim, sol.label)
        if line:
            out.append(("Carrier motion", md_html(line)))
    if s.compared:  # v9 #7 / v10 T22: Optimize -> Compare setups, the project's setup always in
        from ui.report_sections import setup_comparison

        out.append(("Setup comparison", setup_comparison(s, sol)))
    if frame is not None and frame.rows:  # v10 T22: Optimize -> Gantry placement
        from ui.report_sections import gantry_placement

        out.append(("Gantry placement", gantry_placement(s, frame.rows)))
    tol = tolerance_for(s, sol)  # review R8: this plan's, and only while the plan is unchanged
    if tol is not None:
        from ui.report_text import tolerance_summary

        out.append(("Tolerance check", md_html(tolerance_summary(tol))))
    from ui.radius_ui import report_md

    fit = report_md(s)  # v10 T12: points the tool cannot fit
    if fit:
        out.append(("Tool fit", md_html(fit)))
    msg = unreached_report(s.result_points.get(plan_key(sol), {}), sol.unreached, sol.label)
    out.append(("Unreached points", md_html(msg) if msg else "<p>Every point is reached.</p>"))
    return out


def make_csv(session, sol, stale=None) -> str:
    geo, _ = geometry(session)
    info = "; ".join(f"{g['name']} {g['mode']}/{g['kind']} dev {g['deviation_m'] * 1000:.1f} mm" for g in geo)
    return ex.rows_to_csv(rows(session, sol), ex.csv_comments(list(stale or []), info))


def make_userframes(session, sol) -> str:
    """Robot base -> part frame per stop, in the loaded robot's frame convention."""
    from core.userframe import userframe_csv, userframe_rows

    s = session
    if s.robot is None or s.part is None:
        raise ValueError("needs a robot and a part")
    conv = s.robot.spec.frame_convention
    urows = userframe_rows(sol, s.base, s.T_world_base, s.part.T_world_part, conv)
    return userframe_csv(urows, conv)


def make_report(
    session,
    sol,
    *,
    screenshot=None,
    stale=None,
    sim=None,
    length_samples=None,
    notes=None,
    frame=None,
    images=None,
) -> str:
    """The HTML report. ``images`` are rendered 3D views as (caption, PNG bytes | data URI |
    RGB array); ``screenshot`` is one more such view (what png_data_uri takes; None = none);
    ``notes`` come first in the Notes section (e.g. why the 3D views are missing)."""
    from core.report import png_data_uri

    s = session
    tm = tool_of(s)
    table = rows(s, sol)
    shots = list(images or [])
    if screenshot is not None:
        shots.append(("3D view at export time", screenshot))
    images = []
    for caption, img in shots:
        if isinstance(img, bytes):
            img = "data:image/png;base64," + base64.b64encode(img).decode()
        elif not isinstance(img, str):
            img = png_data_uri(img)
        images.append((caption, img))
    notes = list(notes or [])
    pts = s.result_points.get(plan_key(sol), {})
    notes += sorted({n for r in pts.values() for n in getattr(r, "notes", [])})
    if not pts:
        notes.append(
            "Per-point joint solutions are not stored for this solution (loaded from a project): "
            "re-run the plan for the full point table."
        )
    tool = s.tool
    geo, geo_warn = geometry(s)
    motion = None
    if _sim_of(sim, sol) is not None:
        motion = getattr(sim.plan, "motion_checked", None)
    report = ReportInput(
        stale=list(stale or []),
        geometry=geo,
        geometry_warning=geo_warn,
        motion_checked=motion,
        mesh_license=s.robot.spec.mesh_license,
        title=f"Cell plan: {sol.label}",
        robot=s.robot.spec.name,
        tool=tool.name if tool else "-",
        process=tool.process if tool else "-",
        base_mode=f"{s.base.mode}" + (f" ({s.base.carrier})" if s.base.mode == "movable" else ""),
        solution=sol,
        rows=table,
        station_poses=[ex.station_pose(s.base, st, s.T_world_base) for st in sol.stations],
        envelope_svg=ex.envelope_svg(tm, sol.tool_values),
        images=images,
        length_samples=list(length_samples or []),
        notes=notes,
        sections=sections(s, sol, sim, frame),
        settings={
            "part": s.part.mesh_path if s.part else "-",
            "targets": f"{len(s.targets.points)} points in {len(s.targets.groups)} group(s)",
            "standoff min / preferred (m)": f"{s.base.standoff_m.carrier_part_min} / "
            f"{s.base.standoff_m.carrier_part_preferred}",
            "tool values": sol.tool_values.model_dump()
            if sol.tool_values
            else tool.extension.nominal.model_dump(),
        },
    )
    return build_report(report)


def write_text(out_dir: Path, name: str, text: str) -> tuple[Path, bytes]:
    data = text.encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(data)
    return path, data


def export_all(
    session, sol, out_dir, pdf: bool = False, *, stale=None, sim=None, length_samples=None, images=None
) -> dict:
    """Write tcp_poses.csv, userframes.csv and report.html (+ report.pdf) into ``out_dir``.

    Returns {file name: Path}. A file that could not be made maps to the error text (str)
    instead; a PDF failure never raises. ``images`` = [(caption, PNG)] rendered offline (the batch
    runner); without them the report says the 3D views are missing."""
    out_dir = Path(out_dir)
    out: dict = {}
    jobs = [
        ("tcp_poses.csv", lambda: make_csv(session, sol, stale)),
        ("userframes.csv", lambda: make_userframes(session, sol)),
        (
            "report.html",
            lambda: make_report(
                session,
                sol,
                stale=stale,
                sim=sim,
                length_samples=length_samples,
                notes=[] if images else [NO_VIEWS_BATCH],
                images=images,
            ),
        ),
    ]
    for name, fn in jobs:
        try:
            out[name] = write_text(out_dir, name, fn())[0]
        except Exception as err:  # one failed export must not stop the others
            log.exception("export %s failed", name)
            out[name] = f"{type(err).__name__}: {err}"
    if pdf:
        html = out["report.html"]
        if not isinstance(html, Path):
            out["report.pdf"] = "not made: the HTML report failed"
        else:
            from core.report import html_to_pdf

            try:
                html_to_pdf(html, out_dir / "report.pdf")
                out["report.pdf"] = out_dir / "report.pdf"
            except PdfError as err:
                out["report.pdf"] = f"PdfError: {err}"
            except Exception as err:
                out["report.pdf"] = f"{type(err).__name__}: {err}"
    return out
