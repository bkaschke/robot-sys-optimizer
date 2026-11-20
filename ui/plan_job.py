"""Station-plan job (v8 T1): the planner body of ``ui.stations``, runnable in the GUI's thread
or in the worker process (``ui.jobs``).

The payload carries either the live ``Session`` (in-process) or a project snapshot that the
worker turns back into a ``Session`` (cached while the snapshot stays the same). Messages:
``progress`` (msg, fraction), ``solution`` (Solution, {id: PointResult}), ``max_manip`` (float).
Returns {"best": Solution | None, "rules_status": str | None}.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from core import schema
from ui.plan_assets import _asset_stamp

_CACHE: dict = {}  # worker: snapshot hash -> Session
_LOG = logging.getLogger(__name__)


def failure_text(what: str, err: BaseException) -> str:
    """Review L6: the one-line summary of a job error for a status line. The full text (the
    worker's traceback, else this process's) goes to the log panel and to ``logging``."""
    import traceback

    from ui.log_panel import log

    lines = str(err).splitlines()
    summary = getattr(err, "summary", None) or (lines[0] if lines else type(err).__name__)
    tb = getattr(err, "worker_traceback", "") or "".join(
        traceback.format_exception(type(err), err, err.__traceback__)
    )
    log(f"{what} failed: {summary}", "error")
    if tb:
        log(f"{what}: details\n{tb}", "info")
    _LOG.error("%s failed: %s\n%s", what, summary, tb)
    return summary


def payload_for(s, spec: dict, live: bool) -> dict:
    """Job payload: the live session, or a picklable snapshot of it (worker)."""
    if live:
        return {"session": s, "spec": spec}
    p = s.to_project().model_copy(update={"results": [], "work": schema.WorkState()})
    return {
        "project": p.model_dump_json(),
        "project_dir": str(s.project_dir),
        "T_base": np.array(s.T_world_base),
        "T_part": None if s.part is None else np.array(s.part.T_world_part),
        "spec": spec,
    }


def session_from(payload: dict):
    """The payload's session (worker: rebuilt from the snapshot, reused while it is unchanged)."""
    if payload.get("session") is not None:
        return payload["session"]
    from ui.session import Session

    key = hashlib.sha1(
        (payload["project"] + payload["project_dir"]).encode()
        + payload["T_base"].tobytes()
        + (b"" if payload["T_part"] is None else payload["T_part"].tobytes())
        + _asset_stamp(payload["project"], payload["project_dir"])
    ).hexdigest()
    s = _CACHE.get(key)
    if s is None:
        s = Session()
        s.apply_project(schema.Project.model_validate_json(payload["project"]), Path(payload["project_dir"]))
        s.T_world_base = payload["T_base"].copy()  # exact, not through the pose fields
        if payload["T_part"] is not None:
            s.part.T_world_part = payload["T_part"].copy()
        _CACHE.clear()
        _CACHE[key] = s
    return s


def project_margins(s):
    """Review R13: the project's ``optimizer.margins`` (defaults for a project without them)."""
    from core.schema import Optimizer

    return (s.extra.get("optimizer") or Optimizer()).margins


def checker_config(s):
    """(collision Margins, CheckSettings) of every user-facing check, from ``optimizer.margins``."""
    from core.collision import Margins
    from core.feasibility import CheckSettings

    m = project_margins(s)
    return Margins(clearance=m.clearance_m), CheckSettings(
        joint_margin_deg=m.joint_deg, clearance=m.clearance_m
    )


def margins_label(m) -> str:
    """The margins for a plan's provenance text, e.g. ``25 mm / 8 deg``."""
    return f"{m.clearance_m * 1000:g} mm / {m.joint_deg:g} deg"


def build_checker(s, robot, sites, workers: int, say, emit):
    """The exact checker of the planners (part, carrier bodies, stay-out zones), with a check
    pool (v8 T2) when ``workers`` > 1 (kept for the next job with the same setup)."""
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik

    mesh = s.part_mesh()
    T_base0, T_part = s.T_world_base.copy(), s.part.T_world_part.copy()
    bb, wb = sites.scene_bodies() if sites is not None else ([], [])
    obstacles = s.obstacle_entries()  # stay-out zones
    margins, settings = checker_config(s)  # review R13: optimizer.margins of the project
    scene = CollisionScene(
        robot,
        s.tool_model,
        mesh,
        T_base0,
        T_part,
        margins,
        base_bodies=bb,
        world_bodies=wb,
        obstacles=obstacles,
    )
    checker = FeasibilityChecker(robot, make_ik(robot), s.tool_model, scene, T_base0, T_part, settings)
    from core.collision_mesh import deviation_warning

    warning = deviation_warning(scene.geometry_report(), margins.clearance)
    if warning:  # simplified collision copies can deviate more than the clearance
        emit("warn", warning)
    if workers > 1:
        from opt.parallel import CheckerSpec, pool_for

        say(f"starting {workers} check workers", 0.0)
        from opt import parallel

        # pool failures (fallback to serial checks) reach the user; reset when the job ends
        parallel.on_warn = lambda m: emit("warn", m.splitlines()[0] if m else m)
        try:
            cs = CheckerSpec(robot.spec, s.tool_model, mesh, T_base0, T_part, bb, wb, obstacles)
            cs.collision = scene.config  # v12 #10: the collision settings, one value
            cs.margins, cs.settings = margins, settings  # review R13 (read by CheckerSpec.build)
            checker.pool = pool_for(cs, workers)
        except Exception as err:  # e.g. a setup object that doesn't pickle: run serially
            emit("warn", f"parallel checks off for this run ({type(err).__name__}: {err})")
    return checker


def _end_job() -> None:
    from opt import parallel

    parallel.on_warn = None  # a later job must not call this job's emit


def station_job(payload: dict, emit, cancelled) -> dict:
    """Plan stations (normal, quick or rules only, from ``spec``)."""
    try:
        return _station_job(payload, emit, cancelled)
    finally:
        _end_job()


def _cancelled_step(plans: list, step: str, breakdowns: list | None = None) -> list:
    """Review R7: the plain plans after a cancelled Pareto / robust step keep their labels and
    planner, say so in the label, and lose any partial robust score."""
    for i, (sol, _res) in enumerate(plans):
        sol.label = f"{sol.label} (cancelled before the {step} step)"
        if breakdowns is not None:
            sol.breakdown = breakdowns[i]
    return plans


def _station_job(payload: dict, emit, cancelled) -> dict:
    from core.feasibility import CancelledError
    from core.robot import Robot
    from core.schema import length_ref
    from opt.placement import plan_stations
    from opt.reachability import map_for_tcps
    from opt.tolerance import ToleranceSettings
    from ui.stations import part_footprint, world_targets

    s = session_from(payload)
    spec = payload["spec"]
    settings = spec["settings"]
    from core.speed_zones import speed_zones

    settings.speed_zones = tuple(speed_zones(s.stayouts, s.resolve))  # cycle estimate
    robot = Robot(s.robot.spec)
    t = world_targets(s, spec.get("skip", ()))  # v10 T12: points the tool cannot fit left out
    fp = part_footprint(s)
    sites = s.sites()
    T_base0 = s.T_world_base.copy()
    L = s.tool_model.length()
    L_ref = length_ref(s.tool.extension)

    def say(msg: str, f: float) -> None:
        emit("progress", (msg, f))

    from core.provenance import stamp

    prov = stamp(
        {
            "refinement": getattr(settings, "refine_method", "cmaes") if settings.refine else "none",
            "objective": spec.get("objective") or ("rules" if spec.get("rules") else "weighted"),
            "margins": margins_label(project_margins(s)),  # review R13
        },
        settings,
        project_margins(s),
        spec["weights"],
        spec.get("rules"),
        spec.get("pareto_keys"),
    )

    def streamed(sol, res) -> None:
        sol.provenance = dict(prov)  # v8 T21
        emit("solution", (sol, res))

    say("loading reachability map (first time per tool length: ~1 min)", 0.0)
    rmap = map_for_tcps(  # v8 T16: every TCP of the tool (two tips: either may reach a point)
        robot,
        s.tool_model,
        [c.name for c in s.tool.tcps],
        cancelled=cancelled,
        any_offset=s.tool.extension.type in ("pose", "bent2"),  # shared flange map (pose, two bends)
    )
    checker = build_checker(s, robot, sites, int(spec.get("workers", 1)), say, emit)
    emit("max_manip", float(rmap.max_manip))  # the Results card: manipulability in % (not saved)
    args = dict(t=t, T_base0=T_base0, fp=fp, weights=spec["weights"], settings=settings, L=L, L_ref=L_ref)
    rules = spec.get("rules")
    if rules is not None:
        from ui import rules_ui

        out, status = rules_ui.run(
            rules["objective"],
            rules["sing_pct"],
            rmap,
            checker,
            {**args, "sites": sites},
            streamed,
            say,
            cancelled,
        )
        return {"best": out[0][0] if out else None, "rules_status": status}
    pareto = spec.get("objective") == "pareto"
    robust = spec.get("objective") == "robust"
    fin: list = []
    out = plan_stations(
        rmap,
        checker,
        t,
        T_base0,
        fp,
        spec["weights"],
        settings,
        L,
        L_ref,
        on_solution=None if pareto or robust else streamed,
        progress=say,
        cancelled=cancelled,
        sites=sites,
        finisher=fin,
    )
    if pareto and out and fin:  # v8 T11: the trade-off plans instead of the cheapest one
        from opt.pareto import station_front

        finish, used_sites = fin[0]
        keys = spec.get("pareto_keys") or ("stations", "cycle", "clearance")
        try:
            plans = station_front(
                finish,
                used_sites,
                t,
                out,
                keys,
                settings.refine_radius,
                settings.refine_evals,
                1,
                say,
                cancelled,
            )
        except CancelledError:
            plans = _cancelled_step(out, "Pareto")
        else:
            n = len(plans)
            for i, (sol, _res) in enumerate(plans):
                sol.label = f"Pareto {i + 1}/{n}: {sol.label}"
                sol.planner = "pareto"
        for sol, _res in plans:
            sol.provenance = dict(prov)
        emit("replace", plans)
        return {"best": plans[0][0] if plans else None, "rules_status": None}
    if spec.get("objective") == "robust" and out and fin:  # v8 T12: worst case under tolerance
        from opt.robust import station_robust

        finish, used_sites = fin[0]
        r = spec.get("robust") or {}
        breakdowns = [dict(sol.breakdown) for sol, _res in out]  # station_robust adds "robust"
        try:
            plans = station_robust(
                finish,
                used_sites,
                checker,
                t,
                out,
                r.get("tolerance") or ToleranceSettings(samples=10),
                r.get("measure", "Worst case"),
                float(spec["weights"].unreached),
                settings.refine_radius,
                settings.refine_evals,
                say,
                cancelled,
            )
        except CancelledError:
            plans = _cancelled_step(out, "robust", breakdowns)
        else:
            for sol, _res in plans:
                sol.label = f"Robust: {sol.label}"
                sol.planner = "robust"
        for sol, _res in plans:
            sol.provenance = dict(prov)
        emit("replace", plans)
        return {"best": plans[0][0] if plans else None, "rules_status": None}
    return {"best": out[0][0] if out else None, "rules_status": None}


def tool_job(payload: dict, emit, cancelled):
    """Optimize the tool geometry (``spec``: weights, max_stations, st = GeometrySettings).
    Returns the GeometryResult."""
    from core.robot import Robot
    from opt.tool_geometry import optimize_tool
    from ui.stations import part_footprint, world_targets

    s = session_from(payload)
    spec = payload["spec"]
    spec["st"].family = family_parts(spec.get("family_dirs") or [])  # v8 T16
    from core.speed_zones import speed_zones

    spec["st"].planner.speed_zones = tuple(speed_zones(s.stayouts, s.resolve))  # cycle estimate
    ee = s.tool
    if spec.get("tube_od_mm"):  # v8 T17: a standard tube instead of the tool's own capsule radius
        x = ee.extension.model_copy(update={"radius": float(spec["tube_od_mm"]) / 2000})
        ee = ee.model_copy(update={"extension": x})
    sites = s.sites()
    margins, check = checker_config(s)  # review R13: optimizer.margins of the project
    r = optimize_tool(
        Robot(s.robot.spec),
        ee,
        s.resolve(ee.urdf) if ee.urdf else None,
        s.part_mesh(),
        s.part.T_world_part.copy(),
        s.T_world_base.copy(),
        world_targets(s, spec.get("skip", ())),  # v10 T12
        part_footprint(s),
        spec["weights"],
        1 if sites is None else spec["max_stations"],  # static base: one station
        st=spec["st"],
        progress=lambda msg, f: emit("progress", (msg, f)),
        cancelled=cancelled,
        sites=sites,
        obstacles=s.obstacle_entries(),  # stay-out zones
        margins=margins,
        settings=check,
    )
    from core.provenance import stamp

    st = spec["st"]
    prov = stamp(
        {
            "shape": st.shape_method,
            "start": st.start,
            "loop": st.loop,
            "objective": st.objective,
            "margins": margins_label(project_margins(s)),  # review R13
        },
        st,
        spec["weights"],
        project_margins(s),
    )
    for d in r.designs:  # v8 T21
        d.solution.provenance = dict(prov)
    if r.solution is not None:
        r.solution.provenance = dict(prov)
    return r


def family_parts(dirs: list[str]) -> list:
    """opt.tool_geometry.FamilyPart for each project folder (its part, targets and cell)."""
    from opt.tool_geometry import FamilyPart
    from ui.session import Session, load_project
    from ui.stations import part_footprint, world_targets

    out = []
    for d in dirs:
        s = Session()
        load_project(s, Path(d))
        if s.part is None or not s.targets.points:
            raise ValueError(f"part family: {d} has no part or no targets")
        out.append(
            FamilyPart(
                s.part_mesh(),
                s.part.T_world_part.copy(),
                s.T_world_base.copy(),
                world_targets(s),
                part_footprint(s),
                s.sites(),
                s.obstacle_entries(),
                Path(d).name,
            )
        )
    return out


def tolerance_job(payload: dict, emit, cancelled):
    """v8 T5: Monte Carlo tolerance check of a plan (``spec``: sol, results, settings, workers).
    Returns opt.tolerance.ToleranceResult."""
    try:
        return _tolerance_job(payload, emit, cancelled)
    finally:
        _end_job()


def _tolerance_job(payload: dict, emit, cancelled):
    from core.export import station_pose
    from core.robot import Robot
    from opt.tolerance import tolerance_check
    from ui.stations import world_targets

    s = session_from(payload)
    spec = payload["spec"]
    sol = spec["sol"]
    robot = Robot(s.robot.spec)

    def say(msg: str, f: float) -> None:
        emit("progress", (msg, f))

    checker = build_checker(s, robot, s.sites(), int(spec.get("workers", 1)), say, emit)
    T_stops = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    return tolerance_check(
        checker,
        world_targets(s),
        sol,
        spec["results"],
        T_stops,
        spec["settings"],
        pool=checker.pool,
        progress=say,
        cancelled=cancelled,
    )


def edge_job(payload: dict, emit, cancelled):
    """v10 T10: dense reach check of a plan at its zone edges (``spec``: sol, results, workers).
    Returns core.edge_check.EdgeResult."""
    from core import edge_check as ec
    from core import targets as tg
    from core.export import station_pose
    from core.robot import Robot
    from ui.stations import world_targets

    s = session_from(payload)
    spec = payload["spec"]
    sol, res = spec["sol"], spec["results"]
    robot = Robot(s.robot.spec)

    def say(msg: str, f: float) -> None:
        emit("progress", (msg, f))

    say("placing edge points", 0.0)
    ids, P, N, _ = tg.as_arrays(s.targets)
    stop_of, zone_of, low = ec.plan_maps(sol, res)
    h = ec.spacing_of(P)
    pts = ec.edge_points(s.part_mesh(), ids, P, N, stop_of, zone_of, low, spacing=h)
    for g in ec.two_sided(ids, P, N, {int(p.id): p.group for p in s.targets.points}, h):  # v12 #5
        emit("warn", f"group **{g}** has targets on opposite sides of the part; no edge points between them")
    checker = build_checker(s, robot, s.sites(), int(spec.get("workers", 1)), say, emit)
    t = world_targets(s)
    group = {int(pid): g for pid, g in zip(t.ids, t.groups, strict=True)}
    T_stops = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    gaps = ec.check_edges(
        checker,
        pts,
        T_stops,
        lambda pid: group[int(pid)],
        cancelled,
        lambda f: say(f"checking {len(pts)} edge points", f),
        pool=checker.pool,
    )
    return ec.EdgeResult(sol.label, pts, gaps, h, key=sol.key)
