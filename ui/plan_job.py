"""Station-plan job (v8 T1): the planner body of ``ui.stations``, runnable in the GUI's thread
or in the worker process (``ui.jobs``).

The payload carries either the live ``Session`` (in-process) or a project snapshot that the
worker turns back into a ``Session`` (cached while the snapshot stays the same). Messages:
``progress`` (msg, fraction), ``solution`` (Solution, {id: PointResult}), ``max_manip`` (float).
Returns {"best": Solution | None, "rules_status": str | None}.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from core import schema

_CACHE: dict = {}  # worker: snapshot hash -> Session


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


def station_job(payload: dict, emit, cancelled) -> dict:
    """Plan stations (normal, quick or rules only, from ``spec``)."""
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik
    from core.robot import Robot
    from core.schema import length_ref
    from opt.placement import plan_stations
    from opt.reachability import map_for_tool
    from ui.stations import part_footprint, world_targets

    s = session_from(payload)
    spec = payload["spec"]
    settings = spec["settings"]
    robot = Robot(s.robot.spec)
    tcp = s.tool.tcps[0].name
    t = world_targets(s)
    fp = part_footprint(s)
    mesh = s.part_mesh()
    sites = s.sites()
    T_base0, T_part = s.T_world_base.copy(), s.part.T_world_part.copy()
    L = s.tool_model.length()
    L_ref = length_ref(s.tool.extension)

    def say(msg: str, f: float) -> None:
        emit("progress", (msg, f))

    def streamed(sol, res) -> None:
        emit("solution", (sol, res))

    say("loading reachability map (first time per tool length: ~1 min)", 0.0)
    rmap = map_for_tool(
        robot,
        s.tool_model.T_flange_tcp(tcp),
        cancelled=cancelled,
        any_offset=s.tool.extension.type == "pose",  # pose tools: shared flange map
    )
    bb, wb = sites.scene_bodies() if sites is not None else ([], [])
    scene = CollisionScene(
        robot,
        s.tool_model,
        mesh,
        T_base0,
        T_part,
        base_bodies=bb,
        world_bodies=wb,
        obstacles=s.obstacle_entries(),  # stay-out zones
    )
    checker = FeasibilityChecker(robot, make_ik(robot), s.tool_model, scene, T_base0, T_part)
    workers = int(spec.get("workers", 1))
    if workers > 1:  # v8 T2: exact checks in a process pool (kept for the next plan)
        from opt.parallel import CheckerSpec, pool_for

        say(f"starting {workers} check workers", 0.0)
        cs = CheckerSpec(robot.spec, s.tool_model, mesh, T_base0, T_part, bb, wb, s.obstacle_entries())
        try:
            checker.pool = pool_for(cs, workers)
        except Exception as err:  # e.g. a setup object that doesn't pickle: plan serially
            emit("warn", f"parallel checks off for this plan ({type(err).__name__}: {err})")
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
        on_solution=streamed,
        progress=say,
        cancelled=cancelled,
        sites=sites,
    )
    return {"best": out[0][0] if out else None, "rules_status": None}


def tool_job(payload: dict, emit, cancelled):
    """Optimize the tool geometry (``spec``: weights, max_stations, st = GeometrySettings).
    Returns the GeometryResult."""
    from core.robot import Robot
    from opt.tool_geometry import optimize_tool
    from ui.stations import part_footprint, world_targets

    s = session_from(payload)
    spec = payload["spec"]
    ee = s.tool
    sites = s.sites()
    return optimize_tool(
        Robot(s.robot.spec),
        ee,
        s.resolve(ee.urdf) if ee.urdf else None,
        s.part_mesh(),
        s.part.T_world_part.copy(),
        s.T_world_base.copy(),
        world_targets(s),
        part_footprint(s),
        spec["weights"],
        1 if sites is None else spec["max_stations"],  # static base: one station
        st=spec["st"],
        progress=lambda msg, f: emit("progress", (msg, f)),
        cancelled=cancelled,
        sites=sites,
        obstacles=s.obstacle_entries(),  # stay-out zones
    )
