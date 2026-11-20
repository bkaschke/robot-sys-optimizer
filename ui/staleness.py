"""Setup fingerprints (v2 phase 2): what a result or a confirmed step was made from.

``components(session)`` hashes each part of the setup separately (robot, carrier, tool,
part, targets). A result stores the hashes from when its run started (``Session.stamps``);
when today's hashes differ, the result is out of date and ``changed`` names what moved.
Confirmed setup steps store theirs in ``Session.steps`` the same way.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ui.fingerprint import content
from ui.plan_ids import key as plan_key

COMPONENTS = (
    "robot",
    "carrier",
    "tool",
    "part",
    "targets",
    "stayout",
    "collision",
    "collision_geom",
    "margins",
)
LABELS = {
    "robot": "robot / mount",
    "carrier": "external axis",
    "tool": "tool",
    "part": "part",
    "targets": "targets",
    "stayout": "stay-out zones",
    "collision": "collision settings",
    "collision_geom": "collision geometry",
    "margins": "clearance / joint margins",
    "base_pose": "robot base pose",
}
# setup steps (tab names in app.TABS) and the components each one defines
STEP_COMPONENTS = {
    "Cell": ("robot", "part", "stayout", "collision", "collision_geom"),
    "External Axis": ("carrier",),
    "Tool Setup": ("tool",),
    "Targets": ("targets",),
}


def _h(x) -> str:
    text = x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _pose(T: np.ndarray, digits: int = 6) -> list:
    return (np.round(np.asarray(T, float), digits) + 0.0).tolist()  # + 0.0: -0.0 hashes as 0.0


def _prims(s, prims, legacy: bool) -> list:
    """Primitives (carrier bodies, stay-out zones) with mesh paths replaced by their content."""
    out = []
    for p in prims:
        d = p.model_dump(mode="json")
        if not legacy:
            for k in ("mesh", "collision_mesh"):
                if d.get(k):
                    d[k] = content(s, d[k])
        out.append(d)
    return out


def _margins(s) -> dict:
    """The project's clearance / joint margins (defaults when the session has none; review R13)."""
    try:
        from core.schema import Optimizer

        return (s.extra.get("optimizer") or Optimizer()).margins.model_dump()
    except (AttributeError, TypeError, ValueError):
        return {}


def components(s, base_pose: bool = False, tool_nominal: bool = True, legacy: bool = False) -> dict[str, str]:
    """Hash per setup component of session ``s``.

    ``base_pose``: include the full robot base pose (the feasibility check runs from it;
    plans choose their own stations). In static mode the base height and orientation always
    count, because the planner keeps them. ``tool_nominal=False`` leaves out the tool's
    rough placement (the tool optimizer chooses it: applying its result must not make the
    result itself out of date). Files count by content (v10 #2: ui.fingerprint), so copying
    them into the project on save changes nothing; ``legacy``: the v9 hashes (by path), only
    to recognise stamps saved by v9 (``migrate``)."""
    T = np.asarray(s.T_world_base, float)
    urdf = s.robot_urdf if legacy else content(s, s.robot_urdf, folder=True)
    robot = {"name": s.robot_name, "urdf": urdf, "mount": s.mount}
    if s.robot is not None and s.robot.spec.limits_deg is not None:  # v6: user joint limits
        robot["limits"] = s.robot.spec.limits_deg
    if s.base.mode != "movable":
        robot["base_z"] = round(float(T[2, 3]), 6)  # the planner keeps height and orientation
        robot["base_R"] = _pose(T[:3, :3])
    if legacy:
        carrier = s.base.model_dump_json()
    else:
        carrier = {**s.base.model_dump(mode="json"), "bodies": _prims(s, s.base.bodies, legacy)}
    out = {
        "robot": _h(robot),
        "carrier": _h(carrier),
        "tool": "",
        "part": "",
        "targets": _h(s.targets.model_dump_json()),
        "stayout": _h(_prims(s, getattr(s, "stayouts", []), legacy)),
        "margins": _h(_margins(s)),
    }
    if s.tool is not None:
        tool = s.tool.model_dump(mode="json")
        tool.pop("pad_radius_m", None)  # informational (Detect warning only): not a planning input
        if not tool_nominal:
            tool["extension"].pop("nominal", None)
        if not legacy:
            tool["urdf"] = content(s, tool.get("urdf"), folder=True)
        out["tool"] = _h(tool)
    if s.part is not None:
        mesh = s.part.mesh_path if legacy else content(s, s.part.mesh_path)
        out["part"] = _h([mesh, s.part.units, _pose(s.part.T_world_part)])
    if not legacy:  # v10 #5: arm padding, the part's own collision mesh (content), kept faces
        from core.collision_config import current

        part, cfg = s.part, current()
        out["collision"] = _h(
            [
                round(float(cfg.arm_pad), 6),
                sorted((int(k), round(float(v), 6)) for k, v in cfg.link_pad.items()),  # v10 T19
                content(s, part.collision_mesh) if part is not None else None,
                sorted(int(f) for f in part.collision_keep) if part is not None else [],
            ]
        )
        # review R11: the simplified-copy mode per component (reduce / hull / wrap, triangles). Its own
        # key, so stamps saved without it are not flagged (``changed`` compares the stamp's keys)
        out["collision_geom"] = _h(sorted((k, v.key()) for k, v in cfg.geom.items()))
    if base_pose:
        out["base_pose"] = _h(_pose(T))
    return out


def migrate(s) -> int:
    """v10 #2: stamps and confirmed steps saved by v9 hash files by path. Where a stored hash
    equals the v9 hash of the current setup it is replaced by today's (content) hash, so loading
    an older project does not flag every plan. Returns how many hashes were replaced."""
    n = 0
    variants = [
        (components(s, legacy=True, tool_nominal=t), components(s, tool_nominal=t)) for t in (True, False)
    ]
    stamps = [s.stamps, *(c.get("stamps", {}) for c in (getattr(s, "compared", None) or {}).values())]
    now = variants[0][1]
    for d in [v for st in stamps for v in st.values()] + list(s.steps.values()):
        for k in ("collision", "collision_geom"):  # v10 #5 / review R11: new components, taken as now
            if k not in d:
                d[k] = now[k]
                n += 1
        for k in ("robot", "carrier", "tool", "part", "stayout"):
            for old, new in variants:
                if k in d and d[k] == old[k] and old[k] != new[k]:
                    d[k] = new[k]
                    n += 1
                    break
    return n


def stamp_for(s, kind: str) -> dict[str, str]:
    """Fingerprint a run of ``kind``: 'feasibility', 'plan' or 'tool'."""
    return components(s, base_pose=kind == "feasibility", tool_nominal=kind != "tool")


def changed(old: dict | None, new: dict) -> list[str]:
    """Human names of the components that differ (empty: up to date or no stamp)."""
    if not old:
        return []  # results from before v2 carry no stamp: unknown, not flagged
    return [LABELS.get(k, k) for k in old if old.get(k) != new.get(k)]


def solution_changes(s, sol) -> list[str]:
    """What changed since solution ``sol`` was computed ([] = up to date or unknown)."""
    kind = "tool" if getattr(sol, "tool_values", None) is not None else "plan"
    return changed(s.stamps.get(plan_key(sol)), stamp_for(s, kind))


def feasibility_changes(s) -> list[str]:
    return changed(s.stamps.get("feasibility"), stamp_for(s, "feasibility"))


def step_state(s, step: str) -> str:
    """'confirmed', 'changed' (confirmed, then edited) or 'todo'."""
    saved = s.steps.get(step)
    if not saved:
        return "todo"
    now = components(s)
    return "confirmed" if all(saved.get(k) == now[k] for k in STEP_COMPONENTS[step]) else "changed"


def confirm_step(s, step: str) -> None:
    now = components(s)
    s.steps[step] = {k: now[k] for k in STEP_COMPONENTS[step]}


CLEARS = ("part", "tool", "targets")  # v6 feedback: plans made for another part / tool / targets go


def clearing_changes(s, sol) -> list[str]:
    """The components among ``CLEARS`` that changed since ``sol`` was computed (its plan no
    longer applies: the part moved, the tool or the targets changed); [] = keep it."""
    kind = "tool" if getattr(sol, "tool_values", None) is not None else "plan"
    old, new = s.stamps.get(plan_key(sol)), stamp_for(s, kind)
    if not old:
        return []
    return [LABELS[k] for k in CLEARS if k in old and old[k] != new.get(k)]


def drop_stale_results(s, fresh: set[str]) -> dict[str, list[str]]:
    """Remove the results (and their joint solutions and stamps) whose part, tool or targets
    changed after they were seen up to date. ``fresh`` (kept by the caller between calls)
    collects the labels seen up to date: results that load already out of date (e.g. saved
    by an older version) are only flagged, never removed. Returns {label: what changed}."""
    gone, labels = {}, {}  # v12 #8: by plan key; the caller gets labels to show
    for sol in list(s.results):
        c, k = clearing_changes(s, sol), plan_key(sol)
        if not c:
            fresh.add(k)
        elif k in fresh:
            gone[k], labels[sol.label] = c, c
    if gone:
        s.results = [r for r in s.results if plan_key(r) not in gone]
        for k in gone:
            s.result_points.pop(k, None)
            s.stamps.pop(k, None)
            fresh.discard(k)
    return labels
