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

COMPONENTS = ("robot", "carrier", "tool", "part", "targets", "stayout")
LABELS = {
    "robot": "robot / mount",
    "carrier": "external axis",
    "tool": "tool",
    "part": "part",
    "targets": "targets",
    "stayout": "stay-out zones",
    "base_pose": "robot base pose",
}
# setup steps (tab names in app.TABS) and the components each one defines
STEP_COMPONENTS = {
    "Cell": ("robot", "part", "stayout"),
    "External Axis": ("carrier",),
    "Tool Setup": ("tool",),
    "Targets": ("targets",),
}


def _h(x) -> str:
    text = x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _pose(T: np.ndarray, digits: int = 6) -> list:
    return np.round(np.asarray(T, float), digits).tolist()


def components(s, base_pose: bool = False, tool_nominal: bool = True) -> dict[str, str]:
    """Hash per setup component of session ``s``.

    ``base_pose``: include the full robot base pose (the feasibility check runs from it;
    plans choose their own stations). In static mode the base height and orientation always
    count, because the planner keeps them. ``tool_nominal=False`` leaves out the tool's
    rough placement (the tool optimizer chooses it: applying its result must not make the
    result itself out of date)."""
    T = np.asarray(s.T_world_base, float)
    robot = {"name": s.robot_name, "urdf": s.robot_urdf, "mount": s.mount}
    if s.robot is not None and s.robot.spec.limits_deg is not None:  # v6: user joint limits
        robot["limits"] = s.robot.spec.limits_deg
    if s.base.mode != "movable":
        robot["base_z"] = round(float(T[2, 3]), 6)  # the planner keeps height and orientation
        robot["base_R"] = _pose(T[:3, :3])
    out = {
        "robot": _h(robot),
        "carrier": _h(s.base.model_dump_json()),
        "tool": "",
        "part": "",
        "targets": _h(s.targets.model_dump_json()),
        "stayout": _h([p.model_dump(mode="json") for p in getattr(s, "stayouts", [])]),
    }
    if s.tool is not None:
        tool = s.tool.model_dump(mode="json")
        if not tool_nominal:
            tool["extension"].pop("nominal", None)
        out["tool"] = _h(tool)
    if s.part is not None:
        out["part"] = _h([s.part.mesh_path, s.part.units, _pose(s.part.T_world_part)])
    if base_pose:
        out["base_pose"] = _h(_pose(T))
    return out


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
    return changed(s.stamps.get(sol.label), stamp_for(s, kind))


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
