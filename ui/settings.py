"""Panel settings saved with the project (``WorkState.ui``) so the app reopens as it was left.

Only the listed widgets are saved: generation/brush/check/planner settings, display
options and the selected solution. Tool and carrier fields are not listed because they
are saved through the project model itself. Keys are "<panel>.<attribute>" so they stay
stable when labels change. Restoring skips unknown keys, values of the wrong type and
dropdown values that are no longer offered.
"""

from __future__ import annotations

from typing import Any

from viser import GuiCheckboxHandle, GuiDropdownHandle, GuiNumberHandle, GuiSliderHandle, GuiTextHandle

SAVED: dict[str, tuple[str, ...]] = {
    "targets": (
        "click_mode",
        "angle",
        "normal_dir",
        "normal_angle",
        "method",
        "spacing",
        "stepover",
        "direction",
        "normals_mode",
        "color_by",
        "show_normals",
        "normal_len",
        "point_size",
    ),
    "targets.tools": ("row", "col", "row_dir", "radius", "max_angle", "max_edge"),
    "optimize": ("clearance", "joint_margin", "torque"),
    "placement": ("res", "grid", "half"),
    "stations": ("max_stations", "step", "refine"),
    "tool_opt": ("max_stations", "tol", "shape_evals"),
    "results": ("choice", "show_lines", "show_ghosts"),
    "results.export": ("screenshot",),
}


def _panel(state: dict, path: str):
    obj: Any = state
    for part in path.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _handles(state: dict):
    for panel, attrs in SAVED.items():
        obj = _panel(state, panel)
        for a in attrs:
            h = getattr(obj, a, None) if obj is not None else None
            if h is not None and hasattr(h, "value"):
                yield f"{panel}.{a}", h
    if "dark_toggle" in state:
        yield "app.dark", state["dark_toggle"]


def collect(state: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, h in _handles(state):
        v = h.value
        out[key] = list(v) if isinstance(v, tuple) else v
    return out


def apply(state: dict, saved: dict[str, Any]) -> list[str]:
    """Set saved values on the widgets (their callbacks run as for a user edit). Returns the
    keys that were applied."""
    done = []
    for key, h in _handles(state):
        if key not in saved:
            continue
        v = saved[key]
        try:
            if isinstance(h, GuiDropdownHandle):
                if v not in h.options:
                    continue
            elif isinstance(h, GuiCheckboxHandle):
                if not isinstance(v, bool):
                    continue
            elif isinstance(h, GuiNumberHandle | GuiSliderHandle):
                if isinstance(v, bool) or not isinstance(v, int | float):
                    continue
            elif isinstance(h, GuiTextHandle):
                if not isinstance(v, str):
                    continue
            elif isinstance(v, list):  # vector2 / vector3
                cur = h.value
                if not isinstance(cur, tuple) or len(cur) != len(v):
                    continue
                v = tuple(float(x) for x in v)
            if h.value != v:
                h.value = v
            done.append(key)
        except (ValueError, TypeError):
            continue
    return done
