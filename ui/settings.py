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
        "inset",
        "row_angle",
        "angle",
        "normal_dir",
        "normal_angle",
        "method",
        "spacing",
        "stepover",
        "direction",
        "normals_mode",
        "split_sides",
        "color_by",
        "show_normals",
        "normal_len",
        "point_size",
        "show_unreached",
    ),
    "targets.heat": ("choice", "cones"),  # v8 T8
    "targets.tools": ("row", "col", "row_dir", "radius", "max_angle", "max_edge"),
    "optimize": ("check_from",),  # the margins are saved in optimizer.margins, not here
    "objects": ("gizmo_size", "gizmo_top"),
    "view": ("lighting", "shadows"),  # R2
    "placement": ("res", "grid", "half"),
    "stations": (
        "max_stations",
        "skip_unfit",  # v10 T12
        "step",
        "refine",
        "objective_m",
        "robust_measure",
        "robust_samples",
        "areas",
        "zoning",
        "mode",
        "objective",
        "sing_pct",
    ),
    "stations.costs": ("require_all", "require_config"),
    "tool_opt.costs": ("require_all", "require_config"),
    "tool_opt": (
        "max_stations",
        "tol",
        "shape_evals",
        "shape",
        "start_shape",
        "loop",
        "objective",
        "family_cost",
        "tube",
        "bend_step",
    ),
    "tolerance": ("part_mm", "part_deg", "station_mm", "samples", "seed"),
    "edge": ("auto",),  # v10 T10
    "cable": ("mode", "dress", "diameter", "offset", "side", "show", "wrap"),  # v8 T18
    "safety": ("show", "distance"),
    "collision": ("pad_mm", "link_pad"),  # Robot & cell: the arm padding (+ v10 T19 per link)
    "cycle": (
        "joint_speed",
        "joint_acc",
        "process_speed",
        "approach_speed",
        "linear_acc",
        "carrier_speed",
        "carrier_acc",
        "lift_speed",
        "settle_s",
        "joint_jerk",  # v10 T14
        "timing",
    ),
    "picker": ("choice",),  # v10 #6: the chosen Optimizer
    "frame": (
        "standoff",
        "step",
        "shift",
        "z",
        "z_steps",
        "top",
        "beam",
        "beam_steps",
        "refine",
    ),  # Gantry placement
    "results": ("choice", "show_lines", "show_ghosts", "show_areas"),
    "results.export": ("screenshot", "flange", "prog_format"),  # + v10 T21
    "sim": (
        "motion",
        "smooth",  # v10 T13
        "timed",  # v10 T14
        "turn_lift",  # v10 T15
        "motion_mode",  # v10 T16
        "order",
        "step",
        "jump",
        "pattern",
        "angle",
        "corner",
        "row_tol",
        "inward",
        "no_cross",
        "lift",
        "home",
        "between",
        "drive",
        "spin_opt",
    ),
}


# v12 #11: a renamed widget attribute keeps the users' saved value: old key -> new key
RENAMED: dict[str, str] = {}


def missing(state: dict) -> list[str]:
    """Saved keys whose panel or widget does not exist on this app (``SAVED`` out of date: such a
    setting would be dropped without a word). tests/test_v12_settings.py keeps this empty."""
    out = []
    for panel, attrs in SAVED.items():
        obj = _panel(state, panel)
        for a in attrs:
            h = getattr(obj, a, None) if obj is not None else None
            if h is None or not hasattr(h, "value"):
                out.append(f"{panel}.{a}")
    return out


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
    old = {RENAMED[k]: v for k, v in saved.items() if k in RENAMED}  # v12 #11: under the new key
    saved = {**old, **{k: v for k, v in saved.items() if k not in RENAMED}}  # (a new key wins)
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
