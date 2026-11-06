"""Tool Setup -> Extension section: the connector between the flange mount and the head.

Types: **pose** (v2 default for new tools: head pose x, y, z, rx, ry, rz relative to the
mount, each with a min / max range, spline connector), and the legacy **straight** / **bent**
(L, theta, d, roll). Only the fields of the chosen type are shown. The rough-placement
sliders (Extension -> Rough placement) are rebuilt for the type's free values.
"""

from __future__ import annotations

import numpy as np

from core.schema import LEGACY_KEYS, POSE_KEYS

TYPES = ("pose", "straight", "bent")
# key: (label, GUI unit per SI unit)
FIELDS = {
    "L": ("L", 1000.0),
    "theta_deg": ("θ", 1.0),
    "d": ("d", 1000.0),
    "roll_deg": ("roll", 1.0),
    "x": ("x", 1000.0),
    "y": ("y", 1000.0),
    "z": ("z", 1000.0),
    "rx_deg": ("rx / W", 1.0),
    "ry_deg": ("ry / P", 1.0),
    "rz_deg": ("rz / R", 1.0),
}


def _unit(key: str) -> str:
    return "mm" if FIELDS[key][1] == 1000.0 else "deg"


def build(gui, tab) -> None:
    """Create the Extension widgets on ``tab`` (inside its Extension folder)."""
    tab.ext_type = gui.add_dropdown("Type", TYPES, initial_value="straight")
    tab.ext_ranges = {}
    for key, (label, _) in FIELDS.items():
        tab.ext_ranges[key] = gui.add_vector2(f"{label} range ({_unit(key)})", (0.0, 0.0), step=1.0)
    tab.r_L, tab.r_theta, tab.r_d, tab.r_roll = (tab.ext_ranges[k] for k in LEGACY_KEYS)  # old names
    tab.ext_note = gui.add_markdown(
        "_Head pose relative to the flange mount: each value within its range (equal ends fix it). "
        "The connector is a smooth curve from the mount to the head; the optimizer chooses the pose._"
    )
    tab.legacy_note = gui.add_markdown(
        "_Legacy tool (L / θ / d / roll). For **x y z / W P R limits** for the tool optimizer, set "
        "**Type = pose**: the current shape is kept, with every limit fixed at its value; widen the "
        "ones the optimizer may change._"
    )
    tab.radius = gui.add_number("Capsule radius (mm)", 40.0, min=1.0, step=1.0)
    tab.kg_per_m = gui.add_number("Mass per m (kg/m)", 0.0, min=0.0, step=0.1)
    tab.nominal_folder = gui.add_folder("Rough placement")


def widgets(tab) -> tuple:
    return (tab.ext_type, *tab.ext_ranges.values(), tab.radius, tab.kg_per_m)


def keys_for(kind: str) -> tuple[str, ...]:
    return POSE_KEYS if kind == "pose" else LEGACY_KEYS


def commit(tab, d: dict) -> None:
    """Widgets -> the tool dict (``ToolSetupTab._commit``)."""
    x = d["extension"]
    old = x["type"]
    kind = x["type"] = tab.ext_type.value
    if kind == "pose" and old != "pose":
        _legacy_to_pose(x)  # keep the tool's shape; every limit fixed at its value
    else:
        for key in keys_for(kind):
            x[key] = tuple(v / FIELDS[key][1] for v in tab.ext_ranges[key].value)
    if kind != "bent":
        x["theta_deg"], x["d"] = (0.0, 0.0), (0.0, 0.0)
    x["radius"] = tab.radius.value / 1000
    x["kg_per_m"] = tab.kg_per_m.value
    n = x["nominal"]
    for key in keys_for(kind):
        lo, hi = x[key]
        n[key] = float(np.clip(n.get(key, lo), lo, hi))  # keep the placement inside new ranges


def _legacy_to_pose(x: dict) -> None:
    """Legacy L / theta / d / roll -> the same head pose as x y z rx ry rz (nominal), with each
    range fixed at that value."""
    from core.transforms import matrix_to_rpy

    n = x["nominal"]
    c, s = np.cos(np.radians(n["roll_deg"])), np.sin(np.radians(n["roll_deg"]))
    t = np.radians(n["theta_deg"]) if x.get("theta_deg", (0, 0)) != (0.0, 0.0) else 0.0
    d = n["d"] if x.get("d", (0, 0)) != (0.0, 0.0) else 0.0
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    p = Rz @ (np.array([0.0, 0.0, n["L"]]) + Rx @ np.array([0.0, d, 0.0]))
    R = Rz @ Rx
    rpy = np.degrees(matrix_to_rpy(R))
    vals = {"x": p[0], "y": p[1], "z": p[2], "rx_deg": rpy[0], "ry_deg": rpy[1], "rz_deg": rpy[2]}
    for key, v in vals.items():
        n[key] = float(v)
        x[key] = (float(v), float(v))


def refresh(tab, x, on_nominal) -> None:
    """Tool -> widgets; rebuild the rough-placement sliders. ``on_nominal(key, scale)``
    returns a slider callback (``ToolSetupTab._on_nominal``)."""
    tab.ext_type.value = x.type
    shown = set(keys_for(x.type)) - ({"theta_deg", "d"} if x.type == "straight" else set())
    for key, w in tab.ext_ranges.items():
        w.value = tuple(float(v) * FIELDS[key][1] for v in getattr(x, key))
        w.visible = key in shown
    tab.ext_note.visible = x.type == "pose"
    tab.legacy_note.visible = x.type != "pose"
    tab.radius.value, tab.kg_per_m.value = x.radius * 1000, x.kg_per_m
    with tab.nominal_folder:
        for key in keys_for(x.type):
            label, k = FIELDS[key]
            lo, hi = (v * k for v in getattr(x, key))
            if hi - lo < 1e-9:
                continue  # fixed value: nothing to slide
            sl = tab.server.gui.add_slider(
                f"{label} ({_unit(key)})", lo, hi, (hi - lo) / 1000, getattr(x.nominal, key) * k
            )
            sl.on_update(on_nominal(key, 1.0 / k))
            tab._nominal[key] = sl
            tab._dyn.append(sl)
