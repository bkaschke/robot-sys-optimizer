"""External Axis tab: rail / gantry lengths in one setting (v7 round 2, T13).

*Rail length* (rail) or *X rail length* + *Bridge length* (gantry) set ``Base.axis_length_m``;
the bodies marked "follows axis length" take that size and, unless *Set travel limits
manually* (Advanced) is ticked, the travel range follows (core.carrier_axis). The Travel
fields then only show the computed limits. Shown only when some body follows a length, so
older projects look and behave as before. Mobile bases have no lengths.
"""

from __future__ import annotations

from core.carrier_axis import AXES, apply_axis_lengths


def build(gui, panel) -> None:
    panel.len_x = gui.add_number("Rail length (mm)", 6000.0, min=100.0, step=100.0)
    panel.len_y = gui.add_number("Bridge length (mm)", 2000.0, min=100.0, step=100.0)
    panel.axis_adv = gui.add_folder("Advanced", expand_by_default=False)
    with panel.axis_adv:
        panel.end_stop = gui.add_number("End stop (mm)", 50.0, min=0.0, step=5.0)
        panel.manual_travel = gui.add_checkbox(
            "Set travel limits manually", False, hint="off: the travel follows the axis lengths"
        )
    panel.axis_info = gui.add_markdown("")


def widgets(panel) -> list:
    return [panel.manual_travel]


def _stretch_axes(panel, kind: str) -> set[str]:
    return {b.stretch for b in panel.bodies if b.stretch} & set(AXES.get(kind, ()))


def show(panel, movable: bool, kind: str) -> None:
    ax = _stretch_axes(panel, kind) if movable else set()
    panel.len_x.visible = "x" in ax
    panel.len_y.visible = "y" in ax and kind == "gantry"
    panel.axis_adv.visible = bool(ax)
    panel.axis_info.visible = bool(ax)
    try:
        panel.len_x.label = "Rail length (mm)" if kind == "rail" else "X rail length (mm)"
    except AttributeError:
        pass
    computed = bool(ax) and not panel.manual_travel.value
    panel.tx.disabled = computed and "x" in ax  # read-only: they show the computed limits
    panel.ty.disabled = computed and "y" in ax


def load(panel, b) -> None:
    if "x" in b.axis_length_m:
        panel.len_x.value = b.axis_length_m["x"] * 1000
    if "y" in b.axis_length_m:
        panel.len_y.value = b.axis_length_m["y"] * 1000
    panel.end_stop.value = b.end_stop_m * 1000
    panel.manual_travel.value = b.manual_travel
    info(panel, b)


def collect(panel, kind: str) -> dict:
    ax = _stretch_axes(panel, kind)
    lengths = {}
    if "x" in ax:
        lengths["x"] = panel.len_x.value / 1000
    if "y" in ax:
        lengths["y"] = panel.len_y.value / 1000
    return {
        "axis_length_m": lengths,
        "end_stop_m": panel.end_stop.value / 1000,
        "manual_travel": bool(panel.manual_travel.value),
    }


def apply(panel, base):
    """``base`` with lengths applied (stretched bodies, computed travel); the Travel fields
    show the result. Raises ValueError when an axis is too short."""
    base = apply_axis_lengths(base, panel.s.resolve)
    panel.bodies = list(base.bodies) if base.mode == "movable" else panel.bodies
    j = base.joints
    if j.x:
        panel.tx.value = tuple(v * 1000 for v in j.x)
    if j.y:
        panel.ty.value = tuple(v * 1000 for v in j.y)
    info(panel, base)
    return base


def info(panel, b) -> None:
    j = b.joints
    parts = []
    for ax, r in (("x", j.x), ("y", j.y)):
        if r is not None and ax in b.axis_length_m:
            parts.append(f"{ax}: {r[0] * 1000:.0f} … {r[1] * 1000:.0f} mm")
    how = "set manually" if b.manual_travel else "from the lengths"
    panel.axis_info.content = f"_Travel {how}: {', '.join(parts)}_" if parts else ""
