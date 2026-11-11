"""External axis tab: mobile-base sides and track alignment (v3 feedback items 2 and 3).

* Robot sides that may face the part: Front / Side 1 (left) / Side 2 (right) / Rear; several
  may be ticked (the planner picks per stop). None ticked: the free x / y / heading search
  with min / preferred standoff. Any ticked: stops on the contour at a standoff from *Standoff
  from* to *Standoff up to* (equal: one fixed standoff; else the planner picks it per stop).
* Part sides the mobile base may work from, in the part's own frame (+X / -X / +Y / -Y);
  none ticked = all.
* Rail / gantry: *Align track with the part* keeps the track along the part's long axis
  (re-applied when the part moves).
"""

from __future__ import annotations

ROBOT_SIDES = {"front": "Front", "left": "Side 1 (left)", "right": "Side 2 (right)", "back": "Rear"}
PART_SIDES = {"+x": "Part +X side", "-x": "Part −X side", "+y": "Part +Y side", "-y": "Part −Y side"}


def build_align(gui, panel) -> None:
    """Inside the Axis type folder."""
    panel.align = gui.add_checkbox(
        "Align track with the part", False, hint="rail / gantry along the part's long axis, centred on it"
    )


def build_mobile(gui, panel) -> None:
    """Inside the Standoff and safety folder."""
    panel.sides_note = gui.add_markdown("**Robot sides that may face the part** (none: free heading search)")
    panel.side_boxes = {k: gui.add_checkbox(label, False) for k, label in ROBOT_SIDES.items()}
    panel.fixed_so = gui.add_number(
        "Standoff from (mm)", 400.0, min=0.0, step=10.0, hint="mobile base outline to part outline"
    )
    panel.max_so = gui.add_number(
        "Standoff up to (mm)", 400.0, min=0.0, step=10.0, hint="at or below *from* = one fixed standoff"
    )
    panel.part_note = gui.add_markdown("**Part sides it may work from** (part frame; none: all)")
    panel.part_boxes = {k: gui.add_checkbox(label, False) for k, label in PART_SIDES.items()}


def widgets(panel) -> list:
    return [panel.align, *panel.side_boxes.values(), *panel.part_boxes.values()]


def contour_mode(panel) -> bool:
    return any(b.value for b in panel.side_boxes.values())


def show(panel, movable: bool, kind: str) -> None:
    mobile = movable and kind == "mobile"
    panel.align.visible = movable and kind in ("rail", "gantry")
    for w in (panel.sides_note, *panel.side_boxes.values()):
        w.visible = mobile
    contour = mobile and contour_mode(panel)
    for w in (panel.fixed_so, panel.max_so, panel.part_note, *panel.part_boxes.values()):
        w.visible = contour
    panel.so.visible = not contour  # one fixed standoff replaces min / preferred


def load(panel, b) -> None:
    sides = set(b.robot_sides())
    for k, box in panel.side_boxes.items():
        box.value = k in sides
    for k, box in panel.part_boxes.items():
        box.value = k in b.part_sides
    fixed = b.fixed_standoff_m if b.fixed_standoff_m is not None else b.standoff_m.carrier_part_min
    panel.fixed_so.value = fixed * 1000
    panel.max_so.value = (b.max_standoff_m if b.max_standoff_m is not None else fixed) * 1000
    panel.align.value = b.align_to_part


def collect(panel, kind: str) -> dict:
    """Base fields from the widgets."""
    sides = [k for k, box in panel.side_boxes.items() if box.value] if kind == "mobile" else []
    return {
        "facing": None,
        "facings": sides,
        "fixed_standoff_m": panel.fixed_so.value / 1000 if sides else None,
        "max_standoff_m": panel.max_so.value / 1000
        if sides and panel.max_so.value > panel.fixed_so.value
        else None,
        "part_sides": [k for k, box in panel.part_boxes.items() if box.value] if sides else [],
        "align_to_part": bool(panel.align.value) and kind in ("rail", "gantry"),
    }
