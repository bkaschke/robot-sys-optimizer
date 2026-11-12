"""Collision geometry settings (Robot & cell tab, 2026-11-11): how each component's meshes are
simplified for the collision checks (core.collision_mesh), the robot arm padding, and the part
surfaces kept at full detail.

Settings are global (``core.collision.GEOM`` / ``ARM_PAD`` / ``PART_KEEP``, shipped to the
planner workers) and saved with the project's panel settings; the kept faces with the part
(``Part.collision_keep``). The left panel's Collision geometry (ui.collision_view) only shows
or hides the resulting shapes.
"""

from __future__ import annotations

import viser

from core import collision
from core.collision_mesh import MODE_LABELS, MODES, Geom

COMPONENTS = {"Robot": "robot", "Tool": "tool", "Part": "part", "External axis": "carrier"}
LABEL_OF = {MODE_LABELS[m]: m for m in MODES}


class CollisionSettings:
    """Built in the Robot & cell tab (app.py); ``targets`` and ``view`` are set by app.py."""

    def __init__(self, server: viser.ViserServer, session) -> None:
        self.s = session
        self.targets = self.view = None
        gui = server.gui
        options = tuple(MODE_LABELS[m] for m in MODES)
        with gui.add_folder("Collision geometry", expand_by_default=False):
            gui.add_markdown(
                "_The collision checks use simpler copies of large meshes; the 3D view shows the full "
                "ones. Left panel → Collision geometry shows them. Packages carry their own (Package "
                "builder)._"
            )
            self.pad_mm = gui.add_number(
                "Robot arm padding (mm)",
                0.0,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="the arm's collision shapes this much larger against the part, floor, carrier, "
                "stay-out zones and tool (not against itself)",
            )
            self.mode = {
                name: gui.add_dropdown(f"Simplify {name.lower()}", options, initial_value=MODE_LABELS["auto"])
                for name in COMPONENTS
            }
            for name, c in COMPONENTS.items():  # saved panel settings (ui.settings): attributes
                setattr(self, f"mode_{c}", self.mode[name])
            self.faces = gui.add_number(
                "Triangles",
                4000,
                min=100,
                max=200000,
                step=100,
                hint="target triangles per mesh (reduce, wrap)",
            )
            self.wrap_mm = gui.add_number(
                "Wrap offset (mm)",
                5.0,
                min=0.5,
                max=100.0,
                step=0.5,
                hint="the wrap's distance outside the surface",
            )
            self.keep_btn = gui.add_button(
                "Keep selected part surfaces exact",
                hint="the surfaces selected in Targets keep full detail; the rest is simplified",
            )
            self.clear_btn = gui.add_button("Clear kept surfaces")
            self.keep_info = gui.add_markdown("")
        for w in (*self.mode.values(), self.faces, self.wrap_mm, self.pad_mm):
            w.on_update(lambda _e: self.settings_changed())
        self.keep_btn.on_click(lambda _e: self.keep_selection())
        self.clear_btn.on_click(lambda _e: self.set_keep([]))
        self.sync()

    def geom(self, name: str) -> Geom:
        return Geom(
            LABEL_OF.get(self.mode[name].value, "auto"), int(self.faces.value), float(self.wrap_mm.value)
        )

    def sync(self) -> None:
        """Copy the widgets and the part's kept faces into core.collision (all scenes built from
        now on use them; planner jobs take a copy)."""
        part = self.s.part
        keep = tuple(int(f) for f in part.collision_keep) if part is not None else ()
        pad = max(float(self.pad_mm.value), 0.0) / 1000
        collision.set_geom_state(({c: self.geom(n) for n, c in COMPONENTS.items()}, keep, pad))
        n = len(keep)
        text = f"_{n} part faces kept exact._" if n else "_No part surfaces kept exact._"
        if self.keep_info.content != text:
            self.keep_info.content = text

    def settings_changed(self) -> None:
        self.sync()
        if self.view is not None:
            self.view.redraw_soon()

    def keep_selection(self) -> None:
        sel = getattr(self.targets, "selection", None)
        self.set_keep([] if sel is None else [int(f) for f in sel])

    def set_keep(self, faces: list[int]) -> None:
        if self.s.part is None:
            return
        self.s.part.collision_keep = sorted(set(faces))
        self.settings_changed()
