"""Collision geometry in the Robot & cell tab (v9): the **robot arm padding**, and the part's
collision geometry state. Everything else about collision geometry (how parts, tools and carriers
are simplified, their padding, imported collision meshes) is set in the Package builder, which
writes the collision meshes into the packages and the project's part (issues #1, #11).

``sync`` copies the settings into ``core.collision`` (shipped to the planner workers): the arm
padding, the part's own collision mesh (``Part.collision_mesh``, made in the builder) and, for
projects from before v9, the part faces kept exact (``Part.collision_keep``). Meshes without
their own collision copy are reduced automatically when they are large (core.collision_mesh
"auto").
"""

from __future__ import annotations

import viser

from core import collision
from ui.session import UNIT_SCALE

COMPONENTS = {"Robot": "robot", "Tool": "tool", "Part": "part", "External axis": "carrier"}


class CollisionSettings:
    """Built in the Robot & cell tab (app.py); ``view`` is set by app.py."""

    def __init__(self, server: viser.ViserServer, session) -> None:
        self.s = session
        self.view = None
        gui = server.gui
        self.folder = gui.add_folder("Collision geometry", expand_by_default=False)
        with self.folder:
            self.pad_mm = gui.add_number(
                "Robot arm padding (mm)",
                0.0,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="the arm's collision shapes this much larger against the part, floor, carrier, "
                "stay-out zones and tool (not against itself)",
            )
            self.part_info = gui.add_markdown("")
        self.pad_mm.on_update(lambda _e: self.settings_changed())
        self.sync()

    def sync(self) -> None:
        """Copy the settings and the part's collision geometry into core.collision (scenes built
        from now on use them; planner jobs take a copy)."""
        part = self.s.part
        keep = tuple(int(f) for f in part.collision_keep) if part is not None else ()
        low, scale = None, 1.0
        if part is not None and part.collision_mesh:
            try:
                low = str(self.s.resolve(part.collision_mesh))
                scale = UNIT_SCALE[part.units]
            except FileNotFoundError:
                low = None
        pad = max(float(self.pad_mm.value), 0.0) / 1000
        collision.set_geom_state(({}, keep, pad, low, scale))
        if part is None:
            text = "_No part loaded._"
        elif low:
            text = f"_Part: its own collision mesh (`{part.collision_mesh}`), made in the Package builder._"
        else:
            text = (
                "_Part: a large mesh is reduced automatically for collision. Package builder → **Part** "
                "simplifies it your way (wrap, padding, surfaces kept exact)._"
            )
        if self.part_info.content != text:
            self.part_info.content = text

    def settings_changed(self) -> None:
        self.sync()
        if self.view is not None:
            self.view.redraw_soon()
