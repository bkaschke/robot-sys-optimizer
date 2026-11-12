"""Collision geometry in the package builder (2026-11-11): how the package's mesh parts are
simplified for collision (core.urdf_package ``Package.collision`` / ``Part.collision``; written
to ``meshes/collision`` and the URDF's ``<collision>`` elements on export / use), and a preview
of the collision shapes on the package in the builder window.
"""

from __future__ import annotations

import threading

import numpy as np
import viser

from core.collision_mesh import MODE_LABELS, MODES
from core.transforms import matrix_to_wxyz, pose_to_matrix
from core.urdf_package import Collision, part_collision
from ui.widgets import Latest

PACKAGE = "Package setting"
LABEL_OF = {MODE_LABELS[m]: m for m in MODES}
COLOR = (255, 140, 30)


class PackageCollision:
    def __init__(self, server: viser.ViserServer, builder) -> None:
        self.server, self.b = server, builder
        self._handles: list = []
        self._busy = False
        self._lock = threading.Lock()
        gui = server.gui
        options = tuple(MODE_LABELS[m] for m in MODES)
        with gui.add_folder("Collision geometry", expand_by_default=False):
            gui.add_markdown(
                "_Mesh parts get a simpler copy for collision checks (the URDF's `<collision>`); the "
                "visual keeps the full mesh. **Wrap** puts one closed shell around many small pieces._"
            )
            self.mode = gui.add_dropdown("Simplify", options, initial_value=MODE_LABELS["auto"])
            self.faces = gui.add_number(
                "Triangles", 4000, min=50, max=200000, step=100, hint="target per mesh"
            )
            self.wrap_mm = gui.add_number("Wrap offset (mm)", 5.0, min=0.5, max=100.0, step=0.5)
            self.pad_mm = gui.add_number(
                "Padding (mm)",
                0.0,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="every part's collision shape this much larger",
            )
            self.part_mode = gui.add_dropdown(
                "This part", (PACKAGE, *options), hint="the selected part: the package setting or its own"
            )
            self.show = gui.add_checkbox(
                "Show collision shapes", False, hint="orange: what the checks will use"
            )
            self.info = gui.add_markdown("")
        for w in (self.mode, self.faces, self.wrap_mm, self.pad_mm):
            w.on_update(lambda _e: self._package_changed())
        self.part_mode.on_update(lambda _e: self._part_changed())
        self.show.on_update(lambda _e: self.redraw_soon())
        self._redraw = Latest(self.redraw)

    # --- widgets <-> package -------------------------------------------------------------------

    def load(self) -> None:
        """Widgets from the builder's package (after load_widgets)."""
        c, parts = self.b.pkg.collision, self.b.pkg.parts
        self._busy = True
        try:
            self.mode.value = MODE_LABELS[c.mode]
            self.faces.value = c.faces
            self.wrap_mm.value = c.wrap_mm
            self.pad_mm.value = c.pad_mm
            own = parts[self.b.sel].collision if parts else None
            self.part_mode.value = MODE_LABELS[own] if own else PACKAGE
            self.part_mode.visible = bool(parts) and any(x.shape == "mesh" for x in parts)
        finally:
            self._busy = False

    def _package_changed(self) -> None:
        if self._busy:
            return
        c = Collision(
            mode=LABEL_OF.get(self.mode.value, "auto"),
            faces=int(np.clip(self.faces.value, 50, 200000)),
            wrap_mm=float(np.clip(self.wrap_mm.value, 0.5, 200.0)),
            pad_mm=float(np.clip(self.pad_mm.value, 0.0, 100.0)),
        )
        self.b.pkg = self.b.pkg.model_copy(update={"collision": c})
        self.redraw_soon()

    def _part_changed(self) -> None:
        if self._busy or not self.b.pkg.parts:
            return
        own = None if self.part_mode.value == PACKAGE else LABEL_OF.get(self.part_mode.value)
        parts = list(self.b.pkg.parts)
        parts[self.b.sel] = parts[self.b.sel].model_copy(update={"collision": own})
        self.b.pkg = self.b.pkg.model_copy(update={"parts": parts})
        self.redraw_soon()

    # --- preview -------------------------------------------------------------------------------

    def redraw_soon(self) -> None:
        threading.Thread(target=self._redraw, daemon=True).start()

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def redraw(self) -> None:
        """The collision copy of every part under the package frame (wrap can take seconds on a
        large mesh; results are cached)."""
        from core.carrier import primitive_mesh

        with self._lock:
            self.clear()
            if not self.show.value:
                self.info.content = ""
                return
            from ui.package_preview import ROOT

            rows = []
            for k, part in enumerate(self.b.pkg.parts):
                try:
                    m = part_collision(self.b.pkg, part, self.b.work / "meshes")
                    full = primitive_mesh(self.b.primitive(part))
                except (OSError, ValueError) as err:
                    rows.append(f"{part.name}: {type(err).__name__}")
                    continue
                low = full if m is None else m
                if part.shape == "mesh":
                    rows.append(f"{part.name}: {len(full.faces)} → {len(low.faces)} triangles")
                T = pose_to_matrix(part.pose)
                self._handles.append(
                    self.server.scene.add_mesh_simple(
                        f"{ROOT}/collision_{k}",
                        np.asarray(low.vertices, np.float32),
                        np.asarray(low.faces, np.uint32),
                        color=COLOR,
                        opacity=0.5,
                        flat_shading=True,
                        wxyz=matrix_to_wxyz(T[:3, :3]),
                        position=tuple(T[:3, 3]),
                    )
                )
            self.info.content = (
                "  \n".join(rows) if rows else "_No mesh parts: boxes and cylinders are used as they are._"
            )
