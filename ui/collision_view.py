"""Collision geometry in the left panel (2026-11-11): show / hide the shapes the collision checks
use, per component, and choose how each component's meshes are simplified for collision.

The viewer always draws the full visual meshes. The collision copies (core.collision_mesh) are
drawn on top, translucent orange, from the jog's live collision scene (ui.cell_collision), so
what is shown is exactly what the planner, the paths and the jog check. Settings are global
(``core.collision.GEOM``, shipped to the planner workers) and saved with the project's panel
settings; the part's full-detail surfaces (``Part.collision_keep``) with the part.
"""

from __future__ import annotations

import threading

import numpy as np
import viser

from core import collision
from core.collision_mesh import MODE_LABELS, MODES, Geom
from core.transforms import matrix_to_wxyz
from ui.widgets import Latest

COMPONENTS = {"Robot": "robot", "Tool": "tool", "Part": "part", "External axis": "carrier"}
LABEL_OF = {MODE_LABELS[m]: m for m in MODES}
ROOT = "/collision_geometry"
COLOR = (255, 140, 30)


class CollisionPanel:
    """Built inside the left panel (app.py); ``cell`` and ``targets`` are set by app.py."""

    def __init__(self, server: viser.ViserServer, session) -> None:
        self.server, self.s = server, session
        self.cell = self.targets = None
        self._handles: dict[int, object] = {}
        self._drawn: dict[int, int] = {}  # gid -> id of the mesh drawn for it
        self._pads: dict = {}  # (gid, id(mesh), pad) -> bloated copy for drawing
        self._lock = threading.Lock()
        gui = server.gui
        with gui.add_folder("Collision geometry", expand_by_default=False):
            gui.add_markdown(
                "_The checks use simpler copies of large meshes; the viewer shows the full ones. "
                "Tick a component to see its collision shapes (orange)._"
            )
            self.show = {
                name: gui.add_checkbox(
                    f"Show {name.lower()}", False, hint=f"draw the {name.lower()}'s collision shapes"
                )
                for name in COMPONENTS
            }
            self.pad_mm = gui.add_number(
                "Robot arm padding (mm)",
                0.0,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="the arm's collision shapes this much larger against the part, floor, carrier, "
                "stay-out zones and tool (not against itself)",
            )
            with gui.add_folder("Simplify for collision", expand_by_default=False):
                options = tuple(MODE_LABELS[m] for m in MODES)
                self.mode = {
                    name: gui.add_dropdown(name, options, initial_value=MODE_LABELS["auto"])
                    for name in COMPONENTS
                }
                for name, c in COMPONENTS.items():  # saved panel settings (ui.settings): attributes
                    setattr(self, f"mode_{c}", self.mode[name])
                    setattr(self, f"show_{c}", self.show[name])
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
        for box in self.show.values():
            box.on_update(lambda _e: self.redraw_soon())
        for w in (*self.mode.values(), self.faces, self.wrap_mm, self.pad_mm):
            w.on_update(lambda _e: self.settings_changed())
        self.keep_btn.on_click(lambda _e: self.keep_selection())
        self.clear_btn.on_click(lambda _e: self.set_keep([]))
        self._redraw = Latest(self.redraw)
        self.sync()

    # --- settings ------------------------------------------------------------------------------

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
        self.redraw_soon()

    def keep_selection(self) -> None:
        sel = getattr(self.targets, "selection", None)
        self.set_keep([] if sel is None else [int(f) for f in sel])

    def set_keep(self, faces: list[int]) -> None:
        if self.s.part is None:
            return
        self.s.part.collision_keep = sorted(set(faces))
        self.settings_changed()

    # --- drawing -------------------------------------------------------------------------------

    def redraw_soon(self) -> None:
        threading.Thread(target=self._redraw, daemon=True).start()

    def _scene(self):
        jog = getattr(self.cell, "jog_panel", None) if self.cell is not None else None
        coll = getattr(jog, "coll", None)
        if coll is None:
            return None
        try:
            return coll.scene()
        except Exception:  # a broken setup must not break the viewer
            return None

    def _groups(self, sc) -> dict[str, list[int]]:
        return {
            "robot": list(sc.robot_ids),
            "tool": list(sc.tool_ids),
            "part": [] if sc.part_id is None else [sc.part_id],
            "carrier": list(sc.carrier_ids),
        }

    def wanted(self) -> set[str]:
        return {c for n, c in COMPONENTS.items() if self.show[n].value}

    def redraw(self) -> None:
        """Draw / move the collision shapes of the ticked components at the current pose."""
        with self._lock:
            want = self.wanted()
            sc = self._scene() if want else None
            if sc is None or self.s.q is None:
                self.clear()
                return
            sc.set_base(np.asarray(self.s.T_world_base, float))
            poses = {gid: T for gid, _, T in sc.world_shapes(np.asarray(self.s.q, float))}
            keep = set()
            pad = float(getattr(sc, "arm_pad", 0.0))
            for comp, gids in self._groups(sc).items():
                if comp not in want:
                    continue
                for gid in gids:
                    mesh = sc.shape(gid)
                    if mesh is None:
                        continue
                    if comp == "robot" and pad > 0:  # drawn bloated like it is checked
                        mesh = self._padded(gid, mesh, pad)
                    keep.add(gid)
                    T = poses[gid]
                    h = self._handles.get(gid)
                    if h is None or self._drawn.get(gid) != id(mesh):
                        if h is not None:
                            h.remove()
                        h = self.server.scene.add_mesh_simple(
                            f"{ROOT}/g{gid}",
                            np.asarray(mesh.vertices, np.float32),
                            np.asarray(mesh.faces, np.uint32),
                            color=COLOR,
                            opacity=0.45,
                            flat_shading=True,
                        )
                        self._handles[gid], self._drawn[gid] = h, id(mesh)
                    h.wxyz = matrix_to_wxyz(T[:3, :3])
                    h.position = tuple(T[:3, 3])
            for gid in [g for g in self._handles if g not in keep]:
                self._handles.pop(gid).remove()
                self._drawn.pop(gid, None)

    def _padded(self, gid: int, mesh, pad: float):
        """The mesh pushed out along its vertex normals by ``pad`` (display; cached)."""
        key = (gid, id(mesh), round(pad, 6))
        if key not in self._pads:
            out = mesh.copy()
            out.vertices = np.asarray(mesh.vertices) + pad * np.asarray(mesh.vertex_normals)
            self._pads = {k: v for k, v in self._pads.items() if k[0] != gid}
            self._pads[key] = out
        return self._pads[key]

    def clear(self) -> None:
        for h in self._handles.values():
            h.remove()
        self._handles.clear()
        self._drawn.clear()

    def on_moved(self, *_args) -> None:
        """Robot / carrier / part moved: follow it (only when something is shown)."""
        if self.wanted():
            self._redraw()
