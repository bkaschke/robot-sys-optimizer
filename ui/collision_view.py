"""Collision geometry in the left panel (2026-11-11): show / hide the shapes the collision checks
use, per component (how they are simplified: Robot & cell tab, ui.collision_settings).

The viewer always draws the full visual meshes. The collision copies (core.collision_mesh) are
drawn on top, translucent orange, from the jog's live collision scene (ui.cell_collision), so
what is shown is exactly what the planner, the paths and the jog check (the robot arm drawn
bloated by its padding).
"""

from __future__ import annotations

import threading

import numpy as np
import viser

from core.transforms import matrix_to_wxyz
from ui.collision_settings import COMPONENTS
from ui.widgets import Latest

ROOT = "/collision_geometry"
COLOR = (255, 140, 30)


class CollisionView:
    """Built inside the left panel (app.py); ``cell`` is set by app.py."""

    def __init__(self, server: viser.ViserServer, session) -> None:
        self.server, self.s = server, session
        self.cell = None
        self._handles: dict[int, object] = {}
        self._drawn: dict[int, int] = {}  # gid -> id of the mesh drawn for it
        self._pads: dict = {}  # (gid, id(mesh), pad) -> bloated copy for drawing
        self._lock = threading.Lock()
        gui = server.gui
        with gui.add_folder("Collision geometry", expand_by_default=False):
            self.show = {
                name: gui.add_checkbox(
                    f"Show {name.lower()}", False, hint=f"draw the {name.lower()}'s collision shapes (orange)"
                )
                for name in COMPONENTS
            }
        for box in self.show.values():
            box.on_update(lambda _e: self.redraw_soon())
        self._redraw = Latest(self.redraw)

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
            from core.collision_mesh import inflate

            out = inflate(mesh, pad)
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
