"""Targets tab -> Select surface & points: one *Tool* dropdown (v4 plan item 1, after the
sand-optimizer), showing only the chosen tool's settings.

  Fill + / Fill -     click a face: the connected smooth region (neighbour normals within the
                      fill angle limit) is added / removed
  Brush + / Brush -   drag the brush handle over the part: faces within the brush radius of
                      its path are added / removed (a click paints one dab and moves it there)
  Box + / Box -       drag a rectangle in the view (the camera does not orbit meanwhile)
  By normal           faces whose normal is within an angle of a direction
  Full surface        click a body: all of its faces
  Single point, Remove point, Point brush   click tools for individual targets (PointTools)
Ctrl-click inverts + / - for the click tools. Viser reports no mouse drags on meshes, so the
brush is a gizmo kept on the surface rather than the mouse pointer itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser

from core import targets as tg
from core.surface_grid import FacePainter
from ui.widgets import GizmoStyle, Latest

if TYPE_CHECKING:
    from ui.targets_ui import TargetsTab

HANDLE = "/cell/part/brush_handle"
RING = "/cell/part/brush_ring"
TOOLS = (
    "Fill +",
    "Fill −",
    "Brush +",
    "Brush −",
    "Box +",
    "Box −",
    "By normal",
    "Full surface",
    "Single point",
    "Remove point",
    "Point brush",
    "Off",
)
POINT_TOOLS = ("Single point", "Remove point", "Point brush")
HINTS = {
    "Fill +": "_Click a face: its smooth region is added (ctrl-click removes)._",
    "Fill −": "_Click a face: its smooth region is removed._",
    "Brush +": "_Drag the orange-ringed handle over the part to paint; a click paints one dab._",
    "Brush −": "_Drag the handle over the part to erase; a click erases one dab._",
    "Box +": "_Drag a rectangle in the 3D view to add the faces in it (pick another tool to orbit)._",
    "Box −": "_Drag a rectangle in the 3D view to remove the faces in it._",
    "By normal": "_Faces whose normal is within the angle of the direction (part frame)._",
    "Full surface": "_Click a body of the part: all of its faces are added (ctrl-click removes)._",
    "Single point": "_Click: one target at the hit point, normal from the surface._",
    "Remove point": "_Click: removes the target nearest the click (within the brush radius)._",
    "Point brush": "_Click: a dab of rows/columns (Generate points spacing) inside the radius._",
    "Off": "_Clicks on the part do nothing._",
}


class SurfaceTools:
    def __init__(self, server: viser.ViserServer, tab: TargetsTab):
        self.server = server
        self.tab = tab
        self.handle = None
        self._ring = None
        self._rect_on = False
        self._painter: FacePainter | None = None
        self._last: np.ndarray | None = None  # last surface point of the brush
        self._last_n: np.ndarray | None = None
        self._move = Latest(self.brush_to)
        gui = server.gui
        self.tool = gui.add_dropdown("Tool", TOOLS, initial_value="Fill +")
        self.hint = gui.add_markdown(HINTS["Fill +"])
        self.angle = gui.add_slider("Fill angle limit (deg)", 1.0, 45.0, 1.0, 15.0)
        self.radius = gui.add_number("Brush radius (mm)", 60.0, min=1.0, step=5.0)
        self.max_angle = gui.add_number(
            "Brush max surface angle (deg)",
            45.0,
            min=1.0,
            max=180.0,
            hint="faces turned further from the surface under the brush are left alone",
        )
        self.normal_dir = gui.add_vector3("Normal direction", (0.0, 0.0, 1.0), step=0.1)
        self.normal_angle = gui.add_number("within (deg)", 30.0, min=0.0, max=180.0)
        self.by_normal = gui.add_button("Select by normal")
        self.buttons = gui.add_button_group("Selection", ("Select all", "Clear", "Invert"))
        self.sel_info = gui.add_markdown("")
        self.tool.on_update(lambda e: self.on_tool())
        self.by_normal.on_click(lambda e: self.select_by_normal())
        self.buttons.on_click(lambda e: self.bulk(self.buttons.value))
        self.radius.on_update(lambda e: self._draw_ring())
        self.on_tool()

    # --- helpers --------------------------------------------------------------------------------

    def painter(self) -> FacePainter | None:
        m = self.tab.mesh()
        if m is None:
            return None
        if self._painter is None or self._painter.mesh is not m:
            self._painter = FacePainter(m)
        return self._painter

    def _sign(self, modifier=None) -> str:
        mod = (modifier or "").lower()
        add = not self.tool.value.endswith("−")
        if "ctrl" in mod or "control" in mod or "meta" in mod:
            add = not add
        return "add" if add else "remove"

    def bulk(self, what: str) -> None:
        m = self.tab.mesh()
        if m is None:
            return
        if what == "Select all":
            self.tab.select(np.arange(len(m.faces)), "set")
        elif what == "Clear":
            self.tab.select(np.zeros(0, int), "set")
        elif what == "Invert":
            self.tab.select(np.setdiff1d(np.arange(len(m.faces)), self.tab.selection), "set")

    def select_by_normal(self) -> None:
        m = self.tab.mesh()
        if m is not None and np.linalg.norm(self.normal_dir.value) > 0:
            faces = tg.select_by_normal(m, self.normal_dir.value, self.normal_angle.value)
            self.tab.select(faces, "set")

    # --- tool switching ---------------------------------------------------------------------------

    def on_tool(self) -> None:
        t = self.tool.value
        brush = t in ("Brush +", "Brush −")
        self.hint.content = HINTS.get(t, "")
        self.angle.visible = t in ("Fill +", "Fill −")
        self.radius.visible = brush or t in ("Remove point", "Point brush")
        self.max_angle.visible = brush or t == "Point brush"
        for w in (self.normal_dir, self.normal_angle, self.by_normal):
            w.visible = t == "By normal"
        box = t in ("Box +", "Box −")
        if box and not self._rect_on:
            self._rect_cb = self.server.scene.on_rect_select()(self.tab.on_rect)  # a decorator factory
            self._rect_on = True
        elif not box and self._rect_on:
            self.server.scene.remove_rect_select_callback(self._rect_cb)
            self._rect_on = False
        if brush:
            self._make_handle()
        else:
            self.remove_handle()

    def remove_handle(self) -> None:
        for h in (self.handle, self._ring):
            if h is not None:
                h.remove()
        self.handle = self._ring = None

    def _make_handle(self) -> None:
        if self.handle is not None:
            return
        pt = self.painter()
        if pt is None:
            return
        m, sel = pt.mesh, self.tab.selection
        start = m.triangles_center[sel].mean(0) if len(sel) else m.bounds.mean(0) + [0, 0, m.extents[2]]
        p, n, _ = pt.project(start)
        self._last, self._last_n = p, n
        h = self.server.scene.add_transform_controls(
            HANDLE, scale=0.15, disable_rotations=True, position=tuple(p)
        )
        GizmoStyle.register(h, 0.15)
        h.on_update(lambda e: self._move(np.asarray(h.position, float)))
        h.on_drag_end(lambda e: self.snap())
        self.handle = h
        self._draw_ring()

    # --- painting -----------------------------------------------------------------------------------

    def brush_to(self, pos: np.ndarray, sign: str | None = None) -> int:
        """Move the brush to ``pos`` (part frame): paints the path from the last point."""
        pt = self.painter()
        if pt is None:
            return 0
        r = self.radius.value / 1000
        start = self._last if self._last is not None else pos
        steps = max(1, int(np.ceil(np.linalg.norm(pos - start) / (0.4 * r))))
        pts, nrm = [], []
        for k in range(1, steps + 1):
            p, n, _ = pt.project(start + (pos - start) * k / steps)
            pts.append(p)
            nrm.append(n)
        faces = pt.stroke(pts, nrm, r, self.max_angle.value)
        self._last, self._last_n = pts[-1], nrm[-1]
        self._draw_ring()
        if len(faces):
            self.tab.select(faces, sign or self._sign())
        return len(faces)

    def dab(self, p: np.ndarray, sign: str | None = None) -> int:
        """Click with the brush: paint at ``p`` and move the handle there."""
        self._last = None
        n = self.brush_to(np.asarray(p, float), sign)
        self.snap()
        return n

    def snap(self) -> None:
        """Put the handle back on the surface (after a drag ends)."""
        if self.handle is not None and self._last is not None:
            self.handle.position = tuple(self._last)

    def _draw_ring(self) -> None:
        if self.handle is None or self._last is None:
            return
        u, v = tg._frame_from_normal(self._last_n)
        t = np.linspace(0, 2 * np.pi, 49)
        r = self.radius.value / 1000
        ring = self._last + 1e-3 * self._last_n + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        self._ring = self.server.scene.add_line_segments(
            RING,
            np.stack([ring[:-1], ring[1:]], axis=1),
            (240, 120, 20),
            thickness=2.5,
            thickness_units="screen",
        )

    # --- clicks -------------------------------------------------------------------------------------

    def on_click(self, mesh, face: int, o: np.ndarray, d: np.ndarray, event=None) -> None:
        t, mod = self.tool.value, getattr(event, "modifier", None)
        if t in ("Fill +", "Fill −"):
            self.tab.select(tg.grow_region(mesh, face, self.angle.value), self._sign(mod))
        elif t == "Full surface":
            self.tab.select(self.painter().component(face), self._sign(mod))
        elif t in ("Brush +", "Brush −"):
            locs, _, _ = mesh.ray.intersects_location([o], [d / np.linalg.norm(d)], multiple_hits=False)
            if len(locs):
                self.dab(locs[0], self._sign(mod))
        elif t in POINT_TOOLS:
            self.tab.tools.on_click(t, mesh, o, d, event)
