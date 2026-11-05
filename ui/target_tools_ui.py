"""Targets tab, extra tools: single points by click, removing a point by click, the
row/column brush, filling the selection with rows/columns, undo, and refining (further
tessellating) the part mesh.

Click modes (set in "Select surface" -> "Click on part"):
  add single point   exact ray hit on the part, normal from the surface
  remove point       deletes the target nearest the click (within the brush radius)
  brush points       one "dab" per click: row/column lattice inside the brush radius;
                     dabs share one lattice (anchored at the part origin), and points
                     that already exist are skipped, so overlapping dabs do not double up
Viser reports clicks, not drags, so the brush is applied click by click. To cover a large
area in one go, select it (click / box select) and use "Fill selection with rows/columns".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser

from core import target_tools as tt
from core import targets as tg
from ui.session import PartState
from ui.widgets import notify

if TYPE_CHECKING:
    from ui.targets_ui import TargetsTab

BRUSH = "/cell/part/brush"


class PointTools:
    def __init__(self, server: viser.ViserServer, tab: TargetsTab):
        self.server = server
        self.tab = tab
        self.s = tab.s
        self.undo_stack: list[list[int]] = []
        self.on_mesh_replaced = None  # set by the app: reload the part in the scene
        self._ring = None
        gui = server.gui
        with gui.add_folder("Brush & single points"):
            gui.add_markdown(
                "_Set **Click on part** to `add single point`, `remove point` or `brush points`. "
                "Each brush click adds a dab of rows/columns; existing points are skipped._"
            )
            self.row = gui.add_number("Row spacing (mm)", 25.0, min=0.5, step=1.0)
            self.col = gui.add_number("Column spacing (mm)", 25.0, min=0.5, step=1.0)
            self.row_dir = gui.add_vector3("Row direction (part frame)", (1.0, 0.0, 0.0), step=0.1)
            self.radius = gui.add_number("Brush radius (mm)", 60.0, min=1.0, step=5.0)
            self.max_angle = gui.add_number("Brush max surface angle (deg)", 45.0, min=1.0, max=90.0)
            self.fill = gui.add_button("Fill selection with rows/columns")
            self.undo = gui.add_button("Undo last add")
            self.info = gui.add_markdown("")
        with gui.add_folder("Refine part mesh", expand_by_default=False):
            self.max_edge = gui.add_number("Max edge length (mm)", 20.0, min=0.5, step=1.0)
            self.refine = gui.add_button("Tessellate part mesh")
            self.mesh_info = gui.add_markdown("")
            gui.add_markdown(
                "_Splits triangles (the surface shape is unchanged) so clicks, regions and normals "
                "work on a finer mesh. Saved as a new STL in the project; the selection is kept._"
            )
        self.fill.on_click(lambda e: self.fill_selection(e))
        self.undo.on_click(lambda e: self.undo_last(e))
        self.refine.on_click(lambda e: self.refine_part(e))

    # --- helpers ------------------------------------------------------------------------------

    def _dir(self):
        d = np.asarray(self.tab.direction.value, float)
        return d if np.linalg.norm(d) > 0 else None

    def _row_dir(self):
        r = np.asarray(self.row_dir.value, float)
        return r if np.linalg.norm(r) > 0 else (1.0, 0.0, 0.0)

    def _tol(self) -> float:
        return 0.4 * min(self.row.value, self.col.value) / 1000

    def add(self, P: np.ndarray, N: np.ndarray, event=None, what: str = "points") -> int:
        """Append points (deduplicated against existing ones) as one undo step."""
        _, P_old, _, _ = tg.as_arrays(self.s.targets)
        P, N = tt.dedupe(np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3), P_old, self._tol())
        if len(P) == 0:
            self.info.content = f"_No new {what} (all already present)._"
            return 0
        before = {p.id for p in self.s.targets.points}
        self.tab._add(P, N, event)
        self.undo_stack.append([p.id for p in self.s.targets.points if p.id not in before])
        self.info.content = f"Added **{len(P)}** {what}."
        return len(P)

    def undo_last(self, event=None) -> int:
        if not self.undo_stack:
            notify(event, "Nothing to undo", "No points added since the tab opened")
            return 0
        drop = set(self.undo_stack.pop())
        pts = [p for p in self.s.targets.points if p.id not in drop]
        self.s.targets = self.s.targets.model_copy(update={"points": pts})
        self.tab.refresh()
        self.info.content = f"Removed {len(drop)} points."
        return len(drop)

    # --- clicks -------------------------------------------------------------------------------

    def on_click(self, mode: str, mesh, o: np.ndarray, d: np.ndarray, event=None) -> None:
        d = d / np.linalg.norm(d)
        locs, _, tris = mesh.ray.intersects_location([o], [d], multiple_hits=False)
        if not len(locs):
            return
        p, face = locs[0], int(tris[0])
        if mode == "add single point":
            n = tg.surface_normals(mesh, locs[:1], tris[:1], None, self.tab.normals_mode.value)
            self.add(locs[:1], n, event, "point")
        elif mode == "remove point":
            ids, P, _, _ = tg.as_arrays(self.s.targets)
            k = tt.nearest_point(P, p, self.radius.value / 1000)
            if k is None:
                self.info.content = "_No point within the brush radius of the click._"
                return
            pts = [x for x in self.s.targets.points if x.id != int(ids[k])]
            self.s.targets = self.s.targets.model_copy(update={"points": pts})
            self.tab.refresh()
            self.info.content = f"Removed point {int(ids[k])}."
        elif mode == "brush points":
            self.dab(mesh, face, p, event)

    def dab(self, mesh, face: int, p: np.ndarray, event=None) -> int:
        P, N = tt.brush_dab(
            mesh,
            face,
            p,
            self.radius.value / 1000,
            self.row.value / 1000,
            self.col.value / 1000,
            self._dir(),
            self._row_dir(),
            self.max_angle.value,
            self.tab.normals_mode.value,
        )
        self._draw_ring(p, mesh.face_normals[face])
        return self.add(P, N, event, "brush points")

    def _draw_ring(self, p: np.ndarray, n: np.ndarray) -> None:
        u, v = tg._frame_from_normal(n)
        t = np.linspace(0, 2 * np.pi, 49)
        r = self.radius.value / 1000
        ring = p + 1e-3 * n + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        seg = np.stack([ring[:-1], ring[1:]], axis=1)
        if self._ring is not None:
            self._ring.remove()
        self._ring = self.server.scene.add_line_segments(
            BRUSH, seg, (240, 120, 20), thickness=2.0, thickness_units="screen"
        )

    def fill_selection(self, event=None) -> int:
        m = self.tab.mesh()
        if m is None or len(self.tab.selection) == 0:
            notify(event, "No selection", "Select a surface first")
            return 0
        try:
            P, N = tt.lattice_grid(
                m,
                self.tab.selection,
                self.row.value / 1000,
                self.col.value / 1000,
                self._dir(),
                self._row_dir(),
                normals=self.tab.normals_mode.value,
            )
        except ValueError as err:
            notify(event, "Fill failed", str(err))
            return 0
        return self.add(P, N, event, "grid points")

    # --- mesh refinement ------------------------------------------------------------------------

    def refine_part(self, event=None) -> bool:
        tab, part = self.tab, self.s.part
        m = tab.mesh()
        if m is None or part is None:
            notify(event, "No part", "Load a part in the Cell tab first")
            return False
        edge = self.max_edge.value / 1000
        try:
            new, parent = tt.refine_mesh(m, edge)
        except ValueError as err:
            notify(event, "Cannot refine", str(err))
            return False
        stem = part.mesh_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        path = self.s.project_dir / "assets" / "parts" / f"{stem}_refined_{self.max_edge.value:g}mm.stl"
        path.parent.mkdir(parents=True, exist_ok=True)
        new.export(path)
        old_sel = tab.selection
        self.s.part = PartState(self.s.relative(path), "m", part.T_world_part)
        reloaded = self.s.part_mesh()
        # carry the selection over: match faces by centroid (loading may reorder nothing, but
        # never trust indices across a file round trip)
        from scipy.spatial import cKDTree

        _, k = cKDTree(new.triangles_center).query(reloaded.triangles_center, k=1)
        tab._mesh, tab._mesh_key = reloaded, (self.s.part.mesh_path, "m")
        tab.selection = np.flatnonzero(np.isin(parent[k], old_sel))
        if self.on_mesh_replaced is not None:
            self.on_mesh_replaced()
        tab.refresh()
        self.mesh_info.content = f"{len(m.faces):,} → **{len(reloaded.faces):,}** faces (`{path.name}`)"
        notify(event, "Mesh refined", f"{len(reloaded.faces):,} faces", ok=True)
        return True
