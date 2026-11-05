"""Targets tab: surface selection on the part, point generation, groups/tolerances,
point editing, CSV import, normals display and a tool preview at any point.

Points and the selection live in the part frame and are drawn under /cell/part, so they
follow the part gizmo. Positions/normals in m; the GUI shows mm and degrees.
"""

from __future__ import annotations

import numpy as np
import trimesh
import viser
from pydantic import ValidationError

from core import targets as tg
from core.schema import TargetGroup, Tolerances
from core.transforms import make_T, matrix_to_wxyz, wxyz_to_matrix
from ui.session import Session
from ui.target_tools_ui import PointTools
from ui.widgets import MAX_UPLOAD, notify

PART = "/cell/part"
PREVIEW = "/cell/preview"
SEL_COLOR = (60, 200, 120)
PALETTE = [(230, 60, 60), (40, 120, 230), (240, 170, 20), (150, 70, 200), (20, 170, 170), (120, 120, 120)]
STATUS_COLORS = {"reachable": (40, 180, 70), "marginal": (240, 170, 20), "unreachable": (220, 40, 40)}
UNCHECKED = (150, 150, 150)
METHODS = ("Projected grid", "Poisson (even)", "Raster lines", "Selection edges")
CLICK_MODES = ("off", "select surface", "add single point", "remove point", "brush points")


class TargetsTab:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server = server
        self.s = session
        self.selection = np.zeros(0, dtype=int)
        self._mesh: trimesh.Trimesh | None = None
        self._mesh_key = None
        self._handles: dict[str, object] = {}
        self._busy = False
        self.selected_results = None  # set by the app: point results of the selected solution
        gui = server.gui

        with gui.add_folder("Select surface"):
            self.click_mode = gui.add_dropdown("Click on part", CLICK_MODES, initial_value="select surface")
            self.angle = gui.add_slider("Region angle (deg)", 1.0, 45.0, 1.0, 15.0)
            gui.add_markdown(
                "_Select surface: click = new selection, shift-click = add, ctrl-click = remove. "
                "Point modes: see **Brush & single points**._"
            )
            self.normal_dir = gui.add_vector3("Normal direction", (0.0, 0.0, 1.0), step=0.1)
            self.normal_angle = gui.add_number("within (deg)", 30.0, min=0.0, max=180.0)
            self.by_normal = gui.add_button("Select by normal")
            self.box = gui.add_button("Box select (drag in view)")
            self.clear = gui.add_button("Clear selection")
            self.sel_info = gui.add_markdown("")
        with gui.add_folder("Generate points"):
            self.method = gui.add_dropdown("Method", METHODS)
            self.spacing = gui.add_number("Spacing / step (mm)", 10.0, min=0.5, step=1.0)
            self.stepover = gui.add_number("Raster step-over (mm)", 25.0, min=0.5, step=1.0)
            self.direction = gui.add_vector3(
                "Grid ray / raster line dir",
                (0.0, 0.0, 0.0),
                step=0.1,
                hint="(0,0,0) = automatic from the selection",
            )
            self.normals_mode = gui.add_dropdown("Normals", ("interpolated", "face"))
            self.gen_group = gui.add_dropdown("Into group", ("default",))
            self.generate = gui.add_button("Generate on selection")
        self.tools = PointTools(server, self)
        with gui.add_folder("Groups & tolerances"):
            self.group = gui.add_dropdown("Group", ("default",))
            self.new_group = gui.add_text("New group", "")
            self.add_group = gui.add_button("Add group")
            self.spin = gui.add_text("Spin (deg)", "free", hint="'free', a half-width, or 'lo,hi'")
            self.tilt = gui.add_number("Tilt cone (deg)", 0.0, min=0.0, max=90.0, step=0.5)
            self.standoff = gui.add_vector2("Standoff min/max (mm)", (0.0, 0.0), step=1.0)
            self.tcps = gui.add_text("TCPs (comma, empty = any)", "")
            self.apply_tol = gui.add_button("Apply to group")
        with gui.add_folder("Edit points"):
            self.count = gui.add_markdown("")
            self.assign = gui.add_button("Assign selected-surface points to group")
            self.flip = gui.add_button("Flip normals on selected surface")
            self.delete = gui.add_button("Delete points on selected surface")
            self.delete_all = gui.add_button("Delete all points")
            self.csv_units = gui.add_dropdown("CSV units", ("mm", "m"))
            self.csv = gui.add_upload_button("Import CSV (x,y,z,nx,ny,nz[,group])", mime_type="text/csv")
        with gui.add_folder("Display"):
            self.color_by = gui.add_dropdown(
                "Color points by", ("group", "feasibility", "zone", "configuration")
            )
            self.show_normals = gui.add_checkbox("Show normals", True)
            self.normal_len = gui.add_number("Normal length (mm)", 20.0, min=1.0, step=1.0)
            self.point_size = gui.add_number("Point size (mm)", 4.0, min=0.5, step=0.5)
            self.preview_on = gui.add_checkbox("Preview tool at point", False)
            self.preview_idx = gui.add_slider("Point #", 0, 1, 1, 0)
        self._wire()
        self.refresh()

    # --- mesh / selection helpers -------------------------------------------------------------

    def mesh(self) -> trimesh.Trimesh | None:
        part = self.s.part
        key = (part.mesh_path, part.units) if part else None
        if key != self._mesh_key:
            self._mesh = self.s.part_mesh() if part else None
            self._mesh_key = key
            self.selection = np.zeros(0, dtype=int)
        return self._mesh

    def on_part_changed(self) -> None:
        self.mesh()  # drops the selection if the mesh changed
        self.refresh()

    def _to_part(self, event) -> tuple[np.ndarray, np.ndarray]:
        T = np.linalg.inv(self.s.part.T_world_part)
        o = T[:3, :3] @ np.asarray(event.ray_origin) + T[:3, 3]
        d = T[:3, :3] @ np.asarray(event.ray_direction)
        return o, d

    def _set_selection(self, faces: np.ndarray, modifier) -> None:
        faces = np.asarray(faces, dtype=int)
        mod = (modifier or "").lower()
        if "shift" in mod:
            faces = np.union1d(self.selection, faces)
        elif "ctrl" in mod or "control" in mod or "meta" in mod:
            faces = np.setdiff1d(self.selection, faces)
        self.selection = np.unique(faces)
        self._draw_selection()

    def on_part_click(self, event) -> None:
        mode, m = self.click_mode.value, self.mesh()
        if mode == "off" or m is None:
            return
        o, d = self._to_part(event)
        face = tg.pick_face(m, o, d)
        if face is None:
            return
        if mode == "select surface":
            self._set_selection(tg.grow_region(m, face, self.angle.value), getattr(event, "modifier", None))
            return
        self.tools.on_click(mode, m, o, d, event)

    def on_rect(self, event) -> None:
        m = self.mesh()
        if m is None:
            return
        cam = event.client.camera
        T_world_cam = make_T(wxyz_to_matrix(cam.wxyz), cam.position)
        T_cam_part = np.linalg.inv(T_world_cam) @ self.s.part.T_world_part
        faces = tg.select_in_screen_rect(
            m, T_cam_part, cam.fov, cam.aspect, event.screen_min, event.screen_max
        )
        self._set_selection(faces, event.modifier)
        self.server.scene.remove_rect_select_callback(self.on_rect)

    # --- target-set edits ---------------------------------------------------------------------

    def _ensure_group(self, name: str) -> None:
        if name in {g.name for g in self.s.targets.groups}:
            return
        tol = self.s.tool.tolerances if self.s.tool else Tolerances()
        g = TargetGroup(name=name, spin_deg=tol.spin_deg, tilt_deg=tol.tilt_deg, standoff_m=tol.standoff_m)
        self.s.targets = self.s.targets.model_copy(update={"groups": [*self.s.targets.groups, g]})

    def _add(self, P: np.ndarray, N: np.ndarray, event=None) -> None:
        if len(P) == 0:
            return notify(event, "No points", "Nothing generated: check the selection and spacing")
        group = self.gen_group.value
        self._ensure_group(group)
        self.s.targets = tg.append_points(self.s.targets, P, N, group)
        self.refresh()

    def _edit_on_selection(self, fn, event) -> None:
        m = self.mesh()
        if m is None or len(self.selection) == 0:
            return notify(event, "No selection", "Select a surface first")
        ids, P, N, G = tg.as_arrays(self.s.targets)
        mask = tg.points_on_faces(m, P, self.selection)
        ids, P, N, G = fn(mask, ids, P, N, G)
        self.s.targets = tg.with_points(self.s.targets, ids, P, N, G)
        self.refresh()

    # --- callbacks ------------------------------------------------------------------------------

    def _wire(self) -> None:
        @self.by_normal.on_click
        def _(e) -> None:
            m = self.mesh()
            if m is not None and np.linalg.norm(self.normal_dir.value) > 0:
                self._set_selection(
                    tg.select_by_normal(m, self.normal_dir.value, self.normal_angle.value), None
                )

        @self.box.on_click
        def _(e) -> None:
            self.server.scene.on_rect_select(self.on_rect)
            notify(e, "Box select", "Drag a rectangle in the 3D view (shift = add)", ok=True)

        @self.clear.on_click
        def _(e) -> None:
            self.selection = np.zeros(0, dtype=int)
            self._draw_selection()

        self.generate.on_click(lambda e: self.run_generate(e))

        @self.add_group.on_click
        def _(e) -> None:
            name = self.new_group.value.strip()
            if not name:
                return
            self._ensure_group(name)
            self.group_value = name
            self.refresh()

        @self.group.on_update
        def _(e) -> None:
            if not self._busy:
                self.group_value = self.group.value
                self._show_group()

        @self.apply_tol.on_click
        def _(e) -> None:
            name = self.group.value
            try:
                spin = self.spin.value.strip()
                if spin != "free":
                    parts = [float(x) for x in spin.split(",")]
                    spin = parts[0] if len(parts) == 1 else tuple(parts)
                lo, hi = (v / 1000 for v in self.standoff.value)
                tcps = [t.strip() for t in self.tcps.value.split(",") if t.strip()]
                g = TargetGroup(
                    name=name,
                    spin_deg=spin,
                    tilt_deg=self.tilt.value,
                    standoff_m=lo if lo == hi else (lo, hi),
                    tcps=tcps,
                )
                groups = [g if x.name == name else x for x in self.s.targets.groups]
                self.s.targets = self.s.targets.model_copy(update={"groups": groups})
                self.s.to_project()  # validates TCP names against the tool
            except (ValueError, ValidationError) as err:
                return notify(e, "Invalid tolerances", str(err)[:300])
            notify(e, "Saved", f"tolerances for '{name}'", ok=True)

        @self.assign.on_click
        def _(e) -> None:
            grp = self.group.value
            self._ensure_group(grp)
            self._edit_on_selection(
                lambda m, i, P, N, G: (i, P, N, [grp if k else g for k, g in zip(m, G, strict=True)]), e
            )

        @self.flip.on_click
        def _(e) -> None:
            self._edit_on_selection(lambda m, i, P, N, G: (i, P, np.where(m[:, None], -N, N), G), e)

        @self.delete.on_click
        def _(e) -> None:
            self._edit_on_selection(
                lambda m, i, P, N, G: (i[~m], P[~m], N[~m], [g for k, g in zip(m, G, strict=True) if not k]),
                e,
            )

        @self.delete_all.on_click
        def _(e) -> None:
            self.s.targets = self.s.targets.model_copy(update={"points": []})
            self.refresh()

        @self.csv.on_upload
        def _(e) -> None:
            f = self.csv.value
            if len(f.content) > MAX_UPLOAD:
                return notify(e, "Too large", "CSV over 200 MB")
            try:
                rows = tg.parse_csv(
                    f.content.decode("utf-8-sig"), 1e-3 if self.csv_units.value == "mm" else 1.0
                )
            except (ValueError, UnicodeDecodeError) as err:
                return notify(e, "CSV import failed", str(err))
            groups = [r[2] or self.gen_group.value for r in rows]
            for g in set(groups):
                self._ensure_group(g)
            self.s.targets = tg.append_points(
                self.s.targets, np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), groups
            )
            self.refresh()
            notify(e, "Imported", f"{len(rows)} points", ok=True)

        for w in (self.color_by, self.show_normals, self.normal_len, self.point_size):
            w.on_update(lambda e: self._draw_points())
        for w in (self.preview_on, self.preview_idx):
            w.on_update(lambda e: self._draw_preview())

    def run_generate(self, e=None) -> None:
        m = self.mesh()
        if m is None or len(self.selection) == 0:
            return notify(e, "No selection", "Select a surface on the part first")
        step = self.spacing.value / 1000
        d = np.asarray(self.direction.value, float)
        d = d if np.linalg.norm(d) > 0 else None
        method = self.method.value
        try:
            if method == "Projected grid":
                P, N = tg.projected_grid(
                    m, self.selection, step, direction=d, normals=self.normals_mode.value
                )
            elif method == "Poisson (even)":
                P, N = tg.poisson_disk(m, self.selection, step, normals=self.normals_mode.value)
            elif method == "Raster lines":
                P, N = tg.raster(
                    m, self.selection, self.stepover.value / 1000, step, d, self.normals_mode.value
                )
            else:
                P, N = tg.edge_paths(m, self.selection, step)
        except (ValueError, IndexError) as err:
            return notify(e, "Generation failed", str(err))
        self._add(P, N, e)
        notify(e, "Generated", f"{len(P)} points into '{self.gen_group.value}'", ok=True)

    # --- drawing --------------------------------------------------------------------------------

    def _drop(self, key: str) -> None:
        h = self._handles.pop(key, None)
        if h is not None:
            h.remove()

    def refresh(self) -> None:
        self._busy = True
        try:
            names = [g.name for g in self.s.targets.groups] or ["default"]
            cur = getattr(self, "group_value", None)
            for dd in (self.group, self.gen_group):
                dd.options = tuple(names)
            self.group.value = cur if cur in names else names[0]
            self.gen_group.value = self.group.value
            self._show_group()
        finally:
            self._busy = False
        self._draw_selection()
        self._draw_points()
        self._draw_preview()

    def _show_group(self) -> None:
        g = next((x for x in self.s.targets.groups if x.name == self.group.value), None)
        tol = g or (self.s.tool.tolerances if self.s.tool else Tolerances())
        sp = tol.spin_deg
        self.spin.value = (
            sp if sp == "free" else (",".join(map(str, sp)) if isinstance(sp, tuple) else str(sp))
        )
        self.tilt.value = tol.tilt_deg
        so = tol.standoff_m if isinstance(tol.standoff_m, tuple) else (tol.standoff_m, tol.standoff_m)
        self.standoff.value = tuple(v * 1000 for v in so)
        self.tcps.value = ", ".join(g.tcps) if g else ""

    def _draw_selection(self) -> None:
        self._drop("sel")
        m = self.mesh()
        n = len(self.selection)
        if m is not None and n:
            sub = m.submesh([self.selection], append=True)
            verts = sub.vertices + 5e-4 * sub.vertex_normals  # draw just above the part
            self._handles["sel"] = self.server.scene.add_mesh_simple(
                f"{PART}/selection", verts, sub.faces, color=SEL_COLOR, opacity=0.6
            )
        area = m.area_faces[self.selection].sum() * 1e4 if m is not None and n else 0.0
        self.sel_info.content = f"Selected: {n} faces, {area:.1f} cm²" if n else "_Nothing selected._"

    def _draw_points(self) -> None:
        self._drop("points")
        self._drop("normals")
        ids, P, N, G = tg.as_arrays(self.s.targets)
        names = [g.name for g in self.s.targets.groups]
        counts = {g: G.count(g) for g in names}
        self.count.content = f"**{len(P)} points**  \n" + ", ".join(f"{k}: {v}" for k, v in counts.items())
        self.preview_idx.max = max(1, len(P) - 1)
        if len(P) == 0:
            return
        sel = self.selected_results() if self.selected_results else {}
        if self.color_by.value == "feasibility" and self.s.feasibility:
            res = self.s.feasibility
            colors = np.array([STATUS_COLORS[res[i].status] if i in res else UNCHECKED for i in ids])
        elif self.color_by.value in ("zone", "configuration") and sel:
            # selected solution (Results tab): zone index, or one colour per joint configuration
            labels = sorted({r.config for r in sel.values() if r.config})
            key = (lambda r: r.zone) if self.color_by.value == "zone" else (lambda r: labels.index(r.config))
            colors = np.array(
                [
                    PALETTE[key(sel[i]) % len(PALETTE)]
                    if i in sel and sel[i].config and sel[i].zone is not None
                    else UNCHECKED
                    for i in ids
                ]
            )
        else:
            colors = np.array(
                [PALETTE[names.index(g) % len(PALETTE)] if g in names else PALETTE[-1] for g in G]
            )
        self._handles["points"] = self.server.scene.add_point_cloud(
            f"{PART}/targets",
            P,
            colors.astype(np.uint8),
            point_size=self.point_size.value / 1000,
            point_shape="circle",
            precision="float32",
        )
        if self.show_normals.value:
            seg = np.stack([P, P + N * self.normal_len.value / 1000], axis=1)
            self._handles["normals"] = self.server.scene.add_line_segments(
                f"{PART}/normals", seg, (30, 30, 30), thickness=1.5, thickness_units="screen"
            )

    def _draw_preview(self) -> None:
        for k in [k for k in self._handles if k.startswith("pv")]:
            self._drop(k)
        ids, P, N, G = tg.as_arrays(self.s.targets)
        tm = self.s.tool_model
        if not self.preview_on.value or tm is None or len(P) == 0 or self.s.part is None:
            return
        i = int(np.clip(self.preview_idx.value, 0, len(P) - 1))
        T_world_tcp = self.s.part.T_world_part @ target_tcp_pose(P[i], N[i], self._standoff(G[i]))
        tcp = tm.ee.tcps[0].name
        group = next((g for g in self.s.targets.groups if g.name == G[i]), None)
        if group and group.tcps:
            tcp = group.tcps[0]
        T_world_flange = T_world_tcp @ np.linalg.inv(tm.T_flange_tcp(tcp))
        root = self.server.scene.add_frame(
            PREVIEW,
            show_axes=False,
            wxyz=matrix_to_wxyz(T_world_flange[:3, :3]),
            position=T_world_flange[:3, 3],
        )
        self._handles["pv_root"] = root
        ext = tm.extension_mesh()
        if ext is not None:
            self._handles["pv_ext"] = self.server.scene.add_mesh_simple(
                f"{PREVIEW}/ext", ext.vertices, ext.faces, color=(90, 140, 220), opacity=0.5
            )
        T_fh = tm.T_flange_head()
        for hl in tm.links:
            T = T_fh @ hl.T_head_link
            self._handles[f"pv_{hl.name}"] = self.server.scene.add_mesh_simple(
                f"{PREVIEW}/{hl.name}",
                hl.mesh.vertices,
                hl.mesh.faces,
                color=(200, 200, 210),
                opacity=0.5,
                wxyz=matrix_to_wxyz(T[:3, :3]),
                position=T[:3, 3],
            )
        self._handles["pv_tcp"] = self.server.scene.add_frame(
            f"{PREVIEW}/tcp",
            axes_length=0.05,
            axes_radius=0.002,
            wxyz=matrix_to_wxyz(tm.T_flange_tcp(tcp)[:3, :3]),
            position=tm.T_flange_tcp(tcp)[:3, 3],
        )

    def _standoff(self, group: str) -> float:
        g = next((x for x in self.s.targets.groups if x.name == group), None)
        if g is None:
            return 0.0
        return float(np.mean(g.standoff_m)) if isinstance(g.standoff_m, tuple) else float(g.standoff_m)


def target_tcp_pose(p: np.ndarray, n: np.ndarray, standoff: float = 0.0) -> np.ndarray:
    """TCP pose (part frame) that reaches target (p, n): origin at p + standoff*n and
    z = -n (tool points into the surface). Spin about the normal is chosen arbitrarily."""
    z = -np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), np.asarray(p) + standoff * np.asarray(n))
