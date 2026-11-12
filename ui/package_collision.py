"""Collision geometry in the package builder (v9, issues #3, #4, #11): how the package's parts are
simplified for collision (``Package.collision`` / ``Part.collision``), each part's own imported
collision mesh (``Part.collision_file``), the part surfaces kept exact (part packages).

Editing the settings does not compute anything: **Simplify** makes the collision copies in the
background (progress bar, a status per part: not simplified / simplified, N → M triangles / out of
date), **Show collision shapes** draws the last ones, **Revert to original** drops the
simplification (the visual geometry is used). Export and *Use in this project* write the copies
(``meshes/collision`` and the URDF's ``<collision>``; for a part, the project's part collision mesh).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import viser

from core.collision_mesh import MODE_LABELS, MODES
from core.transforms import matrix_to_wxyz, pose_to_matrix
from core.urdf_package import Collision, part_collision, safe
from ui.widgets import MAX_UPLOAD, notify

PACKAGE = "Package setting"
LABEL_OF = {MODE_LABELS[m]: m for m in MODES}
COLOR = (255, 140, 30)


class PackageCollision:
    def __init__(self, server: viser.ViserServer, builder) -> None:
        self.server, self.b = server, builder
        self._handles: list = []
        self._busy = False
        self._lock = threading.Lock()
        self.done: dict[str, tuple] = {}  # part name -> (settings key, faces in, faces out, mesh)
        self.get_selection = None  # app.py: the Targets tab's selected faces (part packages)
        gui = server.gui
        options = tuple(MODE_LABELS[m] for m in MODES)
        with gui.add_folder("Collision geometry", expand_by_default=False):
            gui.add_markdown(
                "_The collision checks use a simpler copy of each mesh part (the URDF's `<collision>`); the "
                "3D view keeps the full mesh. **Wrap** puts one closed shell around many small pieces._"
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
            self.coll_upload = gui.add_upload_button(
                "Collision mesh for this part",
                hint="your own low-detail mesh (same units and origin as the part)",
            )
            self.coll_remove = gui.add_button("Remove its collision mesh", visible=False)
            self.keep_btn = gui.add_button(
                "Keep the project's selected surfaces exact",
                hint="part packages: the faces selected in Targets keep full detail, the rest is simplified",
            )
            self.keep_info = gui.add_markdown("")
            self.run = gui.add_button("Simplify", color="green")
            self.bar = gui.add_progress_bar(0.0, visible=False)
            self.revert = gui.add_button("Revert to original", hint="no simplification: the visual meshes")
            self.show = gui.add_checkbox(
                "Show collision shapes", False, hint="orange: what the checks will use"
            )
            self.info = gui.add_markdown("")
        for w in (self.mode, self.faces, self.wrap_mm, self.pad_mm):
            w.on_update(lambda _e: self._package_changed())
        self.part_mode.on_update(lambda _e: self._part_changed())
        self.coll_upload.on_upload(self._on_collision_mesh)
        self.coll_remove.on_click(lambda _e: self._set_part(collision_file=None))
        self.keep_btn.on_click(lambda e: self.keep_selection(e))
        self.run.on_click(lambda _e: self.simplify_soon())
        self.revert.on_click(lambda _e: self.revert_all())
        self.show.on_update(lambda _e: self.redraw())

    # --- widgets <-> package -------------------------------------------------------------------

    def load(self) -> None:
        """Widgets from the builder's package (after load_widgets)."""
        p = self.b.pkg
        c, parts = p.collision, p.parts
        self._busy = True
        try:
            self.mode.value = MODE_LABELS[c.mode]
            self.faces.value = c.faces
            self.wrap_mm.value = c.wrap_mm
            self.pad_mm.value = c.pad_mm
            x = parts[self.b.sel] if parts else None
            mesh = x is not None and x.shape == "mesh"
            self.part_mode.value = MODE_LABELS[x.collision] if x is not None and x.collision else PACKAGE
            self.part_mode.visible = self.coll_upload.visible = mesh
            self.coll_remove.visible = mesh and bool(x.collision_file)
            self.keep_btn.visible = p.kind == "part"
            n = len(c.keep)
            self.keep_info.content = f"_{n} faces kept exact._" if n and p.kind == "part" else ""
        finally:
            self._busy = False
        self.update_status()

    def _package_changed(self) -> None:
        if self._busy:
            return
        c = self.b.pkg.collision.model_dump()
        c.update(
            mode=LABEL_OF.get(self.mode.value, "auto"),
            faces=int(np.clip(self.faces.value, 50, 200000)),
            wrap_mm=float(np.clip(self.wrap_mm.value, 0.5, 200.0)),
            pad_mm=float(np.clip(self.pad_mm.value, 0.0, 100.0)),
        )
        self.b.pkg = self.b.pkg.model_copy(update={"collision": Collision.model_validate(c)})
        self.update_status()

    def _set_part(self, **upd) -> None:
        if not self.b.pkg.parts:
            return
        parts = list(self.b.pkg.parts)
        parts[self.b.sel] = parts[self.b.sel].model_copy(update=upd)
        self.b.pkg = self.b.pkg.model_copy(update={"parts": parts})
        self.load()

    def _part_changed(self) -> None:
        if self._busy or not self.b.pkg.parts:
            return
        own = None if self.part_mode.value == PACKAGE else LABEL_OF.get(self.part_mode.value)
        self._set_part(collision=own)

    def _on_collision_mesh(self, e) -> None:
        """#3: a part's own collision mesh (STL / OBJ / PLY, in the part's units and frame)."""
        import trimesh

        f = self.coll_upload.value
        ext = Path(f.name).suffix.lower()
        if len(f.content) > MAX_UPLOAD or ext not in (".stl", ".obj", ".ply"):
            return notify(e, "Not a collision mesh", "Use STL, OBJ or PLY under 200 MB")
        x = self.b.pkg.parts[self.b.sel] if self.b.pkg.parts else None
        if x is None or x.shape != "mesh":
            return notify(e, "Pick a mesh part first", "Boxes and cylinders are used as they are")
        rel = f"collision_src/{safe(x.name)}{ext}"
        path = self.b.work / "meshes" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f.content)
        try:
            if len(trimesh.load(str(path), force="mesh").faces) == 0:
                raise ValueError("no triangles")
        except Exception as err:  # a broken file: report, keep the part as it was
            path.unlink(missing_ok=True)
            return notify(e, "Collision mesh not added", f"{type(err).__name__}: {err}")
        self._set_part(collision_file=rel)

    def keep_selection(self, e=None) -> None:
        """Part packages made from the project's part: the faces selected in Targets stay exact."""
        src = getattr(self.b, "edit_source", None)
        if not src or src[0] != "part" or len(self.b.pkg.parts) != 1:
            return notify(e, "Edit the project's part first", "From the project → Edit the project's part")
        sel = self.get_selection() if self.get_selection is not None else None
        keep = sorted({int(f) for f in (sel if sel is not None else [])})
        c = self.b.pkg.collision.model_copy(update={"keep": keep})
        self.b.pkg = self.b.pkg.model_copy(update={"collision": c})
        self.load()

    def revert_all(self) -> None:
        """No simplification: the visual meshes (imported collision meshes stay)."""
        c = self.b.pkg.collision.model_copy(update={"mode": "visual", "pad_mm": 0.0, "keep": []})
        parts = [x.model_copy(update={"collision": None}) for x in self.b.pkg.parts]
        self.b.pkg = self.b.pkg.model_copy(update={"collision": c, "parts": parts})
        self.done.clear()
        self.load()
        self.redraw()

    # --- simplify ------------------------------------------------------------------------------

    def key(self, part) -> str:
        """What a part's collision copy depends on (its geometry and the settings)."""
        c = self.b.pkg.collision
        return json.dumps(
            [part.model_dump(mode="json", exclude={"name", "stage", "pose"}), c.model_dump(mode="json")],
            sort_keys=True,
        )

    def update_status(self) -> None:
        rows = []
        for part in self.b.pkg.parts:
            d = self.done.get(part.name)
            if part.collision_file and d is None:
                state = f"its own collision mesh `{part.collision_file}`"
            elif part.shape != "mesh" and self.b.pkg.collision.pad_mm <= 0:
                state = f"{part.shape}: used as it is"
            elif d is None:
                state = "not simplified"
            elif d[0] != self.key(part):
                state = "**out of date** (settings changed): Simplify again"
            else:
                state = f"ready, {d[1]} → {d[2]} triangles"
            rows.append(f"{part.name}: {state}")
        text = "  \n".join(rows) if rows else "_No parts yet._"
        if self.info.content != text:
            self.info.content = text

    def simplify_soon(self) -> None:
        threading.Thread(target=self.simplify, daemon=True).start()

    def simplify(self) -> None:
        """Make every part's collision copy (cached per mesh and settings), with progress."""
        from core.carrier import primitive_mesh

        with self._lock:
            parts = list(self.b.pkg.parts)
            self.bar.value, self.bar.visible = 0.0, True
            self.run.disabled = True
            errors = []
            try:
                for k, part in enumerate(parts):
                    try:
                        full = primitive_mesh(self.b.primitive(part))
                        low = part_collision(self.b.pkg, part, self.b.work / "meshes")
                    except Exception as err:  # a broken file: shown in the status
                        self.done.pop(part.name, None)
                        errors.append(f"✘ {part.name}: {type(err).__name__}: {err}")
                        continue
                    low = full if low is None else low
                    self.done[part.name] = (self.key(part), len(full.faces), len(low.faces), low)
                    self.bar.value = 100.0 * (k + 1) / max(len(parts), 1)
            finally:
                self.bar.visible = False
                self.run.disabled = False
            self.update_status()
            if errors:
                self.info.content += "  \n" + "  \n".join(errors)
        self.redraw()

    # --- preview -------------------------------------------------------------------------------

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def redraw(self) -> None:
        """The last simplified collision shapes, under the package frame."""
        from ui.package_preview import ROOT

        self.clear()
        if not self.show.value:
            return
        for k, part in enumerate(self.b.pkg.parts):
            d = self.done.get(part.name)
            if d is None:
                continue
            low = d[3]
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

    def redraw_soon(self) -> None:
        """The builder redrew its preview (a part moved): the collision shapes follow."""
        if self.show.value:
            self.redraw()
