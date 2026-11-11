"""Package builder tab: author a URDF package for an end effector, a rail, a gantry or a mobile
base (core.urdf_package), preview it, export it as a zip and use it in the project.

The package being edited lives in ``<project>/assets/packages/_builder`` (uploaded meshes are
copied there). *Use in this project* writes ``<project>/assets/packages/<name>/`` and loads it
as the tool (Tool tab) or the external axis (External axis tab). *Open package (.zip)* reads a
package made here or by hand / from CAD (see the user guide, "Building URDF packages").
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import viser

from core.carrier import primitive_mesh
from core.schema import Primitive
from core.transforms import matrix_to_pose, pose_to_matrix
from core.urdf_package import (
    STAGES,
    Frame,
    Package,
    Part,
    check,
    open_package,
    safe,
    to_base,
    to_tool,
    write_package,
    zip_folder,
)
from ui.package_preview import ROOT, draw
from ui.widgets import MAX_UPLOAD, PoseWidgets, notify

KIND_LABELS = {"End effector (tool)": "tool", "Rail": "rail", "Gantry": "gantry", "Mobile base": "mobile"}
KIND_OF = {v: k for k, v in KIND_LABELS.items()}
STAGE_LABELS = {
    "tool": "tool (on the flange mount)",
    "fixed": "fixed (track / frame)",
    "bridge": "bridge (moves in X)",
    "carriage": "carriage (moves)",
    "chassis": "chassis (the mobile base)",
    "lift": "lift stage (rises with the robot)",
}
STAGE_OF = {v: k for k, v in STAGE_LABELS.items()}
NONE = "(none)"
UNITS = ("mm", "m", "in")


def _mm(v) -> tuple:
    return tuple(float(x) * 1000 for x in v)


class PackageBuilder:
    def __init__(self, server: viser.ViserServer, session, tool_tab=None, carrier_panel=None):
        self.server, self.s = server, session
        self.tool_tab, self.carrier = tool_tab, carrier_panel
        self.pkg = Package(kind="tool", name="my_tool", tcps=[Frame(name="tcp")])
        self.sel = 0  # selected part
        self.sel_tcp = 0
        self._busy = False
        self._handles: list = []
        gui = server.gui
        gui.add_markdown(
            "_Build a URDF package for a **tool**, a **rail / gantry** or a **mobile base** from boxes, "
            "cylinders and meshes (STL / OBJ / PLY / STEP), then export it or use it here. Poses are in the "
            "package frame: the robot flange (tool) or the carrier frame on the floor. See the user guide, "
            "*Building URDF packages*._"
        )
        self.kind = gui.add_dropdown("Package type", tuple(KIND_LABELS), initial_value=KIND_OF["tool"])
        self.name = gui.add_text("Name", self.pkg.name)
        with gui.add_folder("Parts"):
            self.part = gui.add_dropdown("Part", (NONE,))
            self.add_box = gui.add_button("Add box")
            self.add_cyl = gui.add_button("Add cylinder")
            self.units = gui.add_dropdown("Mesh units", UNITS, hint="STL / OBJ / PLY (STEP carries its own)")
            self.upload = gui.add_upload_button("Add mesh (STL/OBJ/PLY/STEP)")
            self.part_name = gui.add_text("Part name", "")
            self.stage = gui.add_dropdown("Stage", (STAGE_LABELS["tool"],))
            self.size = gui.add_vector3("Box size (mm)", (200.0, 200.0, 200.0), min=(1, 1, 1), step=5.0)
            self.cyl = gui.add_vector2("Cylinder r / h (mm)", (100.0, 200.0), min=(1, 1), step=5.0)
            self.part_pose = PoseWidgets(server, f"{ROOT}/edit_part", np.eye(4), self._part_moved, scale=0.2)
            self.remove = gui.add_button("Remove part", color="red")
        with gui.add_folder("Frames and travel"):
            self.tcp = gui.add_dropdown("TCP", ("tcp",))
            self.add_tcp = gui.add_button("Add TCP")
            self.tcp_name = gui.add_text("TCP name", "tcp")
            self.tcp_pose = PoseWidgets(server, f"{ROOT}/edit_tcp", np.eye(4), self._tcp_moved, scale=0.12)
            self.del_tcp = gui.add_button("Remove TCP")
            self.mount_md = gui.add_markdown("**Robot mount** (where the robot's base sits)")
            self.mount_pose = PoseWidgets(
                server, f"{ROOT}/edit_mount", np.eye(4), self._mount_moved, scale=0.25
            )
            self.tx = gui.add_vector2("X travel (mm)", (-1500.0, 1500.0), step=10.0)
            self.ty = gui.add_vector2("Y travel (mm)", (-1000.0, 1000.0), step=10.0)
            self.tlift = gui.add_vector2("Lift travel (mm)", (0.0, 0.0), step=10.0, hint="0, 0 = no lift")
        with gui.add_folder("Preview"):
            self.show = gui.add_checkbox("Show preview", False)
            self.at = gui.add_vector3(
                "Preview at (mm)", (0.0, -3000.0, 0.0), step=50.0, hint="clear of the cell"
            )
        self.check_btn = gui.add_button("Check")
        self.export = gui.add_button("Export package (.zip)", color="green")
        self.use = gui.add_button("Use in this project", color="green")
        self.open = gui.add_upload_button("Open package (.zip)", mime_type="application/zip")
        self.new = gui.add_button("New package")
        self.status = gui.add_markdown("")
        self.root = server.scene.add_frame(
            ROOT, show_axes=True, axes_length=0.3, axes_radius=0.006, visible=False
        )
        self._wire()
        self.load_widgets()

    # --- paths ---------------------------------------------------------------------------------

    @property
    def work(self) -> Path:
        return self.s.project_dir / "assets" / "packages" / "_builder"

    def _resolve(self, rel: str) -> Path:
        return self.work / "meshes" / rel

    # --- widgets <-> package -------------------------------------------------------------------

    def _wire(self) -> None:
        def guard(fn):
            def h(e) -> None:
                if not self._busy:
                    fn(e)

            return h

        self.kind.on_update(guard(lambda e: self._set_kind(KIND_LABELS[self.kind.value])))
        self.name.on_update(guard(lambda e: self._update(name=self.name.value.strip() or "package")))
        self.part.on_update(guard(lambda e: self._select(self.part.value)))
        self.add_box.on_click(
            lambda e: self._add(
                Part(name="box", stage=self._default_stage(), shape="box", size=(0.2, 0.2, 0.2))
            )
        )
        self.add_cyl.on_click(
            lambda e: self._add(
                Part(name="cylinder", stage=self._default_stage(), shape="cylinder", radius=0.1, height=0.2)
            )
        )
        self.upload.on_upload(self._on_mesh)
        for w in (self.part_name, self.stage, self.size, self.cyl):
            w.on_update(guard(lambda e: self._part_fields()))
        self.remove.on_click(lambda e: self._remove())
        self.tcp.on_update(guard(lambda e: self._select_tcp(self.tcp.value)))
        self.add_tcp.on_click(lambda e: self._new_tcp())
        self.tcp_name.on_update(guard(lambda e: self._tcp_fields()))
        self.del_tcp.on_click(lambda e: self._remove_tcp(e))
        for w in (self.tx, self.ty, self.tlift):
            w.on_update(guard(lambda e: self._travel()))
        self.show.on_update(lambda e: self.draw())
        self.at.on_update(lambda e: self.draw())
        self.check_btn.on_click(self._check)
        self.export.on_click(self._export)
        self.use.on_click(self._use)
        self.open.on_upload(self._open)
        self.new.on_click(lambda e: self._reset())

    def _default_stage(self) -> str:
        k = self.pkg.kind
        return {"tool": "tool", "mobile": "chassis"}.get(k, "carriage")

    def _update(self, **kw) -> None:
        self.pkg = self.pkg.model_copy(update=kw)
        self.draw()

    def load_widgets(self) -> None:
        """All fields from ``self.pkg``."""
        p = self.pkg
        self._busy = True
        try:
            self.kind.value = KIND_OF[p.kind]
            self.name.value = p.name
            names = [x.name for x in p.parts]
            self.part.options = tuple(names) or (NONE,)
            self.sel = min(self.sel, max(len(p.parts) - 1, 0))
            self.part.value = names[self.sel] if names else NONE
            self.stage.options = tuple(STAGE_LABELS[s] for s in STAGES[p.kind])
            if p.parts:
                x = p.parts[self.sel]
                self.part_name.value = x.name
                self.stage.value = (
                    STAGE_LABELS[x.stage] if x.stage in STAGES[p.kind] else self.stage.options[0]
                )
                if x.size:
                    self.size.value = _mm(x.size)
                if x.radius and x.height:
                    self.cyl.value = (x.radius * 1000, x.height * 1000)
                self.part_pose.set(pose_to_matrix(x.pose))
            tool = p.kind == "tool"
            self.tcp.options = tuple(t.name for t in p.tcps) or (NONE,)
            if p.tcps:
                self.sel_tcp = min(self.sel_tcp, len(p.tcps) - 1)
                self.tcp.value = p.tcps[self.sel_tcp].name
                self.tcp_name.value = p.tcps[self.sel_tcp].name
                self.tcp_pose.set(pose_to_matrix(p.tcps[self.sel_tcp].pose))
            self.mount_pose.set(pose_to_matrix(p.mount))
            if p.x:
                self.tx.value = _mm(p.x)
            if p.y:
                self.ty.value = _mm(p.y)
            self.tlift.value = _mm(p.lift) if p.lift else (0.0, 0.0)
            for w in (
                self.tcp,
                self.add_tcp,
                self.tcp_name,
                self.del_tcp,
                *self._pose_widgets(self.tcp_pose),
            ):
                w.visible = tool
            for w in (self.mount_md, self.tlift, *self._pose_widgets(self.mount_pose)):
                w.visible = not tool
            self.tx.visible = p.kind in ("rail", "gantry")
            self.ty.visible = p.kind == "gantry"
            has = bool(p.parts)
            for w in (self.part_name, self.stage, self.remove, *self._pose_widgets(self.part_pose)):
                w.visible = has
            shape = p.parts[self.sel].shape if has else None
            self.size.visible, self.cyl.visible = shape == "box", shape == "cylinder"
            for pw, on in ((self.part_pose, has), (self.tcp_pose, tool), (self.mount_pose, not tool)):
                pw.gizmo.visible = on  # only these gizmo nodes: they have no children
        finally:
            self._busy = False
        self.draw()

    @staticmethod
    def _pose_widgets(pw: PoseWidgets) -> tuple:
        return (pw.xyz, pw.rpy, pw.show)

    def _set_kind(self, kind: str) -> None:
        ok = STAGES[kind]
        remap = {"fixed": "chassis", "chassis": "fixed", "bridge": "carriage"}
        parts = []
        for x in self.pkg.parts:
            st = (
                x.stage
                if x.stage in ok
                else remap.get(x.stage, ok[min(1, len(ok) - 1)] if kind != "tool" else "tool")
            )
            parts.append(x.model_copy(update={"stage": st if st in ok else ok[0]}))
        tcps = self.pkg.tcps or ([Frame(name="tcp")] if kind == "tool" else [])
        travel = {"x": self.pkg.x or (-1.5, 1.5)} if kind in ("rail", "gantry") else {"x": None}
        travel["y"] = (self.pkg.y or (-1.0, 1.0)) if kind == "gantry" else None
        self.pkg = self.pkg.model_copy(update={"kind": kind, "parts": parts, "tcps": tcps, **travel})
        self.load_widgets()

    # --- parts ---------------------------------------------------------------------------------

    def _add(self, part: Part) -> None:
        names = {x.name for x in self.pkg.parts}
        base, k = part.name, 2
        while part.name in names:
            part = part.model_copy(update={"name": f"{base}_{k}"})
            k += 1
        self.pkg = self.pkg.model_copy(update={"parts": [*self.pkg.parts, part]})
        self.sel = len(self.pkg.parts) - 1
        self.show.value = True
        self.load_widgets()

    def _on_mesh(self, e) -> None:
        f = self.upload.value
        if len(f.content) > MAX_UPLOAD:
            return notify(e, "Upload too large", "Limit is 200 MB")
        ext = Path(f.name).suffix.lower()
        name = safe(Path(f.name).stem)
        mesh_dir = self.work / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        try:
            if ext in (".step", ".stp"):
                from core.tool_import import step_mesh

                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / f"{name}{ext}"
                    src.write_bytes(f.content)
                    step_mesh(src).export(mesh_dir / f"{name}.stl")
                fname, units = f"{name}.stl", "m"
            elif ext in (".stl", ".obj", ".ply"):
                fname, units = f"{name}{ext}", self.units.value
                (mesh_dir / fname).write_bytes(f.content)
            else:
                return notify(e, "Unsupported file", "Use STL, OBJ, PLY or STEP")
            part = Part(name=name, stage=self._default_stage(), shape="mesh", mesh=fname, units=units)
            primitive_mesh(self.primitive(part))  # fail now on a broken file
        except Exception as err:  # a broken mesh / STEP file: report, keep the package
            return notify(e, "Mesh not added", f"{type(err).__name__}: {err}")
        self._add(part)

    def _select(self, name: str) -> None:
        names = [x.name for x in self.pkg.parts]
        if name in names:
            self.sel = names.index(name)
            self.load_widgets()

    def _part_fields(self) -> None:
        if not self.pkg.parts:
            return
        x = self.pkg.parts[self.sel]
        upd = {"stage": STAGE_OF.get(self.stage.value, x.stage)}
        new = safe(self.part_name.value)
        if new and new not in {p.name for k, p in enumerate(self.pkg.parts) if k != self.sel}:
            upd["name"] = new
        if x.shape == "box":
            upd["size"] = tuple(max(v, 1.0) / 1000 for v in self.size.value)
        if x.shape == "cylinder":
            upd["radius"], upd["height"] = (max(v, 1.0) / 1000 for v in self.cyl.value)
        self._set_part(x.model_copy(update=upd))
        if "name" in upd and upd["name"] != x.name:
            self.load_widgets()

    def _set_part(self, part: Part) -> None:
        parts = list(self.pkg.parts)
        parts[self.sel] = part
        self._update(parts=parts)

    def _part_moved(self, T: np.ndarray) -> None:
        if self.pkg.parts and not self._busy:
            self._set_part(self.pkg.parts[self.sel].model_copy(update={"pose": matrix_to_pose(T)}))

    def _remove(self) -> None:
        if self.pkg.parts:
            parts = [x for k, x in enumerate(self.pkg.parts) if k != self.sel]
            self.pkg = self.pkg.model_copy(update={"parts": parts})
            self.sel = max(self.sel - 1, 0)
            self.load_widgets()

    # --- frames and travel ---------------------------------------------------------------------

    def _select_tcp(self, name: str) -> None:
        names = [t.name for t in self.pkg.tcps]
        if name in names:
            self.sel_tcp = names.index(name)
            self.load_widgets()

    def _new_tcp(self) -> None:
        names = {t.name for t in self.pkg.tcps}
        k = len(names) + 1
        while f"tcp{k}" in names:
            k += 1
        self.pkg = self.pkg.model_copy(update={"tcps": [*self.pkg.tcps, Frame(name=f"tcp{k}")]})
        self.sel_tcp = len(self.pkg.tcps) - 1
        self.load_widgets()

    def _set_tcp(self, frame: Frame) -> None:
        tcps = list(self.pkg.tcps)
        tcps[self.sel_tcp] = frame
        self._update(tcps=tcps)

    def _tcp_fields(self) -> None:
        new = safe(self.tcp_name.value)
        others = {t.name for k, t in enumerate(self.pkg.tcps) if k != self.sel_tcp}
        if self.pkg.tcps and new and new not in others:
            self._set_tcp(self.pkg.tcps[self.sel_tcp].model_copy(update={"name": new}))
            self.load_widgets()

    def _tcp_moved(self, T: np.ndarray) -> None:
        if self.pkg.tcps and not self._busy:
            self._set_tcp(self.pkg.tcps[self.sel_tcp].model_copy(update={"pose": matrix_to_pose(T)}))

    def _remove_tcp(self, e) -> None:
        if len(self.pkg.tcps) <= 1:
            return notify(e, "Keep one TCP", "A tool needs at least one TCP")
        self.pkg = self.pkg.model_copy(
            update={"tcps": [t for k, t in enumerate(self.pkg.tcps) if k != self.sel_tcp]}
        )
        self.sel_tcp = 0
        self.load_widgets()

    def _mount_moved(self, T: np.ndarray) -> None:
        if not self._busy:
            self._update(mount=matrix_to_pose(T))

    def _travel(self) -> None:
        k = self.pkg.kind
        lift = tuple(v / 1000 for v in self.tlift.value)
        self._update(
            x=tuple(v / 1000 for v in self.tx.value) if k in ("rail", "gantry") else None,
            y=tuple(v / 1000 for v in self.ty.value) if k == "gantry" else None,
            lift=lift if k != "tool" and lift != (0.0, 0.0) else None,
        )

    # --- preview -------------------------------------------------------------------------------

    def primitive(self, part: Part) -> Primitive:
        mesh = str(self._resolve(part.mesh)) if part.shape == "mesh" else None
        return Primitive(
            shape=part.shape,
            size=part.size,
            radius=part.radius,
            height=part.height,
            mesh=mesh,
            units=part.units,
        )

    def draw(self) -> None:
        draw(self)

    # --- actions -------------------------------------------------------------------------------

    def _write(self, dest: Path) -> Path:
        if dest.exists():
            shutil.rmtree(dest)
        return write_package(self.pkg, self.work / "meshes", dest)

    def _check(self, e) -> None:
        problems = check(self.pkg)
        self.status.content = "✔ **Ready to export.**" if not problems else "✘ " + "  \n✘ ".join(problems)
        if problems:
            notify(e, "Package not ready", "; ".join(problems))

    def _export(self, e) -> None:
        try:
            urdf = self._write(self.s.project_dir / "export" / "packages" / safe(self.pkg.name))
        except (ValueError, OSError) as err:
            return notify(e, "Export failed", str(err))
        data = zip_folder(urdf.parent)
        zp = urdf.parent.parent / f"{safe(self.pkg.name)}.zip"
        zp.write_bytes(data)
        self.status.content = f"Saved `{zp}`"
        if getattr(e, "client", None) is not None:
            e.client.send_file_download(zp.name, data)
        notify(e, "Package exported", zp.name, ok=True)

    def _use(self, e) -> None:
        dest = self.s.project_dir / "assets" / "packages" / safe(self.pkg.name)
        try:
            urdf = self._write(dest)
            if self.pkg.kind == "tool":
                self.s.set_tool(to_tool(self.pkg, self.s.relative(urdf)))
                if self.tool_tab is not None:
                    self.tool_tab.refresh()
                where = "Tool tab: set the process and the extension ranges, then **Save to library**"
            else:
                base = to_base(self.pkg, self.s.relative(dest / "meshes"), self.s.base)
                if self.carrier is not None:
                    self.carrier.load(base)
                    if not self.carrier.commit(e):
                        return
                else:
                    self.s.base = base
                where = "External axis tab: set the standoffs and keep-outs, then plan"
        except (ValueError, OSError) as err:
            return notify(e, "Not loaded", str(err))
        self.status.content = f"Loaded **{self.pkg.name}** ({self.pkg.kind}). {where}."
        notify(e, "Package loaded", self.pkg.name, ok=True)

    def _open(self, e) -> None:
        f = self.open.value
        if len(f.content) > MAX_UPLOAD:
            return notify(e, "Upload too large", "Limit is 200 MB")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pkg, urdf = open_package(f.content, Path(tmp) / "pkg")
                mesh_dir = self.work / "meshes"
                if mesh_dir.exists():
                    shutil.rmtree(mesh_dir)
                mesh_dir.mkdir(parents=True)
                if (urdf.parent / "meshes").is_dir():
                    shutil.copytree(urdf.parent / "meshes", mesh_dir, dirs_exist_ok=True)
        except Exception as err:  # malformed zips / URDFs: report, keep the current package
            return notify(e, "Could not open the package", f"{type(err).__name__}: {err}")
        self.pkg, self.sel, self.sel_tcp = pkg, 0, 0
        self.show.value = True
        self.load_widgets()
        self.status.content = f"Opened **{pkg.name}**: a {pkg.kind} with {len(pkg.parts)} part(s)."
        notify(e, "Package opened", pkg.name, ok=True)

    def _reset(self) -> None:
        kind = self.pkg.kind
        self.pkg = Package(kind=kind, name=f"my_{kind}", tcps=[Frame(name="tcp")] if kind == "tool" else [])
        self._set_kind(kind)
        self.status.content = ""
