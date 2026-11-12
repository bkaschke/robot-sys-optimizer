"""Package builder tab: author a URDF package for an end effector, a rail, a gantry or a mobile
base (core.urdf_package), preview it, export it as a zip and use it in the project.

The package being edited lives in ``<project>/assets/packages/_builder`` (uploaded meshes are
copied there). *Use in this project* writes ``<project>/assets/packages/<name>/`` and loads it
as the tool (Tool tab) or the external axis (External axis tab). *Open package (.zip)* reads a
package made here or by hand / from CAD (see the user guide, "Building URDF packages").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import viser

from core.schema import Primitive
from core.transforms import matrix_to_pose, pose_to_matrix
from core.urdf_package import (
    STAGES,
    Frame,
    Package,
    Part,
    safe,
)
from ui import package_actions
from ui.package_preview import ROOT, draw
from ui.widgets import PoseWidgets, notify

KIND_LABELS = {
    "End effector (tool)": "tool",
    "Part": "part",
    "Rail": "rail",
    "Gantry": "gantry",
    "Mobile base": "mobile",
}
TABS = {"End effector": "tool", "Part": "part", "Rail": "rail", "Gantry": "gantry", "Mobile robot": "mobile"}
TAB_OF = {v: k for k, v in TABS.items()}
KIND_OF = {v: k for k, v in KIND_LABELS.items()}
STAGE_LABELS = {
    "tool": "tool (on the flange mount)",
    "part": "part (the workpiece)",
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
    def __init__(
        self,
        server: viser.ViserServer,
        session,
        tool_tab=None,
        carrier_panel=None,
        robot_preview: bool = False,
    ):
        self.server, self.s = server, session
        self.tool_tab, self.carrier = tool_tab, carrier_panel
        self.pkg = Package(kind="tool", name="my_tool", tcps=[Frame(name="tcp")])
        self.sel = 0  # selected part
        self.sel_tcp = 0
        self._busy = False
        self._handles: list = []
        gui = server.gui
        gui.add_markdown(
            "_All geometry of the project is made here (v9): an **end effector**, the **part**, a **rail / "
            "gantry** or a **mobile robot**, from boxes, cylinders and meshes (STL / OBJ / PLY / STEP), with "
            "its collision geometry. Export it or use it in the project. Poses are in the package frame: the "
            "robot flange (tool), the part's own frame, or the carrier frame on the floor. User guide: "
            "*Building URDF packages*._"
        )
        self.tabs = gui.add_button_group("Package", tuple(TABS))  # v9 #1: a tab per package type
        self.kind_md = gui.add_markdown("")
        self.kind = gui.add_dropdown(
            "Package type", tuple(KIND_LABELS), initial_value=KIND_OF["tool"], visible=False
        )
        self.edit_source = (
            None  # ("tool", EndEffector) / ("part", None) / ("carrier", None): from the project
        )
        from ui import package_edit

        with gui.add_folder("From the project", expand_by_default=False):
            self.edit_tool = gui.add_button("Edit the project's tool")
            self.edit_part = gui.add_button("Edit the project's part")
            self.edit_carrier = gui.add_button("Edit the project's external axis")
        self.edit_tool.on_click(lambda e: package_edit.edit_tool(self, e))
        self.edit_part.on_click(lambda e: package_edit.edit_part(self, e))
        self.edit_carrier.on_click(lambda e: package_edit.edit_carrier(self, e))
        self.on_part = None  # app.py: the project's part changed (Use in this project)
        self.name = gui.add_text("Name", self.pkg.name)
        with gui.add_folder("Parts"):
            self.part = gui.add_dropdown("Part", (NONE,))
            self.add_box = gui.add_button("Add box")
            self.add_cyl = gui.add_button("Add cylinder")
            self.units = gui.add_dropdown("Mesh units", UNITS, hint="STL / OBJ / PLY (STEP carries its own)")
            self.upload = gui.add_upload_button("Add mesh (STL/OBJ/PLY/STEP)")
            self.import_bar = gui.add_progress_bar(0.0, animated=True, visible=False)
            self.import_info = gui.add_markdown("")
            self.import_errors: dict[str, str] = {}  # file -> why its import failed (Check lists it)
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
            self.show = gui.add_checkbox("Show preview", True)
            self.at = gui.add_vector3(
                "Preview at (mm)",
                (0.0, 0.0, 0.0),
                step=50.0,
                hint="where a rail / gantry / mobile base stands",
            )
        from ui.package_collision import PackageCollision

        self.collision = PackageCollision(server, self)  # how mesh parts are simplified for collision
        self.check_btn = gui.add_button("Check")
        self.export = gui.add_button("Export package (.zip)", color="green")
        self.use = gui.add_button("Use in this project", color="green")
        self.open = gui.add_upload_button("Open package (.zip)", mime_type="application/zip")
        self.new = gui.add_button("New package")
        self.status = gui.add_markdown("")
        self.root = server.scene.add_frame(
            ROOT, show_axes=True, axes_length=0.3, axes_radius=0.006, visible=False
        )
        self.robot_preview = None
        if robot_preview:  # the builder window (ui.builder_window): a robot to try the package on
            from ui.package_robot import RobotPreview

            self.robot_preview = RobotPreview(server, self, session)
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
        self.tabs.on_click(lambda e: self._set_kind(TABS[e.target.value]))
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
        self.use.on_click(self._confirm_use)
        self.open.on_upload(self._open)
        self.new.on_click(lambda e: self._reset())

    def _default_stage(self) -> str:
        k = self.pkg.kind
        return {"tool": "tool", "part": "part", "mobile": "chassis"}.get(k, "carriage")

    def _update(self, **kw) -> None:
        self.pkg = self.pkg.model_copy(update=kw)
        self.draw()

    def load_widgets(self) -> None:
        """All fields from ``self.pkg``."""
        p = self.pkg
        self._busy = True
        try:
            self.kind.value = KIND_OF[p.kind]
            self.kind_md.content = f"**{TAB_OF[p.kind]}** package"
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
            carrier = p.kind in ("rail", "gantry", "mobile")
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
                w.visible = carrier
            self.tx.visible = p.kind in ("rail", "gantry")
            self.ty.visible = p.kind == "gantry"
            has = bool(p.parts)
            for w in (self.part_name, self.stage, self.remove, *self._pose_widgets(self.part_pose)):
                w.visible = has
            self.stage.visible = has and len(STAGES[p.kind]) > 1
            shape = p.parts[self.sel].shape if has else None
            self.size.visible, self.cyl.visible = shape == "box", shape == "cylinder"
            for pw, on in ((self.part_pose, has), (self.tcp_pose, tool), (self.mount_pose, carrier)):
                pw.gizmo.visible = on  # only these gizmo nodes: they have no children
        finally:
            self._busy = False
        self.collision.load()
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
                else remap.get(x.stage, ok[min(1, len(ok) - 1)] if kind not in ("tool", "part") else kind)
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
        package_actions.import_mesh(self, e)  # v9 #2: in the background, with status

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

    def root_pose(self) -> np.ndarray:
        """World pose of the package frame: on the preview robot's flange (tool), else at
        *Preview at*."""
        from ui.package_robot import at_pose

        at = at_pose(self.at.value)
        return at if self.robot_preview is None else self.robot_preview.root_pose(at)

    def draw(self) -> None:
        draw(self)
        if self.collision.show.value:
            self.collision.redraw_soon()
        if self.robot_preview is not None:
            self.robot_preview.place(self.root_pose())

    # --- actions (ui.package_actions) ---------------------------------------------------------

    def _write(self, dest: Path) -> Path:
        return package_actions.write(self, dest)

    def _check(self, e) -> None:
        package_actions.check_package(self, e)

    def _export(self, e) -> None:
        package_actions.export(self, e)

    def _use(self, e) -> bool:
        return package_actions.use(self, e)

    def _confirm_use(self, e) -> None:
        from ui.package_edit import confirm_use

        confirm_use(self, e)  # v9 #5: asks first, closes the window after

    def _open(self, e) -> None:
        package_actions.open_zip(self, e)

    def _reset(self) -> None:
        package_actions.reset(self)
