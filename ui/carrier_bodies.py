"""External Axis tab: body editor and robot mount editor.

The carrier is a list of bodies (schema.Primitive): boxes, cylinders and uploaded meshes,
each attached to the carriage (moves with the carrier DOFs) or fixed in the carrier frame
(rail track, gantry beam). Bodies are placed with fields or a drag gizmo. The robot is
mounted with "Snap robot to top of body" and fine-tuned with fields or a gizmo.

Edits here apply immediately through ``CarrierPanel.commit`` (validated as schema.Base);
an invalid edit is reported and the fields are reloaded from the last valid state.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import viser
from pydantic import ValidationError

from core.carrier import MESH_EXTS, UNIT_SCALE, body_top
from core.schema import Pose, Primitive
from core.transforms import (
    make_T,
    matrix_to_pose,
    matrix_to_wxyz,
    pose_to_matrix,
    rpy_to_matrix,
    wxyz_to_matrix,
)
from ui.widgets import MAX_UPLOAD, notify, safe_name

if TYPE_CHECKING:
    from ui.carrier_ui import CarrierPanel

NONE = "(none)"
ATTACH = {"carriage (moves)": "carriage", "fixed (track / frame)": "fixed"}
ATTACH_LABEL = {v: k for k, v in ATTACH.items()}
GIZMO = "/carrier_edit"


def _deg(rpy) -> tuple:
    return tuple(float(np.degrees(x)) for x in rpy)


class BodyEditor:
    def __init__(self, server: viser.ViserServer, panel: CarrierPanel):
        self.server = server
        self.p = panel
        self._busy = False
        self.gizmo = None
        self.mount_gizmo = None
        gui = server.gui
        with gui.add_folder("Bodies", expand_by_default=True):
            self.pick = gui.add_dropdown("Body", (NONE,))
            self.add_box = gui.add_button("Add box")
            self.add_cyl = gui.add_button("Add cylinder")
            self.upload = gui.add_upload_button("Add STL/OBJ/PLY body", hint="units: set 'Mesh units' first")
            self.mesh_units = gui.add_dropdown("Mesh units", tuple(UNIT_SCALE), initial_value="mm")
            self.dup = gui.add_button("Duplicate body")
            self.remove = gui.add_button("Remove body", color="red")
            self.name = gui.add_text("Name", "")
            self.attach = gui.add_dropdown("Attached to", tuple(ATTACH))
            self.box = gui.add_vector3("Box L W H (mm)", (500.0, 500.0, 200.0), step=10.0)
            self.cyl = gui.add_vector2("Cylinder radius / height (mm)", (100.0, 200.0), step=5.0)
            self.mesh_info = gui.add_markdown("")
            self.xyz = gui.add_vector3("Position (mm)", (0.0, 0.0, 0.0), step=5.0)
            self.rpy = gui.add_vector3("Rotation rpy (deg)", (0.0, 0.0, 0.0), step=5.0)
            self.kg = gui.add_number("Mass (kg, 0 = unknown)", 0.0, min=0.0, step=5.0)
            self.drag = gui.add_checkbox("Drag body with gizmo", False)
        with gui.add_folder("Robot mount", expand_by_default=True):
            self.mount_on = gui.add_dropdown("Mount on body", (NONE,))
            self.snap = gui.add_button("Snap robot to top of body")
            self.mount_xyz = gui.add_vector3("Mount xyz (mm)", (0.0, 0.0, 0.0), step=10.0)
            self.mount_rpy = gui.add_vector3("Mount rpy (deg)", (0.0, 0.0, 0.0), step=5.0)
            self.drag_mount = gui.add_checkbox("Drag mount with gizmo", False)
            gui.add_markdown("_Mount xyz/rpy: robot base in the carriage frame (z up, at the floor)._")
        self._wire()

    # --- state <-> widgets -------------------------------------------------------------

    @property
    def bodies(self) -> list[Primitive]:
        return self.p.bodies

    def index(self) -> int | None:
        names = [b.name for b in self.bodies]
        return names.index(self.pick.value) if self.pick.value in names else None

    def sync(self, select: str | None = None) -> None:
        """Refresh the dropdowns and the fields of the selected body from ``bodies``."""
        self._busy = True
        try:
            names = tuple(b.name for b in self.bodies) or (NONE,)
            keep = select if select in names else (self.pick.value if self.pick.value in names else names[0])
            self.pick.options = names
            self.pick.value = keep
            moving = tuple(b.name for b in self.bodies if b.attach == "carriage") or (NONE,)
            self.mount_on.options = moving
            if self.mount_on.value not in moving:
                self.mount_on.value = moving[-1]
            k = self.index()
            b = self.bodies[k] if k is not None else None
            for h in (self.name, self.attach, self.xyz, self.rpy, self.kg, self.drag, self.dup, self.remove):
                h.disabled = b is None
            self.box.visible = b is not None and b.shape == "box"
            self.cyl.visible = b is not None and b.shape == "cylinder"
            self.mesh_info.visible = b is not None and b.shape == "mesh"
            if b is not None:
                self.name.value = b.name
                self.attach.value = ATTACH_LABEL[b.attach]
                if b.shape == "box":
                    self.box.value = tuple(v * 1000 for v in b.size)
                elif b.shape == "cylinder":
                    self.cyl.value = (b.radius * 1000, b.height * 1000)
                else:
                    self.mesh_info.content = f"Mesh `{b.mesh}` ({b.units})"
                self.xyz.value = tuple(v * 1000 for v in b.pose.xyz)
                self.rpy.value = _deg(b.pose.rpy)
                self.kg.value = b.kg or 0.0
            m = self.p.mount_pose
            self.mount_xyz.value = tuple(v * 1000 for v in m.xyz)
            self.mount_rpy.value = _deg(m.rpy)
        finally:
            self._busy = False
        self.place_gizmos()

    def _from_fields(self, b: Primitive) -> Primitive:
        d = b.model_dump()
        d.update(
            name=self.name.value.strip() or b.name,
            attach=ATTACH[self.attach.value],
            pose=Pose(xyz=tuple(v / 1000 for v in self.xyz.value), rpy=tuple(np.radians(self.rpy.value))),
            kg=self.kg.value or None,
        )
        if b.shape == "box":
            d["size"] = tuple(v / 1000 for v in self.box.value)
        elif b.shape == "cylinder":
            d["radius"], d["height"] = (v / 1000 for v in self.cyl.value)
        return Primitive.model_validate(d)

    def _commit(self, event, bodies: list[Primitive], select: str | None = None) -> bool:
        old = list(self.p.bodies)
        self.p.bodies = bodies
        if not self.p.commit(event):
            self.p.bodies = old
            self.sync()
            return False
        self.sync(select)
        return True

    def _edit(self, event) -> None:
        if self._busy:
            return
        k = self.index()
        if k is None:
            return
        try:
            b = self._from_fields(self.bodies[k])
        except (ValueError, ValidationError) as err:
            msg = err.errors()[0]["msg"] if isinstance(err, ValidationError) else str(err)
            notify(event, "Invalid body", msg)
            return self.sync()
        bodies = list(self.bodies)
        bodies[k] = b
        self._commit(event, bodies, select=b.name)

    def _new_name(self, stem: str) -> str:
        names = {b.name for b in self.bodies}
        k = 1
        while f"{stem}{k}" in names:
            k += 1
        return f"{stem}{k}"

    def add_shape(self, shape: str, event=None) -> bool:
        """'Add box' / 'Add cylinder': a new carriage body at the carriage origin."""
        if shape == "box":
            body = Primitive(shape="box", size=(0.5, 0.5, 0.2), name=self._new_name("box"))
        else:
            body = Primitive(shape="cylinder", radius=0.1, height=0.2, name=self._new_name("cylinder"))
        return self._commit(event, [*self.bodies, body], select=body.name)

    def snap_mount(self, event=None) -> bool:
        """Put the robot base on the top centre of the 'Mount on body' body (keeps the yaw)."""
        names = [b.name for b in self.bodies]
        if self.mount_on.value not in names:
            notify(event, "No body", "Add a carriage body to mount the robot on")
            return False
        b = self.bodies[names.index(self.mount_on.value)]
        try:
            top = body_top(b, self.p.s.resolve)
        except (OSError, ValueError) as err:
            notify(event, "Cannot snap", str(err))
            return False
        yaw = self.p.mount_pose.rpy[2]
        return self._set_mount(event, make_T(rpy_to_matrix((0.0, 0.0, yaw)), top))

    def add_mesh_bytes(self, filename: str, content: bytes, event=None) -> bool:
        """Store an uploaded mesh in <project>/assets/carriers/ and add it as a carriage body
        (units from the 'Mesh units' dropdown). A file that fails to load is removed again."""
        if len(content) > MAX_UPLOAD:
            notify(event, "Upload too large", "Limit is 200 MB")
            return False
        fname = safe_name(filename)
        if not fname.lower().endswith(MESH_EXTS):
            notify(event, "Unsupported file", f"use {', '.join(MESH_EXTS)}")
            return False
        dest = self.p.s.project_dir / "assets" / "carriers" / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        existed = dest.exists()
        dest.write_bytes(content)
        stem = re.sub(r"[^A-Za-z0-9_]", "_", fname.rsplit(".", 1)[0])
        body = Primitive(
            shape="mesh", mesh=self.p.s.relative(dest), units=self.mesh_units.value, name=self._new_name(stem)
        )
        if not self._commit(event, [*self.bodies, body], select=body.name):
            if not existed:
                dest.unlink(missing_ok=True)
            return False
        notify(event, "Body added", f"{fname} ({self.mesh_units.value})", ok=True)
        return True

    # --- callbacks ------------------------------------------------------------------------

    def _wire(self) -> None:
        @self.pick.on_update
        def _(e) -> None:
            if not self._busy:
                self.sync()
                self.p.draw()  # highlight the selected body

        for h in (self.name, self.attach, self.box, self.cyl, self.xyz, self.rpy, self.kg):
            h.on_update(self._edit)

        self.add_box.on_click(lambda e: self.add_shape("box", e))
        self.add_cyl.on_click(lambda e: self.add_shape("cylinder", e))
        self.snap.on_click(lambda e: self.snap_mount(e))

        @self.upload.on_upload
        def _(e) -> None:
            f = self.upload.value
            self.add_mesh_bytes(f.name, f.content, e)

        @self.dup.on_click
        def _(e) -> None:
            k = self.index()
            if k is not None:
                b = self.bodies[k]
                x, y, z = b.pose.xyz
                pose = Pose(xyz=(x, y + 0.1, z), rpy=b.pose.rpy)
                dup = b.model_copy(update={"name": self._new_name(b.name + "_"), "pose": pose})
                self._commit(e, [*self.bodies, dup], select=dup.name)

        @self.remove.on_click
        def _(e) -> None:
            k = self.index()
            if k is not None:
                self._commit(e, [b for i, b in enumerate(self.bodies) if i != k])

        def mount_fields(e) -> None:
            if not self._busy:
                T = make_T(
                    rpy_to_matrix(np.radians(self.mount_rpy.value)), np.array(self.mount_xyz.value) / 1000
                )
                self._set_mount(e, T)

        self.mount_xyz.on_update(mount_fields)
        self.mount_rpy.on_update(mount_fields)
        self.drag.on_update(lambda e: self.place_gizmos())
        self.drag_mount.on_update(lambda e: self.place_gizmos())

    def _set_mount(self, event, T_carriage_base: np.ndarray) -> bool:
        old = self.p.mount_pose
        self.p.mount_pose = matrix_to_pose(T_carriage_base)
        ok = self.p.commit(event)
        if not ok:
            self.p.mount_pose = old
        self.sync()
        return ok

    # --- gizmos -----------------------------------------------------------------------------

    def place_gizmos(self) -> None:
        """Create, move or remove the body and mount gizmos to match the checkboxes."""
        k = self.index()
        movable = self.p.s.base.mode == "movable"
        want_body = self.drag.value and k is not None and movable
        if want_body:
            b = self.bodies[k]
            T = self.p.parent_T(b.attach) @ pose_to_matrix(b.pose)
            self.gizmo = self._place(self.gizmo, f"{GIZMO}/body", T, self._on_body_drag)
        elif self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None
        if self.drag_mount.value and movable:
            T = self.p.parent_T("mount") @ pose_to_matrix(self.p.mount_pose)
            self.mount_gizmo = self._place(self.mount_gizmo, f"{GIZMO}/mount", T, self._on_mount_drag)
        elif self.mount_gizmo is not None:
            self.mount_gizmo.remove()
            self.mount_gizmo = None

    def _place(self, handle, name: str, T: np.ndarray, cb):
        if handle is None:
            handle = self.server.scene.add_transform_controls(name, scale=0.35)
            handle.on_update(cb)
        self._busy = True
        try:
            handle.wxyz = matrix_to_wxyz(T[:3, :3])
            handle.position = T[:3, 3]
        finally:
            self._busy = False
        return handle

    def _gizmo_T(self, h) -> np.ndarray:
        return make_T(wxyz_to_matrix(h.wxyz), h.position)

    def _on_body_drag(self, e) -> None:
        k = self.index()
        if self._busy or k is None or self.gizmo is None:
            return
        b = self.bodies[k]
        T = np.linalg.inv(self.p.parent_T(b.attach)) @ self._gizmo_T(self.gizmo)
        bodies = list(self.bodies)
        bodies[k] = b.model_copy(update={"pose": matrix_to_pose(T)})
        self._commit(e, bodies, select=b.name)

    def _on_mount_drag(self, e) -> None:
        if self._busy or self.mount_gizmo is None:
            return
        self._set_mount(e, np.linalg.inv(self.p.parent_T("mount")) @ self._gizmo_T(self.mount_gizmo))
