"""Robot & cell tab -> Stay-out zones (v3 feedback item 5): boxes, cylinders and uploaded
STL/OBJ/PLY meshes in the world that the arm, the tool and the carrier keep clear of.

Edits go straight into ``session.stayouts`` (validated as schema.Primitive). The zones are
drawn as see-through red volumes under /cell/stayout; *Drag with gizmo* places the selected
zone in 3D. Box / cylinder frames sit at the centre of the bottom face (as carrier bodies).
"""

from __future__ import annotations

import numpy as np
import viser
from pydantic import ValidationError

from core.carrier import primitive_mesh
from core.schema import Pose, Primitive
from core.transforms import make_T, matrix_to_rpy, matrix_to_wxyz, pose_to_matrix, wxyz_to_matrix
from ui.robot_view import quiet_remove
from ui.session import MESH_EXTS, UNIT_SCALE, Session
from ui.widgets import MAX_UPLOAD, GizmoStyle, Latest, notify, safe_name

ROOT = "/cell/stayout"
NONE = "(none)"
COLOR, PICKED = (230, 50, 50), (255, 120, 60)


class StayoutPanel:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server, self.s = server, session
        self._handles: dict = {}
        self.gizmo = None
        self._gizmo_zone: str | None = None  # the zone the current gizmo belongs to (T12)
        self._busy = False
        self.on_alt_click = None  # (zone name, event) -> None: Move mode (app.py, T11)
        gui = server.gui
        self.folder = gui.add_folder("Stay-out zones", expand_by_default=False)
        with self.folder:
            gui.add_markdown(
                "_Volumes the arm, the tool and the carrier keep clear of (planner, check, paths). "
                "The red floor areas of the External axis tab are carrier-only floor keep-outs._"
            )
            self.pick = gui.add_dropdown("Zone", (NONE,))
            self.add_box = gui.add_button("Add box")
            self.add_cyl = gui.add_button("Add cylinder")
            self.units = gui.add_dropdown("Mesh units", tuple(UNIT_SCALE), initial_value="mm")
            self.upload = gui.add_upload_button("Add STL/OBJ/PLY zone")
            self.name = gui.add_text("Name", "")
            self.box = gui.add_vector3("Box L W H (mm)", (500.0, 500.0, 1000.0), step=10.0)
            self.cyl = gui.add_vector2("Cylinder radius / height (mm)", (250.0, 1000.0), step=10.0)
            self.xyz = gui.add_vector3("Position (mm)", (0.0, 0.0, 0.0), step=10.0)
            self.rpy = gui.add_vector3("Rotation rpy (deg)", (0.0, 0.0, 0.0), step=5.0)
            self.drag = gui.add_checkbox("Drag with gizmo", False)
            self.remove = gui.add_button("Remove zone", color="red")
            self.info = gui.add_markdown("")
        self.add_box.on_click(lambda e: self.add_shape("box", e))
        self.add_cyl.on_click(lambda e: self.add_shape("cylinder", e))
        self.upload.on_upload(
            lambda e: self.add_mesh_bytes(self.upload.value.name, self.upload.value.content, e)
        )
        self.remove.on_click(lambda e: self.remove_zone(e))
        self.pick.on_update(lambda e: None if self._busy else self.sync())
        for w in (self.name, self.box, self.cyl, self.xyz, self.rpy):
            w.on_update(lambda e: None if self._busy else self._edit(e))
        self.drag.on_update(lambda e: self.place_gizmo())
        self.refresh()

    # --- list --------------------------------------------------------------------------------

    @property
    def zones(self) -> list[Primitive]:
        return self.s.stayouts

    def index(self) -> int | None:
        names = [z.name for z in self.zones]
        return names.index(self.pick.value) if self.pick.value in names else None

    def _unique(self, stem: str) -> str:
        names = {z.name for z in self.zones}
        k = 1
        while f"{stem}{k}" in names:
            k += 1
        return f"{stem}{k}"

    def _commit(self, zones: list[Primitive], pick: str | None = None, event=None) -> bool:
        try:
            zones = [Primitive.model_validate(z.model_dump()) for z in zones]
            for z in zones:
                primitive_mesh(z, self.s.resolve)  # fail here on a broken mesh file
        except (ValueError, OSError, ValidationError) as err:
            msg = err.errors()[0]["msg"] if isinstance(err, ValidationError) else str(err)
            notify(event, "Invalid stay-out zone", msg)
            return False
        self.s.stayouts = zones
        self.refresh(pick)
        return True

    def add_shape(self, shape: str, event=None) -> bool:
        name = self._unique("zone")
        size = {"size": (0.5, 0.5, 1.0)} if shape == "box" else {"radius": 0.25, "height": 1.0}
        # each new zone a bit further along x, so zones never start on top of each other (T12)
        z = Primitive(shape=shape, name=name, attach="fixed", pose=Pose(xyz=self._new_xyz()), **size)
        return self._commit([*self.zones, z], name, event)

    def _new_xyz(self) -> tuple[float, float, float]:
        return (1.0 + 0.7 * len(self.zones), 1.0, 0.0)

    def select(self, name: str) -> bool:
        """Make ``name`` the selected zone (dropdown, a click on the zone in 3D, the Objects
        tree). False if there is no such zone."""
        if name not in [z.name for z in self.zones]:
            return False
        self.refresh(name)
        return True

    def _alt(self, name: str, event) -> None:
        if self.on_alt_click is not None:
            self.on_alt_click(name, event)

    def add_mesh_bytes(self, filename: str, data: bytes, event=None) -> bool:
        from pathlib import Path

        ext = Path(filename).suffix.lower()
        if ext not in MESH_EXTS or len(data) > MAX_UPLOAD:
            notify(event, "Unsupported mesh", f"Use {', '.join(MESH_EXTS)} under 200 MB")
            return False
        path = self.s.project_dir / "assets" / "stayouts" / safe_name(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        name = self._unique(safe_name(Path(filename).stem).replace(".", "_")[:40] or "mesh")
        z = Primitive(
            shape="mesh", name=name, attach="fixed", mesh=self.s.relative(path), units=self.units.value
        )
        return self._commit([*self.zones, z], name, event)

    def remove_zone(self, event=None) -> bool:
        i = self.index()
        if i is None:
            return False
        return self._commit([z for k, z in enumerate(self.zones) if k != i], None, event)

    def _edit(self, event=None) -> None:
        i = self.index()
        if i is None:
            return
        z = self.zones[i]
        upd = {
            "name": self.name.value.strip() or z.name,
            "pose": Pose(xyz=tuple(v / 1000 for v in self.xyz.value), rpy=tuple(np.radians(self.rpy.value))),
        }
        if z.shape == "box":
            upd["size"] = tuple(max(v, 1.0) / 1000 for v in self.box.value)
        elif z.shape == "cylinder":
            upd["radius"], upd["height"] = (max(v, 1.0) / 1000 for v in self.cyl.value)
        zones = list(self.zones)
        zones[i] = z.model_copy(update=upd)
        self._commit(zones, upd["name"], event)

    # --- widgets and scene -------------------------------------------------------------------

    def refresh(self, pick: str | None = None) -> None:
        names = [z.name for z in self.zones]
        self._busy = True
        try:
            self.pick.options = tuple(names) if names else (NONE,)
            if pick in names:
                self.pick.value = pick
            elif self.pick.value not in names:
                self.pick.value = names[0] if names else NONE
        finally:
            self._busy = False
        self.sync()

    def sync(self) -> None:
        """Selected zone -> fields; redraw."""
        i = self.index()
        self._busy = True
        try:
            for w in (self.name, self.box, self.cyl, self.xyz, self.rpy, self.drag, self.remove):
                w.disabled = i is None
            if i is not None:
                z = self.zones[i]
                self.name.value = z.name
                self.box.visible, self.cyl.visible = z.shape == "box", z.shape == "cylinder"
                if z.shape == "box":
                    self.box.value = tuple(v * 1000 for v in z.size)
                elif z.shape == "cylinder":
                    self.cyl.value = (z.radius * 1000, z.height * 1000)
                self.xyz.value = tuple(v * 1000 for v in z.pose.xyz)
                self.rpy.value = tuple(float(v) for v in np.degrees(z.pose.rpy))
        finally:
            self._busy = False
        n = len(self.zones)
        self.info.content = f"{n} stay-out zone{'s' if n != 1 else ''}" if n else "_No stay-out zones._"
        self.draw()

    def draw(self) -> None:
        for h in self._handles.values():
            quiet_remove(h)
        self._handles = {}
        i = self.index()
        dragging = self.drag.value and i is not None
        for k, z in enumerate(self.zones):
            try:
                m = primitive_mesh(z, self.s.resolve)
            except (ValueError, OSError):
                continue
            T = pose_to_matrix(z.pose)
            h = self._handles[z.name] = self.server.scene.add_mesh_simple(
                f"{ROOT}/{z.name}",
                m.vertices,
                m.faces,
                color=PICKED if k == i else COLOR,
                opacity=0.35,
                side="double",
                wxyz=matrix_to_wxyz(T[:3, :3]),
                position=T[:3, 3],
            )
            if not (dragging and k == i):  # T10: the zone under the gizmo takes no clicks
                h.on_click(lambda e, name=z.name: self.select(name))
                h.on_click(modifier="alt")(lambda e, name=z.name: self._alt(name, e))
        self.place_gizmo()

    def place_gizmo(self) -> None:
        i = self.index()
        name = self.zones[i].name if i is not None else None
        if self.gizmo is not None and (not self.drag.value or name != self._gizmo_zone):
            # a new gizmo per zone: late drag events of the old one cannot move another zone
            quiet_remove(self.gizmo)
            self.gizmo, self._gizmo_zone = None, None
        if not self.drag.value or i is None:
            return
        T = pose_to_matrix(self.zones[i].pose)
        if self.gizmo is None:
            self.gizmo = self.server.scene.add_transform_controls("/stayout_edit/gizmo", scale=0.4)
            self._gizmo_zone = name
            GizmoStyle.register(self.gizmo, 0.4)
            try:  # T10: the gizmo reaches outside the zone
                b = primitive_mesh(self.zones[i], self.s.resolve).bounds
                GizmoStyle.fit(self.gizmo, float(np.max(b[1] - b[0])))
            except (ValueError, OSError):
                pass
            drag = Latest(lambda: self._from_gizmo(name))  # drags: newest pose only
            self.gizmo.on_update(lambda e: None if self._busy else drag())
        self._busy = True
        try:
            self.gizmo.wxyz = matrix_to_wxyz(T[:3, :3])
            self.gizmo.position = T[:3, 3]
        finally:
            self._busy = False

    def _from_gizmo(self, name: str | None = None) -> None:
        """Gizmo -> pose of zone ``name`` (the zone the gizmo was made for), only while that
        zone is still the selected one."""
        i = self.index()
        if i is None or self.gizmo is None or (name is not None and self.zones[i].name != name):
            return
        T = make_T(wxyz_to_matrix(self.gizmo.wxyz), self.gizmo.position)
        zones = list(self.zones)
        zones[i] = zones[i].model_copy(
            update={"pose": Pose(xyz=tuple(float(v) for v in T[:3, 3]), rpy=tuple(matrix_to_rpy(T[:3, :3])))}
        )
        self.s.stayouts = zones
        self._busy = True
        try:
            self.xyz.value = tuple(v * 1000 for v in T[:3, 3])
            self.rpy.value = tuple(float(v) for v in np.degrees(matrix_to_rpy(T[:3, :3])))
        finally:
            self._busy = False
        h = self._handles.get(zones[i].name)
        if h is not None:  # move the drawn zone without re-sending its mesh
            h.wxyz, h.position = matrix_to_wxyz(T[:3, :3]), T[:3, 3]
