"""External Axis tab (M9 + carrier builder): static or movable base, the carrier built from
boxes, cylinders and meshes (ui.carrier_bodies), the robot mount, standoffs, keep-outs,
localization settings, and DOF sliders to jog the carrier.

Axis-type and safety fields are committed with "Apply base settings"; body and mount edits
apply immediately. Both go through ``commit`` (validated as schema.Base).
Scene: carrier bodies under /cell/carrier, standoff shells (min red, preferred amber) and
keep-out boxes on the floor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import viser
from pydantic import ValidationError

from core.carrier import migrate
from core.schema import Base, CarrierJoints, Keepout, Pose, Primitive, Standoffs
from core.standoff import Zones, footprint
from core.transforms import make_T, matrix_to_wxyz, pose_to_matrix
from ui import carrier_mobile
from ui.carrier_bodies import BodyEditor
from ui.session import Session
from ui.widgets import Latest, notify

ROOT = "/cell/carrier"

DEFAULTS = {
    "rail": Base(
        mode="movable",
        carrier="rail",
        pose=Pose(xyz=(0, -0.8, 0)),
        track=Primitive(shape="box", size=(7.0, 0.3, 0.1)),
        chassis=Primitive(shape="box", size=(0.5, 0.5, 0.15)),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.55),
        T_mount=Pose(xyz=(0, 0, 0.7)),
        joints=CarrierJoints(x=(-3.0, 3.0)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4),
    ),
    "gantry": Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0, 0, 2.5)),
        chassis=Primitive(shape="box", size=(0.4, 0.4, 0.15)),
        T_mount=Pose(rpy=(np.pi, 0, 0)),
        joints=CarrierJoints(x=(-1.5, 1.5), y=(-1.0, 1.0)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4),
    ),
    "mobile": Base(
        mode="movable",
        carrier="mobile",
        chassis=Primitive(shape="box", size=(1.2, 0.8, 0.4)),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.3),
        T_mount=Pose(xyz=(0.2, 0, 0.7)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-4.0, 4.0), yaw_deg=(-180, 180)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.5),
        localization_m=0.01,
        localization_deg=1.0,
    ),
}
BODY_COLOR, FIXED_COLOR, PICKED_COLOR = (120, 130, 150), (150, 140, 120), (90, 140, 220)


def _label(handle, text: str) -> None:
    """Relabel a GUI input (no-op on Viser versions without settable labels)."""
    try:
        if handle.label != text:
            handle.label = text
    except AttributeError:
        pass


def _mm(v) -> tuple:
    return tuple(float(x) * 1000 for x in v)


class CarrierPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        set_base: Callable[[np.ndarray], None],
        dof_container=None,
    ):
        self.server = server
        self.s = session
        self.set_base = set_base
        self._handles: list = []
        self._dof: dict = {}
        self._dof_sig = None
        self._busy = False
        self._body_handles: dict = {}  # body name -> scene mesh handle (re-posed on jog)
        self._moving_names: list[str] = []
        self._zones = None  # standoff zones for the current part pose (rebuilt in draw)
        self.bodies: list[Primitive] = []
        self.mount_pose = Pose()
        gui = server.gui
        self.locked_note = gui.add_markdown("", visible=False)
        self.axis_folder = gui.add_folder("Axis type")
        with self.axis_folder:
            self.mode = gui.add_dropdown("Mode", ("static", "movable"))
            self.kind = gui.add_dropdown("Carrier", ("mobile", "rail", "gantry"))
            self.defaults = gui.add_button("Load carrier defaults")
            self.frame_xyz = gui.add_vector3("Track/gantry frame xyz (mm)", (0.0, 0.0, 0.0), step=10.0)
            self.frame_yaw = gui.add_number("Track/gantry yaw (deg)", 0.0, step=1.0)
            self.tx = gui.add_vector2("Travel x / s (mm)", (-3000.0, 3000.0), step=10.0)
            self.ty = gui.add_vector2("Travel y (mm)", (-1000.0, 1000.0), step=10.0)
            self.tyaw = gui.add_vector2("Yaw range (deg)", (-180.0, 180.0), step=5.0)
            self.lift = gui.add_vector2("Lift range (mm, 0 0 = none)", (0.0, 0.0), step=10.0)
            carrier_mobile.build_align(gui, self)
        self.editor = BodyEditor(server, self)
        self.safety_folder = gui.add_folder("Standoff and safety", expand_by_default=False)
        with self.safety_folder:
            carrier_mobile.build_mobile(gui, self)  # robot sides, fixed standoff, part sides
            self.so = gui.add_vector2("Part standoff min / preferred (mm)", (300.0, 400.0), step=10.0)
            self.env = gui.add_number("Keep-out clearance (mm)", 100.0, min=0.0, step=10.0)
            self.keepouts = gui.add_text("Keep-outs (mm)", "", hint="x0,y0,x1,y1; x0,y0,x1,y1 (floor boxes)")
            self.loc = gui.add_vector2("Localization ±mm / ±deg (mobile)", (10.0, 1.0), step=1.0)
        self.apply = gui.add_button("Apply base settings", color="green")
        if dof_container is not None:  # the left panel's Jog folder (app.py)
            self.dof_folder = dof_container
        else:
            self.dof_folder = gui.add_folder("Carrier position")
        self.info = gui.add_markdown("")
        self._jog_latest = Latest(self._jog)  # slider drags: newest value only
        self.apply.on_click(lambda e: self.commit(e))
        # aliases kept for scripts/tests written against the M9 panel
        self.mount_xyz, self.mount_rpy = self.editor.mount_xyz, self.editor.mount_rpy

        @self.defaults.on_click
        def _(e) -> None:
            self.load(DEFAULTS[self.kind.value].model_copy(update={"mode": "movable"}))
            self.commit(e)

        self.locked = False
        for w in (self.mode, self.kind, *carrier_mobile.widgets(self)):
            w.on_update(lambda e: self.update_visibility())
        self.load(session.base)
        self.refresh()

    # --- what is shown --------------------------------------------------------------------------

    def update_visibility(self) -> None:
        """Show only the settings of the selected mode / axis type, and nothing editable while
        the setup is confirmed (locked): the robot then rides on the carrier as one assembly."""
        movable = self.mode.value == "movable"
        kind = self.kind.value
        edit = not self.locked
        self.axis_folder.visible = edit
        for w in (self.kind, self.defaults, self.tx, self.lift):
            w.visible = movable
        self.frame_xyz.visible = self.frame_yaw.visible = movable and kind in ("rail", "gantry")
        self.ty.visible = movable and kind in ("mobile", "gantry")
        self.tyaw.visible = movable and kind == "mobile"
        self.loc.visible = movable and kind == "mobile"
        carrier_mobile.show(self, movable, kind)
        _label(
            self.tx,
            "Planner search x (mm)"
            if kind == "mobile"
            else ("Travel s (mm)" if kind == "rail" else "Travel x (mm)"),
        )
        _label(self.ty, "Planner search y (mm)" if kind == "mobile" else "Travel y (mm)")
        self.editor.bodies_folder.visible = self.editor.mount_folder.visible = movable and edit
        self.safety_folder.visible = movable and edit
        self.apply.visible = edit
        self.dof_folder.visible = movable
        if not edit:  # no editing gizmos on a confirmed setup
            self.editor.drag.value = self.editor.drag_mount.value = False
        self.locked_note.visible = not edit
        self.locked_note.content = (
            "**Setup confirmed:** the robot rides on the carrier. Move them together with the "
            "*Carrier position* sliders or the jog arrows (left panel → Jog). **Edit setup** to change it."
            if movable
            else "**Setup confirmed** (static base). **Edit setup** to change it."
        )

    def set_locked(self, locked: bool) -> None:
        if locked != self.locked:
            self.locked = locked
            self.update_visibility()

    # --- widgets <-> schema ------------------------------------------------------------------

    def load(self, b: Base) -> None:
        b = migrate(b)
        self._busy = True
        try:
            self.mode.value = b.mode
            if b.carrier in ("mobile", "rail", "gantry"):
                self.kind.value = b.carrier
            self.frame_xyz.value = _mm(b.pose.xyz)
            self.frame_yaw.value = float(np.degrees(b.pose.rpy[2]))
            j = b.joints
            if j.x:
                self.tx.value = _mm(j.x)
            if j.y:
                self.ty.value = _mm(j.y)
            if j.yaw_deg:
                self.tyaw.value = tuple(j.yaw_deg)
            self.lift.value = _mm(j.lift) if j.lift else (0.0, 0.0)
            so = b.standoff_m
            self.so.value = _mm((so.carrier_part_min, so.carrier_part_preferred or so.carrier_part_min))
            self.env.value = so.carrier_env_min * 1000
            self.keepouts.value = "; ".join(
                ",".join(f"{v * 1000:.0f}" for v in (*k.min_xy, *k.max_xy)) for k in b.keepouts
            )
            self.loc.value = (b.localization_m * 1000, b.localization_deg)
            carrier_mobile.load(self, b)
            self.bodies = list(b.bodies)
            self.mount_pose = b.T_mount
            self._dof_sig = None  # next refresh re-reads the slider values from the base pose
        finally:
            self._busy = False
        self.editor.sync()
        self.update_visibility()

    def collect(self) -> Base:
        kind = self.kind.value
        movable = self.mode.value == "movable"
        kos = []
        for chunk in filter(None, (c.strip() for c in self.keepouts.value.split(";"))):
            x0, y0, x1, y1 = (float(v) / 1000 for v in chunk.split(","))
            kos.append(Keepout(min_xy=(min(x0, x1), min(y0, y1)), max_xy=(max(x0, x1), max(y0, y1))))
        smin, spref = (v / 1000 for v in self.so.value)
        lift = tuple(v / 1000 for v in self.lift.value)
        joints = CarrierJoints(
            x=tuple(v / 1000 for v in self.tx.value),
            y=tuple(v / 1000 for v in self.ty.value) if kind in ("mobile", "gantry") else None,
            yaw_deg=tuple(self.tyaw.value) if kind == "mobile" else None,
            lift=lift if lift != (0.0, 0.0) else None,
        )
        return Base(
            mode="movable" if movable else "static",
            carrier=kind if movable else None,
            pose=Pose(
                xyz=tuple(v / 1000 for v in self.frame_xyz.value),
                rpy=(0, 0, np.radians(self.frame_yaw.value)),
            ),
            bodies=self.bodies,
            joints=joints,
            T_mount=self.mount_pose,
            keepouts=kos,
            standoff_m=Standoffs(
                carrier_part_min=smin, carrier_part_preferred=spref, carrier_env_min=self.env.value / 1000
            ),
            localization_m=self.loc.value[0] / 1000,
            localization_deg=self.loc.value[1],
            **carrier_mobile.collect(self, kind),
        )

    def realign(self) -> None:
        """Rail / gantry with *Align track with the part*: put the track along the part's long
        axis (after Apply and whenever the part moves)."""
        b = self.s.base
        if not b.align_to_part or b.mode != "movable" or self.s.part is None:
            return
        from core.carrier import aligned_track_pose

        pose = aligned_track_pose(b, footprint(self.s.part_mesh(), self.s.part.T_world_part))
        if np.allclose(pose_to_matrix(pose), pose_to_matrix(b.pose), atol=1e-9):
            return
        self.s.base = b.model_copy(update={"pose": pose})
        self._busy = True
        try:
            self.frame_xyz.value = _mm(pose.xyz)
            self.frame_yaw.value = float(np.degrees(pose.rpy[2]))
        finally:
            self._busy = False
        self.refresh()

    def commit(self, event=None) -> bool:
        try:
            base = self.collect()
            if base.mode == "movable":
                c = self._carrier(base)
                for k in range(len(c.specs)):
                    c.mesh(k)  # fail here (not mid-plan) on a missing or broken mesh file
        except (ValueError, OSError, ValidationError) as err:
            msg = err.errors()[0]["msg"] if isinstance(err, ValidationError) else str(err)
            notify(event, "Invalid base settings", msg)
            return False
        self.s.base = base
        self.realign()  # no-op unless the track follows the part
        self.refresh()
        return True

    def _carrier(self, base: Base | None = None):
        from core.carrier import Carrier

        b = base or self.s.base
        if b.mode == "movable":  # movable carriers do not depend on T_world_base: reuse (caches)
            hit = getattr(self, "_carrier_cache", None)
            if hit is not None and hit[0] is b:
                return hit[1]
            c = Carrier(b, self.s.T_world_base, resolve=self.s.resolve)
            self._carrier_cache = (b, c)
            return c
        return Carrier(b, self.s.T_world_base, resolve=self.s.resolve)

    # --- DOF sliders and drawing ---------------------------------------------------------------

    def refresh(self) -> None:
        b = self.s.base
        c = self._carrier() if b.mode == "movable" else None
        sig = None if c is None else (tuple(c.dofs), c.bounds.round(9).tobytes())
        if sig != self._dof_sig:  # rebuild sliders only when the DOFs or ranges change
            v0 = c.values_for_base(self.s.T_world_base) if c is not None else {}
            for h in self._dof.values():
                h.remove()
            self._dof = {}
            self._dof_sig = sig
            if c is not None:
                with self.dof_folder:
                    for n, (lo, hi) in zip(c.dofs, c.bounds, strict=True):
                        unit = "deg" if n == "yaw" else "mm"
                        conv = np.degrees if n == "yaw" else (lambda x: x * 1000)
                        sl = self.server.gui.add_slider(
                            f"{n} ({unit})",
                            float(conv(lo)),
                            float(conv(hi)),
                            1.0 if n == "yaw" else 10.0,
                            float(conv(v0.get(n, lo))),
                        )
                        sl.on_update(lambda e: self._jog_latest())
                        self._dof[n] = sl
        if c is not None:
            self._jog()
        else:
            self.draw()

    def current(self, c) -> dict:
        """DOF values to show: within the travel, except a mobile base's x / y, which has no
        limits (the travel fields of a mobile base only bound the planner's search)."""
        raw = self.values()
        v = c.clip(raw)
        if c.kind == "mobile":
            v.update({k: raw[k] for k in ("x", "y") if k in raw})
        return v

    def jog_dof(self, dof: str, delta: float) -> bool:
        """Arrow jog: move one carrier DOF by ``delta`` (m or rad). Rail / gantry travel is
        clamped to its range; a mobile base moves freely (its sliders grow as needed) and
        its yaw wraps. Returns False if the carrier has no such DOF."""
        sl = self._dof.get(dof)
        if sl is None or self.s.base.mode != "movable":
            return False
        mobile = self.s.base.carrier == "mobile"
        x = sl.value + (np.degrees(delta) if dof == "yaw" else delta * 1000)
        if dof == "yaw" and mobile:
            x = (x + 180.0) % 360.0 - 180.0
        elif mobile and dof in ("x", "y"):
            sl.min, sl.max = min(sl.min, x), max(sl.max, x)
        else:
            x = float(np.clip(x, sl.min, sl.max))
        sl.value = float(x)  # fires the jog
        return True

    def values(self) -> dict:
        return {n: (np.radians(sl.value) if n == "yaw" else sl.value / 1000) for n, sl in self._dof.items()}

    def parent_T(self, attach: str) -> np.ndarray:
        """World pose of a body's parent frame: carriage, fixed (carrier frame), the X axis (gantry
        bridge), or the lift stage = the mount parent (carriage + lift)."""
        c = self._carrier()
        if c.kind == "static":
            return np.eye(4)
        if attach == "fixed":
            return c.T_frame
        v = self.current(c)
        if attach == "x_axis":
            return c.T_world_axis(v)
        T = c.T_world_carriage(v)
        return T @ make_T(p=(0, 0, v.get("lift", 0.0))) if attach in ("mount", "lift") else T

    def _jog(self) -> None:
        if self._busy or self.s.base.mode != "movable":
            return
        c = self._carrier()
        v = self.current(c)
        self.set_base(c.T_world_base(v))
        self._move_bodies(c, v)

    def set_values(self, values: dict) -> None:
        """Move the carrier to DOF values (SI units, e.g. a planned station): sliders, robot
        base and bodies together."""
        if self.s.base.mode != "movable" or not self._dof:
            return
        self._busy = True
        try:
            for n, sl in self._dof.items():
                if n in values:
                    x = float(np.degrees(values[n]) if n == "yaw" else values[n] * 1000)
                    if self.s.base.carrier == "mobile" and n in ("x", "y"):
                        sl.min, sl.max = min(sl.min, x), max(sl.max, x)  # no limits
                    sl.value = float(np.clip(x, sl.min, sl.max))
        finally:
            self._busy = False
        self._jog()

    def _move_bodies(self, c, v: dict) -> None:
        """Jog: re-pose the existing body meshes (no mesh re-send) and update the readout.
        Falls back to a full draw when the body set changed."""
        moving = c.moving_bodies(v)
        if [b.name for b in moving] != self._moving_names:
            return self.draw()
        for body in moving:
            h = self._body_handles[body.name]
            h.wxyz = matrix_to_wxyz(body.T[:3, :3])
            h.position = body.T[:3, 3]
        self._update_info(c, v)
        self.editor.place_gizmos()

    def _update_info(self, c, v: dict) -> None:
        fp = c.footprint(v)
        if self._zones is not None:
            zc = self._zones.check(fp) if fp is not None else None
            state = "ok" if zc is None or zc.ok else f"violates {zc.reason}"
            dist = f"{zc.part_distance * 1000:.0f} mm" if zc is not None else "n/a"
            self.info.content = f"Carrier footprint to part: **{dist}** ({state})"
        else:
            self.info.content = f"{len(c.specs)} carrier bodies"

    def draw(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles, self._body_handles, self._moving_names = [], {}, []
        b = self.s.base
        zones = None
        if self.s.part is not None:
            zones = Zones(b, footprint(self.s.part_mesh(), self.s.part.T_world_part))
        self._zones = zones
        if b.mode == "movable":
            c = self._carrier()
            v = self.current(c)
            picked = self.editor.pick.value
            moving = c.moving_bodies(v)
            self._moving_names = [body.name for body in moving]
            for body in moving + c.static_bodies():
                fixed = any(p.name == body.name and p.attach == "fixed" for p in c.specs)
                color = PICKED_COLOR if body.name == picked else (FIXED_COLOR if fixed else BODY_COLOR)
                h = self.server.scene.add_mesh_simple(
                    f"{ROOT}/{body.name}",
                    body.mesh.vertices,
                    body.mesh.faces,
                    color=color,
                    opacity=0.6,
                    side="double",  # STL bodies: never see-through from one side
                    wxyz=matrix_to_wxyz(body.T[:3, :3]),
                    position=body.T[:3, 3],
                )
                self._handles.append(h)
                self._body_handles[body.name] = h
            self._update_info(c, v)
        else:
            self.info.content = "_Static base: the planner places one base (k = 1). Set Mode to movable._"
        if zones is not None:
            for name, d, color in (("min", zones.min, (220, 40, 40)), ("pref", zones.pref, (240, 170, 20))):
                ring = zones.shell(d) if d > 0 else None
                if ring is not None:
                    seg = np.stack(
                        [
                            np.column_stack([ring[:-1], np.full(len(ring) - 1, 0.004)]),
                            np.column_stack([ring[1:], np.full(len(ring) - 1, 0.004)]),
                        ],
                        axis=1,
                    )
                    self._handles.append(
                        self.server.scene.add_line_segments(
                            f"{ROOT}/standoff_{name}", seg, color, thickness=2.0, thickness_units="screen"
                        )
                    )
        for k, ko in enumerate(b.keepouts):
            (x0, y0), (x1, y1) = ko.min_xy, ko.max_xy
            self._handles.append(
                self.server.scene.add_box(
                    f"{ROOT}/keepout{k}",
                    color=(230, 60, 60),
                    dimensions=(x1 - x0, y1 - y0, 0.02),
                    position=((x0 + x1) / 2, (y0 + y1) / 2, 0.01),
                    opacity=0.35,
                )
            )
        self.editor.place_gizmos()
