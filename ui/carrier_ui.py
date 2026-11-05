"""Cell tab → Base mode (M9): static or movable base, carrier built from primitives,
standoffs, keep-outs, localization and tip-over settings, DOF sliders to jog the carrier.

Edits are collected in widgets and committed with "Apply" (validated as schema.Base).
Scene: carrier bodies under /cell/carrier, standoff shells (min red, preferred amber) and
keep-out boxes on the floor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import viser
from pydantic import ValidationError

from core.carrier import Carrier
from core.schema import Base, CarrierJoints, Keepout, Pose, Primitive, Standoffs
from core.standoff import Zones, footprint
from core.transforms import matrix_to_wxyz
from ui.session import Session
from ui.widgets import notify

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
        chassis=Primitive(shape="box", size=(1.2, 0.8, 0.4), kg=300),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.3),
        T_mount=Pose(xyz=(0.2, 0, 0.7)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-4.0, 4.0), yaw_deg=(-180, 180)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.5),
        localization_m=0.01,
        localization_deg=1.0,
    ),
}


def _mm(v) -> tuple:
    return tuple(float(x) * 1000 for x in v)


class CarrierPanel:
    def __init__(self, server: viser.ViserServer, session: Session, set_base: Callable[[np.ndarray], None]):
        self.server = server
        self.s = session
        self.set_base = set_base
        self._handles: list = []
        self._dof: dict = {}
        self._busy = False
        gui = server.gui
        with gui.add_folder("Base mode", expand_by_default=False):
            self.mode = gui.add_dropdown("Mode", ("static", "movable"))
            self.kind = gui.add_dropdown("Carrier", ("mobile", "rail", "gantry"))
            self.defaults = gui.add_button("Load carrier defaults")
            self.frame_xyz = gui.add_vector3("Track/gantry frame xyz (mm)", (0.0, 0.0, 0.0), step=10.0)
            self.frame_yaw = gui.add_number("Track/gantry yaw (deg)", 0.0, step=1.0)
            self.tx = gui.add_vector2("Travel x / s (mm)", (-3000.0, 3000.0), step=10.0)
            self.ty = gui.add_vector2("Travel y (mm)", (-1000.0, 1000.0), step=10.0)
            self.tyaw = gui.add_vector2("Yaw range (deg)", (-180.0, 180.0), step=5.0)
            self.chassis = gui.add_vector3("Chassis/carriage L W H (mm)", (1200.0, 800.0, 400.0), step=10.0)
            self.chassis_kg = gui.add_number("Chassis mass (kg, 0 = unknown)", 0.0, min=0.0, step=10.0)
            self.riser = gui.add_vector2("Riser radius / height (mm)", (150.0, 300.0), step=10.0)
            self.track = gui.add_vector3("Rail track L W H (mm)", (7000.0, 300.0, 100.0), step=10.0)
            self.mount_xyz = gui.add_vector3("Mount xyz (mm)", (200.0, 0.0, 700.0), step=10.0)
            self.mount_rpy = gui.add_vector3("Mount rpy (deg)", (0.0, 0.0, 0.0), step=5.0)
            self.so = gui.add_vector2("Part standoff min / preferred (mm)", (300.0, 400.0), step=10.0)
            self.env = gui.add_number("Keep-out clearance (mm)", 100.0, min=0.0, step=10.0)
            self.keepouts = gui.add_text("Keep-outs (mm)", "", hint="x0,y0,x1,y1; x0,y0,x1,y1 (floor boxes)")
            self.loc = gui.add_vector2("Localization ±mm / ±deg (mobile)", (10.0, 1.0), step=1.0)
            self.tip = gui.add_checkbox("Tip-over check (mobile, needs mass)", False)
            self.apply = gui.add_button("Apply base settings")
            self.dof_folder = gui.add_folder("Carrier position")
            self.info = gui.add_markdown("")
        self.apply.on_click(lambda e: self.commit(e))

        @self.defaults.on_click
        def _(e) -> None:
            self.load(DEFAULTS[self.kind.value].model_copy(update={"mode": "movable"}))

        self.load(session.base)
        self.refresh()

    # --- widgets <-> schema ------------------------------------------------------------------

    def load(self, b: Base) -> None:
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
            if b.chassis is not None and b.chassis.shape == "box":
                self.chassis.value = _mm(b.chassis.size)
                self.chassis_kg.value = b.chassis.kg or 0.0
            if b.riser is not None and b.riser.shape == "cylinder":
                self.riser.value = _mm((b.riser.radius, b.riser.height))
            if b.track is not None and b.track.shape == "box":
                self.track.value = _mm(b.track.size)
            self.mount_xyz.value = _mm(b.T_mount.xyz)
            self.mount_rpy.value = tuple(float(np.degrees(x)) for x in b.T_mount.rpy)
            so = b.standoff_m
            self.so.value = _mm((so.carrier_part_min, so.carrier_part_preferred or so.carrier_part_min))
            self.env.value = so.carrier_env_min * 1000
            self.keepouts.value = "; ".join(
                ",".join(f"{v * 1000:.0f}" for v in (*k.min_xy, *k.max_xy)) for k in b.keepouts
            )
            self.loc.value = (b.localization_m * 1000, b.localization_deg)
            self.tip.value = b.tip_over
        finally:
            self._busy = False

    def collect(self) -> Base:
        kind = self.kind.value
        movable = self.mode.value == "movable"
        kos = []
        for chunk in filter(None, (c.strip() for c in self.keepouts.value.split(";"))):
            x0, y0, x1, y1 = (float(v) / 1000 for v in chunk.split(","))
            kos.append(Keepout(min_xy=(min(x0, x1), min(y0, y1)), max_xy=(max(x0, x1), max(y0, y1))))
        L, W, H = (v / 1000 for v in self.chassis.value)
        kg = self.chassis_kg.value or None
        r, h = (v / 1000 for v in self.riser.value)
        smin, spref = (v / 1000 for v in self.so.value)
        joints = CarrierJoints(
            x=tuple(v / 1000 for v in self.tx.value),
            y=tuple(v / 1000 for v in self.ty.value) if kind in ("mobile", "gantry") else None,
            yaw_deg=tuple(self.tyaw.value) if kind == "mobile" else None,
        )
        return Base(
            mode="movable" if movable else "static",
            carrier=kind if movable else None,
            pose=Pose(
                xyz=tuple(v / 1000 for v in self.frame_xyz.value),
                rpy=(0, 0, np.radians(self.frame_yaw.value)),
            ),
            chassis=Primitive(shape="box", size=(L, W, H), kg=kg),
            riser=Primitive(shape="cylinder", radius=r, height=h) if r > 0 and h > 0 else None,
            track=Primitive(shape="box", size=tuple(v / 1000 for v in self.track.value))
            if kind == "rail"
            else None,
            joints=joints,
            T_mount=Pose(
                xyz=tuple(v / 1000 for v in self.mount_xyz.value),
                rpy=tuple(np.radians(v) for v in self.mount_rpy.value),
            ),
            keepouts=kos,
            standoff_m=Standoffs(
                carrier_part_min=smin, carrier_part_preferred=spref, carrier_env_min=self.env.value / 1000
            ),
            localization_m=self.loc.value[0] / 1000,
            localization_deg=self.loc.value[1],
            tip_over=self.tip.value,
        )

    def commit(self, event=None) -> bool:
        try:
            base = self.collect()
        except (ValueError, ValidationError) as err:
            msg = err.errors()[0]["msg"] if isinstance(err, ValidationError) else str(err)
            notify(event, "Invalid base settings", msg)
            return False
        self.s.base = base
        self.refresh()
        return True

    # --- DOF sliders and drawing ---------------------------------------------------------------

    def refresh(self) -> None:
        for h in self._dof.values():
            h.remove()
        self._dof = {}
        b = self.s.base
        if b.mode == "movable":
            c = Carrier(b, self.s.T_world_base)
            v0 = c.values_for_base(self.s.T_world_base)
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
                    sl.on_update(lambda e: self._jog())
                    self._dof[n] = sl
            self._jog()
        else:
            self.draw()

    def values(self) -> dict:
        return {n: (np.radians(sl.value) if n == "yaw" else sl.value / 1000) for n, sl in self._dof.items()}

    def _jog(self) -> None:
        if self._busy or self.s.base.mode != "movable":
            return
        c = Carrier(self.s.base, self.s.T_world_base)
        self.set_base(c.T_world_base(c.clip(self.values())))
        self.draw()

    def draw(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        b = self.s.base
        zones = None
        if self.s.part is not None:
            zones = Zones(b, footprint(self.s.part_mesh(), self.s.part.T_world_part))
        if b.mode == "movable":
            c = Carrier(b, self.s.T_world_base)
            v = c.clip(self.values())
            for body in c.moving_bodies(v) + c.static_bodies():
                self._handles.append(
                    self.server.scene.add_mesh_simple(
                        f"{ROOT}/{body.name}",
                        body.mesh.vertices,
                        body.mesh.faces,
                        color=(120, 130, 150),
                        opacity=0.6,
                        wxyz=matrix_to_wxyz(body.T[:3, :3]),
                        position=body.T[:3, 3],
                    )
                )
            if zones is not None:
                fp = c.footprint(v)
                zc = zones.check(fp) if fp is not None else None
                state = "ok" if zc is None or zc.ok else f"violates {zc.reason}"
                dist = f"{zc.part_distance * 1000:.0f} mm" if zc is not None else "n/a"
                self.info.content = f"Carrier footprint to part: **{dist}** ({state})"
        else:
            self.info.content = "_Static base: the planner places one base (k = 1)._"
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
