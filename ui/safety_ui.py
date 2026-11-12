"""Cable dress and safety (v8 T18).

- Optimize -> Cable & safety: *Hose pack* = none, a clearance margin around the arm from the
  upper arm on (core.collision.DRESS_M) or the hose as a shape on the arm (core.hose,
  core.collision.HOSE), and *Joint 6 turn limit* (per stop; beyond it the cost term *Cost per
  90° of joint-6 wrap* applies).
- Results -> Safety: the reach envelope of each stop of the selected plan (a sphere around the
  shoulder: the robot's reach plus the tool) and a suggested fence: the envelopes' floor
  footprint grown by *Fence distance*, drawn on the floor and listed as corner coordinates.
- ``reduced_speed_fn``: the reduced-speed zones (core.speed_zones); the cycle time of built
  paths stretches the moves that end inside one.
"""

from __future__ import annotations

import numpy as np
import viser

ROOT = "/results/safety"


HOSE_MODES = ("none", "clearance margin", "hose along the arm")


class CablePanel:
    """Optimize -> Cable & safety: the hose / dress pack and the joint-6 turn limit.

    *Hose pack*: **none**; **clearance margin** (v8 T18: extra clearance around the arm from the
    upper arm on, core.collision.DRESS_M); **hose along the arm** (a hose of the given diameter
    riding ``offset`` from the arm's joint line on the chosen side, core.hose: checked as a
    shape against the part, stay-out zones, floor and carrier, drawn on the robot)."""

    def __init__(self, server: viser.ViserServer, session, cell=None):
        from core import collision

        self.s, self.server, self.cell = session, server, cell
        self._handles: list = []
        self._view = None  # the robot view the hose was drawn on
        gui = server.gui
        self.folder = gui.add_folder("Cable & safety", expand_by_default=False)
        with self.folder:
            gui.add_markdown(
                "_Hose / dress pack on the arm and the cable wrap at joint 6. Reduced-speed zones are "
                "under Robot & cell → Reduced-speed zones; the reach envelope and a fence line are "
                "under Results → Safety._"
            )
            self.mode = gui.add_dropdown(
                "Hose pack",
                HOSE_MODES,
                initial_value=HOSE_MODES[1] if collision.DRESS_M > 0 else HOSE_MODES[0],
                hint="a clearance margin around the arm, or the hose as a shape on the arm",
            )
            self.dress = gui.add_number(
                "Hose pack clearance (mm)",
                1000 * collision.DRESS_M,
                min=0.0,
                max=300.0,
                step=5.0,
                hint="extra clearance around the upper arm, forearm and wrist (part, stay-out zones)",
            )
            self.diameter = gui.add_number("Hose diameter (mm)", 60.0, min=5.0, max=300.0, step=5.0)
            self.offset = gui.add_number(
                "Hose offset from the arm (mm)",
                150.0,
                min=0.0,
                max=600.0,
                step=10.0,
                hint="from the line through the joints to the hose axis",
            )
            self.side = gui.add_dropdown(
                "Hose side", ("+", "−"), hint="which side of the arm the hose runs on (see it on the robot)"
            )
            self.show = gui.add_checkbox("Show hose on the robot", True)
            self.wrap = gui.add_number(
                "Joint 6 turn limit (deg)",
                360.0,
                min=0.0,
                max=1440.0,
                step=30.0,
                hint="per stop; beyond it the cost term 'Cost per 90° of joint-6 wrap' applies",
            )
        for w in (self.mode, self.dress, self.diameter, self.offset, self.side, self.show):
            w.on_update(lambda _e: self.commit())
        self.commit()

    def hose(self):
        """core.hose.Hose of the fields, or None (mode is not *hose along the arm*)."""
        if self.mode.value != HOSE_MODES[2]:
            return None
        from core.hose import Hose

        return Hose(
            radius=float(self.diameter.value) / 2000,
            offset=float(self.offset.value) / 1000,
            side=1.0 if self.side.value == "+" else -1.0,
        )

    def commit(self) -> None:
        from core import collision

        margin = self.mode.value == HOSE_MODES[1]
        hose = self.mode.value == HOSE_MODES[2]
        self.dress.visible = margin
        for w in (self.diameter, self.offset, self.side, self.show):
            w.visible = hose
        collision.DRESS_M = float(self.dress.value) / 1000 if margin else 0.0
        collision.HOSE = self.hose()
        self.draw()

    def draw(self) -> None:
        """The hose capsules as children of the robot's joint frames (they move with the arm)."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # already gone with a removed robot view
                pass
        self._handles = []
        view = getattr(self.cell, "view", None) if self.cell is not None else None
        self._view = view
        hose = self.hose()
        if hose is None or not self.show.value or view is None or self.s.robot is None:
            return
        import trimesh

        from core.hose import segments
        from core.transforms import matrix_to_wxyz

        names = self.s.robot.model.names
        for k, (j, T, L) in enumerate(segments(self.s.robot, hose)):
            frame = view.joint_frame(names[j])
            if frame is None:
                continue
            m = trimesh.creation.capsule(height=L, radius=hose.radius)
            self._handles.append(
                self.server.scene.add_mesh_simple(
                    f"{frame.name}/hose_{k}",
                    m.vertices,
                    m.faces,
                    color=(40, 40, 40),
                    opacity=0.85,
                    wxyz=matrix_to_wxyz(T[:3, :3]),
                    position=T[:3, 3],
                )
            )

    def follow_robot(self) -> None:
        """Watcher hook: redraw after the robot view was rebuilt (another robot, a reload)."""
        view = getattr(self.cell, "view", None) if self.cell is not None else None
        if view is not self._view:
            self.draw()


def reach_spheres(s, sol) -> list[tuple[np.ndarray, float]]:
    """(centre, radius) per stop: around the shoulder (base + d1 up), reach + tool length."""
    from core.export import station_pose

    reach = float(getattr(s.robot.spec, "reach_m", None) or 1.0)
    tool = float(s.tool_model.length(sol.tool_values)) if s.tool_model is not None else 0.0
    head = 0.15 if s.tool_model is not None else 0.0  # the head beyond the extension, roughly
    d1 = float(s.robot.spec.dh.d1) if getattr(s.robot.spec, "dh", None) is not None else 0.2
    out = []
    for st in sol.stations:
        T = station_pose(s.base, st, s.T_world_base)
        out.append((T[:3, 3] + T[:3, 2] * d1, reach + tool + head))
    return out


def fence(spheres, distance: float) -> np.ndarray:
    """Floor rectangle (4, 2): the spheres' xy bounding box grown by ``distance``."""
    c = np.array([p for p, _ in spheres])
    r = np.array([r for _, r in spheres])
    lo, hi = (c[:, :2] - r[:, None]).min(axis=0) - distance, (c[:, :2] + r[:, None]).max(axis=0) + distance
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


class SafetyView:
    def __init__(self, server: viser.ViserServer, session, selected):
        self.s, self.server, self.selected = session, server, selected
        gui = server.gui
        with gui.add_folder("Safety", expand_by_default=False):
            self.show = gui.add_checkbox("Show reach envelope", False, hint="per stop of the selected plan")
            self.distance = gui.add_number("Fence distance (mm)", 500.0, min=0.0, step=50.0)
            self.info = gui.add_markdown("")
        self._handles: list = []
        for w in (self.show, self.distance):
            w.on_update(lambda _e: self.refresh())

    def refresh(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        sol = self.selected()
        if sol is None or self.s.robot is None:
            self.info.content = "_Select a plan in the list._"
            return
        sph = reach_spheres(self.s, sol)
        F = fence(sph, float(self.distance.value) / 1000)
        rows = " · ".join(f"({x * 1000:.0f}, {y * 1000:.0f})" for x, y in F)
        size = (F[1, 0] - F[0, 0], F[2, 1] - F[1, 1])
        self.info.content = (
            f"Reach envelope: {len(sph)} stop(s), radius {sph[0][1] * 1000:.0f} mm (robot reach + tool).  \n"
            f"**Suggested fence** ({size[0]:.2f} × {size[1]:.2f} m, corners in mm): {rows}  \n"
            "_A layout aid, not a safety assessment: the risk assessment (ISO 10218-2, ISO 13855 "
            "separation distances) decides the real guarding._"
        )
        if not self.show.value:
            return
        import trimesh

        for k, (c, r) in enumerate(sph):
            m = trimesh.creation.icosphere(subdivisions=3, radius=r)
            self._handles.append(
                self.server.scene.add_mesh_simple(
                    f"{ROOT}/envelope_{k}", m.vertices + c, m.faces, color=(250, 170, 40), opacity=0.12
                )
            )
        P = np.c_[F, np.full(4, 0.002)]
        seg = np.stack([P, np.roll(P, -1, axis=0)], axis=1)
        self._handles.append(
            self.server.scene.add_line_segments(f"{ROOT}/fence", seg, colors=(250, 120, 20), line_width=4.0)
        )


def reduced_speed_fn(s, plan):
    """fn(sample) -> speed share for the cycle time, or None when there are no slow zones
    (core.speed_zones: the TCP's world position)."""
    from core.speed_zones import share_at, speed_zones

    zones = speed_zones(s.stayouts, s.resolve)
    if not zones or s.part is None:
        return None
    Tp = s.part.T_world_part

    def share(sample) -> float:
        return float(share_at(zones, (Tp @ sample.T_part_tcp)[:3, 3])[0])

    return share
