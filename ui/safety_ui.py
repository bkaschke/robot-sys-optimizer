"""Cable dress and safety (v8 T18).

- Optimize -> Cable & safety: *Hose pack clearance* (extra clearance around the arm from the
  upper arm on, against the part and stay-out zones: core.collision.DRESS_M) and *Joint 6 turn
  limit* (per stop; beyond it the cost term *Cost per 90° of joint-6 wrap* applies).
- Results -> Safety: the reach envelope of each stop of the selected plan (a sphere around the
  shoulder: the robot's reach plus the tool) and a suggested fence: the envelopes' floor
  footprint grown by *Fence distance*, drawn on the floor and listed as corner coordinates.
- ``reduced_speed_fn``: stay-out zones with a speed (%) are reduced-speed zones; the cycle time
  of built paths stretches the moves that end inside one.
"""

from __future__ import annotations

import numpy as np
import viser

ROOT = "/results/safety"


class CablePanel:
    def __init__(self, server: viser.ViserServer, session):
        from core import collision

        self.s = session
        gui = server.gui
        with gui.add_folder("Cable & safety", expand_by_default=False):
            gui.add_markdown(
                "_Hose / dress pack on the arm and the cable wrap at joint 6. Reduced-speed zones are "
                "stay-out zones with a speed (Robot & cell → Stay-out zones); the reach envelope "
                "and a fence line are under Results → Safety._"
            )
            self.dress = gui.add_number(
                "Hose pack clearance (mm)",
                1000 * collision.DRESS_M,
                min=0.0,
                max=300.0,
                step=5.0,
                hint="extra clearance around the upper arm, forearm and wrist (part, stay-out zones)",
            )
            self.wrap = gui.add_number(
                "Joint 6 turn limit (deg)",
                360.0,
                min=0.0,
                max=1440.0,
                step=30.0,
                hint="per stop; beyond it the cost term 'Cost per 90° of joint-6 wrap' applies",
            )
        self.dress.on_update(lambda _e: self.commit())
        self.commit()

    def commit(self) -> None:
        from core import collision

        collision.DRESS_M = float(self.dress.value) / 1000


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
    """fn(sample) -> speed share for the cycle time, or None when there are no slow zones."""
    zones = [z for z in s.stayouts if z.speed_pct is not None]
    if not zones or s.part is None:
        return None
    from core.carrier import primitive_mesh
    from core.transforms import pose_to_matrix

    meshes = []
    for z in zones:
        m = primitive_mesh(z, s.resolve).copy()
        m.apply_transform(pose_to_matrix(z.pose))
        meshes.append((m, z.speed_pct / 100.0))
    Tp = s.part.T_world_part

    def share(sample) -> float:
        p = (Tp @ sample.T_part_tcp)[:3, 3]
        f = 1.0
        for m, k in meshes:
            if m.contains(p[None])[0]:
                f = min(f, k)
        return f

    return share
