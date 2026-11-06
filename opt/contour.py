"""Mobile base on a contour around the part (v2 phase 3, user decision).

The user fixes one standoff d (base outline to part outline, top view) and which side of
the base faces the part. A stop is then fully given by one number t, the arc length along
the path of the base centre around the part: the base sits at exactly d from the part with
the chosen side turned towards it. The planner treats this like a rail ("virtual rail"):
one DOF instead of x, y, yaw, so the search is far smaller and every stop keeps the same,
user-chosen safety distance.

Construction: the path is the outline of the part footprint grown by d + e, where e is how
far the base outline reaches from its centre towards the facing side. At each path point the
heading turns the facing side towards the nearest part point, and a short 1-D correction
along that direction makes the gap exactly d (corners and wide bases). The path is sampled
once (``step``, default 2 cm) and stations interpolate between samples.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from opt.sites import CarrierSites

SIDES = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


class ContourSites(CarrierSites):
    """CarrierSites for a mobile base on its fixed-standoff contour: DOF ``t`` (m)."""

    def __init__(self, carrier, part_mesh, T_world_part, step: float = 0.02):
        super().__init__(carrier, part_mesh, T_world_part)
        b = carrier.base
        if self.zones.part is None:
            raise ValueError("contour stops need a part")
        self.d = float(b.fixed_standoff_m)
        side = np.array(SIDES[b.facing])
        self.zones.min = max(self.d - 0.002, 0.0)  # 2 mm for the interpolation between samples
        self.zones.pref = self.zones.min  # one fixed standoff: no preferred-distance penalty
        self.side_angle = float(np.arctan2(side[1], side[0]))
        local = carrier._local("fp", lambda: carrier._hulls(carrier._bodies("carriage", np.eye(4))))
        reach = float(np.max(np.asarray(local.exterior.coords) @ side))  # centre -> facing edge
        self.table = self._build(reach, step)
        L = float(self.table[-1, 0])
        self.dofs = ["t"]
        self.bounds = np.array([[0.0, L]])
        self.kind = "mobile"

    # --- path ---------------------------------------------------------------------------------

    def _pose(self, c: np.ndarray, toward: np.ndarray) -> dict:
        """Base centre c, facing side turned to point ``toward`` (world xy)."""
        u = toward - c
        yaw = float(np.arctan2(u[1], u[0])) - self.side_angle
        return {"x": float(c[0]), "y": float(c[1]), "yaw": float((yaw + np.pi) % (2 * np.pi) - np.pi)}

    def _gap(self, v: dict) -> float:
        return float(self.c.footprint(v).distance(self.zones.part))

    def _build(self, reach: float, step: float) -> np.ndarray:
        """Rows (t, x, y, yaw) along the path, closed (last row = first)."""
        part = self.zones.part
        ring = part.buffer(self.d + reach, quad_segs=16).exterior
        n = max(16, int(np.ceil(ring.length / step)))
        rows = []
        for s in np.linspace(0.0, ring.length, n, endpoint=False):
            c = np.asarray(ring.interpolate(s).coords[0])
            near = np.asarray(part.exterior.interpolate(part.exterior.project(Point(c))).coords[0])
            out = (c - near) / max(np.linalg.norm(c - near), 1e-12)
            v = self._pose(c, near)
            for _ in range(8):  # slide along the normal until the gap is d (within 0.1 mm)
                err = self.d - self._gap(v)
                if abs(err) < 1e-4:
                    break
                c = c + err * out
                v = self._pose(c, near)
            rows.append((v["x"], v["y"], v["yaw"]))
        P = np.array(rows + rows[:1])
        seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        return np.column_stack([t, P])

    def full(self, v: dict) -> dict:
        if "x" in v:  # already a full pose (localization perturbations)
            return {**v, **self._lift}
        tb = self.table
        t = float(np.clip(v["t"], tb[0, 0], tb[-1, 0]))
        i = int(np.clip(np.searchsorted(tb[:, 0], t) - 1, 0, len(tb) - 2))
        w = (t - tb[i, 0]) / max(tb[i + 1, 0] - tb[i, 0], 1e-12)
        dyaw = (tb[i + 1, 3] - tb[i, 3] + np.pi) % (2 * np.pi) - np.pi
        return {
            "t": t,
            "x": float(tb[i, 1] + w * (tb[i + 1, 1] - tb[i, 1])),
            "y": float(tb[i, 2] + w * (tb[i + 1, 2] - tb[i, 2])),
            "yaw": float(tb[i, 3] + w * dyaw),
            **self._lift,
        }

    # --- planner interface ------------------------------------------------------------------------

    def candidates(self, P_world: np.ndarray, step: float, cancelled=None) -> list[dict]:
        from opt.sites import _accepted

        L = self.bounds[0, 1]
        n = max(4, int(np.ceil(L / step)))
        return _accepted(self, ({"t": float(t)} for t in np.linspace(0.0, L, n, endpoint=False)), cancelled)

    def accept(self, v: dict) -> tuple[bool, float]:
        """Keep-outs and the standoff (exact on the path, so only keep-outs usually fail)."""
        f = self.full(v)
        fp = self.c.footprint(f)
        zc = self.zones.check(fp)
        return zc.ok, 0.0

    def travel(self, a: dict, b: dict) -> float:
        return self.c.travel(self.full(a), self.full(b))

    def perturbations(self, v: dict) -> list[dict]:
        return super().perturbations(self.full(v))

    def start(self, T_world_base: np.ndarray) -> dict:
        """The path point closest to a base pose (to order the stops from the current one)."""
        v = self.c.values_for_base(T_world_base)
        k = int(np.argmin(np.hypot(self.table[:, 1] - v["x"], self.table[:, 2] - v["y"])))
        return {"t": float(self.table[k, 0])}
