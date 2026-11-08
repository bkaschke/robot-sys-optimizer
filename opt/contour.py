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
    """CarrierSites for a mobile base on its fixed-standoff contour: DOF ``t`` (m).

    One path per robot side allowed to face the part (``Base.robot_sides``), laid end to end
    in ``t``; samples along part sides that are not allowed (``Base.part_sides``, part frame)
    are kept in the table but never accepted."""

    def __init__(self, carrier, part_mesh, T_world_part, step: float = 0.02, obstacles=None):
        super().__init__(carrier, part_mesh, T_world_part, obstacles)
        b = carrier.base
        if self.zones.part is None:
            raise ValueError("contour stops need a part")
        self.d = float(b.fixed_standoff_m)
        self.zones.min = max(self.d - 0.002, 0.0)  # 2 mm for the interpolation between samples
        self.zones.pref = self.zones.min  # one fixed standoff: no preferred-distance penalty
        self.part_sides = set(b.part_sides)
        R = np.asarray(T_world_part, float)[:3, :3] if T_world_part is not None else np.eye(3)
        self._R_part = R
        local = carrier._local("fp", lambda: carrier._hulls(carrier._bodies("carriage", np.eye(4))))
        tables, t0 = [], 0.0
        self.sides = b.robot_sides()
        for k, name in enumerate(self.sides):
            side = np.array(SIDES[name])
            reach = float(np.max(np.asarray(local.exterior.coords) @ side))  # centre -> facing edge
            tb = self._build(reach, step, float(np.arctan2(side[1], side[0])))
            tb[:, 0] += t0
            t0 = float(tb[-1, 0]) + 1e-6  # segments end to end (no interpolation across them)
            tables.append(np.column_stack([tb, np.full(len(tb), k)]))
        self.table = np.vstack(tables)  # t, x, y, yaw, allowed, segment
        self.dofs = ["t"]
        self.bounds = np.array([[0.0, float(self.table[-1, 0])]])
        self.kind = "mobile"

    # --- path ---------------------------------------------------------------------------------

    def _pose(self, c: np.ndarray, toward: np.ndarray, side_angle: float) -> dict:
        """Base centre c, the facing side (at ``side_angle`` in the base) turned to ``toward``."""
        u = toward - c
        yaw = float(np.arctan2(u[1], u[0])) - side_angle
        return {"x": float(c[0]), "y": float(c[1]), "yaw": float((yaw + np.pi) % (2 * np.pi) - np.pi)}

    def _gap(self, v: dict) -> float:
        return float(self.c.footprint(v).distance(self.zones.part))

    def _part_side(self, out: np.ndarray) -> str:
        """Which side of the part (its own frame) a world floor direction points to."""
        n = self._R_part.T @ np.array([out[0], out[1], 0.0])
        if abs(n[0]) >= abs(n[1]):
            return "+x" if n[0] > 0 else "-x"
        return "+y" if n[1] > 0 else "-y"

    def _build(self, reach: float, step: float, side_angle: float) -> np.ndarray:
        """Rows (t, x, y, yaw, allowed) along one side's path, closed (last row = first)."""
        part = self.zones.part
        ring = part.buffer(self.d + reach, quad_segs=16).exterior
        n = max(16, int(np.ceil(ring.length / step)))
        rows = []
        for s in np.linspace(0.0, ring.length, n, endpoint=False):
            c = np.asarray(ring.interpolate(s).coords[0])
            near = np.asarray(part.exterior.interpolate(part.exterior.project(Point(c))).coords[0])
            out = (c - near) / max(np.linalg.norm(c - near), 1e-12)
            v = self._pose(c, near, side_angle)
            for _ in range(8):  # slide along the normal until the gap is d (within 0.1 mm)
                err = self.d - self._gap(v)
                if abs(err) < 1e-4:
                    break
                c = c + err * out
                v = self._pose(c, near, side_angle)
            ok = not self.part_sides or self._part_side(out) in self.part_sides
            rows.append((v["x"], v["y"], v["yaw"], 1.0 if ok else 0.0))
        P = np.array(rows + rows[:1])
        seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        return np.column_stack([t, P])

    def _row(self, t: float) -> int:
        tb = self.table
        return int(np.clip(np.searchsorted(tb[:, 0], t) - 1, 0, len(tb) - 2))

    def full(self, v: dict) -> dict:
        if "x" in v:  # already a full pose (localization perturbations)
            return {**v, **self._lift}
        tb = self.table
        t = float(np.clip(v["t"], tb[0, 0], tb[-1, 0]))
        i = self._row(t)
        j = i + 1 if tb[i + 1, 5] == tb[i, 5] else i  # never interpolate across two sides
        w = (t - tb[i, 0]) / max(tb[j, 0] - tb[i, 0], 1e-12) if j != i else 0.0
        dyaw = (tb[j, 3] - tb[i, 3] + np.pi) % (2 * np.pi) - np.pi
        return {
            "t": t,
            "x": float(tb[i, 1] + w * (tb[j, 1] - tb[i, 1])),
            "y": float(tb[i, 2] + w * (tb[j, 2] - tb[i, 2])),
            "yaw": float(tb[i, 3] + w * dyaw),
            **self._lift,
        }

    def side_at(self, v: dict) -> str:
        """Robot side facing the part at stop v."""
        return self.sides[int(self.table[self._row(float(v["t"])), 5])]

    # --- planner interface ------------------------------------------------------------------------

    def candidates(self, P_world: np.ndarray, step: float, cancelled=None) -> list[dict]:
        from opt.sites import _accepted

        L = self.bounds[0, 1]
        n = max(4, int(np.ceil(L / step)))
        return _accepted(self, ({"t": float(t)} for t in np.linspace(0.0, L, n, endpoint=False)), cancelled)

    def accept(self, v: dict) -> tuple[bool, float]:
        """Allowed part side, keep-outs, stay-out zones and the standoff (exact on the path)."""
        if "t" in v and self.table[self._row(float(v["t"])), 4] < 0.5:
            return False, 1.0  # along a part side the base may not work from
        f = self.full(v)
        fp = self.c.footprint(f)
        zc = self.zones.check(fp)
        return zc.ok and self.clear_of_obstacles(f), 0.0

    def travel(self, a: dict, b: dict) -> float:
        return self.c.travel(self.full(a), self.full(b))

    def transit(self, a: dict, b: dict, step: float) -> list[tuple[dict, bool]]:
        """Along the contour path when both stops face the part with the same robot side;
        otherwise a straight move (x, y, yaw)."""
        A, B = self.full(a), self.full(b)
        if "t" in A and "t" in B and self.table[self._row(A["t"]), 5] == self.table[self._row(B["t"]), 5]:
            n = max(1, int(np.ceil(abs(B["t"] - A["t"]) / step)))
            ts = A["t"] + (B["t"] - A["t"]) * np.arange(1, n + 1) / n
            return [(f, self.transit_ok(f)) for f in (self.full({"t": float(t)}) for t in ts)]
        A = {k: v for k, v in A.items() if k != "t"}
        B = {k: v for k, v in B.items() if k != "t"}
        return super().transit(A, B, step)

    def transit_ok(self, v: dict) -> bool:
        """Keep-outs, part standoff and stay-out zones (any part side may be passed)."""
        f = self.full(v)
        return self.zones.check(self.c.footprint(f)).ok and self.clear_of_obstacles(f)

    def perturbations(self, v: dict) -> list[dict]:
        return super().perturbations(self.full(v))

    def start(self, T_world_base: np.ndarray) -> dict:
        """The path point closest to a base pose (to order the stops from the current one)."""
        v = self.c.values_for_base(T_world_base)
        k = int(np.argmin(np.hypot(self.table[:, 1] - v["x"], self.table[:, 2] - v["y"])))
        return {"t": float(self.table[k, 0])}
