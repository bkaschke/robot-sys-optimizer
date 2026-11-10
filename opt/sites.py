"""Where stations may go: the search space of the placement layer (plan § Base mode).

Every carrier is "a small set of extra DOFs that place the robot base". A Sites object
exposes that space to the planner:
    dofs / bounds         carrier DOF names and ranges
    candidates(t, st)     DOF dicts to seed the set cover
    T(v)                  robot base pose (world) for DOF values v
    accept(v)             (ok, soft penalty) from standoff / keep-out zones
    travel(a, b)          carrier travel metric between two stations

FloorSites is the M7 behaviour (free floor positions, part AABB keep-out).
CarrierSites wraps core.carrier.Carrier with core.standoff zones.
"""

from __future__ import annotations

import numpy as np
import trimesh

from core.carrier import Carrier
from core.standoff import Zones, carrier_ok, footprint
from core.transforms import make_T


def _accepted(sites, samples, cancelled=None) -> list[dict]:
    """Samples passing ``sites.accept``; polls ``cancelled`` every 256 samples."""
    from core.feasibility import CancelledError

    out = []
    for i, v in enumerate(samples):
        if cancelled is not None and i % 256 == 0 and cancelled():
            raise CancelledError
        if sites.accept(v)[0]:
            out.append(v)
    return out


class CarriageFollow:
    """Pose of a carriage body in the robot base frame for any base pose of a carrier with a
    lift: the base is carriage @ Tz(lift) @ mount and the carriage height is the same at every
    stop, so the lift is read from the base pose (core.collision re-places the body)."""

    def __init__(self, carrier: Carrier, T_carriage_body: np.ndarray):
        self.M_inv = np.linalg.inv(carrier.T_mount)
        self.T_cb = T_carriage_body
        v = {d: float(np.mean(b)) for d, b in zip(carrier.dofs, carrier.bounds, strict=True)}
        Tc = carrier.T_world_carriage(v)
        self.h0 = float(Tc[:3, 3] @ Tc[:3, 2])  # carriage origin along its own z axis

    def __call__(self, T_world_base: np.ndarray) -> np.ndarray:
        T = T_world_base @ self.M_inv  # the carriage frame raised by the lift
        lift = float(T[:3, 3] @ T[:3, 2]) - self.h0
        return self.M_inv @ make_T(p=(0, 0, -lift)) @ self.T_cb


class FloorSites:
    """Robot base anywhere on the floor at fixed height/orientation (no carrier model)."""

    kind = "floor"
    localization = (0.0, 0.0)

    def __init__(
        self, T_world_base0: np.ndarray, footprint_xy: np.ndarray | None, margin: float, half: float
    ):
        self.z, self.R = float(T_world_base0[2, 3]), T_world_base0[:3, :3]
        self.fp, self.margin, self.half = footprint_xy, margin, half
        self.dofs = ["x", "y"]
        self.bounds = None  # set from the targets in candidates()

    def candidates(self, P_world: np.ndarray, step: float, cancelled=None) -> list[dict]:
        lo = P_world[:, :2].min(axis=0) - self.half
        hi = P_world[:, :2].max(axis=0) + self.half
        self.bounds = np.array([[lo[0], hi[0]], [lo[1], hi[1]]])
        xs = np.arange(lo[0], hi[0] + 1e-9, step)
        ys = np.arange(lo[1], hi[1] + 1e-9, step)
        return _accepted(self, ({"x": float(x), "y": float(y)} for y in ys for x in xs), cancelled)

    def T(self, v: dict) -> np.ndarray:
        return make_T(self.R, (v["x"], v["y"], self.z))

    def accept(self, v: dict) -> tuple[bool, float]:
        if self.fp is None:
            return True, 0.0
        xy = np.array([v["x"], v["y"]])
        inside = np.all((xy >= self.fp[0] - self.margin) & (xy <= self.fp[1] + self.margin))
        return (not inside), 0.0

    def travel(self, a: dict, b: dict) -> float:
        return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))

    def perturbations(self, v: dict) -> list[dict]:
        return []


class CarrierSites:
    """Stations of a rail / gantry / mobile carrier, filtered by standoff and keep-outs."""

    def __init__(
        self,
        carrier: Carrier,
        part_mesh: trimesh.Trimesh | None,
        T_world_part: np.ndarray | None,
        obstacles: list | None = None,
    ):
        self.c = carrier
        self.obstacles = list(obstacles or [])  # core.stayout.Obstacle
        self.kind = carrier.kind
        self.part, self.T_part = part_mesh, T_world_part
        fp = footprint(part_mesh, T_world_part) if part_mesh is not None else None
        self.zones = Zones(carrier.base, fp)
        # v6: the lift is planned per stop like the other carrier axes
        self.dofs = list(carrier.dofs)
        self.bounds = np.array(carrier.bounds, float).reshape(-1, 2)
        b = carrier.base
        self.localization = (
            (b.localization_m, np.radians(b.localization_deg)) if self.kind == "mobile" else (0.0, 0.0)
        )
        self._lift = (
            {"lift": float(np.mean(carrier.bounds[carrier.dofs.index("lift")]))}
            if "lift" in carrier.dofs
            else {}
        )

    def scene_bodies(self) -> tuple[list, list]:
        """(base_bodies, world_bodies) for core.collision.CollisionScene: the carriage bodies
        and the lift stage ride with the robot base (with a lift, the carriage bodies get a
        4th entry that re-places them for each base height: the lift moves the robot, not the
        carriage); the rail track is fixed in the world."""
        v = self.full({d: float(np.mean(b)) for d, b in zip(self.dofs, self.bounds, strict=True)})
        T_base_inv = np.linalg.inv(self.c.T_world_base(v))
        Tc = self.c.T_world_carriage(v)
        lifted = "lift" in self.c.dofs
        base = []
        for b in self.c._bodies("carriage", Tc):
            entry = (b.name, b.mesh, T_base_inv @ b.T)
            base.append((*entry, CarriageFollow(self.c, np.linalg.inv(Tc) @ b.T)) if lifted else entry)
        lift_T = Tc @ make_T(p=(0, 0, v.get("lift", 0.0)))
        base += [(b.name, b.mesh, T_base_inv @ b.T) for b in self.c._bodies("lift", lift_T)]
        Ta = self.c.T_world_axis(v)  # gantry bridge: moves in x only
        for b in self.c._bodies("x_axis", Ta):
            base.append((b.name, b.mesh, T_base_inv @ b.T, self.c.axis_follow(np.linalg.inv(Ta) @ b.T)))
        world = [(b.name, b.mesh, b.T) for b in self.c.static_bodies()]
        return base, world

    def full(self, v: dict) -> dict:
        """DOF values with the lift filled in (its mid height when v has none)."""
        return {**self._lift, **v}

    def candidates(self, P_world: np.ndarray, step: float, cancelled=None) -> list[dict]:
        return _accepted(self, self.c.samples(step), cancelled)

    def T(self, v: dict) -> np.ndarray:
        return self.c.T_world_base(self.full(v))

    def accept(self, v: dict) -> tuple[bool, float]:
        v = self.full(v)
        if any(
            not lo - 1e-9 <= v[d] <= hi + 1e-9 for d, (lo, hi) in zip(self.c.dofs, self.c.bounds, strict=True)
        ):
            return False, 1.0
        zc = carrier_ok(self.c, self.zones, v, self.part, self.T_part)
        if zc.ok and not self.clear_of_obstacles(v):
            return False, 1.0
        return zc.ok, zc.penalty

    def clear_of_obstacles(self, v: dict, clearance: float | None = None) -> bool:
        """The carrier at DOF values v keeps ``clearance`` (default: the keep-out clearance)
        from every stay-out zone."""
        if not self.obstacles:
            return True
        from core.stayout import carrier_clear

        v = self.full(v)
        bodies = self.c.moving_bodies(v) + self.c.static_bodies()
        if clearance is None:
            clearance = self.c.base.standoff_m.carrier_env_min
        ok, _ = carrier_clear(bodies, self.c.footprint(v), self.obstacles, clearance)
        return ok

    @property
    def nav_clearance(self) -> float:
        """Clearance kept from stay-out zones while a mobile base drives between stops: its
        part standoff (the fixed contour standoff, else the minimum), at least the keep-out
        clearance. Rails and gantries keep the keep-out clearance."""
        so = self.c.base.standoff_m
        if self.kind != "mobile":
            return so.carrier_env_min
        d = getattr(self, "d", None)
        return max(so.carrier_env_min, float(d if d is not None else so.carrier_part_min))

    def travel(self, a: dict, b: dict) -> float:
        return self.c.travel(a, b)

    def transit(self, a: dict, b: dict, step: float) -> list[tuple[dict, bool]]:
        """Carrier poses moving from stop a to stop b (b included, a not), every ``step`` of
        travel, each with whether it is clear (v4 plan item 4: the arm waits at home). A
        mobile base whose straight move is blocked drives around the zones (``route``)."""
        A, B = self.full(a), self.full(b)
        keys = [d for d in self.c.dofs if d in A and d in B]
        n = max(1, int(np.ceil(self.travel(A, B) / step)))
        out = []
        for s in np.arange(1, n + 1) / n:
            v = {d: B[d] for d in keys}
            for d in keys:
                dv = B[d] - A[d]
                if d == "yaw":
                    dv = (dv + np.pi) % (2 * np.pi) - np.pi
                v[d] = float(A[d] + s * dv)
            out.append((v, self.transit_ok(v, (A, B))))
        return self._or_route(A, B, out, step)

    def transit_staged(self, a: dict, b: dict, step: float, travel_lift: float | None = None):
        """Like ``transit``, but with a lift the carrier drives at one ``travel_lift`` height:
        the lift moves there at stop a first, the carrier drives, then the lift moves to b's
        height (the arm waits at home throughout). None = ``transit`` (lift moves en route)."""
        if travel_lift is None or "lift" not in self.dofs:
            return self.transit(a, b, step)
        A, B = self.full(a), self.full(b)
        L = float(travel_lift)

        def lift_move(v: dict, h0: float, h1: float) -> list[tuple[dict, bool]]:
            n = int(np.ceil(abs(h1 - h0) / step - 1e-9))
            out = []
            for s in np.arange(1, n + 1) / max(n, 1):
                f = self.full({**v, "lift": float(h0 + s * (h1 - h0))})
                out.append((f, self.transit_ok(f, (A, B))))
            return out

        drive = self.transit({**a, "lift": L}, {**b, "lift": L}, step)
        return lift_move(a, A["lift"], L) + drive + lift_move(b, L, B["lift"])

    def _or_route(self, A: dict, B: dict, moves: list, step: float) -> list[tuple[dict, bool]]:
        if self.kind != "mobile" or all(ok for _, ok in moves):
            return moves
        r = self.route(A, B, step)
        return r if r is not None else moves  # no route: the blocked direct move is reported

    def transit_ok(self, v: dict, ends: tuple = ()) -> bool:
        """A carrier pose on the way between stops is clear: keep-outs, part standoff and the
        stay-out zones at ``nav_clearance``; within that distance of a stop (``ends``) the
        keep-out clearance the stop itself was planned with is enough."""
        f = self.full(v)
        return self._move_ok(f, self._clearance_at(f, ends))

    def _clearance_at(self, f: dict, ends: tuple) -> float:
        nav, env = self.nav_clearance, self.c.base.standoff_m.carrier_env_min
        if nav > env and "x" in f:
            for e in ends:
                if "x" in e and np.hypot(f["x"] - e["x"], f["y"] - e["y"]) < nav:
                    return env
        return nav

    def _move_ok(self, f: dict, clearance: float) -> bool:
        if any(
            not lo - 1e-9 <= f[d] <= hi + 1e-9 for d, (lo, hi) in zip(self.c.dofs, self.c.bounds, strict=True)
        ):
            return False
        if not carrier_ok(self.c, self.zones, f, self.part, self.T_part).ok:
            return False
        return self.clear_of_obstacles(f, clearance)

    def footprint_radius(self) -> float:
        fp = self.c.footprint(self.full({"x": 0.0, "y": 0.0, "yaw": 0.0}))
        if fp is None:
            return 0.0
        geoms = getattr(fp, "geoms", [fp])
        return float(max(np.max(np.linalg.norm(np.asarray(g.exterior.coords), axis=1)) for g in geoms))

    def route(self, A: dict, B: dict, step: float) -> list[tuple[dict, bool]] | None:
        """Mobile base: poses every ``step`` along a route from A to B around the stay-out
        zones, keep-outs and the part (``opt.navigate``), or None when there is none."""
        from opt.navigate import resample, route

        ends = (A, B)
        lifts = [{"lift": float(E["lift"])} for E in ends if "lift" in E] or [{}]

        def pose(x: float, y: float, yaw: float, lift: dict) -> dict:
            return {"x": float(x), "y": float(y), "yaw": float(yaw), **lift}

        def ok(x: float, y: float, yaw: float) -> bool:  # clear at both stops' lift heights
            return all(self.transit_ok(pose(x, y, yaw, lf), ends) for lf in lifts)

        rad = self.footprint_radius()
        boxes = [(A["x"], A["y"], A["x"], A["y"]), (B["x"], B["y"], B["x"], B["y"])]
        boxes += [o.footprint.bounds for o in self.obstacles]
        boxes += [k.bounds for k in self.zones.keepouts]
        if self.zones.part is not None:
            boxes.append(self.zones.part.bounds)
        Bx = np.array(boxes)
        pad = rad + self.nav_clearance + 0.5
        region = (Bx[:, 0].min() - pad, Bx[:, 1].min() - pad, Bx[:, 2].max() + pad, Bx[:, 3].max() + pad)
        path = route(ok, (A["x"], A["y"], A["yaw"]), (B["x"], B["y"], B["yaw"]), region, rad)
        if path is None:
            return None
        pts = resample(path, step, rad)
        out = []
        for i, p in enumerate(pts):  # the lift moves from A's height to B's along the way
            s = (i + 1) / len(pts)
            lf = {"lift": (1 - s) * lifts[0]["lift"] + s * lifts[-1]["lift"]} if lifts[0] else {}
            out.append(pose(*p, lf))
        return [(v, self.transit_ok(v, ends)) for v in out]

    def perturbations(self, v: dict) -> list[dict]:
        """Mobile localization tolerance: base poses the robot must also work from."""
        dp, dyaw = self.localization
        if dp == 0 and dyaw == 0:
            return []
        out = []
        for dx, dy, dt in ((dp, 0, 0), (-dp, 0, 0), (0, dp, 0), (0, -dp, 0), (0, 0, dyaw), (0, 0, -dyaw)):
            if dx or dy or dt:
                out.append({**v, "x": v["x"] + dx, "y": v["y"] + dy, "yaw": v["yaw"] + dt})
        return out
