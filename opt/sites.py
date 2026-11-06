"""Where stations may go: the search space of the placement layer (plan § Base mode).

Every carrier is "a small set of extra DOFs that place the robot base". A Sites object
exposes that space to the planner:
    dofs / bounds         carrier DOF names and ranges
    candidates(t, st)     DOF dicts to seed the set cover
    T(v)                  robot base pose (world) for DOF values v
    accept(v)             (ok, soft penalty) from standoff / keep-out zones
    travel(a, b)          carrier travel metric between two stations
    point_ok(v, r, ...)   extra per-point checks (tip-over), default: ok

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

    def point_ok(self, v, r, checker, n_world) -> tuple[bool, str]:
        return True, ""

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
        self.dofs = [d for d in carrier.dofs if d != "lift"]
        self.bounds = np.array(
            [b for d, b in zip(carrier.dofs, carrier.bounds, strict=True) if d != "lift"]
        ).reshape(-1, 2)
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
        """(base_bodies, world_bodies) for core.collision.CollisionScene: carriage/riser are
        rigid in the robot base frame; the rail track is fixed in the world."""
        v = self.full({d: float(np.mean(b)) for d, b in zip(self.dofs, self.bounds, strict=True)})
        T_base_inv = np.linalg.inv(self.c.T_world_base(v))
        base = [(b.name, b.mesh, T_base_inv @ b.T) for b in self.c.moving_bodies(v)]
        world = [(b.name, b.mesh, b.T) for b in self.c.static_bodies()]
        return base, world

    def full(self, v: dict) -> dict:
        return {**v, **self._lift}

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

    def clear_of_obstacles(self, v: dict) -> bool:
        """The carrier at DOF values v keeps the keep-out clearance from every stay-out zone."""
        if not self.obstacles:
            return True
        from core.stayout import carrier_clear

        v = self.full(v)
        bodies = self.c.moving_bodies(v) + self.c.static_bodies()
        ok, _ = carrier_clear(
            bodies, self.c.footprint(v), self.obstacles, self.c.base.standoff_m.carrier_env_min
        )
        return ok

    def travel(self, a: dict, b: dict) -> float:
        return self.c.travel(a, b)

    def point_ok(self, v: dict, r, checker, n_world: np.ndarray) -> tuple[bool, str]:
        if self.kind != "mobile" or not self.c.base.tip_over or r.q is None:
            return True, ""
        T_wb = self.T(v)
        T_tcp = T_wb @ checker.robot.fk(r.q) @ checker.T_flange_tcp[r.tcp]
        force = -checker.force * np.asarray(n_world)
        ok, _ = self.c.tip_over(
            self.full(v), checker.robot, r.q, checker.mass, checker.com, force, T_tcp[:3, 3]
        )
        return ok, "" if ok else "tip_over"

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
