"""Carriers built from primitives (plan § Base mode and carriers, M9).

A carrier is a few extra DOFs that place the robot base (station mode: the carrier stops,
then the arm works). See schema.Base for the frame conventions. This module turns a Base
into: the DOF list and bounds, the robot base pose for given DOF values, the primitive
bodies (moving with the carriage, or static like a rail track), their floor footprint, the
travel metric between stations, a static tip-over check, and a URDF of the chain.

Bodies come from ``Base.bodies`` (boxes, cylinders and meshes placed freely on the carriage
or fixed in the carrier frame). Older projects use ``chassis``/``riser``/``track`` instead;
``body_list`` converts those, and ``migrate`` rewrites a Base into the body-list form.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import pinocchio as pin
import trimesh
from shapely.affinity import affine_transform
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from core.schema import Base, Pose, Primitive
from core.transforms import make_T, pose_to_matrix, rpy_to_matrix

G = 9.81
REPO = Path(__file__).resolve().parents[1]
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
MESH_EXTS = (".stl", ".obj", ".ply")
Resolver = Callable[[str], Path]


def default_resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


@functools.lru_cache(maxsize=32)
def _load_mesh(path: str, mtime: float, units: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"{Path(path).name} contains no triangles")
    m.apply_scale(UNIT_SCALE[units])
    return m


def primitive_mesh(p: Primitive, resolve: Resolver | None = None) -> trimesh.Trimesh:
    """Body mesh in its own frame. Box/cylinder: bottom face on z = 0, centred in x/y.
    Mesh bodies keep the file's origin, scaled by ``units``."""
    if p.shape == "mesh":
        path = (resolve or default_resolve)(p.mesh)
        if path.suffix.lower() not in MESH_EXTS:
            raise ValueError(f"unsupported mesh type {path.suffix}; use {', '.join(MESH_EXTS)}")
        if not path.exists():
            raise FileNotFoundError(f"carrier mesh not found: {p.mesh}")
        return _load_mesh(str(path), path.stat().st_mtime, p.units).copy()
    if p.shape == "box":
        m = trimesh.creation.box(extents=p.size)
        m.apply_translation((0, 0, p.size[2] / 2))
    else:
        m = trimesh.creation.cylinder(radius=p.radius, height=p.height, sections=32)
        m.apply_translation((0, 0, p.height / 2))
    return m


def primitive_height(p: Primitive) -> float:
    return p.size[2] if p.shape == "box" else p.height


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "body"


def body_list(base: Base) -> list[Primitive]:
    """The carrier's bodies with unique names. Legacy chassis/riser/track are converted:
    the riser sits on the chassis top, the track is centred on the rail travel range."""
    if base.bodies:
        out = list(base.bodies)
    else:
        out = []
        if base.chassis is not None:
            out.append(
                base.chassis.model_copy(update={"name": base.chassis.name or "chassis", "attach": "carriage"})
            )
            if base.riser is not None:
                z = primitive_height(base.chassis)
                upd = {"name": base.riser.name or "riser", "attach": "carriage", "pose": Pose(xyz=(0, 0, z))}
                out.append(base.riser.model_copy(update=upd))
        if base.track is not None and base.carrier == "rail" and base.joints.x is not None:
            lo, hi = base.joints.x
            upd = {
                "name": base.track.name or "track",
                "attach": "fixed",
                "pose": Pose(xyz=((lo + hi) / 2, 0, 0)),
            }
            out.append(base.track.model_copy(update=upd))
    seen: set[str] = set()
    named = []
    for k, b in enumerate(out):
        n = safe_name(b.name or f"{b.shape}{k + 1}")
        while n in seen:
            n = f"{n}_{k + 1}"
        seen.add(n)
        named.append(b.model_copy(update={"name": n}))
    return named


def migrate(base: Base) -> Base:
    """Same carrier in body-list form (chassis/riser/track moved into ``bodies``)."""
    d = base.model_dump()
    d.update(bodies=[b.model_dump() for b in body_list(base)], chassis=None, riser=None, track=None)
    return Base.model_validate(d)


def body_top(p: Primitive, resolve: Resolver | None = None) -> np.ndarray:
    """Centre of the top of a body's bounding box, in its parent (carriage) frame: where a
    robot mounted 'on top of' the body sits."""
    m = primitive_mesh(p, resolve)
    T = pose_to_matrix(p.pose)
    w = m.vertices @ T[:3, :3].T + T[:3, 3]
    lo, hi = w.min(axis=0), w.max(axis=0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])


def _com(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.center_mass if mesh.is_volume else mesh.bounds.mean(axis=0))


@dataclass
class Body:
    name: str
    mesh: trimesh.Trimesh  # in its own frame
    T: np.ndarray  # world pose of that frame
    kg: float | None = None


class Carrier:
    def __init__(
        self, base: Base, T_world_base_static: np.ndarray | None = None, resolve: Resolver | None = None
    ):
        self.base = base
        self.resolve = resolve
        self.specs = body_list(base)
        self._meshes: dict[int, trimesh.Trimesh] = {}
        self.kind = base.carrier if base.mode == "movable" else "static"
        self.T_static = np.eye(4) if T_world_base_static is None else np.asarray(T_world_base_static, float)
        self.T_frame = pose_to_matrix(base.pose)
        self.T_mount = pose_to_matrix(base.T_mount)
        j = base.joints
        names = {"static": [], "rail": ["s"], "gantry": ["x", "y"], "mobile": ["x", "y", "yaw"]}[self.kind]
        if self.kind != "static" and j.lift is not None:
            names = [*names, "lift"]
        self.dofs = names
        rng = {
            "s": j.x,
            "x": j.x,
            "y": j.y,
            "lift": j.lift,
            "yaw": tuple(np.radians(j.yaw_deg)) if j.yaw_deg is not None else (-np.pi, np.pi),
        }
        self.bounds = np.array([rng[n] for n in names], float).reshape(-1, 2)

    # --- kinematics ------------------------------------------------------------------------

    def clip(self, v: dict) -> dict:
        return {
            n: float(np.clip(v.get(n, lo), lo, hi))
            for n, (lo, hi) in zip(self.dofs, self.bounds, strict=True)
        }

    def T_world_carriage(self, v: dict) -> np.ndarray:
        if self.kind == "rail":
            return self.T_frame @ make_T(p=(v["s"], 0, 0))
        if self.kind == "gantry":
            return self.T_frame @ make_T(p=(v["x"], v["y"], 0))
        if self.kind == "mobile":
            return make_T(rpy_to_matrix((0, 0, v["yaw"])), (v["x"], v["y"], 0))
        return self.T_static @ np.linalg.inv(self.T_mount)

    def T_world_base(self, v: dict) -> np.ndarray:
        if self.kind == "static":
            return self.T_static
        return self.T_world_carriage(v) @ make_T(p=(0, 0, v.get("lift", 0.0))) @ self.T_mount

    def values_for_base(self, T_world_base: np.ndarray) -> dict:
        """Best-effort inverse (GUI): DOF values whose base pose is closest to T_world_base."""
        if self.kind == "static":
            return {}
        T_c = T_world_base @ np.linalg.inv(self.T_mount)
        if self.kind == "mobile":
            v = {"x": T_c[0, 3], "y": T_c[1, 3], "yaw": float(np.arctan2(T_c[1, 0], T_c[0, 0]))}
        else:
            local = np.linalg.inv(self.T_frame) @ T_c
            v = {"s": local[0, 3]} if self.kind == "rail" else {"x": local[0, 3], "y": local[1, 3]}
        return self.clip(v)

    # --- bodies ------------------------------------------------------------------------------

    def mesh(self, k: int) -> trimesh.Trimesh:
        if k not in self._meshes:
            self._meshes[k] = primitive_mesh(self.specs[k], self.resolve)
        return self._meshes[k]

    def _bodies(self, attach: str, T_parent: np.ndarray) -> list[Body]:
        return [
            Body(p.name, self.mesh(k), T_parent @ pose_to_matrix(p.pose), p.kg)
            for k, p in enumerate(self.specs)
            if p.attach == attach
        ]

    def moving_bodies(self, v: dict) -> list[Body]:
        """Bodies on the carriage (world poses for DOF values v)."""
        if self.kind == "static":
            return []
        return self._bodies("carriage", self.T_world_carriage(v))

    def static_bodies(self) -> list[Body]:
        """Bodies fixed in the carrier frame (rail track, gantry beams, ...)."""
        if self.kind == "static":
            return []
        return self._bodies("fixed", self.T_frame)

    @staticmethod
    def _hulls(bodies: list[Body]) -> Polygon | None:
        polys = []
        for body in bodies:
            w = body.mesh.vertices @ body.T[:3, :3].T + body.T[:3, 3]
            polys.append(MultiPoint(w[:, :2]).convex_hull)
        return unary_union(polys) if polys else None

    def _local(self, key: str, make):
        """Cached geometry computed in the carriage frame (the bodies are rigid in it)."""
        cache = self.__dict__.setdefault("_local_cache", {})
        if key not in cache:
            cache[key] = make()
        return cache[key]

    def _moved(self, poly: Polygon | None, T: np.ndarray) -> Polygon | None:
        """Carriage-frame polygon -> world xy, for a carriage pose that only turns about z."""
        if poly is None:
            return None
        return affine_transform(poly, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]])

    def footprint(self, v: dict) -> Polygon | None:
        """Floor footprint (union of each body's convex hull) at DOF values v. The carriage
        outline is computed once in the carriage frame and moved, which keeps the planner's
        standoff filter (thousands of candidates) and the GUI jog cheap."""
        if self.kind == "static":
            return None
        Tc = self.T_world_carriage(v)
        if abs(Tc[2, 2] - 1.0) > 1e-9:  # tilted carriage: project the bodies directly
            moving = self._hulls(self.moving_bodies(v))
        else:
            local = self._local("fp", lambda: self._hulls(self._bodies("carriage", np.eye(4))))
            moving = self._moved(local, Tc)
        static = self._local("fp_static", lambda: self._hulls(self.static_bodies()))
        polys = [p for p in (moving, static) if p is not None]
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else unary_union(polys)

    # --- planning helpers ----------------------------------------------------------------------

    def travel(self, a: dict, b: dict) -> float:
        if self.kind == "rail":
            return abs(a["s"] - b["s"])
        if self.kind == "gantry":
            return max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))  # both axes move at once
        if self.kind == "mobile":
            return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))  # straight line (v1)
        return 0.0

    def samples(self, step: float, yaw_steps: int = 4) -> list[dict]:
        if self.kind == "static":
            return [{}]
        axes = []
        for n, (lo, hi) in zip(self.dofs, self.bounds, strict=True):
            if n == "yaw":
                full = hi - lo >= 2 * np.pi - 1e-6
                k = yaw_steps if hi > lo else 1
                axes.append(np.linspace(lo, hi, k, endpoint=not full) if k > 1 else np.array([lo]))
            elif n == "lift":
                axes.append(np.array([(lo + hi) / 2]))
            else:
                axes.append(np.arange(lo, hi + 1e-9, step) if hi > lo else np.array([lo]))
        grid = np.meshgrid(*axes, indexing="ij")
        return [
            dict(zip(self.dofs, map(float, vals), strict=True))
            for vals in zip(*(g.ravel() for g in grid), strict=True)
        ]

    # --- tip-over -----------------------------------------------------------------------------

    def tip_over(
        self,
        v: dict,
        robot,
        q: np.ndarray,
        tool_kg: float,
        tool_com_flange: np.ndarray,
        force_world: np.ndarray | None,
        tcp_world: np.ndarray | None,
    ) -> tuple[bool, float]:
        """Static stability of a mobile base: the centre of pressure of gravity (chassis,
        riser, arm, tool) plus the part's reaction to the process force must lie inside the
        chassis footprint shrunk by ``tip_margin_m``. Returns (ok, margin to the edge in m)."""
        if self.kind != "mobile" or not self.base.tip_over:
            return True, float("inf")
        forces: list[tuple[np.ndarray, np.ndarray]] = []  # (point, force) in world
        for body in self.moving_bodies(v):
            if body.kg:
                c = body.T[:3, :3] @ _com(body.mesh) + body.T[:3, 3]
                forces.append((c, np.array([0, 0, -G * body.kg])))
        T_wb = self.T_world_base(v)
        pin.framesForwardKinematics(robot.model, robot.data, np.asarray(q))
        T_ub = np.asarray(robot.data.oMf[robot.base_frame_id].homogeneous)
        T_wu = T_wb @ np.linalg.inv(T_ub)
        com = pin.centerOfMass(robot.model, robot.data, np.asarray(q))
        forces.append(
            (T_wu[:3, :3] @ com + T_wu[:3, 3], np.array([0, 0, -G * pin.computeTotalMass(robot.model)]))
        )
        if tool_kg > 0:
            T_wf = T_wb @ robot.fk(q)
            forces.append((T_wf[:3, :3] @ tool_com_flange + T_wf[:3, 3], np.array([0, 0, -G * tool_kg])))
        if force_world is not None and tcp_world is not None:
            forces.append((np.asarray(tcp_world), -np.asarray(force_world)))  # part pushes back
        P = np.array([f[0] for f in forces])
        F = np.array([f[1] for f in forces])
        N = -F[:, 2].sum()
        if N <= 0:
            return False, -np.inf
        cx = (P[:, 2] * F[:, 0] - P[:, 0] * F[:, 2]).sum() / N
        cy = -(P[:, 1] * F[:, 2] - P[:, 2] * F[:, 1]).sum() / N
        support = self.footprint_moving(v)
        from shapely.geometry import Point

        inside = support.buffer(-self.base.tip_margin_m)
        pt = Point(cx, cy)
        margin = (
            float(support.exterior.distance(pt)) if support.contains(pt) else -float(support.distance(pt))
        )
        return bool(not inside.is_empty and inside.contains(pt)), margin - self.base.tip_margin_m

    def footprint_moving(self, v: dict) -> Polygon:
        """Support polygon: convex hull of the carriage bodies that reach the floor (lowest
        within 20 mm of the lowest carriage point), e.g. the chassis or the wheels."""

        def support(bodies: list[Body]) -> Polygon:
            world = [b.mesh.vertices @ b.T[:3, :3].T + b.T[:3, 3] for b in bodies]
            if not world:
                raise ValueError("carrier has no carriage bodies")
            z0 = min(w[:, 2].min() for w in world)
            pts = np.vstack([w for w in world if w[:, 2].min() <= z0 + 0.02])
            return MultiPoint(pts[:, :2]).convex_hull

        Tc = self.T_world_carriage(v)
        if abs(Tc[2, 2] - 1.0) > 1e-9:
            return support(self.moving_bodies(v))
        return self._moved(self._local("support", lambda: support(self._bodies("carriage", np.eye(4)))), Tc)

    # --- URDF ---------------------------------------------------------------------------------

    def urdf(self) -> str:
        """Carrier chain world -> DOF joints -> carriage -> mount, primitives as geometry."""

        def geom(p: Primitive) -> str:
            if p.shape == "box":
                return f'<box size="{p.size[0]} {p.size[1]} {p.size[2]}"/>'
            if p.shape == "mesh":
                path = (self.resolve or default_resolve)(p.mesh).resolve().as_posix()
                sc = UNIT_SCALE[p.units]
                return f'<mesh filename={quoteattr(path)} scale="{sc} {sc} {sc}"/>'
            return f'<cylinder radius="{p.radius}" length="{p.height}"/>'

        def link(name: str, p: Primitive | None = None) -> str:
            if p is None:
                return f'  <link name="{name}"/>'
            h = 0.0 if p.shape == "mesh" else primitive_height(p) / 2
            g = f'<origin xyz="0 0 {h}"/><geometry>{geom(p)}</geometry>'
            return f'  <link name="{name}"><visual>{g}</visual><collision>{g}</collision></link>'

        def joint(name, kind, parent, child, xyz="0 0 0", rpy="0 0 0", axis="1 0 0", lim=None) -> str:
            lim_s = (
                f'<limit lower="{lim[0]}" upper="{lim[1]}" effort="1000" velocity="1"/>'
                if lim is not None
                else ""
            )
            ax = f'<axis xyz="{axis}"/>' if kind != "fixed" else ""
            return (
                f'  <joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                f'<origin xyz="{xyz}" rpy="{rpy}"/>{ax}{lim_s}</joint>'
            )

        def xyz_rpy(T: np.ndarray) -> tuple[str, str]:
            from core.transforms import matrix_to_rpy

            return " ".join(f"{v:.9g}" for v in T[:3, 3]), " ".join(
                f"{v:.9g}" for v in matrix_to_rpy(T[:3, :3])
            )

        lines = ['<robot name="carrier">', link("world")]
        for p in self.specs:
            if p.attach == "fixed":
                xyz, rpy = xyz_rpy(self.T_frame @ pose_to_matrix(p.pose))
                lines += [
                    link(f"fixed_{p.name}", p),
                    joint(f"fixed_{p.name}_joint", "fixed", "world", f"fixed_{p.name}", xyz, rpy),
                ]
        parent = "world"
        bnd = dict(zip(self.dofs, self.bounds.tolist(), strict=True))
        if self.kind in ("rail", "gantry"):
            xyz, rpy = xyz_rpy(self.T_frame)
            lines += [link("carrier_frame"), joint("frame", "fixed", "world", "carrier_frame", xyz, rpy)]
            parent = "carrier_frame"
        axes = {
            "s": ("1 0 0", "prismatic"),
            "x": ("1 0 0", "prismatic"),
            "y": ("0 1 0", "prismatic"),
            "yaw": ("0 0 1", "revolute"),
        }
        for n in [d for d in self.dofs if d != "lift"]:
            axis, kind = axes[n]
            lines += [
                link(f"{n}_link"),
                joint(f"{n}_joint", kind, parent, f"{n}_link", axis=axis, lim=bnd[n]),
            ]
            parent = f"{n}_link"
        lines += [link("carriage"), joint("carriage_fixed", "fixed", parent, "carriage")]
        parent = "carriage"
        for p in self.specs:
            if p.attach == "carriage":
                xyz, rpy = xyz_rpy(pose_to_matrix(p.pose))
                lines += [
                    link(f"body_{p.name}", p),
                    joint(f"body_{p.name}_joint", "fixed", "carriage", f"body_{p.name}", xyz, rpy),
                ]
        if "lift" in self.dofs:
            lines += [
                link("lift_link"),
                joint("lift_joint", "prismatic", "carriage", "lift_link", axis="0 0 1", lim=bnd["lift"]),
            ]
            parent = "lift_link"
        xyz, rpy = xyz_rpy(self.T_mount)
        lines += [link("mount"), joint("mount_fixed", "fixed", parent, "mount", xyz, rpy), "</robot>"]
        return "\n".join(lines)
