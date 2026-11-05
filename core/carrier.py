"""Carriers built from primitives (plan § Base mode and carriers, M9).

A carrier is a few extra DOFs that place the robot base (station mode: the carrier stops,
then the arm works). See schema.Base for the frame conventions. This module turns a Base
into: the DOF list and bounds, the robot base pose for given DOF values, the primitive
bodies (moving with the carriage, or static like a rail track), their floor footprint, the
travel metric between stations, a static tip-over check, and a URDF of the chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from core.schema import Base, Primitive
from core.transforms import make_T, pose_to_matrix, rpy_to_matrix

G = 9.81


def primitive_mesh(p: Primitive) -> trimesh.Trimesh:
    """Mesh with its bottom face on z = 0, centred in x/y."""
    if p.shape == "box":
        m = trimesh.creation.box(extents=p.size)
        m.apply_translation((0, 0, p.size[2] / 2))
    else:
        m = trimesh.creation.cylinder(radius=p.radius, height=p.height, sections=32)
        m.apply_translation((0, 0, p.height / 2))
    return m


def primitive_height(p: Primitive) -> float:
    return p.size[2] if p.shape == "box" else p.height


@dataclass
class Body:
    name: str
    mesh: trimesh.Trimesh  # in its own frame
    T: np.ndarray  # world pose of that frame
    kg: float | None = None


class Carrier:
    def __init__(self, base: Base, T_world_base_static: np.ndarray | None = None):
        self.base = base
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

    def moving_bodies(self, v: dict) -> list[Body]:
        if self.kind == "static":
            return []
        out = []
        T = self.T_world_carriage(v)
        b = self.base
        if b.chassis is not None:
            out.append(Body("chassis", primitive_mesh(b.chassis), T, b.chassis.kg))
            if b.riser is not None:
                out.append(
                    Body(
                        "riser",
                        primitive_mesh(b.riser),
                        T @ make_T(p=(0, 0, primitive_height(b.chassis))),
                        b.riser.kg,
                    )
                )
        return out

    def static_bodies(self) -> list[Body]:
        b = self.base
        if self.kind != "rail" or b.track is None:
            return []
        lo, hi = self.bounds[0]
        return [
            Body("track", primitive_mesh(b.track), self.T_frame @ make_T(p=((lo + hi) / 2, 0, 0)), b.track.kg)
        ]

    def footprint(self, v: dict) -> Polygon | None:
        polys = []
        for body in self.moving_bodies(v) + self.static_bodies():
            w = body.mesh.vertices @ body.T[:3, :3].T + body.T[:3, 3]
            polys.append(MultiPoint(w[:, :2]).convex_hull)
        return unary_union(polys) if polys else None

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
                c = body.T[:3, :3] @ body.mesh.center_mass + body.T[:3, 3]
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
        bodies = [b for b in self.moving_bodies(v) if b.name == "chassis"]
        w = bodies[0].mesh.vertices @ bodies[0].T[:3, :3].T + bodies[0].T[:3, 3]
        return MultiPoint(w[:, :2]).convex_hull

    # --- URDF ---------------------------------------------------------------------------------

    def urdf(self) -> str:
        """Carrier chain world -> DOF joints -> carriage -> mount, primitives as geometry."""

        def geom(p: Primitive) -> str:
            if p.shape == "box":
                return f'<box size="{p.size[0]} {p.size[1]} {p.size[2]}"/>'
            return f'<cylinder radius="{p.radius}" length="{p.height}"/>'

        def link(name: str, p: Primitive | None = None, z0: float = 0.0) -> str:
            if p is None:
                return f'  <link name="{name}"/>'
            h = primitive_height(p) / 2 + z0
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
        lines += [link("carriage", self.base.chassis), joint("carriage_fixed", "fixed", parent, "carriage")]
        parent = "carriage"
        if self.base.riser is not None and self.base.chassis is not None:
            lines += [
                link("riser", self.base.riser),
                joint(
                    "riser_fixed", "fixed", "carriage", "riser", f"0 0 {primitive_height(self.base.chassis)}"
                ),
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
