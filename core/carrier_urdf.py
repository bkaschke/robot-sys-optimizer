"""URDF of a carrier chain (core.carrier.Carrier.urdf): world -> DOF joints -> carriage ->
mount, the primitives as geometry. Bodies on the X axis (a gantry bridge) hang on the first
DOF link, carriage bodies on the carriage, lift-stage bodies on the lift link."""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

import numpy as np

from core.carrier import UNIT_SCALE, Carrier, default_resolve, primitive_height
from core.schema import Primitive
from core.transforms import pose_to_matrix


def carrier_urdf(c: Carrier) -> str:
    """Carrier chain world -> DOF joints -> carriage -> mount, primitives as geometry."""

    def geom(p: Primitive) -> str:
        if p.shape == "box":
            return f'<box size="{p.size[0]} {p.size[1]} {p.size[2]}"/>'
        if p.shape == "mesh":
            path = (c.resolve or default_resolve)(p.mesh).resolve().as_posix()
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

        return " ".join(f"{v:.9g}" for v in T[:3, 3]), " ".join(f"{v:.9g}" for v in matrix_to_rpy(T[:3, :3]))

    lines = ['<robot name="carrier">', link("world")]
    for p in c.specs:
        if p.attach == "fixed":
            xyz, rpy = xyz_rpy(c.T_frame @ pose_to_matrix(p.pose))
            lines += [
                link(f"fixed_{p.name}", p),
                joint(f"fixed_{p.name}_joint", "fixed", "world", f"fixed_{p.name}", xyz, rpy),
            ]
    parent = "world"
    bnd = dict(zip(c.dofs, c.bounds.tolist(), strict=True))
    if c.kind in ("rail", "gantry"):
        xyz, rpy = xyz_rpy(c.T_frame)
        lines += [link("carrier_frame"), joint("frame", "fixed", "world", "carrier_frame", xyz, rpy)]
        parent = "carrier_frame"
    axes = {
        "s": ("1 0 0", "prismatic"),
        "x": ("1 0 0", "prismatic"),
        "y": ("0 1 0", "prismatic"),
        "yaw": ("0 0 1", "revolute"),
    }
    for n in [d for d in c.dofs if d != "lift"]:
        axis, kind = axes[n]
        lines += [
            link(f"{n}_link"),
            joint(f"{n}_joint", kind, parent, f"{n}_link", axis=axis, lim=bnd[n]),
        ]
        parent = f"{n}_link"
        if n == c.dofs[0]:  # gantry bridge: moves with the first axis only
            for p in c.specs:
                if p.attach == "x_axis":
                    xyz, rpy = xyz_rpy(pose_to_matrix(p.pose))
                    lines += [
                        link(f"body_{p.name}", p),
                        joint(f"body_{p.name}_joint", "fixed", parent, f"body_{p.name}", xyz, rpy),
                    ]
    lines += [link("carriage"), joint("carriage_fixed", "fixed", parent, "carriage")]
    parent = "carriage"
    if "lift" in c.dofs:
        lines += [
            link("lift_link"),
            joint("lift_joint", "prismatic", "carriage", "lift_link", axis="0 0 1", lim=bnd["lift"]),
        ]
    stage = "lift_link" if "lift" in c.dofs else "carriage"
    for p in c.specs:
        if p.attach in ("carriage", "lift"):
            on = stage if p.attach == "lift" else "carriage"
            xyz, rpy = xyz_rpy(pose_to_matrix(p.pose))
            lines += [
                link(f"body_{p.name}", p),
                joint(f"body_{p.name}_joint", "fixed", on, f"body_{p.name}", xyz, rpy),
            ]
    parent = stage
    xyz, rpy = xyz_rpy(c.T_mount)
    lines += [link("mount"), joint("mount_fixed", "fixed", parent, "mount", xyz, rpy), "</robot>"]
    return "\n".join(lines)
