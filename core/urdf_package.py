"""URDF packages for end effectors, external axes and mobile bases (Package builder tab).

A package is a zip ``<name>/<name>.urdf`` + ``meshes/`` + ``package.yaml`` (kind, name, the
parts as authored) + ``README.txt``. The URDF follows one convention so the app (and other
URDF software) can read it back:

    tool    root ``mount`` (sits on the robot flange: z out of the flange), fixed frames
            ``tcp_<name>`` (z = tool axis pointing at the surface)
    rail    ``base_link`` (track frame) --prismatic axis_x (+x)--> ``carriage``
    gantry  ``base_link`` --axis_x (+x)--> ``bridge`` --axis_y (+y)--> ``carriage``
    mobile  ``base_link`` = the chassis (origin on the floor, x = forward)
    carriers optionally --prismatic lift (+z)--> ``lift_stage``, then a fixed frame
            ``robot_mount`` where the robot's base goes.

Joint origins are identities (every stage frame coincides with the root at zero travel), so
a part's pose is simply its pose in the root frame. Reading is tolerant: URDFs exported from
CAD (other link names, joint origins) are read through their kinematics at zero travel; the
roles come from the prismatic joints' axes (x / y / z in the root frame), ``robot_mount`` and
``tcp*`` frames. Box / cylinder parts use the app's convention (bottom face on z = 0, centred;
URDF boxes and cylinders are centred, the offset is added / removed here).

Collision geometry (2026-11-11): mesh parts get a low-detail collision copy
(``Package.collision``, per part ``Part.collision``; core.collision_mesh) written to
``meshes/collision/<part>.stl`` (metres) and used by the URDF's ``<collision>`` element; the
visual element keeps the full mesh. Reading takes the visual geometry.
"""

from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.schema import Base, CarrierJoints, EndEffector, Pose, Primitive, Range
from core.transforms import make_T, matrix_to_pose, pose_to_matrix

Kind = Literal["tool", "rail", "gantry", "mobile"]
KINDS: tuple[str, ...] = ("tool", "rail", "gantry", "mobile")
STAGES = {
    "tool": ("tool",),
    "rail": ("fixed", "carriage", "lift"),
    "gantry": ("fixed", "bridge", "carriage", "lift"),
    "mobile": ("chassis", "lift"),
}
LINK = {"tool": "mount", "fixed": "base_link", "chassis": "base_link", "bridge": "bridge"}
LINK.update(carriage="carriage", lift="lift_stage")
ATTACH = {"fixed": "fixed", "bridge": "x_axis", "carriage": "carriage", "lift": "lift", "chassis": "carriage"}
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
MESH_EXTS = (".stl", ".obj", ".ply")
NAME = re.compile(r"[^A-Za-z0-9_-]+")
CollisionMode = Literal["auto", "visual", "reduce", "hull", "wrap"]
MODES = ("auto", "visual", "reduce", "hull", "wrap")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Part(_M):
    """One body: a box, cylinder or mesh on one stage, posed in the package root frame."""

    name: str = Field(max_length=64)
    stage: str
    shape: Literal["box", "cylinder", "mesh"]
    size: tuple[float, float, float] | None = None  # box L x W x H (m)
    radius: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    mesh: str | None = None  # file name in the package's meshes/ folder
    units: Literal["m", "mm", "in"] = "m"
    pose: Pose = Field(default_factory=Pose)
    collision: CollisionMode | None = None  # None: the package's collision setting


class Collision(_M):
    """How mesh parts are simplified for collision (core.collision_mesh.Geom)."""

    mode: CollisionMode = "auto"
    faces: int = Field(default=4000, ge=50, le=200000)
    wrap_mm: float = Field(default=5.0, gt=0, le=200)
    pad_mm: float = Field(default=0.0, ge=0, le=100)  # every part's collision shape this much larger


class Frame(_M):
    name: str = Field(max_length=64)
    pose: Pose = Field(default_factory=Pose)


class Package(_M):
    kind: Kind
    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    parts: list[Part] = Field(default_factory=list, max_length=64)
    tcps: list[Frame] = Field(default_factory=list)  # tool
    mount: Pose = Field(default_factory=Pose)  # carriers: the robot base on the carriage / lift
    x: Range | None = None  # rail / gantry travel (m)
    y: Range | None = None  # gantry
    lift: Range | None = None  # optional on every carrier
    collision: Collision = Field(default_factory=Collision)


def safe(name: str) -> str:
    return NAME.sub("_", name).strip("_") or "part"


def check(p: Package) -> list[str]:
    """Problems that stop an export (empty: fine)."""
    out = []
    if not p.parts:
        out.append("add at least one part")
    for part in p.parts:
        if part.stage not in STAGES[p.kind]:
            out.append(f"{part.name}: a {p.kind} has no stage '{part.stage}' ({', '.join(STAGES[p.kind])})")
        if part.shape == "box" and (part.size is None or min(part.size) <= 0):
            out.append(f"{part.name}: a box needs three positive sizes")
        if part.shape == "cylinder" and (part.radius is None or part.height is None):
            out.append(f"{part.name}: a cylinder needs a radius and a height")
        if part.shape == "mesh" and not part.mesh:
            out.append(f"{part.name}: no mesh file")
    if p.kind == "tool" and not p.tcps:
        out.append("a tool needs at least one TCP")
    if len({t.name for t in p.tcps}) != len(p.tcps):
        out.append("TCP names must differ")
    if p.kind in ("rail", "gantry") and p.x is None:
        out.append(f"a {p.kind} needs an X travel range")
    if p.kind == "gantry" and p.y is None:
        out.append("a gantry needs a Y travel range")
    if any(part.stage == "lift" for part in p.parts) and p.lift is None:
        out.append("parts on the lift stage need a lift range")
    for r, n in ((p.x, "X"), (p.y, "Y"), (p.lift, "lift")):
        if r is not None and r[1] <= r[0]:
            out.append(f"{n} travel: max must be above min")
    return out


# --- writing ------------------------------------------------------------------------------------


def _xyz_rpy(T: np.ndarray) -> str:
    p = matrix_to_pose(T)
    return f'xyz="{" ".join(f"{v:.12g}" for v in p.xyz)}" rpy="{" ".join(f"{v:.12g}" for v in p.rpy)}"'


def _geometry(part: Part) -> tuple[str, np.ndarray]:
    """(URDF geometry, origin in the link) for a part."""
    T = pose_to_matrix(part.pose)
    if part.shape == "box":
        return f'<box size="{" ".join(f"{v:.6g}" for v in part.size)}"/>', T @ make_T(
            p=(0, 0, part.size[2] / 2)
        )
    if part.shape == "cylinder":
        g = f'<cylinder radius="{part.radius:.6g}" length="{part.height:.6g}"/>'
        return g, T @ make_T(p=(0, 0, part.height / 2))
    s = UNIT_SCALE[part.units]
    return f'<mesh filename="meshes/{part.mesh}" scale="{s:.6g} {s:.6g} {s:.6g}"/>', T


def part_collision(p: Package, part: Part, mesh_dir: Path) -> trimesh.Trimesh | None:
    """A part's collision copy in metres, in the part's own frame (boxes / cylinders: bottom
    face on z = 0), or None when it is the visual geometry itself (no padding, and a box,
    a cylinder, "visual" or a small mesh under "auto"). ``Collision.pad_mm`` makes it larger:
    boxes and cylinders exactly, a wrap by a larger offset, other meshes along their normals
    (core.collision_mesh.inflate)."""
    from core.collision_mesh import Geom, collision_mesh, inflate

    c = p.collision
    pad = c.pad_mm / 1000
    if part.shape == "box" and part.size:
        if pad <= 0:
            return None
        m = trimesh.creation.box(extents=[v + 2 * pad for v in part.size])
        m.apply_translation((0, 0, part.size[2] / 2))  # bottom at -pad: centred on the box
        return m
    if part.shape == "cylinder" and part.radius and part.height:
        if pad <= 0:
            return None
        m = trimesh.creation.cylinder(radius=part.radius + pad, height=part.height + 2 * pad, sections=32)
        m.apply_translation((0, 0, part.height / 2))
        return m
    if part.shape != "mesh" or not part.mesh:
        return None
    mode = part.collision or c.mode
    if mode == "visual" and pad <= 0:
        return None
    m = trimesh.load(str(mesh_dir / part.mesh), force="mesh")
    m.apply_scale(UNIT_SCALE[part.units])
    if mode == "wrap":
        return collision_mesh(m, Geom("wrap", c.faces, c.wrap_mm + c.pad_mm))
    out = m if mode == "visual" else collision_mesh(m, Geom(mode, c.faces, c.wrap_mm))
    if pad > 0:
        return inflate(out, pad)
    return None if out is m else out


def build_urdf(p: Package, collision: dict[str, str] | None = None) -> str:
    """The URDF; ``collision``: part name -> its collision mesh file under meshes/ (metres)."""
    collision = collision or {}
    stages = list(STAGES[p.kind])
    if p.kind != "tool" and p.lift is None:
        stages.remove("lift")
    links: dict[str, list[str]] = {LINK[s]: [] for s in stages}
    for part in p.parts:
        g, T = _geometry(part)
        o = f"<origin {_xyz_rpy(T)}/>"
        el = f"<geometry>{g}</geometry>"
        cf = collision.get(part.name)
        cel = f'<geometry><mesh filename="meshes/{cf}"/></geometry>' if cf else el
        co = f"<origin {_xyz_rpy(pose_to_matrix(part.pose))}/>" if cf else o  # file: the part frame
        links[LINK[part.stage]].append(
            f'    <visual name="{safe(part.name)}">{o}{el}</visual>\n'
            f'    <collision name="{safe(part.name)}">{co}{cel}</collision>\n'
        )
    head = f'<?xml version="1.0"?>\n<!-- {p.kind} package {safe(p.name)} (Package builder) -->\n'
    out = [head + f'<robot name="{safe(p.name)}">']
    for name, geo in links.items():
        out.append(f'  <link name="{name}">\n{"".join(geo)}  </link>' if geo else f'  <link name="{name}"/>')

    def joint(name, kind, parent, child, axis=None, lim=None, T=None):
        s = f'  <joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
        s += f"<origin {_xyz_rpy(np.eye(4) if T is None else T)}/>"
        if axis:
            s += f'<axis xyz="{axis}"/><limit lower="{lim[0]:.6g}" upper="{lim[1]:.6g}" '
            s += 'effort="1000" velocity="1"/>'
        return s + "</joint>"

    if p.kind == "tool":
        for t in p.tcps:
            out.append(f'  <link name="tcp_{safe(t.name)}"/>')
            out.append(
                joint(
                    f"tcp_{safe(t.name)}", "fixed", "mount", f"tcp_{safe(t.name)}", T=pose_to_matrix(t.pose)
                )
            )
    else:
        top = "base_link"
        if p.kind in ("rail", "gantry"):
            out.append(
                joint(
                    "axis_x",
                    "prismatic",
                    "base_link",
                    "bridge" if p.kind == "gantry" else "carriage",
                    "1 0 0",
                    p.x,
                )
            )
            top = "carriage"
        if p.kind == "gantry":
            out.append(joint("axis_y", "prismatic", "bridge", "carriage", "0 1 0", p.y))
        if p.lift is not None:
            out.append(joint("lift", "prismatic", top, "lift_stage", "0 0 1", p.lift))
            top = "lift_stage"
        out.append('  <link name="robot_mount"/>')
        out.append(joint("robot_mount", "fixed", top, "robot_mount", T=pose_to_matrix(p.mount)))
    return "\n".join(out) + "\n</robot>\n"


README = """{name}: a {kind} package (URDF + meshes) written by the robot-sys-optimizer Package builder.

{name}.urdf     the model (metres; frames as in the user guide, "Building URDF packages")
package.yaml    the parts as authored (the builder reopens them from here)
meshes/         mesh files referenced by the URDF (meshes/collision: the low-detail collision copies)

Load it in the app: Package builder -> Open package (.zip) -> Use in this project.
"""


def write_package(p: Package, mesh_dir: Path | None, dest: Path) -> Path:
    """Write the package folder ``dest`` (URDF, package.yaml, README, meshes copied from
    ``mesh_dir``); returns the URDF path."""
    problems = check(p)
    if problems:
        raise ValueError("; ".join(problems))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meshes").mkdir(exist_ok=True)
    for part in p.parts:
        if part.shape == "mesh":
            src = (mesh_dir or dest / "meshes") / part.mesh
            if not src.is_file():
                raise FileNotFoundError(f"{part.name}: mesh file {part.mesh} not found")
            if src.resolve() != (dest / "meshes" / part.mesh).resolve():
                shutil.copy(src, dest / "meshes" / part.mesh)
    coll_dir = dest / "meshes" / "collision"
    if coll_dir.is_dir():
        shutil.rmtree(coll_dir)
    collision: dict[str, str] = {}
    for part in p.parts:
        m = part_collision(p, part, dest / "meshes")
        if m is not None:
            coll_dir.mkdir(exist_ok=True)
            collision[part.name] = f"collision/{safe(part.name)}.stl"
            m.export(dest / "meshes" / collision[part.name])
    urdf = dest / f"{safe(p.name)}.urdf"
    urdf.write_text(build_urdf(p, collision), encoding="utf-8")
    (dest / "package.yaml").write_text(
        yaml.safe_dump(p.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    (dest / "README.txt").write_text(README.format(name=p.name, kind=p.kind), encoding="utf-8")
    return urdf


def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, (Path(folder.name) / f.relative_to(folder)).as_posix())
    return buf.getvalue()


# --- into the project ---------------------------------------------------------------------------


def to_base(p: Package, mesh_rel: str, current: Base, collision: dict[str, str] | None = None) -> Base:
    """The carrier as the External axis tab's Base: bodies, travel, robot mount. Standoffs,
    keep-outs, the carrier frame and mobile settings come from ``current``. ``mesh_rel``:
    the package's meshes folder relative to the project (or repo); ``collision``: part name
    -> collision mesh under it (``collision_files``)."""
    collision = collision or {}
    if p.kind == "tool":
        raise ValueError("a tool package is not a carrier")
    bodies = [
        Primitive(
            name=part.name,
            shape=part.shape,
            size=part.size,
            radius=part.radius,
            height=part.height,
            mesh=f"{mesh_rel}/{part.mesh}" if part.shape == "mesh" else None,
            collision_mesh=f"{mesh_rel}/{collision[part.name]}" if part.name in collision else None,
            units=part.units,
            attach=ATTACH[part.stage],
            pose=part.pose,
        )
        for part in p.parts
    ]
    j = current.joints
    if p.kind == "mobile":  # the floor area is the cell's, not the package's
        was = current.carrier == "mobile"
        x = j.x if was and j.x else (-4.0, 4.0)
        y = j.y if was and j.y else (-4.0, 4.0)
        joints = CarrierJoints(x=x, y=y, yaw_deg=j.yaw_deg or (-180.0, 180.0), lift=p.lift)
    else:
        joints = CarrierJoints(x=p.x, y=p.y if p.kind == "gantry" else None, lift=p.lift)
    keep = current.model_dump(
        exclude={
            "mode",
            "carrier",
            "bodies",
            "joints",
            "T_mount",
            "chassis",
            "riser",
            "track",
            "axis_length_m",
            "manual_travel",
        }
    )
    return Base(
        mode="movable",
        carrier=p.kind,
        bodies=bodies,
        joints=joints,
        T_mount=p.mount,
        manual_travel=True,
        **keep,
    )


def to_tool(p: Package, urdf_rel: str) -> EndEffector:
    """A pose tool on the flange mount (every range 0) with the package's TCPs."""
    from core.schema import TCP, Extension, ExtensionValues

    zero = (0.0, 0.0)
    ext = Extension(type="pose", x=zero, y=zero, z=zero, nominal=ExtensionValues(z=0.0))
    tcps = [TCP(name=t.name, T_head_tcp=t.pose) for t in p.tcps] or [TCP(name="tcp")]
    return EndEffector(name=p.name, urdf=urdf_rel, tcps=tcps, extension=ext)


# --- reading (core.urdf_package_read; callers import these from here) -----------------------------


def read_urdf(urdf_path: Path, kind: str | None = None, name: str | None = None) -> Package:
    """The package of a URDF (convention or CAD export), see core.urdf_package_read."""
    from core.urdf_package_read import read_urdf as read

    return read(urdf_path, kind, name)


def open_package(data: bytes, dest: Path) -> tuple[Package, Path]:
    """Unpack a package zip into ``dest`` and read it (core.urdf_package_read)."""
    from core.urdf_package_read import open_package as op

    return op(data, dest)


def collision_files(p: Package, dest: Path) -> dict[str, str]:
    """Part name -> its collision mesh under ``dest/meshes`` (written by write_package)."""
    out = {}
    for part in p.parts:
        rel = f"collision/{safe(part.name)}.stl"
        if (dest / "meshes" / rel).is_file():
            out[part.name] = rel
    return out
