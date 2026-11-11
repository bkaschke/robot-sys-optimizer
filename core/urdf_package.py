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


def build_urdf(p: Package) -> str:
    stages = list(STAGES[p.kind])
    if p.kind != "tool" and p.lift is None:
        stages.remove("lift")
    links: dict[str, list[str]] = {LINK[s]: [] for s in stages}
    for part in p.parts:
        g, T = _geometry(part)
        o = f"<origin {_xyz_rpy(T)}/>"
        el = f"<geometry>{g}</geometry>"
        links[LINK[part.stage]].append(
            f'    <visual name="{safe(part.name)}">{o}{el}</visual>\n'
            f'    <collision name="{safe(part.name)}">{o}{el}</collision>\n'
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
meshes/         mesh files referenced by the URDF

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
    urdf = dest / f"{safe(p.name)}.urdf"
    urdf.write_text(build_urdf(p), encoding="utf-8")
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


# --- reading ------------------------------------------------------------------------------------


def _scale_units(scale) -> str | None:
    if scale is None:
        return "m"
    s = np.atleast_1d(np.asarray(scale, float))
    if not np.allclose(s, s[0]):
        return None
    for k, v in UNIT_SCALE.items():
        if abs(s[0] - v) < 1e-9:
            return k
    return None


def read_urdf(urdf_path: Path, kind: str | None = None, name: str | None = None) -> Package:
    """A Package from a URDF on disk (convention or CAD export), meshes in its folder. Meshes
    in other formats or units are converted to STL in metres next to the URDF."""
    import yourdfpy

    folder = urdf_path.parent
    u = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False, mesh_dir=str(folder))
    root = u.base_link
    zero = {j: 0.0 for j in u.actuated_joint_names}
    u.update_cfg(zero)

    def T_of(link: str) -> np.ndarray:
        return np.asarray(u.get_transform(link, root), float)

    roles: dict[str, str] = {}  # prismatic joint name -> x / y / lift
    for j in u.robot.joints:
        if j.type == "prismatic":
            Ro = np.eye(3) if j.origin is None else np.asarray(j.origin, float)[:3, :3]
            a = T_of(j.parent)[:3, :3] @ Ro @ np.asarray(j.axis, float)
            k = int(np.argmax(np.abs(a)))
            if abs(abs(a[k]) - 1) > 1e-6 or a[k] < 0:
                raise ValueError(f"joint {j.name}: a travel axis must be +x, +y or +z of {root}")
            roles[j.name] = ("x", "y", "lift")[k]
        elif j.type in ("revolute", "continuous"):
            raise ValueError(f"joint {j.name}: rotary joints are not supported in a package")
    links = [lk.name for lk in u.robot.links]
    tcps = [lk for lk in links if lk.lower().startswith("tcp")]
    if kind is None:
        found = set(roles.values())
        kind = (
            "gantry" if {"x", "y"} <= found else "rail" if "x" in found
            else "mobile" if "robot_mount" in links else "tool"
        )  # fmt: skip
    if kind == "tool" and roles:
        raise ValueError("a tool package has no moving joints")
    # stage of each link: the travel joints above it
    parent_joint = {j.child: j for j in u.robot.joints}

    def stage(link: str) -> str:
        above = set()
        while link in parent_joint:
            j = parent_joint[link]
            if j.name in roles:
                above.add(roles[j.name])
            link = j.parent
        if kind == "tool":
            return "tool"
        if "lift" in above:
            return "lift"
        if kind == "mobile":
            return "chassis"
        if "y" in above or (kind == "rail" and "x" in above):
            return "carriage"
        return "bridge" if "x" in above else "fixed"

    parts: list[Part] = []
    for lk in u.robot.links:
        elems = lk.collisions or lk.visuals
        for n, el in enumerate(elems):
            g = el.geometry
            T = T_of(lk.name) @ (np.eye(4) if el.origin is None else np.asarray(el.origin, float))
            pname = safe(el.name or (f"{lk.name}_{n}" if len(elems) > 1 else lk.name))
            if g.box is not None:
                size = tuple(float(v) for v in g.box.size)
                part = Part(
                    name=pname,
                    stage=stage(lk.name),
                    shape="box",
                    size=size,
                    pose=matrix_to_pose(T @ make_T(p=(0, 0, -size[2] / 2))),
                )
            elif g.cylinder is not None:
                r, h = float(g.cylinder.radius), float(g.cylinder.length)
                part = Part(
                    name=pname,
                    stage=stage(lk.name),
                    shape="cylinder",
                    radius=r,
                    height=h,
                    pose=matrix_to_pose(T @ make_T(p=(0, 0, -h / 2))),
                )
            elif g.mesh is not None:
                src = (folder / g.mesh.filename).resolve()
                if not src.is_relative_to(folder.resolve()) or not src.is_file():
                    raise ValueError(f"{lk.name}: mesh {g.mesh.filename} not found in the package")
                units = _scale_units(g.mesh.scale)
                rel = src.relative_to(folder.resolve()).as_posix()
                if units is None or src.suffix.lower() not in MESH_EXTS:  # convert: STL in metres
                    m = trimesh.load(str(src), force="mesh")
                    m.apply_scale(np.asarray(g.mesh.scale if g.mesh.scale is not None else 1.0, float))
                    rel = f"meshes/{pname}.stl"
                    (folder / "meshes").mkdir(exist_ok=True)
                    m.export(folder / rel)
                    units = "m"
                part = Part(
                    name=pname,
                    stage=stage(lk.name),
                    shape="mesh",
                    mesh=rel.removeprefix("meshes/"),
                    units=units,
                    pose=matrix_to_pose(T),
                )
                if not rel.startswith("meshes/"):
                    (folder / "meshes").mkdir(exist_ok=True)
                    shutil.copy(src, folder / "meshes" / Path(rel).name)
                    part = part.model_copy(update={"mesh": Path(rel).name})
            else:
                continue
            parts.append(part)
    lim = {r: None for r in ("x", "y", "lift")}
    for j in u.robot.joints:
        if j.name in roles and j.limit is not None:
            lim[roles[j.name]] = (float(j.limit.lower), float(j.limit.upper))
    mount = matrix_to_pose(T_of("robot_mount")) if "robot_mount" in links else Pose()
    return Package(
        kind=kind,
        name=name or safe(u.robot.name or urdf_path.stem),
        parts=parts,
        tcps=[Frame(name=t[4:].lstrip("_") or "tcp", pose=matrix_to_pose(T_of(t))) for t in tcps],
        mount=mount,
        x=lim["x"],
        y=lim["y"],
        lift=lim["lift"],
    )


def open_package(data: bytes, dest: Path) -> tuple[Package, Path]:
    """Unpack a package zip into ``dest`` (guarded) and read it: (package, URDF path).
    package.yaml (when there) gives the kind, name and description."""
    from core.urdf_import import import_urdf_zip

    info = import_urdf_zip(data, dest, require_actuated=False)
    meta = {}
    for f in (info.path.parent / "package.yaml", *dest.rglob("package.yaml")):
        if f.is_file():
            meta = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            break
    kind = meta.get("kind") if meta.get("kind") in KINDS else None
    pkg = read_urdf(info.path, kind, meta.get("name"))
    return pkg.model_copy(update={"description": str(meta.get("description", ""))}), info.path


# --- into the project ---------------------------------------------------------------------------


def to_base(p: Package, mesh_rel: str, current: Base) -> Base:
    """The carrier as the External axis tab's Base: bodies, travel, robot mount. Standoffs,
    keep-outs, the carrier frame and mobile settings come from ``current``. ``mesh_rel``:
    the package's meshes folder relative to the project (or repo)."""
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
