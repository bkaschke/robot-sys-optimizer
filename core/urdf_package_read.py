"""Reading URDF packages (split from core.urdf_package, which re-exports these names): the
URDF's kinematics at zero travel -> Package parts / frames / travel; package zips."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import trimesh
import yaml

from core.schema import Pose
from core.transforms import make_T, matrix_to_pose
from core.urdf_package import KINDS, MESH_EXTS, MODES, UNIT_SCALE, Collision, Frame, Package, Part, safe


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
        elems = lk.visuals or lk.collisions  # the full geometry (collision may be simplified)
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
    upd: dict = {"description": str(meta.get("description", ""))}
    try:  # the collision settings as authored (package.yaml)
        upd["collision"] = Collision.model_validate(meta.get("collision") or {})
        authored = [x for x in meta.get("parts") or [] if isinstance(x, dict)]
        modes = {str(x.get("name")): x.get("collision") for x in authored}
        upd["parts"] = [
            x.model_copy(update={"collision": modes.get(x.name)}) if modes.get(x.name) in MODES else x
            for x in pkg.parts
        ]
    except ValueError:
        pass
    return pkg.model_copy(update=upd), info.path
