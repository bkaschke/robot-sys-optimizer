"""A project's files live in its own folder (v9, issue #12).

``bundle`` (every save): each file the project refers to that is neither inside the project folder
nor shipped with the app (``assets/``, ``examples/`` of the repo) is copied into
``<project>/assets/imported/...`` and the project refers to the copy: the part mesh, the tool's URDF
folder, carrier and stay-out meshes (and their collision meshes), an uploaded robot's URDF folder.
A project then opens on any PC and keeps working when the folder it came from is moved or deleted.

``tolerate_missing`` (every open): a file that cannot be found no longer stops the whole project
from opening; that item is left out and a warning says what to load again.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

from core import schema

REPO = Path(__file__).resolve().parents[1]
SHIPPED = (REPO / "assets", REPO / "examples")
IMPORTED = Path("assets") / "imported"


def _inside(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def is_local(path: Path, project_dir: Path) -> bool:
    """In the project folder or shipped with the app: nothing to copy."""
    return _inside(path, project_dir) or any(_inside(path, d) for d in SHIPPED)


def _free(dst: Path, src: Path) -> Path:
    """``dst``, or ``dst`` with _2, _3... when another (different) file / folder has that name."""
    k, out = 2, dst
    while out.exists():
        same = out.is_file() and src.is_file() and filecmp.cmp(out, src, shallow=False)
        if same or (out.is_dir() and src.is_dir() and _inside(src, out)):
            return out
        out = dst.with_name(f"{dst.stem}_{k}{dst.suffix}")
        k += 1
    return out


def _copy_file(src: Path, project_dir: Path, sub: str) -> Path:
    dst = _free(project_dir / IMPORTED / sub / src.name, src)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def _copy_folder(src: Path, project_dir: Path, sub: str) -> Path:
    dst = _free(project_dir / IMPORTED / sub / src.name, src)
    if not dst.exists():
        shutil.copytree(src, dst)
    return dst


def _external(session, rel: str | None) -> Path | None:
    """The file ``rel`` refers to when it must be copied into the project, else None."""
    if not rel:
        return None
    try:
        path = session.resolve(rel)
    except FileNotFoundError:
        return None
    return None if is_local(path, session.project_dir) else path


def _primitive(session, p: schema.Primitive, sub: str, notes: list[str]) -> schema.Primitive:
    upd = {}
    for field in ("mesh", "collision_mesh"):
        src = _external(session, getattr(p, field, None))
        if src is not None:
            upd[field] = session.relative(_copy_file(src, session.project_dir, sub))
            notes.append(f"{src.name} → {upd[field]}")
    return p.model_copy(update=upd) if upd else p


def bundle(session) -> list[str]:
    """Copy the project's outside files into its folder and point the session at the copies.
    Returns one note per copied item."""
    notes: list[str] = []
    s = session
    part = s.part
    src = _external(s, part.mesh_path) if part is not None else None
    if src is not None:
        old = part.mesh_path
        part.mesh_path = s.relative(_copy_file(src, s.project_dir, "parts"))
        if s.selection and s.selection[0] == old:  # the surface selection belongs to this mesh
            s.selection = (part.mesh_path, s.selection[1])
        notes.append(f"part {src.name} → {part.mesh_path}")
    ee = s.tool
    src = _external(s, ee.urdf) if ee is not None else None
    if src is not None:
        folder = _copy_folder(src.parent, s.project_dir, "tools")
        s.tool = ee.model_copy(update={"urdf": s.relative(folder / src.name)})
        notes.append(f"tool {src.parent.name} → {s.tool.urdf}")
    if s.base.bodies:
        bodies = [_primitive(s, p, "carrier", notes) for p in s.base.bodies]
        if any(a is not b for a, b in zip(bodies, s.base.bodies, strict=True)):
            s.base = s.base.model_copy(update={"bodies": bodies})
    if s.stayouts:
        s.stayouts = [_primitive(s, p, "stayouts", notes) for p in s.stayouts]
    src = _external(s, s.robot_urdf) if s.robot_name is None else None
    if src is not None:
        folder = _copy_folder(src.parent, s.project_dir, "robots")
        s.robot_urdf = s.relative(folder / src.name)
        notes.append(f"robot {src.parent.name} → {s.robot_urdf}")
    return notes


def _exists(session, rel: str | None) -> bool:
    if not rel:
        return True
    try:
        return session.resolve(rel).exists()
    except FileNotFoundError:
        return False


def tolerate_missing(session, project: schema.Project) -> tuple[schema.Project, list[str]]:
    """The project without the items whose files are missing, and a warning for each."""
    warn: list[str] = []
    upd: dict = {}
    pr = project.robot
    if pr is not None and pr.name is None and not _exists(session, pr.urdf):
        warn.append(f"robot URDF `{pr.urdf}` not found: using the UR20; load the robot again (Robot & cell)")
        upd["robot"] = pr.model_copy(
            update={"name": "ur20", "urdf": None, "home_deg": None, "joint_limits_deg": None}
        )
    if project.part is not None and not _exists(session, project.part.mesh):
        warn.append(f"part mesh `{project.part.mesh}` not found: load the part again (Robot & cell → Part)")
        upd["part"] = None
    ee = project.end_effector
    if ee is not None and ee.urdf and not _exists(session, ee.urdf):
        warn.append(f"tool `{ee.name}` ({ee.urdf}) not found: load the tool again (Tool tab)")
        upd["end_effector"] = None
    bodies = []
    for p in project.base.bodies:
        if p.shape == "mesh" and not _exists(session, p.mesh):
            warn.append(f"carrier body `{p.name}` ({p.mesh}) not found: left out (External axis tab)")
            continue
        if p.collision_mesh and not _exists(session, p.collision_mesh):
            p = p.model_copy(update={"collision_mesh": None})  # the visual mesh is checked instead
        bodies.append(p)
    if len(bodies) != len(project.base.bodies) or any(
        a is not b for a, b in zip(bodies, project.base.bodies, strict=False)
    ):
        upd["base"] = project.base.model_copy(update={"bodies": bodies})
    keep = []
    for p in project.stayouts:
        if p.shape == "mesh" and not _exists(session, p.mesh):
            warn.append(f"stay-out zone `{p.name}` ({p.mesh}) not found: left out")
            continue
        keep.append(p)
    if len(keep) != len(project.stayouts):
        upd["stayouts"] = keep
    return (project.model_copy(update=upd) if upd else project), warn
