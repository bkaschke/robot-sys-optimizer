"""Package builder <-> the project (v9, issues #1, #5, #11).

``edit_tool`` / ``edit_part`` / ``edit_carrier`` load what the project uses into the builder;
``use_part`` puts a part package into the project (the workpiece mesh, and its collision mesh made
here); ``confirm_use`` asks before *Use in this project* replaces the project's tool / part /
external axis, then closes the builder window.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from core.transforms import pose_to_matrix
from core.urdf_package import (
    UNIT_SCALE,
    Collision,
    Frame,
    Package,
    Part,
    open_package,
    part_collision,
    safe,
    zip_folder,
)
from ui.widgets import notify

ATTACH_STAGE = {"fixed": "fixed", "x_axis": "bridge", "carriage": "carriage", "lift": "lift"}
CLOSE_JS = '<img src="x:" style="display:none" onerror="window.close()">'


def load_zip(b, data: bytes) -> Package:
    """Open package zip bytes into the builder (its meshes into the work folder)."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg, urdf = open_package(data, Path(tmp) / "pkg")
        mesh_dir = b.work / "meshes"
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir)
        mesh_dir.mkdir(parents=True)
        if (urdf.parent / "meshes").is_dir():
            shutil.copytree(urdf.parent / "meshes", mesh_dir, dirs_exist_ok=True)
    b.pkg, b.sel, b.sel_tcp = pkg, 0, 0
    b.show.value = True
    b.load_widgets()
    return pkg


def _fresh_work(b) -> Path:
    mesh_dir = b.work / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True)
    return mesh_dir


def edit_tool(b, e=None) -> bool:
    ee = b.s.tool
    if ee is None or not ee.urdf:
        notify(e, "No tool to edit", "The project's tool has no URDF (Tool tab)")
        return False
    try:
        urdf = b.s.resolve(ee.urdf)
        pkg = load_zip(b, zip_folder(urdf.parent))
    except Exception as err:  # an unusual URDF: report, keep the builder as it was
        notify(e, "Could not load the tool", f"{type(err).__name__}: {err}")
        return False
    tcps = [Frame(name=safe(t.name), pose=t.T_head_tcp) for t in ee.tcps] or pkg.tcps
    b.pkg = pkg.model_copy(update={"kind": "tool", "name": safe(ee.name), "tcps": tcps})
    b.edit_source = ("tool", ee)  # Use keeps the tool's process, extension and contact settings
    b.load_widgets()
    b.status.content = f"Editing the project's tool **{ee.name}**."
    return True


def _package_of(b, rel: str | None) -> Path | None:
    """The package folder (with package.yaml) a project file came from, if any."""
    if not rel:
        return None
    try:
        p = b.s.resolve(rel)
    except FileNotFoundError:
        return None
    for d in (p.parent, *p.parents):
        if (d / "package.yaml").is_file():
            return d
        if d.name in ("assets", "packages"):
            break
    return None


def edit_part(b, e=None) -> bool:
    part = b.s.part
    if part is None:
        notify(e, "No part to edit", "Load a part first (Robot & cell → Part)")
        return False
    src = _package_of(b, part.collision_mesh)
    try:
        if src is not None:  # made here before: its settings come back
            load_zip(b, zip_folder(src))
        else:
            mesh = b.s.resolve(part.mesh_path)
            name = safe(mesh.stem)
            shutil.copy(mesh, _fresh_work(b) / mesh.name)
            b.pkg = Package(
                kind="part",
                name=name,
                parts=[Part(name=name, stage="part", shape="mesh", mesh=mesh.name, units=part.units)],
                collision=Collision(keep=[int(f) for f in part.collision_keep]),
            )
            b.sel = 0
            b.show.value = True
            b.load_widgets()
    except Exception as err:
        notify(e, "Could not load the part", f"{type(err).__name__}: {err}")
        return False
    b.edit_source = ("part", None)
    b.status.content = f"Editing the project's part **{b.pkg.name}**."
    return True


def edit_carrier(b, e=None) -> bool:
    base = b.s.base
    if base.mode != "movable" or base.carrier not in ("rail", "gantry", "mobile"):
        notify(e, "No external axis to edit", "The project has a static base (External axis tab)")
        return False
    mesh_dir = _fresh_work(b)
    parts = []
    try:
        for k, p in enumerate(base.bodies):
            stage = ATTACH_STAGE.get(p.attach, "carriage")
            if base.carrier == "mobile":
                stage = "lift" if stage == "lift" else "chassis"
            upd = {"mesh": None}
            if p.shape == "mesh":
                src = b.s.resolve(p.mesh)
                fname = f"{safe(p.name or f'body{k}')}{src.suffix.lower()}"
                shutil.copy(src, mesh_dir / fname)
                upd["mesh"] = fname
            parts.append(
                Part(
                    name=safe(p.name or f"body{k}"),
                    stage=stage,
                    shape=p.shape,
                    size=p.size,
                    radius=p.radius,
                    height=p.height,
                    units=p.units,
                    pose=p.pose,
                    **upd,
                )
            )
    except Exception as err:
        notify(e, "Could not load the external axis", f"{type(err).__name__}: {err}")
        return False
    j = base.joints
    b.pkg = Package(
        kind=base.carrier,
        name=f"my_{base.carrier}",
        parts=parts,
        mount=base.T_mount,
        x=j.x if base.carrier in ("rail", "gantry") else None,
        y=j.y if base.carrier == "gantry" else None,
        lift=j.lift,
    )
    b.sel = 0
    b.edit_source = ("carrier", None)
    b.show.value = True
    b.load_widgets()
    b.status.content = f"Editing the project's {base.carrier}."
    return True


def _combined(b, dest: Path, collision: bool) -> trimesh.Trimesh | None:
    """All parts in the package frame (metres): visual meshes, or their collision copies."""
    from core.carrier import primitive_mesh

    out = []
    for part in b.pkg.parts:
        m = part_collision(b.pkg, part, dest / "meshes") if collision else None
        if m is None:
            m = primitive_mesh(b.primitive(part)) if part.shape != "mesh" else None
        if m is None:
            m = trimesh.load(str(dest / "meshes" / part.mesh), force="mesh")
            m.apply_scale(UNIT_SCALE[part.units])
        m = m.copy()
        m.apply_transform(pose_to_matrix(part.pose))
        out.append(m)
    return trimesh.util.concatenate(out) if out else None


def use_part(b, dest: Path) -> str:
    """The part package as the project's part: the workpiece mesh (kept as it is when its
    geometry did not change, so targets and the surface selection stay valid) and the collision
    mesh made here."""
    from ui.session import PartState

    s = b.s
    visual = _combined(b, dest, collision=False)
    low = _combined(b, dest, collision=True)
    same = False
    if s.part is not None:
        try:
            cur = s.part_mesh()
            same = cur.vertices.shape == visual.vertices.shape and np.allclose(
                cur.vertices, visual.vertices, atol=1e-9
            )
        except (OSError, ValueError):
            same = False
    if same:
        mesh_rel, units = s.part.mesh_path, s.part.units
        scale = 1.0 / UNIT_SCALE[units]
    else:
        visual.export(dest / "part.stl")
        mesh_rel, units, scale = s.relative(dest / "part.stl"), "m", 1.0
    low = low.copy()
    low.apply_scale(scale)  # in the part mesh's units, like the mesh
    low.export(dest / "part_collision.stl")
    T = s.part.T_world_part if s.part is not None else np.eye(4)
    s.part = PartState(mesh_rel, units, T, [], s.relative(dest / "part_collision.stl"))
    hook = getattr(b, "on_part", None)
    if hook is not None:
        hook()  # the cell redraws the part and its collision settings follow
    kept = "kept" if same else "replaced by the package's"
    return f"Part: the mesh is {kept}, the collision mesh is the one made here (Robot & cell shows it)"


def confirm_use(b, e) -> None:
    """*Use in this project* after a confirmation; on success the builder window closes."""
    from ui import package_actions

    what = {"tool": "tool", "part": "part"}.get(b.pkg.kind, "external axis")
    client = getattr(e, "client", None)
    if client is None:  # scripts / tests: no dialog
        package_actions.use(b, e)
        return
    with client.gui.add_modal("Use in this project?") as modal:
        client.gui.add_markdown(
            f"This replaces the project's **{what}** with **{b.pkg.name}**. Plans made with the old one "
            "become out of date. The package builder window closes afterwards."
        )
        go = client.gui.add_button("Use in this project", color="green")
        stop = client.gui.add_button("Cancel")

    @go.on_click
    def _(ev) -> None:
        modal.close()
        if package_actions.use(b, ev):
            client.gui.add_html(CLOSE_JS)  # the window was opened by the app: it may close itself
            notify(ev, "Loaded into the project", "You can close this window", ok=True)

    stop.on_click(lambda _ev: modal.close())
